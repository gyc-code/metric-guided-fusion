# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import concurrent.futures
import logging
import numpy as np
import time
import datetime
import copy
import os
import random
import weakref
from typing import List, Mapping, Optional
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.nn.modules.dropout import _DropoutNd
from timm.models.layers import DropPath
import pathlib
import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.utils.logger import _log_api_usage
import cv2
from detectron2.structures import Instances, Boxes
import torch.nn.functional as F

from .uda_instance_utils import source_instance_paste_to_target_mix, target_instance_paste_to_source_mix,\
remove_ego_car_logo, break_source_target_match, visulize_color_instances, correct_label_by_CLIP

__all__ = ["HookBase", "TrainerBase", "SimpleTrainer", "AMPTrainer"]

VISUL = False
ITERATION_TO_START_UDA = 25000
MINI_BATCH_LOSS = True
USE_CLIP = False

class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.

    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:
    ::
        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        iter += 1
        hook.after_train()

    Notes:
        1. In the hook method, users can access ``self.trainer`` to access more
           properties about the context (e.g., model, current iteration, or config
           if using :class:`DefaultTrainer`).

        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.

           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.

    """

    trainer: "TrainerBase" = None
    """
    A weak reference to the trainer object. Set by the trainer when the hook is registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_backward(self):
        """
        Called after the backward pass of each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass

    def state_dict(self):
        """
        Hooks are stateless by default, but can be made checkpointable by
        implementing `state_dict` and `load_state_dict`.
        """
        return {}


class TrainerBase:
    """
    Base class for iterative trainer with hooks.

    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.

    Attributes:
        iter(int): the current iteration.

        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.

        max_iter(int): The iteration to end training.

        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self) -> None:
        self._hooks: List[HookBase] = []
        self.iter: int = 0
        self.start_iter: int = 0
        self.max_iter: int
        self.storage: EventStorage
        _log_api_usage("trainer." + self.__class__.__name__)

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        self.storage.iter = self.iter
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        # Maintain the invariant that storage.iter == trainer.iter
        # for the entire execution of each step
        self.storage.iter = self.iter

        for h in self._hooks:
            h.before_step()

    def after_backward(self):
        for h in self._hooks:
            h.after_backward()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def run_step(self):
        raise NotImplementedError

    def state_dict(self):
        ret = {"iteration": self.iter}
        hooks_state = {}
        for h in self._hooks:
            sd = h.state_dict()
            if sd:
                name = type(h).__qualname__
                if name in hooks_state:
                    # TODO handle repetitive stateful hooks
                    continue
                hooks_state[name] = sd
        if hooks_state:
            ret["hooks"] = hooks_state
        return ret

    def load_state_dict(self, state_dict):
        logger = logging.getLogger(__name__)
        self.iter = state_dict["iteration"]
        for key, value in state_dict.get("hooks", {}).items():
            for h in self._hooks:
                try:
                    name = type(h).__qualname__
                except AttributeError:
                    continue
                if name == key:
                    h.load_state_dict(value)
                    break
            else:
                logger.warning(f"Cannot find the hook '{key}', its state_dict is ignored.")


class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(
        self,
        model,
        data_loader,
        optimizer,
        gather_metric_period=1,
        zero_grad_before_forward=False,
        async_write_metrics=False,
    ):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
            gather_metric_period: an int. Every gather_metric_period iterations
                the metrics are gathered from all the ranks to rank 0 and logged.
            zero_grad_before_forward: whether to zero the gradients before the forward.
            async_write_metrics: bool. If True, then write metrics asynchronously to improve
                training speed
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.data_loader = data_loader
        # to access the data loader iterator, call `self._data_loader_iter`
        self._data_loader_iter_obj = None
        self.optimizer = optimizer
        self.gather_metric_period = gather_metric_period
        self.zero_grad_before_forward = zero_grad_before_forward
        self.async_write_metrics = async_write_metrics
        # create a thread pool that can execute non critical logic in run_step asynchronically
        # use only 1 worker so tasks will be executred in order of submitting.
        self.concurrent_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())
        if not self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()
        losses.backward()

        self.after_backward()

        if self.async_write_metrics:
            # write metrics asynchronically
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

    @property
    def _data_loader_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_iter_obj is None:
            self._data_loader_iter_obj = iter(self.data_loader)
        return self._data_loader_iter_obj

    def reset_data_loader(self, data_loader_builder):
        """
        Delete and replace the current data loader with a new one, which will be created
        by calling `data_loader_builder` (without argument).
        """
        del self.data_loader
        data_loader = data_loader_builder()
        self.data_loader = data_loader
        self._data_loader_iter_obj = None

    def _write_metrics(
        self,
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
        iter: Optional[int] = None,
    ) -> None:
        logger = logging.getLogger(__name__)

        iter = self.iter if iter is None else iter
        if (iter + 1) % self.gather_metric_period == 0:
            try:
                SimpleTrainer.write_metrics(loss_dict, data_time, iter, prefix)
            except Exception:
                logger.exception("Exception in writing metrics: ")
                raise

    @staticmethod
    def write_metrics(
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        cur_iter: int,
        prefix: str = "",
    ) -> None:
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix for logging keys
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        storage = get_event_storage()
        # Keep track of data time per rank
        storage.put_scalar("rank_data_time", data_time, cur_iter=cur_iter)

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time, cur_iter=cur_iter)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={cur_iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar(
                "{}total_loss".format(prefix), total_losses_reduced, cur_iter=cur_iter
            )
            if len(metrics_dict) > 1:
                storage.put_scalars(cur_iter=cur_iter, **metrics_dict)

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def after_train(self):
        super().after_train()
        self.concurrent_executor.shutdown(wait=True)


class AMPTrainer(SimpleTrainer):
    """
    Like :class:`SimpleTrainer`, but uses PyTorch's native automatic mixed precision
    in the training loop.
    """

    def __init__(
        self,
        model,
        data_loader,
        optimizer,
        gather_metric_period=1,
        zero_grad_before_forward=False,
        grad_scaler=None,
        precision: torch.dtype = torch.float16,
        log_grad_scaler: bool = False,
        async_write_metrics=False,
    ):
        """
        Args:
            model, data_loader, optimizer, gather_metric_period, zero_grad_before_forward,
                async_write_metrics: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
            precision: torch.dtype as the target precision to cast to in computations
        """
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        super().__init__(
            model, data_loader, optimizer, gather_metric_period, zero_grad_before_forward
        )

        if grad_scaler is None:
            from torch.cuda.amp import GradScaler

            grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler
        self.precision = precision
        self.log_grad_scaler = log_grad_scaler
        ''' cindy add ema'''
        self.source_rare_class_samples = []
        # self.target_rare_class_samples = []
        self.local_iter = 0
        self.alpha = 0.999
        timestamp = time.time()
        human_readable_time = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H:%M:%S')
        self.folder_name = './output/debug_in_img_' + human_readable_time+ '/'
        dir = pathlib.Path(self.folder_name)
        dir.mkdir(parents=True, exist_ok=True)

    def _init_ema_weights(self):
        self.ema_model = copy.deepcopy(self.model).eval() # init , no weight load

    def __print_label__(self, instances, image, save_path):
        image_copy = copy.deepcopy(image.permute(1,2,0).numpy().astype(np.uint8))
        cv2.imwrite(save_path, image_copy)
        image_read = cv2.imread(save_path)

        pred_masks = instances.pred_masks
        scores = instances.scores
        pred_classes = instances.pred_classes
        for i, pred_mask in enumerate(pred_masks):
            score = scores[i].item()
            class_id = pred_classes[i].item()
            text = f"s{score:.2f}|c{class_id}"
            # 获取 pred_mask 的右上角位置
            mask_np = pred_mask.cpu().numpy()
            non_zero_indices = torch.nonzero(pred_mask)
            if len(non_zero_indices) == 0:
                continue
            y_min, x_min = non_zero_indices.min(dim=0)[0]
            y_max, x_max = non_zero_indices.max(dim=0)[0]
            y_center = int((y_min + y_max) // 2)
            x_center = int((x_min + x_max) // 2)
            image_text = cv2.putText(image_read, text, (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(save_path, image_text)


    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)  
        #   try update this way to speedup
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(alpha_teacher).add_(param.data, alpha=1 - alpha_teacher)


    # def second_pseudo_labels_far_region(self, data, pseudo_labels, index):
    #     ''' use far region image to generate pseudo label for far region'''
    #     data_copy_plabel_i = copy.deepcopy(data[index])
    #     if len(data[index]['target']['crop_information']) > 0:
    #         '''  try use far image crop to get pseudo label ,
    #         far crop image 500*200, resize to 2 times to inference, get plabel and
    #         resize the instance pred_mask to 600*200, paste to the original image shape according crop imformation(x0 y0 x1 y1)'''
    #         target = data[index]['target']
    #         # 1. 获取 far_region_image
    #         far_region_image = target['far_region_image']
    #         # 2. 将图片放大两倍
    #         far_region_image = far_region_image.float()
    #         enlarged_image = F.interpolate(far_region_image.unsqueeze(0), scale_factor=2, mode='bilinear', align_corners=False).squeeze(0)
    #         file_id = data[index]['target']['image_id'].split('.')[0]
    #         # 3. 送进 self.ema_model 进行推理
    #         data_copy_plabel_i['target']['image'] = enlarged_image
    #         pseudo_labels_second = self.ema_model([data_copy_plabel_i], target=True) #cindy, image size change due to size_divisibility(32)
    #         # 4. 修改输出结果中的 instance 的 image_height 和 image_width
    #         pseudo_labels_second_instance = pseudo_labels_second[0]['instances']
    #         pseudo_labels_second_instance = pseudo_labels_second_instance[pseudo_labels_second_instance.scores.cpu() > 0.8]
    #         # save_path = self.folder_name + file_id + '_' + str(self.local_iter)+ '_enlarge_image.jpg'
    #         # # cv2.imwrite(save_path.replace('_enlarge_image.jpg', '_tar_ori.jpg'), enlarged_image.permute(1,2,0).numpy())
    #         # self.__print_label__(pseudo_labels_second_instance, enlarged_image, save_path)

    #         # 5. 将 instance 的 pred_masks resize 且 pad 回原图
    #         if len(pseudo_labels_second_instance) > 0:
    #             pred_masks = pseudo_labels_second_instance.pred_masks
    #             resized_pred_masks = F.interpolate(pred_masks.unsqueeze(1).float(), size=(far_region_image.shape[1], far_region_image.shape[2]), mode='bilinear', align_corners=False).squeeze(1)
    #             # 6. 创建空白的 mask，并根据 crop_information 放回对应位置
    #             crop_info = target['crop_information']
    #             x0, y0, x1, y1 = crop_info
    #             final_masks = []
    #             final_scores = []
    #             final_classes = []
    #             for j, resized_pred_mask in enumerate(resized_pred_masks):
    #                 mask_tmp = torch.zeros((target['image'].shape[1], target['image'].shape[2]), dtype=resized_pred_mask.dtype)
    #                 keep = self.filter_instances(resized_pred_mask, pseudo_labels[index]['instances'], crop_info, 0.75)
    #                 if keep is False:
    #                     continue
    #                 # 确保 resized_pred_mask 的大小与 crop_info 一致
    #                 if resized_pred_mask.shape[0] != (y1 - y0) or resized_pred_mask.shape[1] != (x1 - x0):
    #                     print(f"Error: resized_pred_mask size {resized_pred_mask.shape} does not match crop_info size {(y1 - y0, x1 - x0)}")
    #                     continue
    #                 mask_tmp[y0:y1, x0:x1] = resized_pred_mask
    #                 final_masks.append(mask_tmp)
    #                 final_scores.append(pseudo_labels_second_instance.scores[j].item())
    #                 final_classes.append(pseudo_labels_second_instance.pred_classes[j].item())

    #             # 7. 更新 output 中的 pred_masks

    #             if len(final_masks) > 0:
    #                 final_masks = torch.stack(final_masks).cuda()
    #                 final_scores = torch.tensor(final_scores).cuda()
    #                 final_classes = torch.tensor(final_classes).cuda()
    #                 boxes = torch.tensor(np.zeros((len(final_masks), 4)))
    #                 ct, ht, wt = data[index]['target']['image'].shape
    #                 second_ins = Instances((ht, wt))
    #                 second_ins.pred_classes = final_classes
    #                 second_ins.pred_masks = final_masks
    #                 second_ins.scores = final_scores
    #                 second_ins.pred_boxes = Boxes(boxes)
    #                 return second_ins
    #             else:
    #                 return None
    #         else:
    #             return None
    #     else:
    #         return None


    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start
        self.local_iter += 1
        if 'source' in data[0] and self.local_iter > ITERATION_TO_START_UDA:# cindy add 
            
            batch_size = len(data)
            assert batch_size == 3, f"Batch size must be 3, but got {batch_size}"
            
            # Init/update ema model
            if self.local_iter == ITERATION_TO_START_UDA + 1:
                self._init_ema_weights()
            self._update_ema(self.local_iter)
            
            ''' cindy : generate pseudo label for target and do mix '''
            with torch.no_grad():
                ''' cindy: forward with taget data in EMA '''
                # for m in self.ema_model.modules():
                #     m.training = False
                self.ema_model.training = False
                # Generate pseudo-label
                pseudo_labels = self.ema_model(data, target=True) 
                pseudo_instances_num_list = []
                for i in range(len(pseudo_labels)): # filter pseudo instances which score are low
                    template_img = data[i]['target']['template_img']
                    pseudo_instances = pseudo_labels[i]['instances']
                    if USE_CLIP:
                        '''if the score(=class*mask)>0.5,I want to correct the score by CLIP, the class score will be updated by 
                        CLIP output,and then update the score of the instance'''
                        pseudo_labels[i]['instances'] = pseudo_instances[pseudo_instances.scores.cpu() > 0.5]
                        correct_label_by_CLIP(pseudo_labels[i]['instances'], data[i]['target']['image'])

                        pseudo_instances.remove('class_scores')
                        pseudo_instances.remove('mask_scores')
                        pseudo_instances.remove('scores_8')
                        # pseudo_instances.scores = pseudo_instances.class_scores * pseudo_instances.mask_scores
                        
                    pseudo_labels[i]['instances'] = pseudo_instances[pseudo_instances.scores.cpu() > 0.9]

                    if VISUL:
                        file_id = data[i]['target']['image_id'].split('.')[0]
                        save_path = self.folder_name + file_id + '_' + str(self.local_iter)+ '_tar.jpg'
                        self.__print_label__(pseudo_instances, data[i]['target']['image'], save_path)
                        save_path_1 = self.folder_name + file_id + '_' + str(self.local_iter)+ '_template.jpg'
                        cv2.imwrite(save_path_1, template_img.permute(1,2,0).numpy().astype(np.uint8))


                    '''Add pseudo label mask to filter out the object in image and
                    do the inference again to get second p label, fuse two p label  -> only get back 1-2 instances'''
                    ##### Initialize a boolean mask of the same size as the template image
                    # second_instances = self.second_pseudo_labels_mask_out(data, i)
                    # second_instances = self.second_pseudo_labels_far_region(data, pseudo_labels, i)
                    # second_instances = None

                    # if second_instances is not None:
                    #     fuse_instance = Instances.cat([second_instances, pseudo_labels[i]['instances']])
                    #     # fuse_instance = second_instances
                    # else:
                    #     fuse_instance = pseudo_labels[i]['instances']
                    #############################################
                    update_pseudo_label = remove_ego_car_logo(pseudo_labels[i]['instances'], template_img)
                    if update_pseudo_label is None:
                        pseudo_instances_num_list.append(0)
                        continue
                    else:
                        pseudo_labels[i]['instances'] = update_pseudo_label
                        pseudo_instances_num_list.append(len(pseudo_labels[i]['instances']._fields))
                    data[i]['target']['instances'] = pseudo_labels[i]['instances']
                del pseudo_labels
            # pseudo instance use to mix
            any_greater_than_zero = any(x > 0 for x in pseudo_instances_num_list)
            if any_greater_than_zero:
                # data_ori_0 = copy.deepcopy(data[0])
                # data_copy = copy.deepcopy(data)
                if VISUL:
                    self.model.eval()
                    data[0]['source']['height'] = 1024
                    data[0]['source']['width'] = 1024

                    source = self.model(data)
                    source_instances = source[0]['instances']
                    source_instances = source_instances[source_instances.scores.cpu() > 0.8]

                    source_instances_img = visulize_color_instances(source_instances)
                    file_id = data[0]['target']['image_id'].split('.')[0]
                    cv2.imwrite(self.folder_name + file_id + '_' + str(self.local_iter)+ '_source_inference_color_instance.jpg', source_instances_img)
                    del source, source_instances, source_instances_img
                   
                '''since use mini_batch_loss,one batch data=data0: source+ data1: s2t+ data2: t2s
                data[1] : source_instance_paste_to_target_mix, data[2] :target_instance_paste_to_source_mix
                batch_size need to be 3'''
                if pseudo_instances_num_list[1] > 0:
                    data[1], self.source_rare_class_samples = source_instance_paste_to_target_mix(data[1], self.local_iter, self.folder_name, self.source_rare_class_samples)
                    # data_copy[i], self.target_rare_class_samples = target_instance_paste_to_source_mix(data_copy[i], pseudo_labels_copy[i], self.local_iter, self.folder_name, self.target_rare_class_samples)
                if pseudo_instances_num_list[2] > 0:
                    data[2] = target_instance_paste_to_source_mix(data[2], self.local_iter, self.folder_name)

                # if not MINI_BATCH_LOSS: ## cindy: cancle this train
                #     self.model.training = True
                #     mix_loss_dict = self.model(data)
                #     if isinstance(mix_loss_dict, torch.Tensor):
                #         s2t_mix_losses = mix_loss_dict
                #         loss_dict = {"total_mix_loss": mix_loss_dict}
                #     else:
                #         s2t_mix_losses = sum(mix_loss_dict.values())

                #     ''' cindy: train with target2source mix data '''
                #     t2s_mix_loss_dict = self.model(data_copy)
                #     if isinstance(t2s_mix_loss_dict, torch.Tensor):
                #         t2s_mix_losses = t2s_mix_loss_dict
                #         loss_dict = {"total_mix_loss": t2s_mix_loss_dict}
                #     else:
                #         t2s_mix_losses = sum(t2s_mix_loss_dict.values())
                #     # unite_loss = 0.6 * t2s_mix_losses + 0.4 * s2t_mix_losses
                #     # unite_loss = s2t_mix_losses # ablation
                #     # unite_loss = t2s_mix_losses # ablation
                #     # unite_loss = 0.5 * source_losses + 0.25 * t2s_mix_losses + 0.25 * s2t_mix_losses
                #     unite_loss = 0.5 * t2s_mix_losses + 0.5 * s2t_mix_losses
                #     # unite_loss = t2s_mix_losses
                #     unite_loss_dict = t2s_mix_loss_dict
                # else:
                ''' use mini batch loss ,one batch data=data0: source+ data1: s2t+ data2: t2s '''
                self.optimizer.zero_grad()
                self.model.train()
                with autocast(dtype=self.precision):
                    unite_loss_dict = self.model(data)
                    if isinstance(unite_loss_dict, torch.Tensor):
                        unite_loss = unite_loss_dict
                        unite_loss_dict = {"total_source_loss": unite_loss_dict}
                    else:
                        unite_loss = sum(unite_loss_dict.values())

                self.grad_scaler.scale(unite_loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                if self.async_write_metrics:
                    # write metrics asynchronically
                    self.concurrent_executor.submit(self._write_metrics, unite_loss_dict, data_time, iter=self.iter)
                else:
                    self._write_metrics(unite_loss_dict, data_time)

            if self.log_grad_scaler:
                storage = get_event_storage()
                storage.put_scalar("[metric]grad_scaler", self.grad_scaler.get_scale())
            self.after_backward()
        else:
            if 'source' in data[0]: # when local_iter < ITERATION_TO_START_UDA, also use only source training
                data = [x['source'] for x in data]
                # for x in data:
                #     if 0:
                #     # if random.randint(0, 1):
                #         x['image'] = x['far_region_image']
                #         x['instances'] = x['far_region_instances']

            if self.zero_grad_before_forward:
                self.optimizer.zero_grad()
            with autocast(dtype=self.precision):
                loss_dict = self.model(data)
                if isinstance(loss_dict, torch.Tensor):
                    losses = loss_dict
                    loss_dict = {"total_loss": loss_dict}
                else:
                    losses = sum(loss_dict.values())

            if not self.zero_grad_before_forward:
                self.optimizer.zero_grad()

            self.grad_scaler.scale(losses).backward()

            if self.log_grad_scaler:
                storage = get_event_storage()
                storage.put_scalar("[metric]grad_scaler", self.grad_scaler.get_scale())

            self.after_backward()

            if self.async_write_metrics:
                # write metrics asynchronically
                self.concurrent_executor.submit(
                    self._write_metrics, loss_dict, data_time, iter=self.iter
                )
            else:
                self._write_metrics(loss_dict, data_time)

            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

    def state_dict(self):
        ret = super().state_dict()
        ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.grad_scaler.load_state_dict(state_dict["grad_scaler"])
