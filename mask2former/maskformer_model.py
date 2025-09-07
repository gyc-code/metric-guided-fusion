# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
import os
import matplotlib.pyplot as plt
import numpy as np

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
# cindy add
from detectron2.structures import BitMasks
from detectron2.layers import batched_nms
import math
from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from sklearn.decomposition import PCA


@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        self.local_count = 0

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        if cfg.MODEL.MASK_FORMER.FROZE_BACKBONE:
            backbone.freeze_backbone()
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_preprocess(self, images):
        if not isinstance(images, ImageList):
            print("Please pass the original 'images' object to visualize.")
            return
        # Denormalize
        img_np = images.tensor[0].detach().cpu().numpy()
        mean_np = self.pixel_mean.detach().cpu().numpy().squeeze()
        std_np = self.pixel_std.detach().cpu().numpy().squeeze()
        img_np = (img_np * std_np[:, None, None]) + mean_np[:, None, None]
        img_np = np.clip(img_np, 0, 255).astype(np.uint8).transpose(1, 2, 0)
        return img_np

    def visualize_features_with_pca(self, features: torch.Tensor, save_dir, img_id, n_components=3):
        """
        Args:
            features: 输入特征图，形状 (B, C, H, W)
            n_components: PCA降维后的维度通常为3 对应RGB
        Returns:
            pca_heatmap: 形状 (H, W, 3)，范围 [0,1]
        """
        keys = list(features.keys())
        for k, feat in features.items():
            # 取第 1 个样本，shape = (C, H, W)
            features_k = feat[0]  
            # 转到 (H, W, C) numpy
            features_np = features_k.permute(1, 2, 0).cpu().numpy()
            H, W, C = features_np.shape
            # 展平并标准化
            flattened = features_np.reshape(-1, C)
            flattened = (flattened - flattened.mean(axis=0)) / (flattened.std(axis=0) + 1e-6)
            # PCA 降到 n_components
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(flattened)      # (H*W, n_components)
            pca_map = pca_result.reshape(H, W, n_components)

            # 归一化到 [0,1]
            pca_map -= pca_map.min()
            pca_map /= (pca_map.max() + 1e-6)

            # 如果 n_components==3，就当作 RGB 图；否则用第一个分量的 Jet 颜色映射
            if n_components == 3:
                heatmap_img = pca_map

            # 新建 figure
            fig, ax = plt.subplots(figsize=(6 * W / H, 6))
            ax.imshow(heatmap_img, interpolation='nearest')
            ax.set_title(f"PCA heatmap — feature {k}")
            ax.axis('off')

            # 带 key 的文件名，避免覆盖
            save_name = f"{img_id[:-4]}_{k}_pca.png"
            save_path = os.path.join(save_dir, save_name)
            fig.savefig(save_path, bbox_inches='tight', dpi=100)
            plt.close(fig)

    def visualize_and_save_features_on_images(self, features, img_np, save_dir, img_id):
        """
        Visualize and save all feature maps overlaid on a selected image.
        The original image is denormalized using inside pixel_mean/std buffers.
        """

        for k, feat in features.items():
            # 1) 如果你真的需要保存原图，建议把 key 也加到文件名里
            fig, ax = plt.subplots()
            ax.imshow(img_np)
            ax.axis('off')
            fig.savefig(os.path.join(save_dir, f"{img_id[:-4]}_{k}_orig.png"),
                        bbox_inches='tight')
            plt.close(fig)

            # 2) 计算 channel 平均后的热力图
            #    feat: Tensor of shape [batch, C, H, W]
            heatmap = feat[0].mean(dim=0, keepdim=True)  # [1, H, W]

            # 3) 上采样到原图大小（比如 img_np.shape[:2] = (H0, W0)）
            H0, W0 = img_np.shape[:2]
            heatmap_up = F.interpolate(
                heatmap.unsqueeze(0),                # -> [1,1,H,W] -> [1,1,H0,W0]
                size=(H0, W0),
                mode='bilinear',
                align_corners=False
            )[0, 0]

            # 4) 单独开一个 figure，用 “nearest” 插值看清每个像素块
            fig, ax = plt.subplots(figsize=(6, 6 * H0 / W0))
            ax.imshow(heatmap_up.detach().cpu().numpy(),
                    interpolation='nearest',
                    cmap='jet')
            ax.set_title(f"Feature {k} (upsampled)")
            ax.axis('off')
            fig.savefig(os.path.join(save_dir, f"{img_id[:-4]}_{k}_mean.png"),
                        dpi=100,
                        bbox_inches='tight')
            plt.close(fig)


    def forward(self, batched_inputs, target=False):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        if 'source' in batched_inputs[0]:
            if target:
                images = [x['target']["image"].to(self.device) for x in batched_inputs]
            else:
                images = [x['source']["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)
            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features)

            if self.training:
                # mask classification target
                if "instances" in batched_inputs[0]['source']:
                    gt_instances = [x['source']["instances"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, images)
                else:
                    targets = None

                # bipartite matching-based loss
                losses = self.criterion(outputs, targets)

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        losses.pop(k)
                return losses
            else:
                
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image['source'].get("height", image_size[0])
                    width = input_per_image['source'].get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        if not target:
                            mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                                mask_pred_result, image_size, height, width
                            )
                            mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r
                    
                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r

                return processed_results

        else:
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)
            features = self.backbone(images.tensor)
            if 0: #  only for fuse train with sam/sam2
                if self.backbone.__class__.__name__ != "ImageEncoderViT":
                    features = self.backbone(images.tensor)

                else:
                ### when test, use square bracket to be the input for sam backbone
                    if self.training:
                        #### pad for sam when input size is 512*1024
                        padded_images = []
                        for img in images:
                            # img: Tensor[C, H, W], 这里 H=512, W=1024
                            C, H, W = img.shape
                            assert H == 512 and W == 1024, "for training, input size 3*512*1024"
                            # 在高度维度上重复两遍
                            img_tiled = torch.cat([img, img], dim=1)  # -> [C, 1024, 1024]
                            padded_images.append(img_tiled)

                        # 构建 ImageList
                        pad_images_4sam = ImageList.from_tensors(padded_images, self.size_divisibility)
                        features = self.backbone(pad_images_4sam.tensor)
                        for k in features.keys():
                            feature_height = int(features[k].shape[2] / 2)
                            features[k] = features[k][:, :, 0:feature_height, :].half()

                    else:
                        #### crop image to be two parts for test, when input is 1024*2048
                        left_images = []
                        right_images = []
                        # mid_images = []
                        for img in images:
                            # img: Tensor[C, H, W], 这里 H=1024, W=2048
                            C, H, W = img.shape
                            assert H == 1024 and W == 2048, "for testing, input size 3*1024*2048"
                            left_images.append(img[:, :, :1024])
                            right_images.append(img[:, :, 1024:])
                            # mid_images.append(img[:, :, 512:1536])  # cindy add, for auxiliary backbone

                        # 构建 ImageList
                        left_images = ImageList.from_tensors(left_images, self.size_divisibility)
                        features_left = self.backbone(left_images.tensor)  # cindy add auxiliary backbone
                        right_images = ImageList.from_tensors(right_images, self.size_divisibility)
                        features_right = self.backbone(right_images.tensor)  # cindy add auxiliary backbone
                        # mid_images = ImageList.from_tensors(mid_images, self.size_divisibility)
                        # features_mid = self.backbone(mid_images.tensor)  # cindy add auxiliary backbone

                        features= {}
                        for k in features_left.keys():
                            # Concatenate left and right features along the channel dimension
                            features[k] = torch.cat((features_left[k], features_right[k]), dim=3)
                            # feature_width = features[k].shape[3]
                            # replace_band = int(feature_width / 8)
                            # features[k] = features[k][:, :, :, int(feature_width / 2) - replace_band:int(feature_width / 2) + replace_band]  # Convert to half precision
                            # print(f"features_aux[{k}].shape: {features_aux[k].shape}")  # Debugging output

            # visualize and save features
            if 0:
                print('draw----')
                image_ids = [x["image_id"] for x in batched_inputs]
                save_dir = "./debug_image/backbone_feature_fuse_model_0826/"
                # Create the directory if it doesn't exist
                os.makedirs(save_dir, exist_ok=True)
                img_id = image_ids[0]
                img_np = self.visualize_preprocess(images)
                self.visualize_and_save_features_on_images(features, img_np, save_dir, img_id)
                self.visualize_features_with_pca(features, save_dir, img_id)

            outputs = self.sem_seg_head(features)
            if self.training:
                # mask classification target
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, images)
                else:
                    targets = None

                # bipartite matching-based loss
                losses = self.criterion(outputs, targets)

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        losses.pop(k)
                return losses
            else:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r
                    
                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r

                self.local_count += 1
                return processed_results


    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        # cindy comment: score is 200*8, flatten is 1600 in total, keep top 100 instances (maybe make too much false positive)
        # cindy comment: possible to keep more than 1 big score for same instance
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False) 
        
        ''' replace the above line with the following line to keep better instances'''
        # # Step 1: 计算每个掩码的面积
        # mask_areas = (mask_pred > 0).float().flatten(1).sum(dim=1)
        # # Step 2: 设定最小面积阈值，并根据面积进行筛选
        # min_area_threshold = 10  # 根据数据集和需求调整阈值
        # valid_indices = torch.nonzero(mask_areas > min_area_threshold, as_tuple=False).squeeze(1)  # 保留tensor类型的索引
        # # Step 3: 只对筛选后的有效掩码进行排序，并选出前 top k
        # valid_scores = scores.flatten(0, 1).index_select(0, valid_indices)   # 获取这些掩码对应的分数
        # # 选出前 self.test_topk_per_image 个结果
        # # 检查 valid_scores 的数量是否小于 self.test_topk_per_image
        # num_valid_scores = valid_scores.size(0)  # 有效分数的数量
        # topk_number = min(self.test_topk_per_image, num_valid_scores)  # 动态调整 topk 数量
        # # 选出前 topk_number 个结果
        # scores_per_image, topk_relative_indices = valid_scores.topk(topk_number, sorted=False)

        # # 转换相对索引为原始索引
        # topk_indices = valid_indices.index_select(0, topk_relative_indices)
        ''' above: replace  to keep better instances'''
        
        labels_per_image = labels[topk_indices]
        # cindy comment: topk_indices back to 100 object
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]
        
        ############# cindy find repeat instances
        # unique_values, counts = topk_indices.unique(return_counts=True)
        # # 找到重复值（计数大于1的值）
        # duplicate_values = unique_values[counts > 1]
        # # 创建布尔掩码，标记重复值的位置为True
        # mask = torch.isin(topk_indices, duplicate_values)
        # print(len(topk_indices)-len(unique_values))
        # print(mask)
        # labels_per_image[mask],  scores_per_image[mask]
        ############# cindy find repeat instances

        # mask_pred = mask_pred[unique_indices]

        # np.save("./vehicle_feature/" + str(self.local_count) + "_topk_indices.npy", topk_indices.cpu())
        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()

        # #### cindy add, filter mask  area for one instance where score is low
        # mask_pred_0_filter= torch.clamp(mask_pred, min=0)
        # mask_pred_0_1 = mask_pred_0_filter.sigmoid()
        # result.pred_masks = (mask_pred_0_1 > 0.75).float()  #  try 0.75 also

        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))

        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        # cindy add 
        # result.class_scores = scores_per_image
        # result.mask_scores = mask_scores_per_image
        # result.scores_8 = scores[topk_indices]
        
        # np.save("./vehicle_feature/" + str(self.local_count) + "_labels_per_image.npy", labels_per_image.cpu())
        # np.save("./vehicle_feature/" + str(self.local_count) + "_scores.npy", result.scores.cpu())
        return result
