# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import cv2
import os

import numpy as np
import pycocotools.mask as mask_util
import torch
from torch.nn import functional as F

import matplotlib.pyplot as plt


from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances, polygons_to_bitmask
from fvcore.transforms.transform import Transform, TransformList

__all__ = ["MaskFormerInstanceDatasetMapper"]

FAR_REGION_RANDOM = True

def cv_read_exr(exrpath):
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    depth = cv2.imread(exrpath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # return (depth*100000).astype(int)
    return (depth*100000).astype(float)


class MaskFormerInstanceDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for instance segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

    def __transform_annotation__(self, transforms, dataset_dict, data_key, image):
        assert "annotations" in dataset_dict[data_key]
        for anno in dataset_dict[data_key]["annotations"]:
            anno.pop("keypoints", None)

        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict[data_key].pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        if len(annos):
            assert "segmentation" in annos[0]
        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            if isinstance(segm, list):
                # polygon
                masks.append(polygons_to_bitmask(segm, *image.shape[:2]))
            elif isinstance(segm, dict):
                # COCO RLE
                masks.append(mask_util.decode(segm))
            elif isinstance(segm, np.ndarray):
                assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                    segm.ndim
                )
                # mask array
                masks.append(segm)
            else:
                raise ValueError(
                    "Cannot convert segmentation of type '{}' to BitMasks!"
                    "Supported types are: polygons as list[list[float] or ndarray],"
                    " COCO-style RLE as a dict, or a binary segmentation mask "
                    " in a 2D numpy array of shape HxW.".format(type(segm))
                )
        return annos, masks
                        
    def __mask_to_instance__(self, image, annos, masks):
        classes = [int(obj["category_id"]) for obj in annos]
        classes = torch.tensor(classes, dtype=torch.int64)
        image_shape = (image.shape[-2], image.shape[-1])  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        # Prepare per-category binary masks
        instances = Instances(image_shape)
        instances.gt_classes = classes
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, image.shape[-2], image.shape[-1]))
        else:
            masks = BitMasks(torch.stack(masks))
            instances.gt_masks = masks.tensor
        return  instances

    def __get_depth_information__(self, dataset_dict, transforms, data_key):
        if data_key == "source":
            depth_file_name = dataset_dict[data_key]["file_name"].replace('/rgb_translated_cityscapes/', '/depth/')\
                .replace('_beauty_', '_depth_').replace('.png','.exr')
            depth_map = cv_read_exr(depth_file_name)

        elif data_key == "target":
            depth_path = './datasets/cityscapes/depth_SwinMTL_train/'
            image_id = dataset_dict[data_key]["image_id"].replace('leftImg8bit.png','depth.npy')
            depth_file_name = depth_path + image_id
            depth_map = np.load(depth_file_name)
            # cv2.imwrite('./datasets/cityscapes/depth_SwinMTL_visual/' + dataset_dict[data_key]["image_id"], depth_map)
        for index, t in enumerate(transforms):
            # print(t.__class__.__name__)
            if 'Color' in t.__class__.__name__:
                continue
            depth_map = t.apply_image(depth_map)
        return depth_map

    def __process_template_for_target__(self, transforms, dataset_dict):
        ''' make template image for target'''
        if "/KITTI360/" in dataset_dict["target"]["file_name"]:
            return None
        else:
            template_mask = utils.read_image('template/cityscapes_ego_car_template.png', format=self.img_format)
            for index, t in enumerate(transforms):
                # print(t.__class__.__name__)
                if 'Color' in t.__class__.__name__:
                    continue
                if 'Resize' in t.__class__.__name__:
                    t.h, t.w = template_mask.shape[:2]
                template_mask = t.apply_image(template_mask)
                
            template_mask = torch.as_tensor(np.ascontiguousarray(template_mask.transpose(2, 0, 1)))
            return template_mask

    def __get_far_region__(self, depth_map, transforms, dataset_dict, data_key, image):
        # for street, more far away place is in the smaller row. the max row is the boundry to cut. here we need to find max_row_index, mean_col_index
        far_region = depth_map > 50
        true_indices = np.where(far_region)
        far_region_transforms = None
        far_region_image = copy.deepcopy(image)

        ''' here, we have far region crop for target and source image'''
        if len(true_indices[0]) > 0:
            ''' far region exist in the image '''  # synthetic depth is regular
            max_row_index = np.max(true_indices[0]) # far region max row
            mean_col_index = int(np.mean(true_indices[1]))

            ''' design a crop box with 300*300 from input image shape(eg,512*1024), and center is the lowest row of the far region '''
            ''' we will crop far region from orignal image '''
            crop_x0 = max(mean_col_index - 256, 0)
            crop_y0 = max(max_row_index - 150, 0)
            crop_x1 = min(mean_col_index + 256, dataset_dict[data_key]["width"]) if crop_x0 != 0 \
                else min(crop_x0 + 512, dataset_dict[data_key]["width"])
            crop_y1 = min(max_row_index + 74, dataset_dict[data_key]["height"]) if crop_y0 != 0 else \
                min(crop_y0 + 224, dataset_dict[data_key]["height"])
            # cv2.imwrite('source_' + dataset_dict[data_key]["image_id"] + '_far.png',((far_region*1)*255))
            far_region_transforms = TransformList([copy.deepcopy(t) for t in transforms if 'Crop' in t.__class__.__name__])
            ''' rewrite the crop param'''
            far_region_transforms[0].y0 = crop_y0
            far_region_transforms[0].x0 = crop_x0
            far_region_transforms[0].w = crop_x1 - crop_x0
            far_region_transforms[0].h = crop_y1 - crop_y0
            far_region_image = far_region_transforms[0].apply_image(far_region_image)
            h, w, c = far_region_image.shape
            crop_info = [crop_x0, crop_y0, crop_x0 + w, crop_y0 + h]
        else:
            ''' if no far region in image, we keep the key far_region_image and make it the same as the processd image'''
            far_region_transforms = transforms
            crop_info = []

        return far_region_image, far_region_transforms, far_region, crop_info

    def visulise(self, far_region_image, full_image, depth_map, far_region, file_name):
        image_name = file_name.split('/')[-1]
        far_region_image_vis = far_region_image.cpu().permute(1,2,0).numpy()
        far_region_image_vis = cv2.cvtColor(far_region_image_vis,cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_name.replace('.png', '_far.png'), far_region_image_vis)

        full_image_vis = full_image.cpu().permute(1,2,0).numpy()
        full_image_vis = cv2.cvtColor(full_image_vis,cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_name.replace('.png', '_full.png'), full_image_vis)

        depth_map_vis = depth_map.astype(int)
        cv2.imwrite(image_name.replace('.png', '_depth.png'), depth_map_vis)
        
        far_region_vis = far_region.astype(int) * 250
        cv2.imwrite(image_name.replace('.png', '_depth_region.png'), far_region_vis)


    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerPanopticDatasetMapper should only be used for training!"
        # dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below, cindy: from orinal code, seems useless
        if "source" in dataset_dict:
            for data_key in dataset_dict.keys():
                image = utils.read_image(dataset_dict[data_key]["file_name"], format=self.img_format)
                utils.check_image_size(dataset_dict[data_key], image)
                aug_input = T.AugInput(image)
                aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
                image = aug_input.image

                # introduce depth label
                # depth_map = self.__get_depth_information__(dataset_dict, transforms, data_key)
                template_mask = self.__process_template_for_target__(transforms, dataset_dict)
                # far_region_image, far_region_transforms, far_region, crop_info = self.__get_far_region__(depth_map, transforms, dataset_dict, data_key, image)
                # far_region_image = torch.as_tensor(np.ascontiguousarray(far_region_image.transpose(2, 0, 1)))

                if data_key == "source":
                    # transform instance masks
                    # dataset_dict_bp = copy.deepcopy(dataset_dict)
                    annos, masks = self.__transform_annotation__(transforms, dataset_dict, data_key, image)
                    # far_region_annos, far_region_masks = self.__transform_annotation__(far_region_transforms, dataset_dict_bp, data_key, far_region_image)
                    # far_region_masks = [torch.from_numpy(np.ascontiguousarray(x)) for x in far_region_masks]
                    masks = [torch.from_numpy(np.ascontiguousarray(x)) for x in masks]

                image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
                if self.size_divisibility > 0:
                    image_size = (image.shape[-2], image.shape[-1])
                    padding_size = [
                        0,
                        self.size_divisibility - image_size[1],
                        0,
                        self.size_divisibility - image_size[0],
                    ]
                    # pad image
                    image = F.pad(image, padding_size, value=128).contiguous()
                    # depth_map = F.pad(depth_map, padding_size, value=128).contiguous()
                    # far_region_image = F.pad(far_region_image, padding_size, value=128).contiguous()
                    if data_key == "source":
                        # pad mask
                        masks = [F.pad(x, padding_size, value=0).contiguous() for x in masks]
                        # far_region_masks = [F.pad(x, padding_size, value=0).contiguous() for x in far_region_masks]
                    else:
                        if template_mask is not None:
                            template_mask = F.pad(template_mask, padding_size, value=128).contiguous()

                if data_key == "target":
                    dataset_dict["target"]['template_img'] = template_mask
                else:
                    instances = self.__mask_to_instance__(image, annos, masks)
                    dataset_dict[data_key]["instances"] = instances
                    # far_region_instances = self.__mask_to_instance__(far_region_image, far_region_annos, far_region_masks)
                    # dataset_dict[data_key]["far_region_instances"] = far_region_instances
                # if 0:
                    # self.visulise(far_region_image, image, depth_map, far_region, dataset_dict[data_key]["file_name"])

                dataset_dict[data_key]["image"] = image
                # dataset_dict[data_key]['depth'] = depth_map
                # dataset_dict[data_key]["far_region_image"] = far_region_image
                # dataset_dict[data_key]['crop_information'] = crop_info
        else:
            image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
            utils.check_image_size(dataset_dict, image)

            aug_input = T.AugInput(image)
            aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
            image = aug_input.image

            # transform instnace masks
            assert "annotations" in dataset_dict
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(obj, transforms, image.shape[:2])
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            if len(annos):
                assert "segmentation" in annos[0]
            segms = [obj["segmentation"] for obj in annos]
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image.shape[:2]))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a binary segmentation mask "
                        " in a 2D numpy array of shape HxW.".format(type(segm))
                    )

            # Pad image and segmentation label here!
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            masks = [torch.from_numpy(np.ascontiguousarray(x)) for x in masks]

            classes = [int(obj["category_id"]) for obj in annos]
            classes = torch.tensor(classes, dtype=torch.int64)

            if self.size_divisibility > 0:
                image_size = (image.shape[-2], image.shape[-1])
                padding_size = [
                    0,
                    self.size_divisibility - image_size[1],
                    0,
                    self.size_divisibility - image_size[0],
                ]
                # pad image
                image = F.pad(image, padding_size, value=128).contiguous()
                # pad mask
                masks = [F.pad(x, padding_size, value=0).contiguous() for x in masks]

            image_shape = (image.shape[-2], image.shape[-1])  # h, w

            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"] = image

            # Prepare per-category binary masks
            instances = Instances(image_shape)
            instances.gt_classes = classes
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, image.shape[-2], image.shape[-1]))
            else:
                masks = BitMasks(torch.stack(masks))
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances
        # print(dataset_dict['source']['image'].shape, dataset_dict['target']['image'].shape)
        return dataset_dict
