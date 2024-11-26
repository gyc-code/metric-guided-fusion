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
import time


from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances, polygons_to_bitmask
from fvcore.transforms.transform import Transform, TransformList

__all__ = ["MaskFormerInstanceDatasetMapper"]
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
        self.template_mask = utils.read_image('template/cityscapes_ego_car_template.png', format=self.img_format)
        

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

    def __process_template_for_target__(self, transforms, dataset_dict):
        ''' make template image for target'''
        if "/KITTI360/" in dataset_dict["target"]["file_name"]:
            return None
        else:
            new_template_mask = copy.deepcopy(self.template_mask)
            for index, t in enumerate(transforms):
                # print(t.__class__.__name__)
                if 'Color' in t.__class__.__name__:
                    continue
                if 'Resize' in t.__class__.__name__:
                    t.h, t.w = new_template_mask.shape[:2]
                new_template_mask = t.apply_image(new_template_mask)               
            new_template_mask = torch.as_tensor(np.ascontiguousarray(new_template_mask.transpose(2, 0, 1)))
            return new_template_mask

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerPanopticDatasetMapper should only be used for training!"
        if "source" in dataset_dict:            
            for data_key in dataset_dict.keys():
                image = utils.read_image(dataset_dict[data_key]["file_name"], format=self.img_format)
                utils.check_image_size(dataset_dict[data_key], image)
                aug_input = T.AugInput(image)
                aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
                image = aug_input.image
                s = time.time()
                new_template_mask = self.__process_template_for_target__(transforms, dataset_dict)
                if data_key == "source":
                    annos, masks = self.__transform_annotation__(transforms, dataset_dict, data_key, image)
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
                    if data_key == "source":
                        # pad mask
                        masks = [F.pad(x, padding_size, value=0).contiguous() for x in masks]
                    else:
                        if new_template_mask is not None:
                            new_template_mask = F.pad(new_template_mask, padding_size, value=128).contiguous()

                if data_key == "target":
                    dataset_dict["target"]['template_img'] = new_template_mask
                else:
                    instances = self.__mask_to_instance__(image, annos, masks)
                    dataset_dict[data_key]["instances"] = instances
                dataset_dict[data_key]["image"] = image
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
