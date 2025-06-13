# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling
from . import vlm_fusion  # register all new VLM fusion models

# config
from .config import add_maskformer2_config
from .config_dual_backbone import add_maskformer2_dual_backbone_config

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

# models
from .maskformer_model import MaskFormer
from .dual_backbone_maskformer_model import DualBackboneMaskFormer
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
from .vlm_fusion.create_fusion_model import (
    CBAM,
    SemanticEdgeFusion
)
from .vlm_fusion.create_fusion_model_1 import (
    ImprovedFusion,
    MultiStageFusion,
    get_sobel_filters,
    conv3x3
)
