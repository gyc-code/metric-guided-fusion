# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hieradet import Hiera 
from .position_encoding import PositionEmbeddingSine
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

@BACKBONE_REGISTRY.register()
class ImageEncoder2(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        ####  for sam2 large 
        # self.trunk = Hiera(
        #     embed_dim=144,
        #     num_heads=2,
        #     stages=[2, 6, 36, 4],
        #     global_att_blocks=[23, 33, 43],
        #     window_pos_embed_bkg_spatial_size=[7, 7],
        #     window_spec=[8, 4, 16, 8])
        # self.neck = FpnNeck(
        #     d_model=256,
        #     backbone_channel_list=[1152, 576, 288, 144],
        #     fpn_top_down_levels=[2, 3],
        #     fpn_interp_model='nearest',
        # )
        #### for sam2 base+
        self.trunk = Hiera(
            embed_dim=112,
            num_heads=2,)
        self.neck = FpnNeck(
            d_model=256,
            backbone_channel_list=[896, 448, 224, 112],
            fpn_top_down_levels=[2, 3],
            fpn_interp_model='nearest',
        )
                
        self.scalp = 1
        assert (
            self.trunk.channel_list == self.neck.backbone_channel_list
        ), f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"
        
        ####  take place temprarily of the config
        self._out_feature_strides = { #placeholder,
            "res2": 16, #placeholder,
            "res3": 16,
            "res4": 16,
            "res5": 16,                                    
        }
        self._out_feature_channels = { #placeholder,
            "res2": 256,
            "res3": 256,
            "res4": 256,
            "res5": 256,   
        }
        self._out_features = ["res2", "res3", "res4", "res5"] #placeholder,
        ####  take place temprarily of the config   
        
        # ==== 在这里加载预训练权重 ====
        ##  only load backbone weights, not the whole model
        # 1) 加载 checkpoint（直接拿它当 state_dict）
        
        if cfg.MODEL.BACKBONE.NAME == "ImageEncoder2":
            # model_weight_file = cfg.MODEL.WEIGHTS
            model_weight_file = "./pretrain/sam2.1_hiera_base_plus.pt"

        else:
            model_weight_file = cfg.MODEL.WEIGHTS_AUX
            
        ckpt = torch.load(model_weight_file, map_location="cuda")
        # 2) strip 掉所有 key 的前缀 "image_encoder."
        prefix = "image_encoder."
        stripped_dict = {}
        for k, v in ckpt['model'].items():  
            if k.startswith(prefix):
                new_k = k[len(prefix):]
                stripped_dict[new_k] = v
            else:
                # 如果有些 key 本来就匹配模型，也可以直接保留：
                stripped_dict[k] = v
            
        print('model dict : ', len(self.state_dict().keys()))
        print('ckpt dict : ', len(stripped_dict.keys()))
        missing, unexpected = self.load_state_dict(stripped_dict, strict=False)
        # 4) 打印一下，让你确认哪些没对上
        print("correct keys:", len(stripped_dict) - len(missing) - len(unexpected) )
        if missing or unexpected:
            print("total key :", len(stripped_dict))
            print("missing keys:", len(missing))    
            print("unexpected keys:", len(unexpected))
            print(f"[Warning] SAM2 ViT backbone loaded with missing keys: {missing}")
            print(f"[Warning] SAM2 ViT backbone loaded with unexpected keys: {unexpected}")
            print('loaded sam2 backbone')
        # ==== 权重加载完毕，后续可以正常 forward ====
    
    def freeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = False
            
            
    def forward(self, sample: torch.Tensor):
        # Forward through backbone
        truck_output = self.trunk(sample)
        # "for input 1024*2014: 
        # outputs[0].shape torch.Size([3, 112, 128, 256])
        # outputs[1].shape torch.Size([3, 224, 64, 128])
        # outputs[2].shape torch.Size([3, 448, 32, 64])
        # outputs[3].shape torch.Size([3, 896, 16, 32])

        features, pos = self.neck(truck_output)
        if self.scalp > 0: #######  remove this, keep all feature
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        
        ###  change return for mask2former
        output = {
            "res2": features[0],
            "res3": features[1],
            "res4": features[2],
            "res5": features[2],
        }
        ####  try only trunk feature
        # output = {
        #     "res2": truck_output[0],
        #     "res3": truck_output[1],
        #     "res4": truck_output[2],
        #     "res5": truck_output[3],
        # }
        return output


class FpnNeck(nn.Module):
    """
    A modified variant of Feature Pyramid Network (FPN) neck
    (we remove output conv and also do bicubic interpolation similar to ViT
    pos embed interpolation)
    """

    def __init__(
        self,
        # position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        """Initialize the neck
        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        """
        super().__init__()
        self.position_encoding = PositionEmbeddingSine(
            num_pos_feats=256,)
        
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        self.d_model = d_model
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        # levels to have top-down features in its outputs
        # e.g. if fpn_top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level.
        if fpn_top_down_levels is None:
            # default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # fpn forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos
