# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

@BACKBONE_REGISTRY.register()
class DinoV3Backbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        # 1) 你的本地 DINOv3 仓库路径（确保里面有 hubconf.py）
        REPO_DIR = "/home/yguo/Documents/other/dinov3"
        # 2) 你下载好的权重路径（举例 vit7b16）
        # ckpt = "/home/yguo/.cache/torch/hub/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
        # ckpt = "/home/yguo/.cache/torch/hub/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        # ckpt = "/home/yguo/.cache/torch/hub/checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
        # ckpt = "/home/yguo/.cache/torch/hub/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        # emb_dim = 1280  #4096-7B, 1536-giant2, 1280-huge2,  1024-L, 768-base

        ckpt = cfg.MODEL.WEIGHTS_BACKBONE
        emb_dim = cfg.MODEL.BACKBONE_EMB_DIM
        # model_name = 'dinov3_vit7b16'
        model_name = cfg.MODEL.WEIGHTS_BACKBONE_NAME #'dinov3_vith16plus'

        self.half_flag = True if model_name is 'dinov3_vit7b16' else False
        self.model = torch.hub.load(
            REPO_DIR,
            model_name, 
            source='local',
            weights=ckpt, 
            pretrained=False,  
        )
        ###########  load weight
        ckpt = torch.load(ckpt, map_location='cpu')
        msg = self.model.load_state_dict(ckpt, strict=False) ##  cindy: load again weight to make sure correct weight
        print("missing:", len(msg.missing_keys), "unexpected:", len(msg.unexpected_keys))
        ##########

        self.model.train()
        # self.model = torch.hub.load(REPO_DIR, 'dinov3_vit7b16', source='local', weights=dinov3_vith16plus).train()
        self.qkv_out = None
        self.token_size = 16
        self.factors = {
            'res2': 4,
            'res3': 8,
            'res4': 16,
            'res5': 32,
        }
        self.base=128
        # self.w, self.h = input_shape
        self._out_features = cfg.MODEL.SWIN.OUT_FEATURES
        # self.model.blocks[11].attn.qkv.register_forward_hook(self.extract_hook())

        self.convs = nn.ModuleList([nn.Conv2d(emb_dim, self.base*fact//4, kernel_size=1) for fact in self.factors.values()])
        
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": 128,
            "res3": 256,
            "res4": 512,
            "res5": 1024,
        }

    def get_divisible_size(self, w, h):
        return w + (16 - w%16), h+ (16 - h%16)

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 16

    def forward(self, x):
        w, h = x.shape[-2:]
        dw, dh = self.get_divisible_size(w, h)
        x_inp = F.interpolate(x, size=(dw, dh))
        pw, ph = dw//self.token_size, dh//self.token_size

        if self.half_flag:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                feat = self.model.forward_features(x_inp.half())
        else:
            feat = self.model.forward_features(x_inp)

        patch = feat["x_norm_patchtokens"]

        patch = patch.reshape(patch.shape[0], pw, ph, patch.shape[-1]).permute(0, 3, 1, 2)

        feat_dict = {}
        for (k, scale), conv in zip(self.factors.items(), self.convs):
            new_patch = F.interpolate(patch, size=(w//scale, h//scale))
            feat_dict[k] = conv(new_patch)

        return feat_dict
        
    def freeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = False
            
