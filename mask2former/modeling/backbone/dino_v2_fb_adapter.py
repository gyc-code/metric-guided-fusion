# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from mask2former.modeling.backbone.conditional_modules import TAA, ConditionalBottleNeck, TaskScaledNorm

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

"""" use method from https://github.com/IVRL/VTAGML/blob/main/script/model/swin_transformer.py """


class AdapterWrapper(nn.Module):
    """
    为现有的Transformer Block添加Adapter功能的包装器
    """
    def __init__(self, original_block, task_configs=None, use_adapter=True, use_tsn=False):
        super().__init__()
        self.original_block = original_block
        self.task_configs = task_configs
        self.use_adapter = use_adapter
        self.use_tsn = use_tsn
        
        # 获取embedding维度
        self.embed_dim = original_block.attn.qkv.in_features
        
        # 添加Adapter层
        if use_adapter and task_configs is not None:
            self.adapter1 = ConditionalBottleNeck(
                task_configs["hidden_size"], self.embed_dim
            )
            self.adapter2 = ConditionalBottleNeck(
                task_configs["hidden_size"], self.embed_dim
            )
        else:
            self.adapter1 = None
            self.adapter2 = None
        
        # 替换normalization层
        if use_tsn and task_configs is not None:
            self.norm1 = TaskScaledNorm(self.embed_dim, task_configs["hidden_size"])
            self.norm2 = TaskScaledNorm(self.embed_dim, task_configs["hidden_size"])
            # 保存原始norm层的参数
            self.norm1.norm.load_state_dict(original_block.norm1.state_dict())
            self.norm2.norm.load_state_dict(original_block.norm2.state_dict())
        else:
            self.norm1 = original_block.norm1
            self.norm2 = original_block.norm2
    
    def forward(self, x, task_embedding=None):
        """
        Args:
            x: 输入特征 [batch_size, seq_len, dim]
            task_embedding: 任务嵌入 [batch_size, hidden_size]
        """
        # 第一个残差连接 (Attention)
        if self.use_tsn and task_embedding is not None:
            normed1 = self.norm1(x, task_embedding)
        else:
            normed1 = self.norm1(x)
        
        # 注意力计算
        attn_out = self.original_block.attn(normed1)
        
        # 第一个Adapter
        if self.adapter1 is not None and task_embedding is not None:
            adapter_out1 = self.adapter1(task_embedding, attn_out)
            attn_out = attn_out + adapter_out1
        
        x = x + attn_out
        
        # 第二个残差连接 (MLP)
        if self.use_tsn and task_embedding is not None:
            normed2 = self.norm2(x, task_embedding)
        else:
            normed2 = self.norm2(x)
        
        # MLP计算
        mlp_out = self.original_block.mlp(normed2)
        
        # 第二个Adapter
        if self.adapter2 is not None and task_embedding is not None:
            adapter_out2 = self.adapter2(task_embedding, mlp_out)
            mlp_out = mlp_out + adapter_out2
        
        x = x + mlp_out
        
        return x

@BACKBONE_REGISTRY.register()
class DinoV2BaseAdapterBackbone(Backbone):
    """
    DinoV2 Backbone with optional Adapter support
    完全保留原有功能,可选择启用Adapter
    """
    def __init__(self, cfg, input_shape, use_adapter=True, use_tsn=False):
        super().__init__()
        # === 完全保留原有初始化 ===
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').train()
        del self.model.mask_token
        self.qkv_out = None
        self.token_size = 14
        self.factors = {
            'res2': 4,
            'res3': 8,
            'res4': 16,
            'res5': 32,
        }
        self.base = 128
        # self.w, self.h = input_shape
        self._out_features = cfg.MODEL.SWIN.OUT_FEATURES
        # self.model.blocks[11].attn.qkv.register_forward_hook(self.extract_hook())
        
        self.convs = nn.ModuleList([nn.Conv2d(768, self.base*fact//4, kernel_size=1) for fact in self.factors.values()])
                
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
        
        # === 新增Adapter相关配置 ===

        self.use_adapter = cfg.ViT_USE_ADAPTER
        self.use_tsn = use_tsn
        
        # 只有在启用adapter时才初始化相关组件
        if self.use_adapter or self.use_tsn:
            # 获取embedding维度
            self.embed_dim = self.model.blocks[0].attn.qkv.in_features  # 768 for ViT-B

            self.task_embedding_dim = 128  # 可以调整，推荐128或256
            self.norm = nn.LayerNorm(self.embed_dim)

            self.task_configs = {
                "hidden_size": self.task_embedding_dim,
                "max_seq_length": (224 // self.token_size)**2 + 1
            }

            # 任务嵌入维度 (task_embedding维度)
            self.task_embedding = nn.Parameter(torch.randn(1, self.task_embedding_dim) * 0.02)

            # 为每个transformer block添加adapter
            self.adapter_blocks = nn.ModuleList()
            for i, block in enumerate(self.model.blocks):
                # 冻结原始参数
                for param in block.parameters():
                    param.requires_grad = False
                
                # 添加adapter wrapper
                adapter_block = AdapterWrapper(
                    block, 
                    task_configs=self.task_configs,
                    use_adapter=self.use_adapter,
                    use_tsn=self.use_tsn
                )
                self.adapter_blocks.append(adapter_block)
            
            # 冻结其他预训练参数
            for name, param in self.model.named_parameters():
                if 'blocks' not in name:
                    param.requires_grad = False
        else:
            self.adapter_blocks = None
            self.task_embedding = None
            self.task_configs = None

    # === 完全保留原有方法 ===
    # def extract_hook(self):
    #     def hook(module, input, output):
    #         self.qkv_out = output
    #     return hook

    def get_divisible_size(self, w, h):
        return w + (14 - w%14), h+ (14 - h%14)

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 14


    def forward_features_with_adapter(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.model.prepare_tokens_with_masks(x, masks)
        # 获取任务嵌入（扩展到batch维度）
        batch_size = x.shape[0]
        task_embedding = self.task_embedding.expand(batch_size, -1)
        
        for adapter_block in self.adapter_blocks:
            x = adapter_block(x, task_embedding)
            
        x_norm = self.norm(x)
        self.num_register_tokens = 0
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }


    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        # 获取任务嵌入（扩展到batch维度）
        batch_size = x[0].shape[0]
        task_embedding = self.task_embedding.expand(batch_size, -1)
        for adapter_block in self.adapter_blocks:
            xi = adapter_block(xi, task_embedding)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward(self, x):
        w, h = x.shape[-2:]
        dw, dh = self.get_divisible_size(w, h)
        x_inp = F.interpolate(x, size=(dw, dh))
        pw, ph = dw//self.token_size, dh//self.token_size
                
        # 根据是否启用adapter选择特征提取方式
        if self.use_adapter or self.use_tsn:
            feat = self.forward_features_with_adapter(x_inp)
        else:
            feat = self.model.forward_features(x_inp)
        
        patch = feat["x_norm_patchtokens"]
        
        patch = patch.reshape(patch.shape[0], pw, ph, patch.shape[-1]).permute(0, 3, 1, 2)
        
        feat_dict = {}
        for (k, scale), conv in zip(self.factors.items(), self.convs):
            new_patch = F.interpolate(patch, size=(w//scale, h//scale))
            feat_dict[k] = conv(new_patch)
        
        return feat_dict
    
    def get_adapter_parameters(self):
        """获取adapter相关的参数"""
        if not (self.use_adapter or self.use_tsn):
            return []
        
        adapter_params = []
        
        # 任务嵌入参数
        if self.task_embedding is not None:
            adapter_params.append(self.task_embedding)
        
        # Adapter参数
        if hasattr(self, 'adapter_blocks') and self.adapter_blocks is not None:
            for adapter_block in self.adapter_blocks:
                if adapter_block.adapter1 is not None:
                    adapter_params.extend(list(adapter_block.adapter1.parameters()))
                if adapter_block.adapter2 is not None:
                    adapter_params.extend(list(adapter_block.adapter2.parameters()))
                if adapter_block.use_tsn:
                    adapter_params.extend(list(adapter_block.norm1.task_scale.parameters()))
                    adapter_params.extend(list(adapter_block.norm1.task_shift.parameters()))
                    adapter_params.extend(list(adapter_block.norm2.task_scale.parameters()))
                    adapter_params.extend(list(adapter_block.norm2.task_shift.parameters()))
        
        # 特征转换层参数（始终可训练）
        adapter_params.extend(list(self.convs.parameters()))
        
        return adapter_params
    
    def freeze_backbone(self):
        """冻结backbone参数,只训练adapter"""
        if not (self.use_adapter or self.use_tsn):
            print("Warning: No adapter enabled, freezing backbone will freeze all parameters")
            return
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 解冻adapter参数
        for param in self.get_adapter_parameters():
            param.requires_grad = True
    
    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True


# 使用示例
if __name__ == "__main__":
    # 模拟配置
    class MockConfig:
        def __init__(self):
            self.MODEL = MockModel()
    
    class MockModel:
        def __init__(self):
            self.SWIN = MockSwin()
    
    class MockSwin:
        def __init__(self):
            self.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    
    cfg = MockConfig()
    input_shape = (224, 224)
    
    print("=== 测试1: 原始功能（不使用adapter） ===")
    backbone_original = DinoV2BaseBackbone(cfg, input_shape, use_adapter=False, use_tsn=False)
    x = torch.randn(2, 3, 224, 224)
    features_original = backbone_original(x)
    
    print("原始版本输出:")
    for name, feat in features_original.items():
        print(f"{name}: {feat.shape}")
    
    print("\n=== 测试2: 带adapter功能 ===")
    backbone_adapter = DinoV2BaseBackbone(cfg, input_shape, use_adapter=True, use_tsn=True)
    backbone_adapter.freeze_backbone()  # 只训练adapter
    
    features_adapter = backbone_adapter(x)
    
    print("Adapter版本输出:")
    for name, feat in features_adapter.items():
        print(f"{name}: {feat.shape}")
    
    # 验证输出形状一致
    print(f"\n=== 验证输出一致性 ===")
    for name in features_original.keys():
        original_shape = features_original[name].shape
        adapter_shape = features_adapter[name].shape
        print(f"{name}: 原始{original_shape} vs Adapter{adapter_shape} - {'✓' if original_shape == adapter_shape else '✗'}")
    
    # 统计参数量
    total_params_original = sum(p.numel() for p in backbone_original.parameters())
    total_params_adapter = sum(p.numel() for p in backbone_adapter.parameters())
    adapter_params = sum(p.numel() for p in backbone_adapter.get_adapter_parameters())
    trainable_params = sum(p.numel() for p in backbone_adapter.parameters() if p.requires_grad)
    
    print(f"\n=== 参数统计 ===")
    print(f"原始模型参数: {total_params_original:,}")
    print(f"Adapter模型总参数: {total_params_adapter:,}")
    print(f"Adapter相关参数: {adapter_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"Adapter参数占比: {adapter_params/total_params_adapter:.2%}")
    print(f"可训练参数占比: {trainable_params/total_params_adapter:.2%}")