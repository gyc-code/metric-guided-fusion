# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional
import logging
from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from iopath.common.file_io import g_pathmgr
import math

from .hieradet import Hiera 
from .position_encoding import PositionEmbeddingSine
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from mask2former.modeling.backbone.conditional_modules import TAA, ConditionalBottleNeck, TaskScaledNorm
from .utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)
from .sam2_utils import DropPath, MLP
import numbers


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
        training: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)
        self.training = training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            training=self.training,
        )
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False, training=True) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=training)
    return attn_weight @ value


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
        # Adapter相关参数
        task_configs: dict = None,
        use_adapter: bool = False,
        use_tsn: bool = False,
        training: bool = True,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.task_configs = task_configs
        self.use_adapter = use_adapter
        self.use_tsn = use_tsn
        
        # 归一化层
        if use_tsn and task_configs is not None:
            self.norm1 = TaskScaledNorm(dim, task_configs["hidden_size"])
            self.norm2 = TaskScaledNorm(dim_out, task_configs["hidden_size"])
        else:
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim_out)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
            training=training,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)
        
        # Adapter层
        if use_adapter and task_configs is not None:
            self.adapter1 = ConditionalBottleNeck(
                task_configs["hidden_size"], dim_out
            )
            self.adapter2 = ConditionalBottleNeck(
                task_configs["hidden_size"], dim_out
            )
        else:
            self.adapter1 = None
            self.adapter2 = None

    def forward(self, x: torch.Tensor, task_embedding: torch.Tensor = None) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        
        # 第一个归一化
        if self.use_tsn and task_embedding is not None:
            x = self.norm1(x, task_embedding)
        else:
            x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        
        # 第一个Adapter
        if self.adapter1 is not None and task_embedding is not None:
            B, H, W, C = x.shape
            x_reshaped = x.view(B, H * W, C)
            task_emb = task_embedding.expand(B, -1)
            adapter_out1 = self.adapter1(task_embedding, x_reshaped)
            adapter_out1 = adapter_out1.view(B, H, W, C)
            x = x + adapter_out1
            
        
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        
        # 第二个归一化
        if self.use_tsn and task_embedding is not None:
            normed_x = self.norm2(x, task_embedding)
        else:
            normed_x = self.norm2(x)
        
        # MLP
        mlp_out = self.mlp(normed_x)
        
        # 第二个Adapter
        if self.adapter2 is not None and task_embedding is not None:
            B, H, W, C = mlp_out.shape
            mlp_out_reshaped = mlp_out.view(B, H * W, C)
            adapter_out2 = self.adapter2(task_embedding, mlp_out_reshaped)
            adapter_out2 = adapter_out2.view(B, H, W, C)
            mlp_out = mlp_out + adapter_out2
        
        x = x + self.drop_path(mlp_out)
        return x


class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """
      
    def __init__(
        self,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        weights_path=None,
        return_interm_layers=True,  # return feats from every stage
        # Adapter相关参数
        task_configs: dict = None,
        use_adapter: bool = False,
        use_tsn: bool = False,
        training: bool = True,
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec
        self.task_configs = task_configs
        self.use_adapter = use_adapter
        self.use_tsn = use_tsn

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )
        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
                task_configs=task_configs,
                use_adapter=use_adapter,
                use_tsn=use_tsn,
                training=training,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

        if weights_path is not None:
            with g_pathmgr.open(weights_path, "rb") as f:
                chkpt = torch.load(f, map_location="cpu")
            logging.info("loading Hiera", self.load_state_dict(chkpt, strict=False))

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor, task_embedding: torch.Tensor = None) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, task_embedding)
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs

    def get_layer_id(self, layer_name):
        # https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        num_layers = self.get_num_layers()

        if layer_name.find("rel_pos") != -1:
            return num_layers + 1
        elif layer_name.find("pos_embed") != -1:
            return 0
        elif layer_name.find("patch_embed") != -1:
            return 0
        elif layer_name.find("blocks") != -1:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers + 1

    def get_num_layers(self) -> int:
        return len(self.blocks)


@BACKBONE_REGISTRY.register()
class ImageEncoder2Adapter(Backbone):
    """
    SAM2 Backbone with optional Adapter support
    """
    def __init__(self, cfg, input_shape, use_adapter=True, use_tsn=False):
        super().__init__()
        
        # === 完全保留原有初始化 ===
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
        
        # === 新增Adapter相关配置 ===
        self.use_adapter = use_adapter
        self.use_tsn = use_tsn
        
        # 任务配置（简化为单任务）
        if self.use_adapter or self.use_tsn:
            self.task_configs = {
                "hidden_size": 256,  # 使用FPN的d_model作为hidden_size
                "max_seq_length": 256 * 256  # 假设最大feature map尺寸
            }
            
            # 实例分割任务嵌入（固定为单个任务）
            self.task_embedding = nn.Parameter(torch.randn(1, 256) * 0.02)
        else:
            self.task_configs = None
            self.task_embedding = None
        
        #### for sam2 base+
        self.trunk = Hiera(
            embed_dim=112,
            num_heads=2,
            task_configs=self.task_configs,
            use_adapter=self.use_adapter,
            use_tsn=self.use_tsn,
            training=self.training,
        )
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
        
        # === 完全保留原有权重加载逻辑 ===
        # ==== 在这里加载预训练权重 ====
        ##  only load backbone weights, not the whole model
        # 1) 加载 checkpoint（直接拿它当 state_dict）
        
        if cfg.MODEL.BACKBONE.NAME == "ImageEncoder2Adapter":
            model_weight_file = cfg.MODEL.WEIGHTS
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
        
        # 只加载非adapter相关的权重
        self.use_adapter = cfg.ViT_USE_ADAPTER
        if self.use_adapter or self.use_tsn:
            # 冻结trunk的原始参数
            for param in self.trunk.parameters():
                param.requires_grad = False
            
            # 过滤掉adapter相关的key
            filtered_dict = {}
            for k, v in stripped_dict.items():
                if not any(adapter_key in k for adapter_key in ['adapter', 'task_scale', 'task_shift']):
                    filtered_dict[k] = v
            stripped_dict = filtered_dict
            
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
    
    def get_adapter_parameters(self):
        """获取adapter相关的参数"""
        if not (self.use_adapter or self.use_tsn):
            return []
        
        adapter_params = []
        
        # 任务嵌入参数
        if self.task_embedding is not None:
            adapter_params.append(self.task_embedding)
        
        # Trunk中的Adapter参数
        for block in self.trunk.blocks:
            if hasattr(block, 'adapter1') and block.adapter1 is not None:
                adapter_params.extend(list(block.adapter1.parameters()))
            if hasattr(block, 'adapter2') and block.adapter2 is not None:
                adapter_params.extend(list(block.adapter2.parameters()))
            if hasattr(block, 'use_tsn') and block.use_tsn:
                if hasattr(block.norm1, 'task_scale'):
                    adapter_params.extend(list(block.norm1.task_scale.parameters()))
                    adapter_params.extend(list(block.norm1.task_shift.parameters()))
                if hasattr(block.norm2, 'task_scale'):
                    adapter_params.extend(list(block.norm2.task_scale.parameters()))
                    adapter_params.extend(list(block.norm2.task_shift.parameters()))
        
        # Neck参数（始终可训练）
        adapter_params.extend(list(self.neck.parameters()))
        
        return adapter_params
    
    def freeze_backbone(self):
        """冻结backbone参数只训练adapter"""
        if not (self.use_adapter or self.use_tsn):
            # 保留原有的freeze_backbone行为
            for param in self.parameters():
                param.requires_grad = False
            return
        
        # 新的adapter模式下的冻结策略
        for param in self.trunk.parameters():
            param.requires_grad = False
        
        # 解冻adapter参数
        for param in self.get_adapter_parameters():
            param.requires_grad = True
    
    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
            
    def forward(self, sample: torch.Tensor):
        """
        完全保留原有forward接口
        """
        # 获取任务嵌入
        if self.use_adapter or self.use_tsn:
            batch_size = sample.shape[0]
            task_embedding = self.task_embedding.expand(batch_size, -1)
        else:
            task_embedding = None
        
        # Forward through backbone
        truck_output = self.trunk(sample, task_embedding)
        # "for input 512*2014: 
        # outputs[0].shape torch.Size([3, 144, 128, 256])
        # outputs[1].shape torch.Size([3, 288, 64, 128])
        # outputs[2].shape torch.Size([3, 576, 32, 64])
        # outputs[3].shape torch.Size([3, 1152, 16, 32])

        features, pos = self.neck(truck_output)
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        # output is dict_keys(['vision_features'：torch.Size([3, 256, 32, 64]), 
        # 'vision_pos_enc':
        # features['vision_pos_enc'][0].shape torch.Size([3, 256, 128, 256]) 
        # features['vision_pos_enc'][1].shape torch.Size([3, 256, 64, 128]) 
        # features['vision_pos_enc'][2].shape torch.Size([3, 256, 32, 64]), 
        # 'backbone_fpn':
        # features['backbone_fpn'][0].shape torch.Size([3, 256, 128, 256]) 
        # features['backbone_fpn'][1].shape torch.Size([3, 256, 64, 128])  
        # features['backbone_fpn'][2].shape torch.Size([3, 256, 32, 64])])
        
        ###  change return for mask2former
        output = {
            "res2": features[0],
            "res3": features[1],
            "res4": features[2],
            "res5": features[2],}
        
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
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        self.d_model = d_model
        self.position_encoding = PositionEmbeddingSine(
            num_pos_feats=self.d_model//2,)
                
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


# 使用示例
if __name__ == "__main__":
    # 模拟配置
    class MockConfig:
        def __init__(self):
            self.MODEL = MockModel()
    
    class MockModel:
        def __init__(self):
            self.BACKBONE = MockBackbone()
            self.WEIGHTS = "path/to/sam2_weights.pth"
            self.WEIGHTS_AUX = "path/to/sam2_weights_aux.pth"
    
    class MockBackbone:
        def __init__(self):
            self.NAME = "ImageEncoder2"
    
    cfg = MockConfig()
    input_shape = (512, 1024)
    
    print("=== 测试1: 原始SAM2功能不使用adapter ===")
    try:
        backbone_original = ImageEncoder2(cfg, input_shape, use_adapter=False, use_tsn=False)
        x = torch.randn(2, 3, 512, 1024)
        features_original = backbone_original(x)
        
        print("原始SAM2版本输出:")
        for name, feat in features_original.items():
            print(f"{name}: {feat.shape}")
    except Exception as e:
        print(f"原始版本测试失败（可能缺少权重文件）: {e}")
    
    print("\n=== 测试2: SAM2带adapter功能 ===")
    try:
        backbone_adapter = ImageEncoder2(cfg, input_shape, use_adapter=True, use_tsn=True)
        backbone_adapter.freeze_backbone()  # 只训练adapter
        
        features_adapter = backbone_adapter(x)
        
        print("SAM2 Adapter版本输出:")
        for name, feat in features_adapter.items():
            print(f"{name}: {feat.shape}")
        
        # 统计参数量
        total_params = sum(p.numel() for p in backbone_adapter.parameters())
        adapter_params = sum(p.numel() for p in backbone_adapter.get_adapter_parameters())
        trainable_params = sum(p.numel() for p in backbone_adapter.parameters() if p.requires_grad)
        
        print(f"\n=== SAM2参数统计 ===")
        print(f"总参数: {total_params:,}")
        print(f"Adapter相关参数: {adapter_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"Adapter参数占比: {adapter_params/total_params:.2%}")
        print(f"可训练参数占比: {trainable_params/total_params:.2%}")
        
    except Exception as e:
        print(f"Adapter版本测试失败可能缺少权重文件: {e}")