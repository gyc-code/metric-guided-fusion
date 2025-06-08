import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class CrossAttnFusion(nn.Module):
    """在单尺度上，用 SAM 做 Query,DINO 做 Key/Value 的跨注意力融合"""
    def __init__(self, dim_q, dim_kv, dim_out, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim_out // num_heads) ** -0.5
        self.to_q = nn.Conv2d(dim_q, dim_out, 1, bias=False)
        self.to_k = nn.Conv2d(dim_kv, dim_out, 1, bias=False)
        self.to_v = nn.Conv2d(dim_kv, dim_out, 1, bias=False)
        self.proj = nn.Conv2d(dim_out, dim_out, 1)

    def forward(self, sam, dino):
        # sam: [B, Cq, H, W]; dino: [B, Ckv, H, W]
        B, _, H, W = sam.shape
        q = self.to_q(sam).flatten(2).view(B, self.num_heads, -1, H*W)    # [B, h, d', L]
        k = self.to_k(dino).flatten(2).view(B, self.num_heads, -1, H*W)
        v = self.to_v(dino).flatten(2).view(B, self.num_heads, -1, H*W)

        attn = (q.transpose(-2,-1) @ k) * self.scale  # [B, h, L, L]
        attn = attn.softmax(-1)

        out = (attn @ v.transpose(-2,-1))             # [B, h, L, d']
        out = out.transpose(-2,-1).contiguous().view(B, -1, H, W)
        return self.proj(out)                         # [B, dim_out, H, W]


# === 通用 Conv–BN–ReLU 模块 ===
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding,
                      dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

# === 改进版 FusionFPN ===
class FusionFPNv2(nn.Module):
    def __init__(self, dino_channels, sam_channel=256, out_channels=256):
        super().__init__()
        # 1×1 投影 + BN + ReLU
        self.lat_dino = nn.ModuleList([
            ConvBNReLU(c, out_channels, kernel_size=1)
            for c in dino_channels
        ])
        self.lat_sam = nn.ModuleList([
            ConvBNReLU(sam_channel, out_channels, kernel_size=1)
            for _ in dino_channels
        ])

        # Cross-Attn（保持你原来的实现）
        self.cross_attn = nn.ModuleList([
            CrossAttnFusion(out_channels, out_channels, out_channels)
            for _ in dino_channels
        ])

        # 融合后通道拼接降维
        self.fuse_conv = nn.ModuleList([
            ConvBNReLU(out_channels * 3, out_channels, kernel_size=1)
            for _ in dino_channels
        ])

        # 可学习融合权重
        self.weights = nn.ParameterList([
            nn.Parameter(torch.ones(3), requires_grad=True)
            for _ in dino_channels
        ])

        # 平滑卷积 + BN + ReLU
        self.smooth = nn.ModuleList([
            ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in dino_channels
        ])

    def forward(self, feats_dino, feat_sam):
        # 1) 投影
        pd = [l(f) for l, f in zip(self.lat_dino, feats_dino)]

        # 2) 对齐并投影 SAM 特征
        ps = []
        for i, p in enumerate(pd):
            up = F.interpolate(feat_sam, size=p.shape[-2:], 
                               mode='bilinear', align_corners=False)
            ps.append(self.lat_sam[i](up))

        # 3) 融合：cross-attn + 可学习权重 + 拼接降维
        fused = []
        for i in range(len(pd)):
            attn_o = self.cross_attn[i](ps[i], pd[i])

            # 规范化为正、归一化
            w = F.relu(self.weights[i])
            w = w / (w.sum() + 1e-4)

            # 拼接三个分支
            cat = torch.cat([pd[i], ps[i], attn_o], dim=1)
            fused_i = self.fuse_conv[i](cat)

            # 加权组合
            fused.append(w[0] * fused_i +
                         w[1] * pd[i] +
                         w[2] * ps[i])

        # 4) 自顶向下金字塔融合 + 平滑
        out = []
        x = fused[-1]
        out.append(self.smooth[-1](x))
        for i in range(len(fused) - 2, -1, -1):
            x = F.interpolate(x, size=fused[i].shape[-2:], 
                              mode='nearest') + fused[i]
            out.insert(0, self.smooth[i](x))

        return out



class FusionFPN(nn.Module):
    """
    输入：
      feats_dino: list of 4 tensors,shape = [
          (B,1024,256,256), (B,1024,128,128),
          (B,1024,64,64),   (B,1024,32,32)
      ]
      feat_sam: single tensor (B,256,64,64)
    输出：
      fused_feats: list of 4 tensors (P2-P5)，每个都是 (B,256,Hi,Wi)
    """
    def __init__(self, dino_channels, sam_channel=256, out_channels=256):
        super().__init__()
        # 1×1 投影
        self.lat_dino = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in dino_channels
        ])
        self.lat_sam  = nn.ModuleList([
            nn.Conv2d(sam_channel, out_channels, 1) for _ in dino_channels
        ])
        # 跨注意力融合块
        self.cross_attn = nn.ModuleList([
            CrossAttnFusion(out_channels, out_channels, out_channels)
            for _ in dino_channels
        ])
        # 平滑卷积
        self.smooth = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in dino_channels
        ])

    def forward(self, feats_dino, feat_sam):
        # 1) 先做 1×1 投影
        pd = [l(feat) for l, feat in zip(self.lat_dino, feats_dino)]
        # 2) 对齐 SAM 到每个尺度、再投影
        ps = []
        for i, p_d in enumerate(pd):
            ps_i = F.interpolate(feat_sam, size=p_d.shape[-2:],
                                 mode='bilinear', align_corners=False)
            ps.append(self.lat_sam[i](ps_i))

        # 3) 跨注意力融合 + 相加残差
        fused = []
        for i in range(len(pd)):
            attn_out = self.cross_attn[i](ps[i], pd[i])
            fused.append(pd[i] + ps[i] + attn_out)

        # 4) 自顶向下金字塔融合 & 平滑
        out_feats = []
        x = fused[-1]
        out_feats.append(self.smooth[-1](x))
        for i in range(len(fused)-2, -1, -1):
            x = F.interpolate(x, size=fused[i].shape[-2:], mode='nearest') \
                + fused[i]
            out_feats.insert(0, self.smooth[i](x))

        return out_feats

class FusionFPNvDS(nn.Module):
    """
    输入：
      feats_dino: list of 4 tensors,shape = [
          (B,1024,256,256), (B,1024,128,128),
          (B,1024,64,64),   (B,1024,32,32)
      ]
      feat_sam: single tensor (B,256,64,64)
    输出：
      fused_feats: list of 4 tensors (P2-P5)，每个都是 (B,256,Hi,Wi)
    """
    def __init__(self, dino_channels, sam_channel=256, out_channels=256):  # 关键修改：out_channels=256
        super().__init__()
        
        # 1×1 投影（增加BN和ReLU）
        self.lat_dino = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ) for c in dino_channels
        ])
        
        self.lat_sam = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(sam_channel, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ) for _ in dino_channels
        ])

        # 改进的跨注意力融合（带位置编码）
        self.cross_attn = nn.ModuleList([
            EnhancedCrossAttn(out_channels, num_heads=8) 
            for _ in dino_channels
        ])

        # 自适应特征融合（代替简单相加）
        self.fuse_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3*out_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ) for _ in dino_channels
        ])

        # 增强的金字塔平滑模块
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            ) for _ in dino_channels
        ])

        # 可学习的上采样修正
        self.upsample_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in dino_channels
        ])

        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feats_dino, feat_sam):
        # 1) 投影DINO特征
        pd = [l(feat) for l, feat in zip(self.lat_dino, feats_dino)]
        
        # 2) 对齐并投影SAM特征
        ps = []
        for i, p_d in enumerate(pd):
            # 可学习的上采样
            scale_factor = p_d.shape[-1] / feat_sam.shape[-1]
            ps_i = F.interpolate(
                feat_sam, 
                scale_factor=scale_factor, 
                mode='bilinear',
                align_corners=False
            )
            ps_i = self.upsample_convs[i](ps_i)  # 上采样修正
            ps.append(self.lat_sam[i](ps_i))

        # 3) 增强的特征融合
        fused = []
        for i in range(len(pd)):
            attn_out = self.cross_attn[i](ps[i], pd[i])
            # 自适应融合代替简单相加
            combined = torch.cat([pd[i], ps[i], attn_out], dim=1)
            fused_feat = self.fuse_conv[i](combined)
            fused.append(fused_feat)

        # 4) 改进的金字塔融合
        out_feats = []
        x = fused[-1]
        out_feats.append(self.smooth[-1](x))
        
        # 自顶向下融合
        for i in range(len(fused)-2, -1, -1):
            x = F.interpolate(x, size=fused[i].shape[-2:], mode='nearest')
            x = x + fused[i]
            x = self.smooth[i](x)
            out_feats.insert(0, x)

        return out_feats


class EnhancedCrossAttn(nn.Module):
    """改进的跨注意力模块，包含位置编码"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.pos_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.q_proj = nn.Linear(channels, channels)
        self.kv_proj = nn.Linear(channels, 2*channels)
        self.attn = nn.MultiheadAttention(channels, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels*4),
            nn.ReLU(),
            nn.Linear(channels*4, channels)
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, sam_feat, dino_feat):
        B, C, H, W = sam_feat.shape
        
        # 位置编码
        pos = self.pos_conv(dino_feat).flatten(2).permute(2,0,1)  # (H*W, B, C)
        
        # Query来自SAM特征
        q = sam_feat.flatten(2).permute(2,0,1)  # (HW, B, C)
        q = self.q_proj(q) + pos  # 加入位置信息
        
        # Key/Value来自DINO特征
        kv = self.kv_proj(dino_feat.flatten(2).permute(2,0,1))
        k, v = torch.chunk(kv, 2, dim=-1)
        
        # 注意力机制
        attn_out, _ = self.attn(q, k, v)
        attn_out = self.norm1(attn_out + q)
        
        # FFN
        ffn_out = self.ffn(attn_out)
        ffn_out = self.norm2(ffn_out + attn_out)
        
        # 恢复形状
        ffn_out = ffn_out.permute(1,2,0).view(B, C, H, W)
        return ffn_out

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 基础组件：Conv -> BN -> ReLU
# -----------------------------
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k, stride=1, padding=0):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

# -----------------------------
# BiFPN 样式加权融合块
# -----------------------------
class WeightedFuse(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        # 初始化为 1
        self.w = nn.Parameter(torch.ones(num_inputs), requires_grad=True)
        # 小常数防止除零
        self.eps = 1e-4
    
    def forward(self, feats):
        # feats: list of tensors to融合，shape 都一致
        w = F.relu(self.w)
        w_sum = w.sum() + self.eps
        out = 0
        for i, f in enumerate(feats):
            out = out + f * (w[i] / w_sum)
        return out

# -----------------------------
# 主 FusionFPNv3 实现
# -----------------------------
class FusionFPNv3(nn.Module):
    def __init__(self, dino_channels, sam_channel=256, out_ch=256):
        """
        dino_channels: list[int]，比如 [1024,1024,1024,1024]
        sam_channel: SAM 输出通道数
        out_ch: FPN 输出通道数
        """
        super().__init__()
        N = len(dino_channels)
        
        # 1) DINO 侧 1x1 投影
        self.proj_d = nn.ModuleList([
            ConvBNReLU(c, out_ch, k=1) for c in dino_channels
        ])
        # 2) SAM 侧：先整体降到 out_ch，再上/下采样到各尺度
        self.proj_s = ConvBNReLU(sam_channel, out_ch, k=1)
        
        # 3) cross-attn 模块（保持你现有实现）
        self.cross_attn = nn.ModuleList([
            CrossAttnFusion(out_ch, out_ch, out_ch) for _ in range(N)
        ])
        
        # 4) BiFPN 样式加权融合：三个分支 pd, ps, attn
        self.weighted = nn.ModuleList([
            WeightedFuse(3) for _ in range(N)
        ])
        
        # 5) 融合后再来一次 BN+ReLU
        self.post = nn.ModuleList([
            ConvBNReLU(out_ch, out_ch, k=3, padding=1) for _ in range(N)
        ])
        
        # 6) bottom-up 路径，用于自底向上回流（可选）
        self.bu_conv = nn.ModuleList([
            ConvBNReLU(out_ch*2, out_ch, k=3, padding=1) for _ in range(N-1)
        ])

    def forward(self, feats_dino, feat_sam):
        # --- 1. DINO 投影 ---
        pd = [p(f) for p, f in zip(self.proj_d, feats_dino)]
        
        # --- 2. SAM 多尺度投影 ---
        # 先 1×1 投影
        ps_base = self.proj_s(feat_sam)
        # 然后对齐到各层
        ps = [
            F.interpolate(ps_base, size=p.shape[-2:], 
                          mode='bilinear', align_corners=False)
            for p in pd
        ]
        
        # --- 3. 跨注意力 + BiFPN 加权融合 + 后处理 ---
        feats_td = []
        for i in range(len(pd)):
            attn = self.cross_attn[i](ps[i], pd[i])
            fused = self.weighted[i]([pd[i], ps[i], attn])
            feats_td.append(self.post[i](fused))
        
        # --- 4. 自顶向下输出 ---
        # 这里可以直接用 feats_td 作为 P2–P5，也可做额外上采样融合
        # --- 5. （可选）自底向上回流增强 ---
        bu = [feats_td[0]]
        for i in range(1, len(feats_td)):
            up = F.interpolate(bu[-1], size=feats_td[i].shape[-2:],
                               mode='nearest')
            cat = torch.cat([up, feats_td[i]], dim=1)
            bu.append(self.bu_conv[i-1](cat))
        
        # 返回双向融合后的结果
        return bu  # list of length N, 每个 (B, out_ch, H_i, W_i)


class SimpleEdgeFusion1024(nn.Module):
    def __init__(self, dino_channels, sam_channel=256):
        """
        dino_channels: list[int]，比如 [1024,1024,1024,1024]
        sam_channel: SAM 特征通道数，这里是 256
        输出通道固定为 1024
        """
        super().__init__()
        N = len(dino_channels)
        out_ch = dino_channels[0]  # 假设所有 dino_channels 都相同
        
        # 1) DINO 投影：1×1 conv 保持 1024→1024
        self.proj_d = nn.ModuleList([
            nn.Conv2d(c, out_ch, kernel_size=1, bias=False)
            for c in dino_channels
        ])
        self.bn_d = nn.ModuleList([
            nn.BatchNorm2d(out_ch) for _ in range(N)
        ])
        
        # 2) SAM 投影一次：256→1024
        self.proj_s = nn.Conv2d(sam_channel, out_ch, kernel_size=1, bias=False)
        self.bn_s   = nn.BatchNorm2d(out_ch)
        
        # 3) 可学习残差缩放 α_i
        self.alpha = nn.Parameter(torch.ones(N) * 0.1)
        
        # 4) 融合后平滑：3×3 conv 保持通道 1024
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ) for _ in range(N)
        ])

    def forward(self, dino_feats, sam_feat):
        """
        dino_feats: list of 4 tensors
            [ (B,1024,256,256), (B,1024,128,128),
              (B,1024,64,64),   (B,1024,32,32) ]
        sam_feat: tensor (B,256,64,64)
        返回：
          out: list of 4 tensors,每个 (B,1024,Hi,Wi)
        """
        # 1) DINO 投影
        pd = []
        for i, f in enumerate(dino_feats):
            x = self.proj_d[i](f)
            x = self.bn_d[i](x)
            pd.append(x)
        
        # 2) SAM 投影一次
        ps_base = self.bn_s(self.proj_s(sam_feat))
        
        out = []
        for i, p in enumerate(pd):
            # 插值到与 DINO 同尺度
            ps_i = F.interpolate(ps_base, size=p.shape[-2:], 
                                 mode='bilinear', align_corners=False)
            
            # 归一化
            p_norm  = p   / (p.abs().mean(dim=[1,2,3], keepdim=True) + 1e-6)
            ps_norm = ps_i / (ps_i.abs().mean(dim=[1,2,3], keepdim=True) + 1e-6)
            
            # 残差融合
            fused = p_norm + self.alpha[i] * ps_norm
            
            # 平滑
            out.append(self.smooth[i](fused))
        
        return out


import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def visualize_with_pca(fused_feats, output_dir='vlm_fusion'):
    """
    对 fused_feats 中的每个特征层同时可视化平均热图和 PCA RGB 映射。
    保存结果到 output_dir/fused_p{i}_comparison.png
    """
    for i, f in enumerate(fused_feats):
        # f: tensor (B, C, H, W)
        feat = f[0]  # 取 batch=0
        C, H, W = feat.shape

        # 1. 平均热图
        heatmap = feat.mean(0).detach().cpu().numpy()

        # 2. PCA 映射到 3 维，作为 RGB 图
        # 重塑为 (H*W, C)
        data = feat.permute(1, 2, 0).reshape(-1, C).detach().cpu().numpy()
        pca = PCA(n_components=3)
        components = pca.fit_transform(data)  # (H*W, 3)
        # 归一化到 [0,1]
        comps_min = components.min(axis=0, keepdims=True)
        comps_max = components.max(axis=0, keepdims=True)
        comps_norm = (components - comps_min) / (comps_max - comps_min + 1e-6)
        # 重塑为图像 (H, W, 3)
        pca_img = comps_norm.reshape(H, W, 3)

        # 3. 绘图
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        # 平均热图
        axes[0].imshow(heatmap, cmap='jet', interpolation='nearest')
        axes[0].set_title(f"P{i+2} Mean Heatmap")
        axes[0].axis('off')
        # PCA RGB
        axes[1].imshow(pca_img)
        axes[1].set_title(f"P{i+2} PCA RGB")
        axes[1].axis('off')

        plt.tight_layout()
        # 保存对比图
        fig_path = f"{output_dir}/fuse_p{i}_comparison.png"
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()

# -----------------------------
# 用法示例
# -----------------------------
# 假设
#   features = {'res7':..., 'res11':..., 'res15':..., 'res23':...}
#   sam_feat = torch.rand(1,256,64,64)
device = 'cuda'
features = np.load('vlm_fusion/dinov2_feature.npy', allow_pickle=True).item()  # 假设 DINO 特征已保存为 NumPy 数组
dino_feats = [
    features['res7'],   # [1,1024,256,256]
    features['res11'],  # [1,1024,128,128]
    features['res15'],  # [1,1024,64,64]
    features['res23'],  # [1,1024,32,32]
]
dino_feats = [f.to(device) for f in dino_feats]
sam_feat = torch.from_numpy(np.load('vlm_fusion/debug_1024_1024.npy', allow_pickle=True)).to(device)    # 假设 SAM 特征已保存为 NumPy 数组

print("DINO features shape:", [f.shape for f in dino_feats])
print("SAM feature shape:", sam_feat.shape)

model = SimpleEdgeFusion1024(
    dino_channels=[1024,1024,1024,1024],
    sam_channel=256
).to(device)  
fused_feats = model(dino_feats, sam_feat)

for lvl, f in enumerate(fused_feats):
    print(f"Fused P{lvl+2} shape:", f.shape)



visualize_with_pca(fused_feats)
