import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Dict, List


def visualize_and_save_feature_comparison(features_1: dict, features_2: dict, features_3: dict, features_4: dict,
                                           save_dir: str, img_id: str, 
                                          n_components: int = 3):
    """
    For four feature dictionaries, saves two comparison figures for each feature key:
    1) PCA comparison: 5 columns [RGB | SwinL | DINOv2 | SAM | Fused]
    2) Mean comparison: 5 columns [RGB | SwinL | DINOv2 | SAM | Fused]
    """
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.splitext(img_id)[0]
    H0, W0 = 1024, 1024
    feature_keys = list(features_1.keys())

    # Column titles
    titles = ["RGB", "SwinL", "DINOv2", "SAM", "Fused"]

    for k in feature_keys:
        feats = [features_1[k], features_2[k], features_3[k], features_4[k]]
        
        # Compute mean heatmaps
        mean_heatmaps = []
        for feat in feats:
            heat_mean = feat[0].mean(dim=0, keepdim=True)  # [1, H, W]
            heat_up = F.interpolate(
                heat_mean.unsqueeze(0),  # [1,1,H,W]
                size=(H0, W0),
                mode='bilinear',
                align_corners=False
            )[0, 0].detach().cpu().numpy()
            mean_heatmaps.append(heat_up)
        
        # Compute PCA visualizations
        pca_heatmaps = []
        for feat in feats:
            arr = feat[0].permute(1, 2, 0).detach().cpu().numpy()
            H, W, C = arr.shape
            flat = arr.reshape(-1, C)
            flat = (flat - flat.mean(0)) / (flat.std(0) + 1e-6)
            pca = PCA(n_components=n_components)
            pca_r = pca.fit_transform(flat).reshape(H, W, n_components)
            # normalize
            pca_r -= pca_r.min()
            pca_r /= (pca_r.max() + 1e-6)
            if n_components == 1:
                vis = pca_r[:, :, 0]
            else:
                vis = pca_r
            # upsample
            tensor_vis = (torch.from_numpy(vis).permute(2,0,1).unsqueeze(0) 
                          if n_components>1 else torch.from_numpy(vis)[None,None])
            up = F.interpolate(tensor_vis, size=(H0, W0), mode='bilinear', align_corners=False)
            if n_components == 1:
                pca_heatmaps.append((up[0,0].cpu().numpy(), 'jet'))
            else:
                pca_heatmaps.append((up[0].permute(1,2,0).cpu().numpy(), None))

        # PCA figure
        fig, axes = plt.subplots(1, 5, figsize=(30, 6))
        for i, ax in enumerate(axes):
            if i == 0:
                # ax.imshow(img_np)
                pass
            else:
                vis, cmap = pca_heatmaps[i-1]
                ax.imshow(vis, cmap=cmap, interpolation='nearest')
            ax.set_title(titles[i])
            ax.axis('off')
        pca_path = os.path.join(save_dir, f"{base}_{k}_pca_comparison_debug.png")
        fig.tight_layout()
        fig.savefig(pca_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        # Mean figure
        fig, axes = plt.subplots(1, 5, figsize=(30, 6))
        for i, ax in enumerate(axes):
            if i == 0:
                # ax.imshow(img_np)
                pass
            else:
                ax.imshow(mean_heatmaps[i-1], cmap='jet', interpolation='nearest')
            ax.set_title(titles[i])
            ax.axis('off')
        mean_path = os.path.join(save_dir, f"{base}_{k}_mean_comparison_debug.png")
        fig.tight_layout()
        fig.savefig(mean_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        

def conv3x3(in_channels, out_channels, groups=1, dilation=1):
    """
    3x3 convolution with padding and optional groups (for depthwise conv when groups=in_channels)
    """
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=3,
        stride=1,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        bias=False
    )


def get_sobel_filters(channel):
    """
    Create fixed Sobel filters for edge extraction, applied per channel (groups=channel).
    Returns conv_x and conv_y modules.
    """
    sobel_x = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]).view(1,1,3,3)
    sobel_y = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]).view(1,1,3,3)
    # repeat for all channels
    sobel_x = sobel_x.repeat(channel, 1, 1, 1)
    sobel_y = sobel_y.repeat(channel, 1, 1, 1)
    conv_x = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
    conv_y = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
    conv_x.weight.data.copy_(sobel_x)
    conv_y.weight.data.copy_(sobel_y)
    # freeze
    for p in conv_x.parameters(): p.requires_grad = False
    for p in conv_y.parameters(): p.requires_grad = False
    return conv_x, conv_y

class ImprovedFusion(nn.Module):
    def __init__(self, C_dino, C_sam=256):
        super().__init__()
        self.proj_s = nn.Conv2d(C_sam, C_dino, kernel_size=1, bias=False)
        self.dwconv = conv3x3(C_dino, C_dino, groups=C_dino)
        self.sobel_x, self.sobel_y = get_sobel_filters(C_dino)
        self.beta = nn.Parameter(torch.tensor(1.0))  # 稳定起见从1.0开始
        self.fuse_conv = nn.Conv2d(2 * C_dino, C_dino, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, F_dino, F_sam):
        # 特征对齐
        F_s = F.interpolate(F_sam, size=F_dino.shape[-2:], mode='bilinear', align_corners=False)
        if F_s.dtype != F_dino.dtype:
            F_s = F_s.type_as(F_dino)
        # 投影到同一维度
        F_s = self.proj_s(F_s)
        # 深度卷积
        F_s = self.dwconv(F_s)
        # Sobel边缘提取
        edge_x = self.sobel_x(F_s)
        edge_y = self.sobel_y(F_s)
        edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        # 边缘增强
        F_s_enhanced = F_s * (1 + self.beta * edge)

        # 拼接并融合
        F_cat = torch.cat([F_dino, F_s_enhanced], dim=1)
        F_fuse = self.act(self.fuse_conv(F_cat))

        # 输出融合后的特征
        return F_dino + self.alpha * F_fuse


class MultiStageFusion(nn.Module):
    """
    对 DINOv2 res2/res5 四阶段分别应用 ImprovedFusion。
    """
    def __init__(self):
        super().__init__()
        self.fuse2 = ImprovedFusion(C_dino=128,  C_sam=256)
        self.fuse3 = ImprovedFusion(C_dino=256,  C_sam=256)
        self.fuse4 = ImprovedFusion(C_dino=512,  C_sam=256)
        self.fuse5 = ImprovedFusion(C_dino=1024, C_sam=256)

    def forward(self, dino_feats: dict, sam_feats: dict) -> dict:
        return {
            'res2': self.fuse2(dino_feats['res2'], sam_feats['res2']),
            'res3': self.fuse3(dino_feats['res3'], sam_feats['res3']),
            'res4': self.fuse4(dino_feats['res4'], sam_feats['res4']),
            'res5': self.fuse5(dino_feats['res5'], sam_feats['res5']),
        }


def align_channels_and_spatial(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """
    Align `src` tensor to the channel and spatial dimensions of `tgt`.

    Uses evenly spaced channel sampling when src has more channels, and fills with
    the mean feature when src has fewer channels.

    Args:
        src: Tensor of shape (N, C_src, h, w)
        tgt: Tensor of shape (N, C_tgt, H, W)
    Returns:
        Tensor of shape (N, C_tgt, H, W)
    """
    # 1. Spatial resize
    N, C_tgt, H, W = tgt.shape
    src_resized = F.interpolate(src, size=(H, W), mode='bilinear', align_corners=False)
    C_src = src_resized.shape[1]

    # 2. Channel align
    if C_src >= C_tgt:
        # Sample channels evenly across src_resized using floor of uniform steps
        step = C_src / C_tgt
        # indices = floor(arange(0, C_tgt) * step)
        indices = (torch.arange(C_tgt, device=src_resized.device) * step).floor().long()
        return src_resized[:, indices, :, :]
    else:
        # Compute mean across channels as filler when C_src < C_tgt
        mean_feat = src_resized.mean(dim=1, keepdim=True)
        pad = mean_feat.repeat(1, C_tgt - C_src, 1, 1)
        return torch.cat([src_resized, pad], dim=1)


def align_and_replace(dino_feats: dict, sam_feats: dict,
                      mapping: dict = None) -> dict:
    """
    Replace selected keys in dino_feats with aligned sam_feats.

    Args:
        dino_feats: dict of feature maps, e.g. {'res2': Tensor, ...}
        sam_feats : dict of feature maps, same keys
        mapping   : dict mapping sam_keys -> dino_keys
                    defaults to {'res2': 'res2', 'res3': 'res4'}

    Returns:
        Updated copy of dino_feats with replaced tensors.
    """
    # Work on a shallow copy to avoid side-effects
    out_feats = dino_feats.copy()
    if mapping is None:
        mapping = {'res2': 'res2', 'res3': 'res4'}

    for sam_key, dino_key in mapping.items():
        src = sam_feats.get(sam_key)
        tgt = out_feats.get(dino_key)
        if src is None or tgt is None:
            continue
        out_feats[dino_key] = align_channels_and_spatial(src, tgt)

    return out_feats


if __name__ == '__main__':
    device = 'cuda'
    dino_feats = {
        'res2': torch.randn(1, 128, 256, 256, device=device).half(),
        'res3': torch.randn(1, 256, 128, 128, device=device).half(),
        'res4': torch.randn(1, 512, 64,  64, device=device).half(),
        'res5': torch.randn(1,1024, 32,  32, device=device).half(),
    }
    sam_feats = {
        'res2': torch.randn(1,256,64,64, device=device).half(),
        'res3': torch.randn(1,256,64,64, device=device).half(),
        'res4': torch.randn(1,256,64,64, device=device).half(),
        'res5': torch.randn(1,256,64,64, device=device).half(),
    }

    # 加载：
    dino_feat = np.load('dino.npy', allow_pickle=True)
    # np.load 返回一个 0-d ndarray，需要取出其中的 Python 对象
    dino_feat = dino_feat.item()
    sam_feat = np.load('sam.npy', allow_pickle=True)
    sam_feat = sam_feat.item()
    bench_feat = np.load('bench.npy', allow_pickle=True)
    # np.load 返回一个 0-d ndarray，需要取出其中的 Python 对象
    bench_feat = bench_feat.item()

    # 转换为 PyTorch Tensor
    dino_feats = {k: torch.tensor(v, device=device).half() for k, v in dino_feat.items()}
    sam_feats = {k: torch.tensor(v, device=device).half() for k, v in sam_feat.items()}
    bench_feats = {k: torch.tensor(v, device=device).half() for k, v in bench_feat.items()}


    fusion_model = MultiStageFusion().to(device).half()
    fused = fusion_model(dino_feats, sam_feats)

    # with torch.no_grad():
    #     fused = fuse_features_with_sobel(dino_feats, sam_feats, alpha=1.0, beta=3.0)
    #     assert fused['res4'].shape == dino_feats['res4'].shape

    for k,v in fused.items():
        print(f"{k}: {v.shape}, alpha={fusion_model.__getattr__('fuse'+k[-1]).alpha.item():.3f}, beta={fusion_model.__getattr__('fuse'+k[-1]).beta.item():.3f}")

    visualize_and_save_feature_comparison(bench_feats, dino_feats, sam_feats, fused, "./", "test_edge_", n_components=3)
