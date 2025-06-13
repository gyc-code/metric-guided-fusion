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
    """
    进一步增强版融合模块：
    1) SAM 投影
    2) Depthwise 3x3 捕捉边缘
    3) Sobel 边缘图门控加强空间细节
    4) 拼接 + 1x1 Conv 通道融合
    5) 可学习残差缩放 alpha
    """
    def __init__(self, C_dino, C_sam=256):
        super().__init__()
        # 1) SAM 特征投影
        self.proj_s = nn.Conv2d(C_sam, C_dino, kernel_size=1, bias=False)
        # 2) Depthwise 3x3
        self.dwconv = conv3x3(C_dino, C_dino, groups=C_dino)
        # 3) Sobel filters
        self.sobel_x, self.sobel_y = get_sobel_filters(C_dino)
        # 4) 门控系数 beta
        self.beta = nn.Parameter(torch.tensor(3.0))
        # 5) 融合 Conv
        self.fuse_conv = nn.Conv2d(2 * C_dino, C_dino, kernel_size=1, bias=False)
        # 6) 激活 & 缩放 alpha
        self.act = nn.ReLU(inplace=True)
        # self.alpha = nn.Parameter(torch.ones(1))
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, F_dino: torch.Tensor, F_sam: torch.Tensor) -> torch.Tensor:
        # 对齐
        F_s = F.interpolate(F_sam, size=F_dino.shape[-2:], mode='bilinear', align_corners=False)
        # SAM 投影
        F_s = self.proj_s(F_s)
        # Depthwise 卷积增强
        F_s = self.dwconv(F_s)
        # Sobel 边缘图
        edge_x = self.sobel_x(F_s)
        edge_y = self.sobel_y(F_s)
        edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        # 门控：在 F_s 基础上加强边缘
        F_s = F_s * (1 + self.beta * edge)
        F_s = self.beta * edge
        # 拼接融合
        # F_cat = torch.cat([F_dino, F_s], dim=1)
        # F_fuse = self.act(self.fuse_conv(F_cat))
        # 残差叠加
        # return F_dino + self.alpha * F_s
        return self.alpha * F_s



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


# Predefined Sobel kernels for edge detection
# _sobel_x = torch.tensor([
#     [1.0, 0.0, -1.0],
#     [2.0, 0.0, -2.0],
#     [1.0, 0.0, -1.0]
# ], dtype=torch.float32)
# _sobel_y = torch.tensor([
#     [1.0, 2.0, 1.0],
#     [0.0, 0.0, 0.0],
#     [-1.0, -2.0, -1.0]
# ], dtype=torch.float32)


# def fuse_features_with_sobel(dino_feats: dict,
#                               sam_feats: dict,
#                               alpha: float = 1.0,
#                               beta: float = 3.0) -> dict:
#     """
#     Inference-only feature fusion using SAM edge contours, without any learnable parameters.

#     Args:
#         dino_feats: dict of DINOv2 features, keys 'res2'...'res5', shapes [B, C_d, H, W]
#         sam_feats : dict of SAM features, same keys, shapes [B, C_s, h, w]
#         alpha     : weight for SAM branch in final sum
#         beta      : strength of edge gating on SAM features

#     Returns:
#         fused_feats: dict with same keys and shapes as dino_feats
#     """
#     fused_feats = {}
#     for lvl, F_d in dino_feats.items():
#         # Upsample SAM to match DINO spatial resolution
#         F_s = F.interpolate(
#             sam_feats[lvl], size=F_d.shape[-2:], mode='bilinear', align_corners=False
#         )
#         B, C_s, H, W = F_s.shape
#         device = F_s.device

#         # Prepare Sobel kernels on the fly, replicated for depthwise conv
#         # Shape for depthwise: [C_s, 1, 3, 3]
#         kx = _sobel_x.view(1,1,3,3).repeat(C_s,1,1,1).to(device).half()
#         ky = _sobel_y.view(1,1,3,3).repeat(C_s,1,1,1).to(device).half()

#         # Compute edge maps per channel (depthwise conv)
#         edge_x = F.conv2d(F_s, weight=kx, padding=1, groups=C_s).to(device).half()
#         edge_y = F.conv2d(F_s, weight=ky, padding=1, groups=C_s).to(device).half()
#         edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)

#         # Edge gating: amplify SAM features at contour locations
#         F_s_enh = F_s * (1.0 + beta * edge)

#         # Final fusion: simple weighted sum, no concat/conv required
#         fused_feats[lvl] = F_d + alpha * F_s_enh

#     return fused_feats



if __name__ == '__main__':
    device = 'cuda'
    # dino_feats = {
    #     'res2': torch.randn(1, 128, 256, 256, device=device).half(),
    #     'res3': torch.randn(1, 256, 128, 128, device=device).half(),
    #     'res4': torch.randn(1, 512, 64,  64, device=device).half(),
    #     'res5': torch.randn(1,1024, 32,  32, device=device).half(),
    # }
    # sam_feats = {
    #     'res2': torch.randn(1,256,64,64, device=device).half(),
    #     'res3': torch.randn(1,256,64,64, device=device).half(),
    #     'res4': torch.randn(1,256,64,64, device=device).half(),
    #     'res5': torch.randn(1,256,64,64, device=device).half(),
    # }

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
