# -*- coding: utf-8 -*-
"""
VFM GT-free Diagnostics (feature-only) — fixed & complete (no name collisions)
---------------------------------------------------------
实现 5 个无监督指标：
- EAI (Edge Amplification Index)
- FER (Mid-band Frequency Ratio)
- GGini (Gradient Gini)
- ECI (Edge Continuity Index)
- ALI (Alignment with Image edges)
"""

from typing import Dict, Tuple
import math
import torch
import torch.nn.functional as Fnn  # 改名，避免与变量冲突


# ------------------------ Utilities ------------------------

def _to_float_tensor(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(x):
        x = x.float()
    return x


def _normalize_per_map(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    对每幅图（B 维）做零均值单位方差归一化。
    适配非连续张量（使用 reshape 而非 view）。
    """
    B = x.shape[0]
    xf = x.reshape(B, -1)
    mean = xf.mean(dim=1, keepdim=True)
    std  = xf.std(dim=1, keepdim=True).clamp_min(eps)
    xn = (xf - mean) / std
    return xn.reshape_as(x)


def gini_coefficient(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    修正后的 Gini 系数（含 n*sums 分母；并限制到 [0,1]）
    x: (B,H,W) 非负数
    return: (B,) Gini in [0,1]
    """
    B = x.shape[0]
    xf = x.reshape(B, -1).clamp_min(0)
    sums = xf.sum(dim=1, keepdim=True)  # (B,1)
    n = xf.shape[1]
    nonzero = (sums > eps).squeeze(1)

    xf_sorted, _ = torch.sort(xf, dim=1)
    i = (torch.arange(n, device=x.device, dtype=xf.dtype) + 1).unsqueeze(0)  # (1,n)
    num = ((n + 1 - i) * xf_sorted).sum(dim=1, keepdim=True)  # (B,1)

    denom = (n * sums + eps)
    G = (1.0 + 1.0 / float(n) - 2.0 * num / denom).squeeze(1)
    G = torch.where(nonzero, G.clamp(0.0, 1.0), torch.zeros_like(G))
    return G


def mean_run_length(binmap: torch.Tensor, min_val: float = 0.5) -> torch.Tensor:
    """
    纯向量化的平均连通 run-length（横纵各一半）
    binmap: (B,1,H,W) in [0,1]
    return: (B,)
    """
    b = (binmap >= min_val).float()        # (B,1,H,W)

    # 水平：每行 ones / 片段数
    bh = b
    bh_pad = Fnn.pad(bh, (1, 0, 0, 0))       # 左侧补0 -> (B,1,H,W+1)
    starts_h = (bh_pad[:, :, :, 1:] - bh_pad[:, :, :, :-1]) == 1  # (B,1,H,W)
    runs_h = starts_h.sum(dim=3).to(bh.dtype).clamp_min(1)        # (B,1,H)
    ones_h = bh.sum(dim=3)                                        # (B,1,H)
    mrl_h = (ones_h / runs_h).mean(dim=2).squeeze(1)              # (B,)

    # 垂直：每列 ones / 片段数
    bv = b
    bv_pad = Fnn.pad(bv, (0, 0, 1, 0))       # 上侧补0 -> (B,1,H+1,W)
    starts_v = (bv_pad[:, :, 1:, :] - bv_pad[:, :, :-1, :]) == 1  # (B,1,H,W)
    runs_v = starts_v.sum(dim=2).to(bv.dtype).clamp_min(1)        # (B,1,W)
    ones_v = bv.sum(dim=2)                                        # (B,1,W)
    mrl_v = (ones_v / runs_v).mean(dim=2).squeeze(1)              # (B,)

    return 0.5 * (mrl_h + mrl_v)


# ------------------------ Gradients & NMS ------------------------

def _sobel_kernels(device, dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor([[-1., 0., 1.],
                       [-2., 0., 2.],
                       [-1., 0., 1.]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.],
                       [ 0.,  0.,  0.],
                       [ 1.,  2.,  1.]], device=device, dtype=dtype).view(1, 1, 3, 3)
    return kx, ky


def gradient_mag_and_orientation(M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    输入:
        M: (B,H,W) float
    输出:
        G:  (B,H,W) 梯度幅值
        gx: (B,1,H,W)
        gy: (B,1,H,W)
    """
    B, H, W = M.shape
    x = M.unsqueeze(1)  # (B,1,H,W)
    kx, ky = _sobel_kernels(M.device, M.dtype)
    gx = Fnn.conv2d(x, kx, padding=1)
    gy = Fnn.conv2d(x, ky, padding=1)
    G = torch.sqrt(gx.squeeze(1).pow(2) + gy.squeeze(1).pow(2) + 1e-12)  # (B,H,W)
    return G, gx, gy


def _shift_2d(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """
    零填充的位移（不环绕），x:(B,1,H,W)
    """
    B, C, H, W = x.shape
    out = torch.zeros_like(x)
    ys_src = slice(max(0, -dy), H - max(0, dy))
    xs_src = slice(max(0, -dx), W - max(0, dx))
    ys_dst = slice(max(0,  dy), H - max(0, -dy))
    xs_dst = slice(max(0,  dx), W - max(0, -dx))
    out[:, :, ys_dst, xs_dst] = x[:, :, ys_src, xs_src]
    return out


def non_max_suppression_thin(G: torch.Tensor, gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
    """
    方向敏感的 NMS，输出细化的边强度 (B,1,H,W)
    输入:
        G:  (B,H,W)
        gx: (B,1,H,W)
        gy: (B,1,H,W)
    """
    B, H, W = G.shape
    G1 = G.unsqueeze(1)  # (B,1,H,W)
    # 方向角度 [0, 180)
    angle = torch.atan2(gy, gx + 1e-12)  # (B,1,H,W), rad, [-pi, pi]
    angle = torch.remainder(angle, math.pi)  # [0, pi)
    angle_deg = angle * (180.0 / math.pi)

    # 4 个方向桶
    d0 = (angle_deg < 22.5) | (angle_deg >= 157.5)                   # 0°
    d45 = (angle_deg >= 22.5) & (angle_deg < 67.5)                   # 45°
    d90 = (angle_deg >= 67.5) & (angle_deg < 112.5)                  # 90°
    d135 = (angle_deg >= 112.5) & (angle_deg < 157.5)                # 135°

    # 邻居比较
    # 0°: 左/右
    n1 = _shift_2d(G1, 0, -1)
    n2 = _shift_2d(G1, 0,  1)
    keep0 = (G1 >= n1) & (G1 >= n2) & d0

    # 45°: 左下/右上  (dy,dx)=(+1,-1)/(-1,+1)
    n1 = _shift_2d(G1, +1, -1)
    n2 = _shift_2d(G1, -1, +1)
    keep45 = (G1 >= n1) & (G1 >= n2) & d45

    # 90°: 上/下
    n1 = _shift_2d(G1, -1, 0)
    n2 = _shift_2d(G1, +1, 0)
    keep90 = (G1 >= n1) & (G1 >= n2) & d90

    # 135°: 左上/右下
    n1 = _shift_2d(G1, -1, -1)
    n2 = _shift_2d(G1, +1, +1)
    keep135 = (G1 >= n1) & (G1 >= n2) & d135

    keep = keep0 | keep45 | keep90 | keep135
    S = torch.where(keep, G1, torch.zeros_like(G1))  # (B,1,H,W)
    return S


# ------------------------ Frequency-domain energy ------------------------

def _fftshift2d(x: torch.Tensor) -> torch.Tensor:
    """等价于 numpy.fft.fftshift for last two dims."""
    h, w = x.shape[-2], x.shape[-1]
    return torch.roll(torch.roll(x, shifts=h // 2, dims=-2), shifts=w // 2, dims=-1)


def rfft2_radial_energy(M: torch.Tensor,
                        low_thr: float = 0.25,
                        mid_thr: float = 0.50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    基于径向频率的能量分段（低/中/高）。
    输入:
        M: (B,H,W)   (建议已做 per-map 标准化)
        low_thr: 半径阈值 r∈[0,low_thr) 为低频
        mid_thr: r∈[low_thr, mid_thr) 为中频；r∈[mid_thr,1] 为高频
    返回:
        (low_energy, mid_energy, high_energy): 形状均为 (B,)
    """
    B, H, W = M.shape
    # FFT
    F2 = torch.fft.fft2(M)                 # (B,H,W), complex
    P = (F2.real ** 2 + F2.imag ** 2)      # 功率谱
    P = _fftshift2d(P)                     # 中心化

    # 归一化半径网格 r∈[0,1]
    yy = torch.linspace(-1.0, 1.0, H, device=M.device, dtype=M.dtype).view(H, 1)
    xx = torch.linspace(-1.0, 1.0, W, device=M.device, dtype=M.dtype).view(1, W)
    rr = torch.sqrt(yy ** 2 + xx ** 2)     # (H,W), ~[0,sqrt(2)]
    rr = rr / rr.max().clamp_min(1e-6)     # 归一化到 [0,1]

    low_mask = (rr < low_thr).to(P.dtype)
    mid_mask = ((rr >= low_thr) & (rr < mid_thr)).to(P.dtype)
    high_mask = (rr >= mid_thr).to(P.dtype)

    # 按批次求和（能量和）
    low = (P * low_mask).reshape(B, -1).sum(dim=1)
    mid = (P * mid_mask).reshape(B, -1).sum(dim=1)
    high = (P * high_mask).reshape(B, -1).sum(dim=1)

    return low, mid, high


# ------------------------ Image edges & NCC ------------------------

def image_edge_map(images: torch.Tensor) -> torch.Tensor:
    """
    基于 Sobel 的图像边缘图
    输入:
        images: (B,1/3,H,W), in [0,1]
    返回:
        (B,1,H,W), 已做 per-map 线性归一化到 [0,1]
    """
    B, C, H, W = images.shape
    if C == 3:
        # RGB -> Gray
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        gray = (images * weights).sum(dim=1, keepdim=True)
    else:
        gray = images[:, :1]  # 取单通道

    kx, ky = _sobel_kernels(images.device, images.dtype)
    gx = Fnn.conv2d(gray, kx, padding=1)
    gy = Fnn.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)  # (B,1,H,W)

    # 每图归一化到 [0,1]（95 分位数缩放，避免尖峰）
    mag_flat = mag.reshape(B, -1)
    q = torch.quantile(mag_flat, 0.95, dim=1, keepdim=True).clamp_min(1e-6)  # (B,1)
    mag_norm = (mag_flat / q).clamp(0, 1).reshape_as(mag)
    return mag_norm


def ncc(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    归一化互相关（per-map）
    输入:
        A, B: (B,1,H,W) 或 (B,H,W)
    返回:
        (B,)
    """
    if A.dim() == 4:
        Af = A.reshape(A.shape[0], -1)
    else:
        Af = A.reshape(A.shape[0], -1)

    if B.dim() == 4:
        Bf = B.reshape(B.shape[0], -1)
    else:
        Bf = B.reshape(B.shape[0], -1)

    ma = Af.mean(dim=1, keepdim=True)
    mb = Bf.mean(dim=1, keepdim=True)
    za = Af - ma
    zb = Bf - mb
    sa = torch.sqrt((za ** 2).sum(dim=1, keepdim=True)).clamp_min(eps)
    sb = torch.sqrt((zb ** 2).sum(dim=1, keepdim=True)).clamp_min(eps)
    r = (za * zb).sum(dim=1, keepdim=True) / (sa * sb)
    return r.squeeze(1).clamp(-1.0, 1.0)


# ------------------------ Main metrics ------------------------

@torch.no_grad()
def compute_metrics_from_features(feat: torch.Tensor,
                                  images: torch.Tensor = None,
                                  q_top: float = 0.10) -> Dict[str, torch.Tensor]:
    """
    计算 5 个指标
    输入:
        feat: (B,C,H,W) 特征
        images: (B,1/3,H,W) in [0,1], 可选，用于 ALI
        q_top: EAI 的 top 比例
    输出:
        dict: {'EAI','FER','GGini','ECI','ALI'} ，每个为 (B,)
    """
    # 诊断用 float32 更稳
    F32 = feat.detach().to(torch.float32)
    B, C, H, W = F32.shape

    # 标量强度图 (L2)
    M = torch.linalg.vector_norm(F32, ord=2, dim=1)  # (B,H,W)
    M = _normalize_per_map(M)

    # 梯度
    G, gx, gy = gradient_mag_and_orientation(M)      # (B,H,W), (B,1,H,W), (B,1,H,W)
    G = G.clamp_min(0)

    # --- EAI（简洁高效版本）
    N = H * W
    top_k = max(1, min(int(N * q_top), N - 1))
    Gf = G.reshape(B, -1)
    top_vals, _ = torch.topk(Gf, top_k, dim=1)
    rest_sum = Gf.sum(dim=1) - top_vals.sum(dim=1)
    rest_mean = rest_sum / float(N - top_k)
    EAI = top_vals.mean(dim=1) / (rest_mean.clamp_min(1e-8))

    # --- FER（中频 / 低频）
    low, mid, high = rfft2_radial_energy(M)
    FER = (mid + 1e-8) / (low + 1e-8)

    # --- GGini（修正版）
    GGini = gini_coefficient(G)

    # --- ECI（NMS+分位阈值，处理退化图）
    S = non_max_suppression_thin(G, gx, gy).to(torch.float32)  # (B,1,H,W)
    Sflat = S.reshape(B, -1)
    Smax = Sflat.amax(dim=1, keepdim=True)                      # (B,1)
    has_edge = Smax > 0

    # 仅对有边的图计算分位数；防止 kth==0 时整图入选
    kth = torch.where(
        has_edge,
        torch.quantile(Sflat, 0.90, dim=1, keepdim=True),
        torch.zeros_like(Smax)
    )

    Sb_flat = torch.where(has_edge, (Sflat > kth).float(), torch.zeros_like(Sflat))
    Sb = Sb_flat.view_as(S)                                     # (B,1,H,W)
    ECI = mean_run_length(Sb, min_val=0.5)

    # --- ALI（尺寸安全 & 零方差返回 0）
    if images is not None:
        Emap = image_edge_map(images.to(torch.float32).clamp(0, 1))  # (B,1,H?,W?)
        if Emap.shape[-2:] != S.shape[-2:]:
            Emap = Fnn.interpolate(Emap, size=S.shape[-2:], mode='bilinear', align_corners=False)
        ALI = ncc(Sb, Emap)
    else:
        ALI = torch.zeros(B, device=feat.device, dtype=torch.float32)

    return {'EAI': EAI, 'FER': FER, 'GGini': GGini, 'ECI': ECI, 'ALI': ALI}


# ------------------------ Quick self-test (optional) ------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    feat_rand = torch.randn(4, 8, 32, 32)
    imgs = torch.rand(4, 3, 32, 32)
    m = compute_metrics_from_features(feat_rand, imgs)
    for k, v in m.items():
        print(k, v.shape, v)
    # 退化图检查
    feat_flat = torch.zeros(2, 8, 32, 32)
    mflat = compute_metrics_from_features(feat_flat)
    print("Flat:", {k: v.tolist() for k, v in mflat.items()})
