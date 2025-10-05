# -*- coding: utf-8 -*-
"""
VFM GT-free Diagnostics (feature-only) — fixed & complete (no name collisions)
---------------------------------------------------------
实现 5 个无监督指标：
- ECR (Edge Amplification Index)
- FER (Mid-band Frequency Ratio)
- GIC (Gradient Gini)
- ECI (Edge Continuity Index)
- ALI (Alignment with Image edges)
"""

import math
import torch
import torch.nn.functional as Fnn  # 改名，避免与变量冲突
from typing import Dict, Iterable, Tuple
import torch.nn.functional as F
import numpy as np
EPS = 1e-8


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


def _as_float(x: torch.Tensor) -> torch.Tensor:
    """Ensure float tensor (keeps device), promote lower precision to float32."""
    if not torch.is_floating_point(x):
        x = x.float()
    elif x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    return x


def _scalar_map_from_feat(Fmap: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Collapse channels by L2 to get a scalar feature map M (B,H,W).
    Optionally standardize per-image to zero-mean unit-std for correlation fairness.
    """
    # Fmap: (B,C,H,W)
    M = torch.linalg.vector_norm(_as_float(Fmap), ord=2, dim=1)  # (B,H,W)
    if normalize:
        B = M.shape[0]
        flat = M.reshape(B, -1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True).clamp_min(EPS)
        M = ((flat - mean) / std).reshape_as(M)
    return M


# --------------------------- NCC helpers ---------------------------

def _ncc_same_size(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Normalized cross-correlation per batch between two maps of same shape (B,1,H,W).
    Returns (B,)
    """
    a = A.reshape(A.shape[0], -1)
    b = B.reshape(B.shape[0], -1)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    num = (a * b).sum(dim=1)
    den = a.norm(dim=1) * b.norm(dim=1) + EPS
    return num / den


def _shift_with_replicate(X: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """
    Shift X (B,1,H,W) by integer offsets (dx,dy) using replicate padding.
    Positive dx -> right, positive dy -> down.
    """
    B, _, H, W = X.shape
    pad = (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0))  # left, right, top, bottom
    Y = F.pad(X, pad, mode="replicate")
    y0 = max(-dy, 0)
    x0 = max(-dx, 0)
    return Y[:, :, y0:y0 + H, x0:x0 + W]


def _overlap_pair(M: torch.Tensor, dx: int, dy: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return overlapping crops (A, B) from M and its (dx,dy)-shifted version,
    so that A and B have identical shape without any padding bias.
    M: (B,1,H,W). Positive dx -> right, positive dy -> down.
    """
    B, _, H, W = M.shape
    # X (columns)
    if dx >= 0:
        xa0, xa1 = 0, W - dx
        xb0, xb1 = dx, W
    else:
        xa0, xa1 = -dx, W
        xb0, xb1 = 0, W + dx  # dx < 0
    # Y (rows)
    if dy >= 0:
        ya0, ya1 = 0, H - dy
        yb0, yb1 = dy, H
    else:
        ya0, ya1 = -dy, H
        yb0, yb1 = 0, H + dy  # dy < 0

    # Guard (shouldn't happen with sane r), but keep safe.
    if xa1 <= xa0 or xb1 <= xb0 or ya1 <= ya0 or yb1 <= yb0:
        # No overlap: return 1x1 replicated center pixels to avoid crash; NCC ~ 1
        cY, cX = H // 2, W // 2
        A = M[:, :, cY:cY + 1, cX:cX + 1]
        Bm = A.clone()
        return A, Bm

    A = M[:, :, ya0:ya1, xa0:xa1]
    Bm = M[:, :, yb0:yb1, xb0:xb1]
    return A, Bm


# --------------------------- SCL ---------------------------

@torch.no_grad()
def compute_SCL(
    Fmap: torch.Tensor,
    tau: float = 0.2,
    num_radii: int = 8,
    r_max_frac: float = 0.4,
    directions: Iterable[Tuple[int, int]] = ((1, 0), (-1, 0), (0, 1), (0, -1)),
    pad_mode: str = "replicate",         # "replicate" | "crop"
    return_normalized: bool = False,     # also return SCL / min(H,W)
) -> torch.Tensor:
    """
    Semantic Aggregation Length: first radius r where NCC(M, shift_r(M)) < tau.
    - Fmap: (B,C,H,W)
    - tau:   correlation threshold (0.2 default)
    - num_radii: how many radii to evaluate (log/lin spaced to r_max)
    - r_max_frac: r_max = floor(r_max_frac * min(H,W)), upper bound for search
    - directions: offsets to average NCC over (default: 4 axial directions)
    - pad_mode: "replicate" (default) or "crop" (padding-free overlap)
    - return_normalized: if True, return SCL / min(H,W)
    Returns: (B,) SCL in pixels (float), or normalized if return_normalized=True.
             If correlation never drops below tau, returns r_max.
    """
    assert Fmap.dim() == 4, "Fmap must be (B,C,H,W)"
    B, C, H, W = Fmap.shape
    device = Fmap.device

    # Scalar standardized map
    M = _scalar_map_from_feat(Fmap, normalize=True).unsqueeze(1)  # (B,1,H,W)

    # Set radii list (avoid 0; ensure at least 1)
    r_max = max(1, int(r_max_frac * min(H, W)))
    if num_radii <= 0:
        num_radii = 1

    # approx geometric spacing, device-agnostic
    log_end = torch.log10(torch.tensor(float(r_max), dtype=torch.float32))
    radii = torch.logspace(0, log_end, steps=num_radii)
    radii = torch.round(radii).clamp(min=1, max=r_max).to(torch.int64)
    radii = torch.unique(radii).tolist()
    if len(radii) == 0:
        radii = [1]

    # Compute NCC vs r
    corr = torch.empty(B, len(radii), device=device)
    use_crop = (pad_mode == "crop")
    for j, r in enumerate(radii):
        vals = []
        for dx, dy in directions:
            if use_crop:
                A, Bm = _overlap_pair(M, dx * r, dy * r)
                vals.append(_ncc_same_size(A, Bm))
            else:
                A = _shift_with_replicate(M, dx * r, dy * r)
                vals.append(_ncc_same_size(A, M))
        corr[:, j] = torch.stack(vals, dim=0).mean(dim=0)

    # First r where NCC < tau; else r_max
    SCL = torch.full((B,), float(radii[-1]), device=device)
    below = (corr < tau)
    for b in range(B):
        hits = torch.where(below[b])[0]
        if hits.numel() > 0:
            SCL[b] = float(radii[int(hits[0])])

    if return_normalized:
        SCL = SCL / float(min(H, W))

    return SCL


# --------------------------- SFC ---------------------------

@torch.no_grad()
def compute_SFC(
    Fmap: torch.Tensor,
    grid: Tuple[int, int] = (16, 16),
    vectorized: bool = True,
) -> torch.Tensor:
    """
    Superpixel-like Feature Coherence using a regular grid over scalar map M.
    SFC = var(cell_means) / (mean(cell_variances) + eps).
    - Fmap: (B,C,H,W)
    - grid: (#cells_y, #cells_x). Will be clipped to image size.
    - vectorized: if True (default) uses unfold-based fast path with uniform cells.
    Returns: (B,) tensor. Higher -> stronger regional homogeneity and inter-region contrast.
    Notes:
      * Uses per-image standardized scalar map to remove global scale.
      * Gracefully handles tiny H/W (cells collapse to at least 1x1).
    """
    assert Fmap.dim() == 4, "Fmap must be (B,C,H,W)"
    B, C, H, W = Fmap.shape
    M = _scalar_map_from_feat(Fmap, normalize=True)  # (B,H,W)

    gy = max(1, min(int(grid[0]), H))
    gx = max(1, min(int(grid[1]), W))

    if not vectorized:
        # Reference implementation with ragged last cells (fine & robust)
        out = []
        cell_h = max(1, H // gy)
        cell_w = max(1, W // gx)
        for b in range(B):
            means = []
            variances = []
            y = 0
            while y < H:
                x = 0
                y1 = min(y + cell_h, H)
                while x < W:
                    x1 = min(x + cell_w, W)
                    patch = M[b, y:y1, x:x1]
                    means.append(patch.mean())
                    variances.append(patch.var(unbiased=False))
                    x = x1
                y = y1
            means = torch.stack(means)             # (Ncells,)
            variances = torch.stack(variances)     # (Ncells,)
            inter = means.var(unbiased=False)      # scalar
            intra = variances.mean()               # scalar
            sfc = inter / (intra + EPS)
            out.append(sfc)
        return torch.stack(out, dim=0)             # (B,)

    # Vectorized path: make all cells uniform by cropping to multiples
    M1 = M.unsqueeze(1)  # (B,1,H,W)
    H1 = (H // gy) * gy
    W1 = (W // gx) * gx
    # When H or W is very small, ensure not dropping everything
    if H1 == 0:
        H1 = H
        gy = 1
    if W1 == 0:
        W1 = W
        gx = 1
    M1 = M1[:, :, :H1, :W1]
    kh, kw = H1 // gy, W1 // gx  # each cell size >= 1
    patches = F.unfold(M1, kernel_size=(kh, kw), stride=(kh, kw))  # (B, kh*kw, gy*gx)
    means = patches.mean(dim=1)                                     # (B, gy*gx)
    vars_ = patches.var(dim=1, unbiased=False)                      # (B, gy*gx)
    inter = means.var(dim=1, unbiased=False)                        # (B,)
    intra = vars_.mean(dim=1)                                       # (B,)
    sfc = inter / (intra + EPS)
    return sfc


# --------------------------- CPR ---------------------------

@torch.no_grad()
def compute_CPR_nan(Fmap: torch.Tensor) -> torch.Tensor:
    """
    Channel Participation Ratio (effective dimensionality across channels).
    PR = (tr(C))^2 / tr(C^2), where C is the channel covariance over spatial samples.
    We return PR/C in (0,1]. Lower -> lower effective rank -> stronger semantic aggregation.
    - Fmap: (B,C,H,W)
    Returns: (B,) tensor in (0,1].
    Implementation details:
      * Center per-sample across spatial positions to form covariance.
      * Uses safe denominators for tiny spatial sizes.
      * If N<2 (not enough spatial samples), returns 1.0 (uninformative / maximal CPR).
    """
    assert Fmap.dim() == 4, "Fmap must be (B,C,H,W)"
    B, C, H, W = Fmap.shape
    X = _as_float(Fmap).reshape(B, C, -1)           # (B,C,N)
    X = X - X.mean(dim=2, keepdim=True)             # center across spatial samples
    N = X.shape[2]
    if N < 2:
        return torch.ones(B, device=Fmap.device)
    Cmat = torch.bmm(X, X.transpose(1, 2)) / (N - 1.0)  # (B,C,C)
    tr = Cmat.diagonal(dim1=1, dim2=2).sum(dim=1)       # (B,)
    tr2 = (Cmat * Cmat).sum(dim=(1, 2))                 # (B,)
    PR = (tr * tr) / (tr2 + EPS)                        # (B,)
    PR_norm = (PR / float(C)).clamp_(0.0, 1.0)
    return PR_norm

def compute_CPR(Fmap: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Channel Participation Ratio, 数值稳健版本。
    输入:
        Fmap: (B, C, H, W) 任意 dtype
    返回:
        CPR: (B,) in [0, 1]
    说明:
        - 先中心化(空间维), 再按样本做 RMS 归一化 (尺度不变, 防溢出)
        - 再提前除以 sqrt(N-1) 后做 bmm，避免中间乘法溢出
    """
    assert Fmap.dim() == 4, "Fmap must be (B, C, H, W)"
    B, C, H, W = Fmap.shape
    N = H * W
    if N < 2:
        # 样本太少，返回“无信息”的 1.0 或按需设为 0.0。
        return torch.ones(B, device=Fmap.device, dtype=torch.float32)

    # to float32 并拉平成 (B, C, N)
    X = Fmap.detach().to(torch.float32).reshape(B, C, N)

    # 去均值（空间维）
    X = X - X.mean(dim=2, keepdim=True)

    # 样本级 RMS 归一化（尺度不变；防止大幅值导致乘法溢出）
    rms = torch.sqrt((X * X).mean(dim=(1, 2), keepdim=True)).clamp_min(eps)
    X = X / rms

    # 在乘法之前先除以 sqrt(N-1)，与 X@X^T/(N-1) 等价，但数值更稳
    scale = math.sqrt(max(N - 1.0, 1.0))
    X = X / scale  # (B, C, N)

    # 现在计算协方差矩阵（已经规范化），C = X X^T
    Cmat = torch.bmm(X, X.transpose(1, 2))  # (B, C, C)

    # tr(C) 与 tr(C^2)
    tr  = Cmat.diag_embed().sum(dim=(1, 2)) if Cmat.dim()==4 else Cmat.diagonal(dim1=1, dim2=2).sum(dim=1)
    tr2 = (Cmat * Cmat).sum(dim=(1, 2))

    # PR = (tr(C))^2 / tr(C^2)；退化(tr2==0)时置 0
    PR = torch.where(tr2 > eps, (tr * tr) / (tr2 + eps), torch.zeros_like(tr))

    # 归一化到 [0,1]：CPR = PR / C
    CPR = (PR / float(C)).clamp(0.0, 1.0)

    # 屏蔽任何残余 NaN/inf
    CPR = torch.nan_to_num(CPR, nan=0.0, posinf=0.0, neginf=0.0)
    return CPR


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
        q_top: ECR 的 top 比例
    输出:
        dict: {'ECR','FER','GIC','ECI','ALI'} ，每个为 (B,)
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

    # --- ECR（简洁高效版本）
    N = H * W
    top_k = max(1, min(int(N * q_top), N - 1))
    Gf = G.reshape(B, -1)
    top_vals, _ = torch.topk(Gf, top_k, dim=1)
    rest_sum = Gf.sum(dim=1) - top_vals.sum(dim=1)
    rest_mean = rest_sum / float(N - top_k)
    ECR = top_vals.mean(dim=1) / (rest_mean.clamp_min(1e-8))

    # --- FER（中频 / 低频）
    low, mid, high = rfft2_radial_energy(M)
    FER = (mid + 1e-8) / (low + 1e-8)

    # --- GIC（修正版）
    GIC = gini_coefficient(G)

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
    # --- 
    SCL = compute_SCL(F32)
    sfc = compute_SFC(F32)
    cpr = compute_CPR(F32)

    stage_metrics = {'ECR': round(ECR.item(), 2), 'FER': round(FER.item(), 2), \
                     'GIC': round(GIC.item(), 2), 'SCL': round(SCL.item(), 2), 'SFC': round(sfc.item(), 2), 'CPR': round(cpr.item(), 2)}
    return stage_metrics

# ------------------------ Quick self-test (optional) ------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    # 退化图检查
    feat_flat = torch.zeros(2, 8, 32, 32)
    mflat = compute_metrics_from_features(feat_flat)
    print("Flat:", {k: v.tolist() for k, v in mflat.items()})
