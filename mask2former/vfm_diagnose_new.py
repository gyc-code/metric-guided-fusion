# -*- coding: utf-8 -*-
"""
Core edge metrics + stepwise visualization (fixed, cityscapes-proof)
- 可视化：Y / GI / PCA / G / E_pca / E_img / valid
- 指标：ECR_purity, Overlap_F1, FER, SCL, SFC(0~1)+SFC_raw, SCS, CPR, FCD
"""

import os, math
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

EPS = 1e-8

# ========================== 结构聚类评分（SCS） ==========================
@torch.no_grad()
def compute_SCS(
    Fmap: torch.Tensor,
    n_samples: int = 4096,     # 聚类前最大采样数
    k_list=(6, 8, 10),         # 评估簇数
    pca_dim: int = 32,         # PCA 上限
    seed: int = 0,
    sil_max: int = 2000,       # silhouette 二次采样上限（O(n^2) 控制）
) -> torch.Tensor:
    assert Fmap.dim() == 4, "Fmap must be (B,C,H,W)"
    B, C, H, W = Fmap.shape
    outs = []
    rng = np.random.default_rng(seed)

    for b in range(B):
        X = (
            Fmap[b].detach().to(torch.float32)
            .permute(1, 2, 0).reshape(-1, C)
            .cpu().numpy()
        )

        # 非法/退化
        if not np.isfinite(X).all():
            outs.append(torch.tensor(0.0, device=Fmap.device)); continue
        ch_std = X.std(axis=0, ddof=0)
        if float(np.max(ch_std)) < 1e-12:
            outs.append(torch.tensor(0.0, device=Fmap.device)); continue

        # z-score
        X = (X - X.mean(0, keepdims=True)) / (ch_std + 1e-6)

        # 聚类前下采样
        if X.shape[0] > n_samples:
            X = X[rng.choice(X.shape[0], n_samples, replace=False)]
        n = X.shape[0]
        if n < 8:
            outs.append(torch.tensor(0.0, device=Fmap.device)); continue

        # PCA（安全上限）
        ncomp_cap = int(max(1, min(pca_dim, X.shape[1], n - 1)))
        if X.shape[1] > ncomp_cap:
            X = PCA(n_components=ncomp_cap, random_state=seed).fit_transform(X)

        if float(np.max(X.std(axis=0, ddof=0))) < 1e-12:
            outs.append(torch.tensor(0.0, device=Fmap.device)); continue

        # 合法 k
        k_eff_list = [k for k in k_list if 2 <= k <= min(max(n // 2, 2), n - 1)]
        if not k_eff_list:
            outs.append(torch.tensor(0.0, device=Fmap.device)); continue

        scores = []
        for k in k_eff_list:
            try:
                km = KMeans(n_clusters=k, n_init=10, random_state=seed)
                labels = km.fit_predict(X)

                uniq, counts = np.unique(labels, return_counts=True)
                if uniq.size < 2 or np.min(counts) < 2:
                    continue

                # —— 分层子采样，保证每簇至少 2 个样本 —— #
                if n > sil_max:
                    take_per_label = {
                        u: max(2, int(round(sil_max * c / n)))
                        for u, c in zip(uniq, counts)
                    }
                    idx_chunks = []
                    for u, take in take_per_label.items():
                        idx_u = np.where(labels == u)[0]
                        if idx_u.size <= take:
                            idx_chunks.append(idx_u)
                        else:
                            idx_chunks.append(rng.choice(idx_u, take, replace=False))
                    idx_sil = np.concatenate(idx_chunks, axis=0)
                    y_sil = labels[idx_sil]
                    uniq_s, counts_s = np.unique(y_sil, return_counts=True)
                    if uniq_s.size < 2 or np.min(counts_s) < 2:
                        continue
                    X_sil = X[idx_sil]
                else:
                    X_sil = X
                    y_sil = labels
                    uniq_s, counts_s = uniq, counts
                    if uniq_s.size < 2 or np.min(counts_s) < 2:
                        continue

                s = silhouette_score(X_sil, y_sil, metric="euclidean")
                if np.isfinite(s):
                    scores.append(s)
            except Exception:
                continue

        if scores:
            val = (float(np.median(scores)) + 1.0) / 2.0  # [-1,1]→[0,1]
            outs.append(torch.tensor(val, device=Fmap.device, dtype=torch.float32))
        else:
            outs.append(torch.tensor(0.0, device=Fmap.device, dtype=torch.float32))

    return torch.stack(outs, dim=0)

# ========================== FCD 标定工具（可选） ==========================
def fcd_power_lift(prod_or_geom: torch.Tensor,
                   gamma: float = 0.25,
                   eps: float = 1e-12) -> torch.Tensor:
    """
    P_lift = P**gamma（0<gamma<1）但严格保证 0→0。
    """
    x = prod_or_geom.clone()
    zero_mask = (x <= 0)
    x = x.clamp_min(eps).pow(gamma)
    x = torch.where(zero_mask, torch.zeros_like(x), x)  # 保证 0 映射到 0
    return x


def fcd_exp_calibrate_01(x: torch.Tensor,
                         target_median: float = 0.5,
                         tau_fixed: float | None = None,
                         eps: float = 1e-12) -> torch.Tensor:
    """
    y = 1 - exp(-x/tau) 映射到 [0,1]，保证：
      - 单调；
      - 0→0；
      - 若 tau_fixed=None，用“x>0 的中位数”来解 tau，使 median(y)=target_median。
    """
    x = torch.clamp(x, min=0.0)

    if tau_fixed is None:
        xs = x.detach().flatten()
        xs = xs[torch.isfinite(xs)]
        xs_pos = xs[xs > 0]
        if xs_pos.numel() == 0:
            return torch.zeros_like(x)
        med = torch.quantile(xs_pos, 0.5)
        t_m = float(min(max(target_median, 1e-6), 1.0 - 1e-6))
        tau = med / (-math.log1p(-t_m))
        tau = torch.as_tensor(tau, device=x.device, dtype=x.dtype)
    else:
        tau = torch.as_tensor(tau_fixed, device=x.device, dtype=x.dtype)

    y = 1.0 - torch.exp(-x / (tau + eps))
    y = torch.where(x == 0, torch.zeros_like(y), y)
    return y.clamp(0, 1)

# ========================== FCD（改：用中心线构环带） ==========================
@torch.no_grad()
def compute_FCD_from_cached(
    *,
    G_raw: torch.Tensor,            # (B,H,W)  PCA梯度“原始幅值”（未做min-max）
    E_pca: torch.Tensor,            # (B,H,W)  特征强边掩码（未膨胀）
    centerline_img: torch.Tensor,   # (B,H,W)  图像强边中心线（~1px）
    valid: torch.Tensor,            # (B,H,W)  有效域掩码
    hf_share: torch.Tensor,         # (B,)     频域比例：(mid+high)/total
    SCL_feat: torch.Tensor,         # (B,)     SCL（单位=特征像素）
    image_hw: tuple,                # (H0, W0) 输入图像的空间分辨率
    feat_hw: tuple,                 # (Hf, Wf) 特征的空间分辨率
    r1: int = 1, r2: int = 3,       # 环带半径（单位=特征像素）
    scl_ref: float = 64.0,          # SCL 像素映射参考尺度
) -> torch.Tensor:
    """
    改进：环带仅由“图像边中心线”产生，避免稠密 E_img 导致 ring 为空。
    """
    assert G_raw.dim() == 3 and E_pca.shape == G_raw.shape and centerline_img.shape == G_raw.shape and valid.shape == G_raw.shape
    B, Hf, Wf = G_raw.shape
    H0, W0 = map(float, image_hw)
    hf, wf = map(float, feat_hw)

    # 1) 平均步幅与细尺度
    stride_eff = 0.5 * (H0 / max(hf, 1.0) + W0 / max(wf, 1.0))
    SCL_pix = SCL_feat * float(stride_eff)       # (B,)
    fine_scale = (scl_ref / (SCL_pix + scl_ref)).clamp(0, 1)  # (B,)

    # 2) 环带：以中心线为核
    pool_r1 = torch.nn.MaxPool2d(2*r1+1, stride=1, padding=r1)
    pool_r2 = torch.nn.MaxPool2d(2*r2+1, stride=1, padding=r2)
    d_in  = (pool_r1(centerline_img.float().unsqueeze(1))[:, 0] > 0)
    d_out = (pool_r2(centerline_img.float().unsqueeze(1))[:, 0] > 0)
    ring  = (d_out & (~d_in) & valid)
    far   = ((~d_out) & valid)

    # 3) 贴边增益 / 远区噪声（以 G_raw 为权）
    Gw = (G_raw * valid.float()).clamp_min(0)        # (B,Hf,Wf)
    total_e = (Gw * E_pca.float()).reshape(B, -1).sum(1).clamp_min(1e-8)
    near_e  = (Gw * (E_pca & ring).float()).reshape(B, -1).sum(1)
    far_e   = (Gw * (E_pca & far ).float()).reshape(B, -1).sum(1)

    near_gain_img = (near_e / total_e).clamp(0, 1)
    far_noise_img = (far_e  / total_e).clamp(0, 1)

    # 4) 组合（严格在 [0,1]），最后缩放到 0~100
    FCD = 100 * (near_gain_img * (1.0 - far_noise_img) * hf_share * fine_scale).clamp(0, 1)
    return FCD

# ========================== I/O & 基础工具 ==========================
def _save_gray(g: torch.Tensor, image_name: str, tag: str, k: str, model_name: str):
    """按要求的命名/范围保存灰度 PNG：{image_name}_{tag}_{k}_{model_name}.png"""
    os.makedirs('./diagnose_1023/', exist_ok=True)
    g = torch.nan_to_num(g.float(), 0.0).clamp(0, 1)
    img = (g * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    Image.fromarray(img, 'L').save(f'./diagnose_1023/{image_name}_{k}_{tag}_{model_name}.png')

def _prep_to_bchw(img: torch.Tensor, target_hw, device):
    """把输入图变成 (B,3,H,W) or (B,1,H,W)，缩放到 target_hw，归一化到[0,1]"""
    Ht, Wt = target_hw
    x = img
    if x.ndim == 2: x = x[None, None, ...]
    elif x.ndim == 3:
        if x.shape[0] in (1,3): x = x[None, ...]
        elif x.shape[-1] in (1,3): x = x.permute(2,0,1)[None, ...]
        else: raise ValueError(f"bad image shape {x.shape}")
    elif x.ndim == 4 and x.shape[1] not in (1,3) and x.shape[-1] in (1,3):
        x = x.permute(0,3,1,2)
    x = x.to(device=device, dtype=torch.float32)
    if x.max() > 1: x = x/255.0
    if x.shape[-2:] != (Ht, Wt):
        x = F.interpolate(x, size=(Ht,Wt), mode="bilinear", align_corners=False)
    return x.clamp_(0,1)

def _srgb_to_linear(x: torch.Tensor):
    return torch.where(x <= 0.04045, x/12.92, ((x+0.055)/1.055).pow(2.4))

def _rgb_to_luma_Y(bchw: torch.Tensor):
    """线性 RGB → 亮度 Y"""
    x = _srgb_to_linear(bchw)
    return (0.2126*x[:,0:1] + 0.7152*x[:,1:2] + 0.0722*x[:,2:3]).clamp(0,1)

def _sobel_replicate(x_1ch: torch.Tensor):
    """Sobel + replicate padding（避免外圈伪亮边）"""
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=x_1ch.dtype, device=x_1ch.device).view(1,1,3,3)/4.0
    ky = kx.transpose(2,3)
    xpad = F.pad(x_1ch, (1,1,1,1), mode='replicate')
    gx = F.conv2d(xpad, kx, padding=0)
    gy = F.conv2d(xpad, ky, padding=0)
    return (gx.square()+gy.square()).sqrt(), gx, gy

def _valid_mask(B:int, H:int, W:int, border:int, device):
    """去掉外圈 border 像素的有效域"""
    m = torch.ones(B,H,W, dtype=torch.bool, device=device)
    if border>0:
        m[:,:border,:] = False; m[:,-border:,:] = False
        m[:,:,:border] = False; m[:,:,-border:] = False
    return m

def _minmax_in_mask(x: torch.Tensor, mask: torch.Tensor):
    """在 mask 内做 min-max 归一化 -> [0,1]"""
    x = torch.nan_to_num(x.float(), 0.0, 0.0, 0.0)
    out = torch.zeros_like(x)
    for i in range(x.shape[0]):
        vals = x[i][mask[i]]
        if vals.numel()==0: continue
        mn, mx = vals.min(), vals.max()
        den = (mx - mn) if (mx - mn) > 1e-8 else torch.tensor(1.0, device=x.device)
        out[i] = ((x[i]-mn)/den).clamp(0,1)
    return out

def _top_mask_in_mask(G: torch.Tensor, q: float, valid: torch.Tensor, dr_thr: float = 1e-6):
    """
    在 valid 内“严格选 k=ceil(q*|valid|) 个像素”为 True；若动态范围太小或有效像素过少→全 False。
    返回 bool 掩码 (B,H,W)
    """
    import math
    B, H, W = G.shape
    M = torch.zeros_like(valid, dtype=torch.bool)
    for i in range(B):
        vi = valid[i]
        idxs = vi.flatten().nonzero().squeeze(1)
        if idxs.numel() == 0:
            continue
        vals = G[i].flatten()[idxs].to(torch.float32)
        vals = vals[torch.isfinite(vals)]
        if vals.numel() < 4:
            continue
        # 动态范围检查
        vmin, vmax = vals.min(), vals.max()
        if (vmax - vmin) < dr_thr:
            continue
        k = max(1, int(math.ceil(q * float(vi.sum().item()))))
        # 选 top-k（稳定）
        topk = torch.topk(vals, k, sorted=False)
        sel = torch.zeros_like(vi.flatten(), dtype=torch.bool)
        sel[idxs[topk.indices]] = True
        M[i] = sel.view(H, W)
    return M

# ========================== PCA 标量图（可视化） ==========================
def _scalar_map_from_feat_pca(
    Fmap: torch.Tensor,
    take_abs: bool = False,
    pool_down: int = 1,
    model_name: str = "",
    image_name: str = "",
    k: str = "",
    n_components: int = 1,
    upsample_to: tuple = None
) -> torch.Tensor:
    """按通道 z-score 后做 sklearn PCA；保存 PCA 可视化；返回 (B,H,W)∈[0,1]"""
    assert Fmap.dim() == 4, f"Fmap must be (B,C,H,W), got {Fmap.shape}"
    B, C, H, W = Fmap.shape
    X = torch.nan_to_num(Fmap.detach().to(torch.float32), 0.0)

    os.makedirs("./diagnose_1020/", exist_ok=True)
    outs = []

    for b in range(B):
        arr = X[b].permute(1, 2, 0).cpu().numpy()  # (H,W,C)
        Hs, Ws, Cs = arr.shape
        flat = arr.reshape(-1, Cs).astype(np.float32)

        mu  = flat.mean(0, keepdims=True)
        std = flat.std(0, keepdims=True)
        flat = (flat - mu) / (std + 1e-6)

        pca = PCA(n_components=n_components)
        Z = pca.fit_transform(flat).astype(np.float32)           # (HW,K)
        pca_r = Z.reshape(Hs, Ws, n_components)

        vis = pca_r[:, :, 0]
        if take_abs: vis = np.abs(vis)

        vmin, vmax = float(vis.min()), float(vis.max())
        vis01 = np.zeros_like(vis, dtype=np.float32) if (vmax-vmin)<1e-8 else ((vis - vmin) / (vmax - vmin)).astype(np.float32)

        if upsample_to is not None and (Hs, Ws) != tuple(upsample_to):
            t = torch.from_numpy(vis01)[None, None]
            t = F.interpolate(t, size=upsample_to, mode='bilinear', align_corners=False)
            vis01 = t[0, 0].cpu().numpy()

        out_path = f'./diagnose_1020/{image_name}_PCA_{k}_{model_name}.png'
        if B > 1:
            root, ext = os.path.splitext(out_path); out_path = f"{root}_b{b}{ext}"
        Image.fromarray((vis01 * 255).round().astype(np.uint8), mode="L").save(out_path)

        outs.append(torch.from_numpy(vis01))

    return torch.stack(outs, dim=0).to(Fmap.device, dtype=torch.float32)  # (B,H,W)

# ========================== 频域能量 (FER) ==========================
def _fftshift2d(x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[-2], x.shape[-1]
    return torch.roll(torch.roll(x, shifts=h // 2, dims=-2), shifts=w // 2, dims=-1)

def rfft2_radial_energy(M: torch.Tensor, low_thr: float = 0.25, mid_thr: float = 0.50):
    """
    输入 M:(B,H,W)（建议 z-score 后再进来）
    返回 (low, mid, high) 各 (B,)
    """
    B, H, W = M.shape
    F2 = torch.fft.fft2(M)                 # (B,H,W), complex
    P = (F2.real ** 2 + F2.imag ** 2)      # 功率谱
    P = _fftshift2d(P)                     # 中心化

    yy = torch.linspace(-1.0, 1.0, H, device=M.device, dtype=M.dtype).view(H, 1)
    xx = torch.linspace(-1.0, 1.0, W, device=M.device, dtype=M.dtype).view(1, W)
    rr = torch.sqrt(yy ** 2 + xx ** 2)
    rr = rr / rr.max().clamp_min(1e-6)

    low_mask  = (rr < low_thr).to(P.dtype)
    mid_mask  = ((rr >= low_thr) & (rr < mid_thr)).to(P.dtype)
    high_mask = (rr >= mid_thr).to(P.dtype)

    low  = (P * low_mask ).reshape(B,-1).sum(dim=1)
    mid  = (P * mid_mask ).reshape(B,-1).sum(dim=1)
    high = (P * high_mask).reshape(B,-1).sum(dim=1)
    return low, mid, high

# ========================== SCL / SFC / CPR ==========================
def _as_float(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(x): x = x.float()
    elif x.dtype in (torch.float16, torch.bfloat16): x = x.float()
    return x

def _scalar_map_from_feat(Fmap: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """通道 L2 → 标量图；可选 per-map z-score"""
    M = torch.linalg.vector_norm(_as_float(Fmap), ord=2, dim=1)  # (B,H,W)
    if normalize:
        B = M.shape[0]
        flat = M.reshape(B, -1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True).clamp_min(EPS)
        M = ((flat - mean) / std).reshape_as(M)
    return M

def _ncc_same_size(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    a = A.reshape(A.shape[0], -1); b = B.reshape(B.shape[0], -1)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    num = (a * b).sum(dim=1)
    den = a.norm(dim=1) * b.norm(dim=1) + EPS
    return num / den

def _shift_with_replicate(X: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    B, _, H, W = X.shape
    pad = (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0))
    Y = F.pad(X, pad, mode="replicate")
    y0 = max(-dy, 0); x0 = max(-dx, 0)
    return Y[:, :, y0:y0 + H, x0:x0 + W]

def _overlap_pair(M: torch.Tensor, dx: int, dy: int):
    B, _, H, W = M.shape
    if dx >= 0: xa0, xa1, xb0, xb1 = 0, W-dx, dx, W
    else:       xa0, xa1, xb0, xb1 = -dx, W, 0, W+dx
    if dy >= 0: ya0, ya1, yb0, yb1 = 0, H-dy, dy, H
    else:       ya0, ya1, yb0, yb1 = -dy, H, 0, H+dy
    if xa1<=xa0 or xb1<=xb0 or ya1<=ya0 or yb1<=yb0:
        cY, cX = H//2, W//2
        A = M[:, :, cY:cY+1, cX:cX+1]; Bm = A.clone(); return A, Bm
    A  = M[:, :, ya0:ya1, xa0:xa1]
    Bm = M[:, :, yb0:yb1, xb0:xb1]
    return A, Bm

@torch.no_grad()
def compute_SCL(
    Fmap: torch.Tensor,
    tau: float = 0.7,              # ↑ 提高阈值（调用时会覆盖成更稳的 0.5）
    num_radii: int = 12,           # 调用时会设置为 16
    r_max_frac: float = 0.6,
    directions=((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)),
    pad_mode: str = "crop",
    return_normalized: bool = False,
    highpass_ks: int = 5           # 调用时会设为 3
) -> torch.Tensor:
    """SCL：NCC(M, shift_r(M)) 首次小于 tau 的 r；改进版避免饱和到 r_max。"""
    assert Fmap.dim() == 4, "Fmap must be (B,C,H,W)"
    B, C, H, W = Fmap.shape

    # 1) 通道L2 -> 标量
    M0 = torch.linalg.vector_norm(_as_float(Fmap), ord=2, dim=1)  # (B,H,W)

    # 2) 轻量高通（去低频趋势）
    if highpass_ks and highpass_ks > 1:
        ks = int(highpass_ks) + (highpass_ks % 2 == 0)
        pad = ks // 2
        M_blur = F.avg_pool2d(M0.unsqueeze(1), kernel_size=ks, stride=1, padding=pad)[:,0]
        M = (M0 - M_blur)
    else:
        M = M0

    # per-map z-score
    flat = M.reshape(B, -1)
    mu   = flat.mean(1, keepdim=True)
    std  = flat.std(1, keepdim=True).clamp_min(EPS)
    M = ((flat - mu) / std).reshape_as(M).unsqueeze(1)  # (B,1,H,W)

    # 3) 半径集合（对数均匀）
    r_max = max(1, int(r_max_frac * min(H, W)))
    log_end = torch.log10(torch.tensor(float(r_max), dtype=torch.float32))
    radii = torch.logspace(0, log_end, steps=max(1, num_radii))
    radii = torch.round(radii).clamp(min=1, max=r_max).to(torch.int64)
    radii = torch.unique(radii).tolist() or [1]

    # 4) NCC vs r
    corr = torch.empty(B, len(radii), device=Fmap.device)
    use_crop = (pad_mode == "crop")
    for j, r in enumerate(radii):
        vals = []
        for dx, dy in directions:
            if use_crop:
                A, Bm = _overlap_pair(M, dx*r, dy*r)
                vals.append(_ncc_same_size(A, Bm))
            else:
                A = _shift_with_replicate(M, dx*r, dy*r)
                vals.append(_ncc_same_size(A, M))
        corr[:, j] = torch.stack(vals, dim=0).mean(dim=0)

    # 5) 首次击穿；若依然未击穿，则返回 r_max
    SCL = torch.full((B,), float(radii[-1]), device=Fmap.device)
    below = (corr < tau)
    for b in range(B):
        idx = torch.where(below[b])[0]
        if idx.numel() > 0:
            SCL[b] = float(radii[int(idx[0])])

    if return_normalized:
        SCL = SCL / float(min(H, W))
    return SCL


@torch.no_grad()
def compute_SFC(
    Fmap: torch.Tensor,
    grid: tuple | None = None,          # 兼容旧接口
    cell: tuple | None = (8, 8),        # ★ 推荐：固定格子大小（特征像素）
    pre_smooth_ks: int = 0,             # 可选：3/5 轻平滑，降格内噪声
    clip_sigma: float | None = 5.0,     # 可选：重尾剪裁（μ±kσ）
    return_both: bool = False,
    return_debug: bool = False,
):
    assert Fmap.dim() == 4
    B, C, H, W = Fmap.shape

    # 标量化 + z-score
    M = torch.linalg.vector_norm(Fmap.to(torch.float32), ord=2, dim=1)  # (B,H,W)
    flat = M.view(B, -1)
    mu = flat.mean(1, keepdim=True); std = flat.std(1, keepdim=True).clamp_min(1e-6)
    M = ((flat - mu) / std).view_as(M)

    # 轻剪裁（稳定重尾）
    if clip_sigma and clip_sigma > 0:
        mu2 = M.view(B, -1).mean(1, keepdim=True).view(B,1,1).expand_as(M)
        std2 = M.view(B, -1).std(1, keepdim=True).view(B,1,1).expand_as(M).clamp_min(1e-6)
        M = M.clamp(mu2 - clip_sigma*std2, mu2 + clip_sigma*std2)

    # 可选平滑，降低格内微纹理造成的 intra 偏大
    if pre_smooth_ks and pre_smooth_ks > 1:
        ks = int(pre_smooth_ks) + (pre_smooth_ks % 2 == 0)
        pad = ks // 2
        M = F.avg_pool2d(M.unsqueeze(1), kernel_size=ks, stride=1, padding=pad)[:,0]

    # 计算格子大小 / 数量
    if cell is not None:
        ch, cw = max(1, int(cell[0])), max(1, int(cell[1]))
        kh = min(ch, H); kw = min(cw, W)          # 单格像素尺寸（尽量固定）
        H1 = (H // kh) * kh; W1 = (W // kw) * kw
        gy = max(1, H1 // kh); gx = max(1, W1 // kw)
    else:
        # 回退到旧逻辑（不推荐）
        gy = max(1, min(int(grid[0]), H)); gx = max(1, min(int(grid[1]), W))
        H1 = (H // gy) * gy; W1 = (W // gx) * gx
        kh, kw = max(1, H1 // gy), max(1, W1 // gx)

    M1 = M[:, :H1, :W1].unsqueeze(1)               # (B,1,H1,W1)
    patches = F.unfold(M1, kernel_size=(kh, kw), stride=(kh, kw))  # (B, kh*kw, gy*gx)
    means = patches.mean(dim=1)                    # (B, gy*gx)
    vars_ = patches.var(dim=1, unbiased=False)     # (B, gy*gx)

    inter = means.var(dim=1, unbiased=False)
    intra = vars_.mean(dim=1).clamp_min(1e-6)

    sfc_raw = inter / intra
    sfc01   = (inter / (inter + intra)).clamp(0, 1)

    if return_debug:
        dbg = {"H":H, "W":W, "gy":int(gy), "gx":int(gx), "kh":int(kh), "kw":int(kw),
               "inter_mean": float(inter.mean().item()), "intra_mean": float(intra.mean().item())}
        return (sfc_raw, sfc01, dbg) if return_both else (sfc01, dbg)
    return (sfc_raw, sfc01) if return_both else sfc01


@torch.no_grad()
def compute_SFC_old(Fmap: torch.Tensor, grid=(16,16), return_both: bool = False):
    """SFC：网格结构性（inter vs intra）。返回有界版避免爆表；可同时给回 raw。"""
    assert Fmap.dim() == 4
    B, C, H, W = Fmap.shape
    M = _scalar_map_from_feat(Fmap, normalize=True)   # (B,H,W)
    gy = max(1, min(int(grid[0]), H)); gx = max(1, min(int(grid[1]), W))
    M1 = M.unsqueeze(1)
    H1 = (H // gy) * gy; W1 = (W // gx) * gx
    if H1 == 0: H1 = H; gy = 1
    if W1 == 0: W1 = W; gx = 1
    M1 = M1[:, :, :H1, :W1]
    kh, kw = H1 // gy, W1 // gx

    patches = F.unfold(M1, kernel_size=(kh, kw), stride=(kh, kw))  # (B, kh*kw, gy*gx)
    means = patches.mean(dim=1)                          # (B, gy*gx)
    vars_ = patches.var(dim=1, unbiased=False)           # (B, gy*gx)

    inter = means.var(dim=1, unbiased=False)             # 跨cell方差
    intra = vars_.mean(dim=1).clamp_min(1e-6)            # cell内平均方差

    sfc_raw = (inter / intra)                            # 原定义（可能很大）
    sfc01   = (inter / (inter + intra)).clamp(0, 1)      # 有界、单调、可比

    return (sfc_raw, sfc01) if return_both else sfc01

@torch.no_grad()
def compute_CPR(Fmap: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """CPR：频道参与率（有效秩） PR/C ∈ [0,1]"""
    assert Fmap.dim() == 4, "Fmap must be (B, C, H, W)"
    B, C, H, W = Fmap.shape
    N = H*W
    if N < 2: return torch.ones(B, device=Fmap.device, dtype=torch.float32)
    X = Fmap.detach().to(torch.float32).reshape(B, C, N)
    X = X - X.mean(dim=2, keepdim=True)
    rms = torch.sqrt((X*X).mean(dim=(1,2), keepdim=True)).clamp_min(eps)
    X = X / rms
    scale = math.sqrt(max(N-1.0,1.0)); X = X / scale
    Cmat = torch.bmm(X, X.transpose(1,2))
    tr  = Cmat.diagonal(dim1=1, dim2=2).sum(dim=1)
    tr2 = (Cmat*Cmat).sum(dim=(1,2))
    PR = torch.where(tr2 > eps, (tr*tr)/(tr2+eps), torch.zeros_like(tr))
    CPR = (PR / float(C)).clamp(0.0, 1.0)
    return torch.nan_to_num(CPR, nan=0.0, posinf=0.0, neginf=0.0)

def _hann2d(h: int, w: int, device, dtype):
    wy = torch.hann_window(h, periodic=False, device=device, dtype=dtype)
    wx = torch.hann_window(w, periodic=False, device=device, dtype=dtype)
    return (wy.view(h,1) * wx.view(1,w)).clamp_min(0)

def _centerline_localmax(Gmag01: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    Gmag01: (B,H,W) 已在 valid 内做了 [0,1] 归一化的梯度幅值（GI 或 PCA 梯度都可）
    返回 bool(B,H,W) 的中心线（约1px厚），用于构造环带
    """
    B, H, W = Gmag01.shape
    pool = torch.nn.MaxPool2d(3, stride=1, padding=1)
    G1 = Gmag01.clamp(0,1).unsqueeze(1)
    Gmax = pool(G1)[:,0]
    center = (Gmag01 >= (Gmax - 1e-6)) & valid  # 允许微小数值误差
    return center


@torch.no_grad()
def _feature_grad_mag(Fmap: torch.Tensor, valid: torch.Tensor, zscore: bool = True):
    """
    多通道特征梯度幅值（不降维）:
      - 可选 per-channel z-score（按图内统计）
      - 每通道 Sobel（replicate pad）
      - 通道维 L2 聚合 -> (B,H,W)
      - 返回: G_raw（能量用）, G_vis（在 valid 内 min-max，出图/阈值用）
    """
    assert Fmap.dim()==4
    B, C, H, W = Fmap.shape
    X = Fmap.detach().to(torch.float32)

    if zscore:
        flat = X.reshape(B, C, -1)
        mu   = flat.mean(dim=2, keepdim=True)
        std  = flat.std(dim=2, keepdim=True).clamp_min(1e-6)
        X = (flat - mu) / std
        X = X.reshape(B, C, H, W)

    # Sobel per-channel, group conv（/8 更“规范”）
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                      dtype=X.dtype, device=X.device).view(1,1,3,3) / 8.0
    ky = kx.transpose(2,3)
    Kx = kx.repeat(C,1,1,1)
    Ky = ky.repeat(C,1,1,1)
    Xpad = F.pad(X, (1,1,1,1), mode='replicate')
    gx = F.conv2d(Xpad, Kx, groups=C)
    gy = F.conv2d(Xpad, Ky, groups=C)
    G_raw = torch.sqrt((gx*gx + gy*gy).sum(dim=1)).clamp_min(0)   # (B,H,W)

    # 在 valid 内 min-max 仅供显示/阈值
    G_vis = _minmax_in_mask(G_raw.clone(), valid)
    return G_raw, G_vis

@torch.no_grad()
def compute_FER_interior(
    Fmap: torch.Tensor,         # (B,C,H,W) float32
    E_img: torch.Tensor,        # (B,H,W) bool, 图像强边
    E_feat: torch.Tensor,       # (B,H,W) bool, 特征强边
    valid: torch.Tensor,        # (B,H,W) bool
    dilate_r: int = 2,
    low_thr: float = 0.15,
    mid_thr: float = 0.45,
):
    B, C, H, W = Fmap.shape
    # 1) 通道L2标量 + z-score（与你现有 FER 保持一致）
    M = _scalar_map_from_feat(Fmap, normalize=True)  # (B,H,W)

    # 2) 形态学膨胀后做“内域”掩码（排除边及其邻域）
    if dilate_r > 0:
        pool = torch.nn.MaxPool2d(2*dilate_r+1, stride=1, padding=dilate_r)
        eimg_d  = (pool(E_img.float().unsqueeze(1))[:,0] > 0)
        efeat_d = (pool(E_feat.float().unsqueeze(1))[:,0] > 0)
    else:
        eimg_d, efeat_d = E_img, E_feat
    interior = valid & (~eimg_d) & (~efeat_d)        # (B,H,W)

    # 若内域太小，回退到全图 FER
    area = interior.flatten(1).float().mean(1)       # 占比
    too_small = (area < 0.05)

    # 3) 软掩码（apodize）：避免硬裁剪带来的频谱泄漏
    # 用均值池平滑掩码，相当于把边界变成 0~1 过渡
    soft = interior.float().unsqueeze(1)             # (B,1,H,W)
    soft = F.avg_pool2d(soft, kernel_size=5, stride=1, padding=2)
    soft = F.avg_pool2d(soft, kernel_size=5, stride=1, padding=2)
    soft = soft[:,0].clamp(0,1)                      # (B,H,W)

    # 4) 乘 Hann 窗 + 软掩码再做频域分带
    win = _hann2d(H, W, M.device, M.dtype)
    Mw = M * win * soft                              # (B,H,W)
    low, mid, high = rfft2_radial_energy(Mw, low_thr=low_thr, mid_thr=mid_thr)
    total = (low + mid + high).clamp_min(1e-8)
    fer_int = (mid / total).clamp(0, 1)              # (B,)

    # 小区域回退：用全图 FER（更稳）
    if too_small.any():
        win_full = _hann2d(H, W, M.device, M.dtype)
        M_full = M * win_full
        l2, m2, h2 = rfft2_radial_energy(M_full, low_thr=low_thr, mid_thr=mid_thr)
        fer_full = (m2 / (l2+m2+h2).clamp_min(1e-8)).clamp(0,1)
        fer_int = torch.where(too_small, fer_full, fer_int)

    return fer_int  # (B,)

def triangular_gate(x: torch.Tensor, center=0.48, width=0.28):
    # 在 [center-width/2, center+width/2] 线性升降；区间外线性衰减至 0
    left  = center - width/2
    right = center + width/2
    y = 1 - (2*torch.abs(x - center) / width)
    return y.clamp(0, 1)


@torch.no_grad()
def compute_metrics_from_features(
    feat: torch.Tensor,              # (B,C,H,W)
    image_chw_uint8: torch.Tensor,   # (3,H0,W0) or (B,3,H0,W0)
    model_name: str = "test_model",
    image_name: str = "test_image",
    k: str = "test_k",
    q_top: float = 0.40,             # 特征强边 top 比例
    q_img: float = 0.40,             # 图像强边 top 比例
    dilate_r: int = 2,
    border: int = 1,
    save_pca_vis: bool = False       # 仅可视化，不参与计算
):
    """
    保存：Y, GI, (可选)PCAvis, G_feat, E_feat, E_img, valid
    返回：ECR, Overlap_F1, FCD, FER, SCL, SFC(0~1)+SFC_raw, CPR, SCS
    """
    assert feat.ndim == 4
    B,C,H,W = feat.shape
    dev = feat.device
    valid = _valid_mask(B,H,W,border, dev)

    # 1) 灰度 Y & 图像梯度 GI（显示/阈值）
    img = _prep_to_bchw(image_chw_uint8, (H,W), dev)
    if img.shape[0] != B: img = img.expand(B,-1,-1,-1)
    Y = _rgb_to_luma_Y(img)                          # (B,1,H,W)
    Y01 = _minmax_in_mask(Y.squeeze(1), valid)
    _save_gray(Y01[0], image_name, 'Y',  k, model_name)

    GI, _, _ = _sobel_replicate(Y)                   # (B,1,H,W)
    GI = _minmax_in_mask(GI.squeeze(1), valid)       # (B,H,W)
    _save_gray(GI[0], image_name, 'GI', k, model_name)

    # 2) （可选）仅做 PCA 可视化，不参与任何指标
    if save_pca_vis:
        M_vis = _scalar_map_from_feat_pca(
            feat, model_name=model_name, image_name=image_name, k=k,
            n_components=1, upsample_to=(H,W)
        ).clamp(0,1)
        _save_gray(M_vis[0], image_name, 'PCAvis', k, model_name)

    # 3) 多通道特征梯度（不降维）
    G_feat_raw, G_feat_vis = _feature_grad_mag(feat, valid, zscore=True)   # (B,H,W)
    _save_gray(G_feat_vis[0], image_name, 'Gfeat', k, model_name)

    # 4) 强边掩码（特征 & 图像）
    E_feat    = _top_mask_in_mask(G_feat_vis, q_top, valid)   # 特征强边（阈值用可视化幅值）
    E_img_top = _top_mask_in_mask(GI,         q_img, valid)   # 图像强边

    # 中心线（用于 FCD 环带 & 内部掩码）
    center_img = _centerline_localmax(GI, valid) & E_img_top

    _save_gray(E_feat[0].float(),    image_name, 'E_feat', k, model_name)
    _save_gray(E_img_top[0].float(), image_name, 'E_img',  k, model_name)
    _save_gray(valid[0].float(),     image_name, 'valid',  k, model_name)

    # 5) 膨胀容忍（F1 用）
    if dilate_r > 0:
        pool = torch.nn.MaxPool2d(2*dilate_r+1, stride=1, padding=dilate_r)
        E_img_d  = (pool(E_img_top.float().unsqueeze(1))[:,0] > 0) & valid
        E_feat_d = (pool(E_feat.float().unsqueeze(1))[:,0] > 0) & valid
    else:
        E_img_d, E_feat_d = E_img_top & valid, E_feat & valid

    # 6) ECR / F1（能量用 raw）
    Gw = (G_feat_raw * valid.float()).clamp_min(0)  # (B,H,W)
    true_energy  = (Gw * (E_feat & E_img_top).float()).flatten(1).sum(1)
    false_energy = (Gw * (E_feat & (~E_img_top) & valid).float()).flatten(1).sum(1)
    ECR_purity   = true_energy / (true_energy + false_energy + EPS)

    TP = (E_feat & E_img_d).flatten(1).sum(1).float()
    P  = (E_feat & valid).flatten(1).sum(1).float().clamp_min(1.0)
    R_ = (E_img_top & valid).flatten(1).sum(1).float().clamp_min(1.0)
    precision = TP / P
    recall    = TP / R_
    Overlap_F1 = (2 * precision * recall) / (precision + recall + EPS)
    Overlap_F1 = Overlap_F1.clamp(0, 1)

    # 7) FER / SCL / SFC / CPR / SCS（都基于原特征，不用 PCA）
    F32 = torch.nan_to_num(feat.detach().to(torch.float32), 0.0)

    # FER: 特征 L2 标量图 -> Hann -> 频带占比
    M_scalar = _scalar_map_from_feat(F32, normalize=True)   # (B,H,W)
    win = _hann2d(H, W, M_scalar.device, M_scalar.dtype)
    Mz_win = M_scalar * win
    low, mid, high = rfft2_radial_energy(Mz_win, low_thr=0.15, mid_thr=0.45)
    total = (low + mid + high).clamp_min(1e-8)
    FER = (mid / total).clamp(0, 1)
    hf_share = ((mid + high) / total).clamp(0, 1)

    # SCL / SFC / CPR / SCS
    SCL = compute_SCL(
        F32, tau=0.5, num_radii=16, r_max_frac=0.6,
        directions=((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)),
        pad_mode="crop", highpass_ks=3,
    )
    # SFC_raw, SFC_01 = compute_SFC(F32, grid=(16,16), return_both=True)
    # 固定单格大小 8×8（特征像素）；必要时用 12×12 做对照
    SFC_raw, SFC_01 = compute_SFC(F32, cell=(8,8), pre_smooth_ks=0, clip_sigma=5.0, return_both=True)

    CPR = compute_CPR(F32)
    SCS = compute_SCS(F32)

    FER_int = compute_FER_interior(F32, E_img_top, E_feat, valid, dilate_r=2)
    gate = triangular_gate(FER_int, center=0.48, width=0.28)

    # 8) FCD（用特征强边 & 特征 raw 梯度）
    FCD = compute_FCD_from_cached(
        G_raw=G_feat_raw,               # (B,H,W)
        E_pca=E_feat,                   # (B,H,W) —— 这里就是特征强边
        centerline_img=center_img,      # (B,H,W)
        valid=valid,
        hf_share=hf_share,              # (B,)
        SCL_feat=SCL,                   # (B,)
        image_hw=img.shape[-2:],        # (H0,W0)
        feat_hw=(H, W),                 # (Hf,Wf)
        r1=dilate_r if dilate_r>0 else 1,
        r2=max(dilate_r*2, 3) if dilate_r>0 else 3,
        scl_ref=64.0,
    )

    return {
        "ECR": round(ECR_purity.item(), 2),
        # "Overlap_F1": round(Overlap_F1.item(), 2),
        "FCD": round(FCD.item(), 2),


        "FER": round(FER_int.item(), 2),
        # "SCL": round(SCL.item(), 2),
        "SFC": round(SFC_01.item(), 2),       # 0~1（稳定）
        # "SFC_raw": round(SFC_raw.item(), 2),  # 原始比值（可能很大，仅参考）
        "SCS": round(SCS.item(), 2),
        "CPR": round(CPR.item(), 2)
    }

def fuse_two_indices(
    metrics: dict,
    calib: dict | None = None
):
    return metrics 


# ------------------------ Quick self-test (optional) ------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    # 退化图检查
    feat_flat = torch.zeros(1, 8, 32, 32)
    imgs = torch.rand(3, 1024, 2048)
    mflat = compute_metrics_from_features(
        feat_flat, imgs, model_name="main", image_name="a", k="test",
        q_top=0.10, q_img=0.10, dilate_r=2
    )
    print("Flat:", {k: v for k, v in mflat.items()})
