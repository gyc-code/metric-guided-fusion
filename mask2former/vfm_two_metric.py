# -*- coding: utf-8 -*-
"""
Core edge/semantic metrics + fusion indices (cityscapes-proof, decoupled FER)
- 可视化：Y / GI / PCA / G / E_pca / E_img / valid
- 指标：ECR, Overlap_F1, FCD, FER(全图), FER_interior(非边), SCL, SFC(0~1)+SFC_raw, SCS, CPR
- 融合：SemIdx（语义综合），EdgeIdx（边缘综合）
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

        if not np.isfinite(X).all():
            outs.append(torch.tensor(0.0, device=Fmap.device)); continue
        ch_std = X.std(axis=0, ddof=0)
        if float(np.max(ch_std)) < 1e-12:
            outs.append(torch.tensor(0.0, device=Fmap.device)); continue

        X = (X - X.mean(0, keepdims=True)) / (ch_std + 1e-6)

        if X.shape[0] > n_samples:
            X = X[rng.choice(X.shape[0], n_samples, replace=False)]
        n = X.shape[0]
        if n < 8:
            outs.append(torch.tensor(0.0, device=Fmap.device)); continue

        ncomp_cap = int(max(1, min(pca_dim, X.shape[1], n - 1)))
        if X.shape[1] > ncomp_cap:
            X = PCA(n_components=ncomp_cap, random_state=seed).fit_transform(X)

        if float(np.max(X.std(axis=0, ddof=0))) < 1e-12:
            outs.append(torch.tensor(0.0, device=Fmap.device)); continue

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

                if n > sil_max:
                    take_per_label = {u: max(2, int(round(sil_max * c / n))) for u, c in zip(uniq, counts)}
                    idx_chunks = []
                    for u, take in take_per_label.items():
                        idx_u = np.where(labels == u)[0]
                        idx_chunks.append(idx_u if idx_u.size <= take else rng.choice(idx_u, take, replace=False))
                    idx_sil = np.concatenate(idx_chunks, axis=0)
                    y_sil = labels[idx_sil]
                    uniq_s, counts_s = np.unique(y_sil, return_counts=True)
                    if uniq_s.size < 2 or np.min(counts_s) < 2:
                        continue
                    X_sil = X[idx_sil]
                else:
                    X_sil = X; y_sil = labels
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
    x = prod_or_geom.clone()
    zero_mask = (x <= 0)
    x = x.clamp_min(eps).pow(gamma)
    x = torch.where(zero_mask, torch.zeros_like(x), x)
    return x

def fcd_exp_calibrate_01(x: torch.Tensor,
                         target_median: float = 0.5,
                         tau_fixed: float | None = None,
                         eps: float = 1e-12) -> torch.Tensor:
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

# ========================== FCD（用中心线构环带） ==========================
@torch.no_grad()
def compute_FCD_from_cached(
    *,
    G_raw: torch.Tensor,            # (B,H,W)  PCA梯度“原始幅值”
    E_pca: torch.Tensor,            # (B,H,W)  特征强边（未膨胀）
    centerline_img: torch.Tensor,   # (B,H,W)  图像边中心线（~1px）
    valid: torch.Tensor,            # (B,H,W)
    hf_share: torch.Tensor,         # (B,)     (mid+high)/total
    SCL_feat: torch.Tensor,         # (B,)
    image_hw: tuple,                # (H0, W0)
    feat_hw: tuple,                 # (Hf, Wf)
    r1: int = 1, r2: int = 3,
    scl_ref: float = 64.0,
) -> torch.Tensor:
    assert G_raw.dim() == 3 and E_pca.shape == G_raw.shape and centerline_img.shape == G_raw.shape and valid.shape == G_raw.shape
    B, Hf, Wf = G_raw.shape
    H0, W0 = map(float, image_hw); hf, wf = map(float, feat_hw)

    stride_eff = 0.5 * (H0 / max(hf, 1.0) + W0 / max(wf, 1.0))
    SCL_pix = SCL_feat * float(stride_eff)
    fine_scale = (scl_ref / (SCL_pix + scl_ref)).clamp(0, 1)

    pool_r1 = torch.nn.MaxPool2d(2*r1+1, stride=1, padding=r1)
    pool_r2 = torch.nn.MaxPool2d(2*r2+1, stride=1, padding=r2)
    d_in  = (pool_r1(centerline_img.float().unsqueeze(1))[:, 0] > 0)
    d_out = (pool_r2(centerline_img.float().unsqueeze(1))[:, 0] > 0)
    ring  = (d_out & (~d_in) & valid)
    far   = ((~d_out) & valid)

    Gw = (G_raw * valid.float()).clamp_min(0)
    total_e = (Gw * E_pca.float()).reshape(B, -1).sum(1).clamp_min(1e-8)
    near_e  = (Gw * (E_pca & ring).float()).reshape(B, -1).sum(1)
    far_e   = (Gw * (E_pca & far ).float()).reshape(B, -1).sum(1)

    near_gain_img = (near_e / total_e).clamp(0, 1)
    far_noise_img = (far_e  / total_e).clamp(0, 1)

    FCD = 100.0 * (near_gain_img * (1.0 - far_noise_img) * hf_share * fine_scale).clamp(0, 1)
    return FCD

# ========================== I/O & 基础工具 ==========================
def _save_gray(g: torch.Tensor, image_name: str, tag: str, k: str, model_name: str):
    os.makedirs('./diagnose_1020/', exist_ok=True)
    g = torch.nan_to_num(g.float(), 0.0).clamp(0, 1)
    img = (g * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    Image.fromarray(img, 'L').save(f'./diagnose_1020/{image_name}_{tag}_{k}_{model_name}.png')

def _prep_to_bchw(img: torch.Tensor, target_hw, device):
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
    x = _srgb_to_linear(bchw)
    return (0.2126*x[:,0:1] + 0.7152*x[:,1:2] + 0.0722*x[:,2:3]).clamp(0,1)

def _sobel_replicate(x_1ch: torch.Tensor):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=x_1ch.dtype, device=x_1ch.device).view(1,1,3,3)/4.0
    ky = kx.transpose(2,3)
    xpad = F.pad(x_1ch, (1,1,1,1), mode='replicate')
    gx = F.conv2d(xpad, kx, padding=0)
    gy = F.conv2d(xpad, ky, padding=0)
    return (gx.square()+gy.square()).sqrt(), gx, gy

def _valid_mask(B:int, H:int, W:int, border:int, device):
    m = torch.ones(B,H,W, dtype=torch.bool, device=device)
    if border>0:
        m[:,:border,:] = False; m[:,-border:,:] = False
        m[:,:,:border] = False; m[:,:,-border:] = False
    return m

def _minmax_in_mask(x: torch.Tensor, mask: torch.Tensor):
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
        vmin, vmax = vals.min(), vals.max()
        if (vmax - vmin) < dr_thr:
            continue
        k = max(1, int(math.ceil(q * float(vi.sum().item()))))
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
    tau: float = 0.5,
    num_radii: int = 16,
    r_max_frac: float = 0.6,
    directions=((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)),
    pad_mode: str = "crop",
    return_normalized: bool = False,
    highpass_ks: int = 3
) -> torch.Tensor:
    assert Fmap.dim() == 4, "Fmap must be (B,C,H,W)"
    B, C, H, W = Fmap.shape

    M0 = torch.linalg.vector_norm(_as_float(Fmap), ord=2, dim=1)  # (B,H,W)

    if highpass_ks and highpass_ks > 1:
        ks = int(highpass_ks) + (highpass_ks % 2 == 0)
        pad = ks // 2
        M_blur = F.avg_pool2d(M0.unsqueeze(1), kernel_size=ks, stride=1, padding=pad)[:,0]
        M = (M0 - M_blur)
    else:
        M = M0

    flat = M.reshape(B, -1)
    mu   = flat.mean(1, keepdim=True)
    std  = flat.std(1, keepdim=True).clamp_min(EPS)
    M = ((flat - mu) / std).reshape_as(M).unsqueeze(1)

    r_max = max(1, int(r_max_frac * min(H, W)))
    log_end = torch.log10(torch.tensor(float(r_max), dtype=torch.float32))
    radii = torch.logspace(0, log_end, steps=max(1, num_radii))
    radii = torch.round(radii).clamp(min=1, max=r_max).to(torch.int64)
    radii = torch.unique(radii).tolist() or [1]

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
def compute_SFC(Fmap: torch.Tensor, grid=(16,16), return_both: bool = False):
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

    patches = F.unfold(M1, kernel_size=(kh, kw), stride=(kh, kw))
    means = patches.mean(dim=1)
    vars_ = patches.var(dim=1, unbiased=False)

    inter = means.var(dim=1, unbiased=False)
    intra = vars_.mean(dim=1).clamp_min(1e-6)

    sfc_raw = (inter / intra)
    sfc01   = (inter / (inter + intra)).clamp(0, 1)

    return (sfc_raw, sfc01) if return_both else sfc01

@torch.no_grad()
def compute_CPR(Fmap: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
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
    return (wy.view(h, 1) * wx.view(1, w)).clamp_min(0)

def _centerline_localmax(Gmag01: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    B, H, W = Gmag01.shape
    pool = torch.nn.MaxPool2d(3, stride=1, padding=1)
    G1 = Gmag01.clamp(0,1).unsqueeze(1)
    Gmax = pool(G1)[:,0]
    center = (Gmag01 >= (Gmax - 1e-6)) & valid
    return center

# ========================== 主函数（保存 + 指标 + FER_interior） ==========================
@torch.no_grad()
def compute_metrics_from_features(
    feat: torch.Tensor,              # (B,C,H,W)
    image_chw_uint8: torch.Tensor,   # (3,H0,W0) or (B,3,H0,W0)
    model_name: str = "model",
    image_name: str = "image",
    k: str = "k",
    q_top: float = 0.10,             # PCA-强边 top 比例
    q_img: float = 0.10,             # 图像强边 top 比例
    dilate_r: int = 2,
    border: int = 1
):
    """
    保存：Y, GI, PCA, G, E_pca, E_img, valid
    返回：ECR, Overlap_F1, FCD, FER(全图), FER_interior(非边), SCL, SFC(0~1)+SFC_raw, SCS, CPR
    """
    assert feat.ndim == 4
    B,C,H,W = feat.shape
    dev = feat.device
    valid = _valid_mask(B,H,W,border, dev)

    # 1) 灰度 Y & 梯度 GI
    img = _prep_to_bchw(image_chw_uint8, (H,W), dev)
    if img.shape[0] != B: img = img.expand(B,-1,-1,-1)
    Y = _rgb_to_luma_Y(img)                          # (B,1,H,W)
    Y01 = _minmax_in_mask(Y.squeeze(1), valid)
    _save_gray(Y01[0], image_name, 'Y',  k, model_name)

    GI, _, _ = _sobel_replicate(Y)                   # (B,1,H,W)
    GI = _minmax_in_mask(GI.squeeze(1), valid)       # (B,H,W)
    _save_gray(GI[0], image_name, 'GI', k, model_name)

    # 2) PCA 标量图 M ——保存
    M = _scalar_map_from_feat_pca(
        feat, model_name=model_name, image_name=image_name, k=k,
        n_components=1, upsample_to=(H,W)
    ).clamp(0,1)                                     # (B,H,W)
    _save_gray(M[0], image_name, 'PCA', k, model_name)

    # 3) PCA 梯度（raw & 可视化）
    G_raw, _, _ = _sobel_replicate(M.unsqueeze(1))   # (B,1,H,W) raw
    G = _minmax_in_mask(G_raw.squeeze(1), valid)     # (B,H,W)   归一
    _save_gray(G[0], image_name, 'G', k, model_name)

    # 4) 掩码：PCA-强边 & 图像-强边（严格 top-k）
    E_pca = _top_mask_in_mask(G,   q_top, valid)     # 稀疏
    E_img_top = _top_mask_in_mask(GI, q_img, valid)  # 稀疏（真正 top 比例）

    # 中心线（用于 FCD 环带 & 内部掩码）
    center_img = _centerline_localmax(GI, valid) & E_img_top

    _save_gray(E_pca[0].float(),    image_name, 'E_pca', k, model_name)
    _save_gray(E_img_top[0].float(),image_name, 'E_img', k, model_name)
    _save_gray(valid[0].float(),    image_name, 'valid', k, model_name)

    # 5) 容忍膨胀（用于 F1）
    if dilate_r > 0:
        pool = torch.nn.MaxPool2d(2*dilate_r+1, stride=1, padding=dilate_r)
        E_img_d = (pool(E_img_top.float().unsqueeze(1))[:,0] > 0) & valid
        E_pca_d = (pool(E_pca.float().unsqueeze(1))[:,0] > 0) & valid
    else:
        E_img_d, E_pca_d = E_img_top & valid, E_pca & valid

    # 6) 两个精简指标（ECR/F1）
    Gw = ( (G_raw.squeeze(1) if G_raw.dim()==4 else G_raw) * valid.float()).clamp_min(0)
    true_energy  = (Gw * (E_pca & E_img_top).float()).flatten(1).sum(1)
    false_energy = (Gw * (E_pca & (~E_img_top) & valid).float()).flatten(1).sum(1)
    ECR_purity   = true_energy / (true_energy + false_energy + EPS)

    TP = (E_pca & E_img_d).flatten(1).sum(1).float()
    P  = (E_pca & valid).flatten(1).sum(1).float().clamp_min(1.0)
    R_ = (E_img_top & valid).flatten(1).sum(1).float().clamp_min(1.0)
    precision = TP / P
    recall    = TP / R_
    Overlap_F1 = (2 * precision * recall) / (precision + recall + EPS)
    Overlap_F1 = Overlap_F1.clamp(0, 1)

    # 7) FER（全图） + FER_interior（非边）
    Mz = M.clone()
    flat = Mz.reshape(B, -1)
    mean = flat.mean(dim=1, keepdim=True)
    std  = flat.std(dim=1, keepdim=True).clamp_min(EPS)
    Mz = ((flat - mean) / std).reshape_as(Mz)
    mu = Mz.mean(dim=(1,2), keepdim=True)
    Mz = Mz.clamp(mu - 8.0, mu + 8.0)

    win = _hann2d(H, W, Mz.device, Mz.dtype)
    Mz_win = Mz * win
    low, mid, high = rfft2_radial_energy(Mz_win, low_thr=0.15, mid_thr=0.45)
    total = (low + mid + high).clamp_min(1e-8)
    FER_global = (mid / total).clamp(0, 1)
    hf_share = ((mid + high) / total).clamp(0, 1)    # FCD 用

    # —— 边带/内部掩码 —— #
    ring_r = max(dilate_r*2, 3)
    _pool = torch.nn.MaxPool2d(2*ring_r+1, stride=1, padding=ring_r)
    edge_band = (_pool(center_img.float().unsqueeze(1))[:,0] > 0) & valid
    interior  = valid & (~edge_band)
    interior_count = interior.flatten(1).sum(1)

    Mz_in = Mz_win * interior.float()
    low_i, mid_i, high_i = rfft2_radial_energy(Mz_in, low_thr=0.15, mid_thr=0.45)
    total_i = (low_i + mid_i + high_i).clamp_min(1e-8)
    FER_interior = (mid_i / total_i).clamp(0,1)
    FER_interior = torch.where(interior_count > 16, FER_interior, FER_global)

    F32 = torch.nan_to_num(feat.detach().to(torch.float32), 0.0)

    # SCL（稳健参数）
    SCL = compute_SCL(
        F32,
        tau=0.5, num_radii=16, r_max_frac=0.6,
        directions=((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)),
        pad_mode="crop",
        highpass_ks=3,
    )

    # SFC：有界+原始
    SFC_raw, SFC_01 = compute_SFC(F32, grid=(16,16), return_both=True)
    CPR = compute_CPR(F32)

    # 8) FCD（严格复用已有变量）
    FCD = compute_FCD_from_cached(
        G_raw=G_raw[:, 0],
        E_pca=E_pca,
        centerline_img=center_img,
        valid=valid,
        hf_share=hf_share,
        SCL_feat=SCL,
        image_hw=img.shape[-2:],
        feat_hw=(H, W),
        r1=dilate_r if dilate_r>0 else 1,
        r2=max(dilate_r*2, 3) if dilate_r>0 else 3,
        scl_ref=64.0,
    )

    SCS = compute_SCS(F32)

    return {
        "ECR": ECR_purity.round(decimals=2),
        # "Overlap_F1": Overlap_F1.round(decimals=2),
        "FCD": FCD.round(decimals=2),
        "FER": FER_global.round(decimals=2),
        # "FER_interior": FER_interior.round(decimals=2),
        # "SCL": SCL.round(decimals=2),
        "SFC": SFC_01.round(decimals=2),       # 0~1（稳定）
        # "SFC_raw": SFC_raw.round(decimals=2),  # 原始比值（可能很大，仅参考）
        "SCS": SCS.round(decimals=2),
        # "CPR": CPR.round(decimals=2)
    }

# ========================== 融合：SemIdx & EdgeIdx ==========================
def _clip01(x): 
    return max(0.0, min(1.0, float(x)))

def _bump_mid(x, m, bw):
    """驼峰打分: 接近目标 m 得 1，偏离按带宽 bw 线性降到 0"""
    if bw <= 1e-8: 
        return 0.0
    return _clip01(1.0 - abs(float(x) - float(m)) / float(bw))

def _gm_weighted(vals, weights):
    """带权几何均值；任何项<=0 则返回0"""
    s = 0.0; wsum = 0.0
    for v, w in zip(vals, weights):
        v = max(1e-8, float(v))  # 避免 log(0)
        s += w * math.log(v)
        wsum += w
    return math.exp(s / max(wsum, 1e-8))

def fuse_two_indices(
    metrics: dict,
    calib: dict | None = None
):
    """
    metrics: 单图指标字典，需含：
        FCD(0-100), Overlap_F1, ECR, SCS, CPR, SFC, FER_interior, SCL
    calib: 可选校准参数：
        FCD_tau (默认0.5) —— FCD 指数标定的 tau
        FER_m (默认0.35), FER_bw (默认0.20)
        SCL_m (默认3.0),  SCL_bw (默认2.0)
    """
    calib = calib or {}
    FCD       = float(metrics.get("FCD", 0.0))
    F1        = _clip01(metrics.get("Overlap_F1", 0.0))
    ECR       = _clip01(metrics.get("ECR", 0.0))
    SCS       = _clip01(metrics.get("SCS", 0.0))
    CPR       = _clip01(metrics.get("CPR", 0.0))
    SFC       = _clip01(metrics.get("SFC", 0.0))
    FER_int   = _clip01(metrics.get("FER_interior", 0.0))
    SCL_val   = float(metrics.get("SCL", 0.0))

    # --- FCD -> [0,1] 并做指数标定（固定 tau，更稳）
    FCD01 = _clip01(FCD / 100.0)
    tau = float(calib.get("FCD_tau", 0.5))
    FCD_cal = 1.0 - math.exp(-FCD01 / max(tau, 1e-6))
    FCD_cal = _clip01(FCD_cal)

    # --- 中频/尺度合意度（语义只看非边区域频带）
    FER_m  = float(calib.get("FER_m", 0.35))
    FER_bw = float(calib.get("FER_bw", 0.20))
    FER_mid = _bump_mid(FER_int, FER_m, FER_bw)

    SCL_m  = float(calib.get("SCL_m", 3.0))
    SCL_bw = float(calib.get("SCL_bw", 2.0))
    SCL_fit = _bump_mid(SCL_val, SCL_m, SCL_bw)

    # 语义综合：SCS, CPR, SFC, FER_interior_mid, SCL_fit
    print(" SCS :",SCS.round(decimals=2), " CPR :", CPR.round(decimals=2), " SFC :", SFC.round(decimals=2), " FER_mid :", FER_mid.round(decimals=2), " SCL_fit :", SCL_fit.round(decimals=2))
    SemIdx  = _gm_weighted([SCS, CPR, SFC, FER_mid, SCL_fit], [0.50, 0.20, 0.15, 0.10, 0.05])
    print(" SemIdx :",SemIdx.round(decimals=4))

    # 边缘综合：FCD_cal, F1, ECR
    print(" FCD_cal :",FCD_cal.round(decimals=2), " F1 :", F1.round(decimals=2), " ECR :", ECR.round(decimals=2))
    EdgeIdx = _gm_weighted([FCD_cal, F1,  ECR], [0.60, 0.25, 0.15])
    print(" EdgeIdx :",EdgeIdx.round(decimals=4))

    return {
        "SemIdx": SemIdx.round(decimals=4),
        "EdgeIdx": EdgeIdx.round(decimals=4),
    }

# ------------------------ Quick self-test (optional) ------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    feat_flat = torch.zeros(1, 8, 32, 32)
    imgs = torch.rand(3, 1024, 2048)
    m = compute_metrics_from_features(
        feat_flat, imgs, model_name="main", image_name="a", k="test",
        q_top=0.10, q_img=0.10, dilate_r=2
    )
    # idx = fuse_two_indices(m, calib={"FCD_tau": 0.5})
    print("metrics:", m)
    # print("fused:", idx)
