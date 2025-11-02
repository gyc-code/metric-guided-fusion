import torch, math
import torch.nn.functional as F
from contextlib import nullcontext
import time

EPS = 1e-8

# ------------------ 已有缓存（Hann/径向掩码） ------------------
_HANN_CACHE = {}
_RADIAL_MASK_CACHE = {}


@torch.no_grad()
def _hf_share_from_pca_map(
    M: torch.Tensor,                # (B,H,W)  已 z-score+clamp 前的 Mz
    low_thr=0.15, mid_thr=0.45,
    fft_cap: int = 128              # 长边上限（0/None 表示不用缩）
):
    B, H, W = M.shape
    Ht, Wt = H, W
    if fft_cap and max(H, W) > fft_cap:
        scale = max(H, W) / float(fft_cap)
        Ht = max(16, int(round(H / scale)))
        Wt = max(16, int(round(W / scale)))
    if (Ht, Wt) != (H, W):
        try:
            Md = F.interpolate(M.unsqueeze(1), size=(Ht, Wt), mode="bilinear", align_corners=False, antialias=True)[:,0]
        except TypeError:
            Md = F.interpolate(M.unsqueeze(1), size=(Ht, Wt), mode="bilinear", align_corners=False)[:,0]
    else:
        Md = M
    win = _get_hann_cached(Ht, Wt, Md.device, Md.dtype)
    low, mid, high = rfft2_radial_energy_cached(Md * win, low_thr=low_thr, mid_thr=mid_thr)
    total    = (low + mid + high).clamp_min(1e-8)
    hf_share = ((mid + high) / total).clamp(0, 1)
    return hf_share



def _get_hann_cached(H, W, device, dtype):
    key = (H, W, device.type, device.index, str(dtype))
    win = _HANN_CACHE.get(key, None)
    if win is None or win.device != device:
        wy = torch.hann_window(H, periodic=False, device=device, dtype=dtype)
        wx = torch.hann_window(W, periodic=False, device=device, dtype=dtype)
        win = (wy.view(H, 1) * wx.view(1, W)).clamp_min(0)
        _HANN_CACHE[key] = win
    return win

def _get_radial_masks_cached(H, W, low_thr, mid_thr, device, dtype):
    key = (H, W, float(low_thr), float(mid_thr), device.type, device.index, str(dtype))
    masks = _RADIAL_MASK_CACHE.get(key, None)
    if masks is None:
        yy = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype).view(H, 1)
        xx = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype).view(1, W)
        rr = torch.sqrt(yy ** 2 + xx ** 2)
        rr = rr / rr.max().clamp_min(1e-6)
        low_mask  = (rr < low_thr).to(dtype)
        mid_mask  = ((rr >= low_thr) & (rr <  mid_thr)).to(dtype)
        high_mask = (rr >= mid_thr).to(dtype)
        masks = (low_mask, mid_mask, high_mask)
        _RADIAL_MASK_CACHE[key] = masks
    return masks

@torch.no_grad()
def rfft2_radial_energy_cached(M: torch.Tensor, low_thr=0.15, mid_thr=0.45):
    # 复用原来的复数 FFT（实现稳妥），已缓存掩码
    B, H, W = M.shape
    F2 = torch.fft.fft2(M)
    P  = (F2.real ** 2 + F2.imag ** 2)
    P  = torch.roll(torch.roll(P, shifts=H // 2, dims=-2), shifts=W // 2, dims=-1)
    low_mask, mid_mask, high_mask = _get_radial_masks_cached(H, W, low_thr, mid_thr, M.device, M.dtype)
    low  = (P * low_mask ).reshape(B, -1).sum(1)
    mid  = (P * mid_mask ).reshape(B, -1).sum(1)
    high = (P * high_mask).reshape(B, -1).sum(1)
    return low, mid, high

# ------------------ 预计算 Y_full：一次 sRGB->Linear->Y ------------------
def _srgb_to_linear(x: torch.Tensor):
    return torch.where(x <= 0.04045, x/12.92, ((x+0.055)/1.055).pow(2.4))

def precompute_Y_full(batch_images: torch.Tensor) -> torch.Tensor:
    """
    输入：batch_images (B,3,H0,W0)，0~255 或 0~1 均可
    输出：Y_full (B,1,H0,W0)，float32，线性亮度
    """
    x = batch_images
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"expect (B,3,H0,W0), got {x.shape}")
    x = x.to(dtype=torch.float32)
    if x.max() > 1: x = x / 255.0
    x = x.clamp(0, 1)
    x_lin = _srgb_to_linear(x)
    Y = (0.2126 * x_lin[:,0:1] + 0.7152 * x_lin[:,1:2] + 0.0722 * x_lin[:,2:3]).clamp(0,1)
    return Y

# ------------------ Sobel / 归一化 / 中心线（你已有） ------------------
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
    x = torch.nan_to_num(x.float(), 0.0, 0.0, 0.0).cuda()
    out = torch.zeros_like(x)
    for i in range(x.shape[0]):
        vals = x[i][mask[i]]
        if vals.numel()==0: continue
        mn, mx = vals.min(), vals.max()
        den = (mx - mn) if (mx - mn) > 1e-8 else torch.tensor(1.0, device=x.device)
        out[i] = ((x[i]-mn)/den).clamp(0,1)
    return out

def _centerline_localmax(Gmag01: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    B, H, W = Gmag01.shape
    pool = torch.nn.MaxPool2d(3, stride=1, padding=1)
    G1 = Gmag01.clamp(0,1).unsqueeze(1)
    Gmax = pool(G1)[:,0]
    center = (Gmag01 >= (Gmax - 1e-6)) & valid
    return center

# ------------------ SCL（保留原语义，但支持“快速模式”） ------------------
def _as_float(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(x): x = x.float()
    elif x.dtype in (torch.float16, torch.bfloat16): x = x.float()
    return x

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
    highpass_ks: int = 3,
    early_stop: bool = True
) -> torch.Tensor:
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
        if early_stop and torch.all(corr[:, j] < tau):
            # 全 batch 在该半径已低于阈值，后续半径可提前结束
            corr = corr[:, :j+1]
            radii = radii[:j+1]
            break

    SCL = torch.full((B,), float(radii[-1]), device=Fmap.device)
    below = (corr < tau)
    for b in range(B):
        idx = torch.where(below[b])[0]
        if idx.numel() > 0:
            SCL[b] = float(radii[int(idx[0])])
    return SCL

# ------------------ PCA：禁用 autocast；支持 eigh/power 两种后端 ------------------
def _no_autocast_ctx(t: torch.Tensor):
    if t.is_cuda:
        try:
            return torch.cuda.amp.autocast(enabled=False)
        except Exception:
            return nullcontext()
    return nullcontext()

@torch.no_grad()
def _pca_first_component_map_torch(
    Fmap: torch.Tensor,
    solver: str = "eigh",   # "eigh" | "power"
    power_iters: int = 5
) -> torch.Tensor:
    assert Fmap.dim() == 4
    B, C, H, W = Fmap.shape
    X = torch.nan_to_num(Fmap.detach().to(torch.float32), 0.0).permute(0,2,3,1).reshape(B, -1, C) # (B,N,C)
    mu  = X.mean(dim=1, keepdim=True)
    std = X.std(dim=1, keepdim=True).clamp_min(1e-6)
    Xz  = (X - mu) / std                                         # (B,N,C)
    N   = Xz.shape[1]

    with _no_autocast_ctx(Fmap):
        Xz32 = Xz.float()

        if solver == "power":
            # 幂迭代：避免构 CxC 和 C^3 的 eig
            v = torch.randn(B, C, device=Xz32.device, dtype=Xz32.dtype)
            v = v / (v.norm(dim=1, keepdim=True) + 1e-6)
            for _ in range(max(1, int(power_iters))):
                w = torch.einsum('bnc,bc->bn', Xz32, v)                   # (B,N)
                v = torch.einsum('bnc,bn->bc', Xz32, w) / float(max(N-1,1)) # (B,C)
                v = v / (v.norm(dim=1, keepdim=True) + 1e-6)
        else:
            Cov = (Xz32.transpose(1,2) @ Xz32) / float(max(N-1,1))        # (B,C,C)
            Cov = 0.5 * (Cov + Cov.transpose(1,2))
            eigvals, eigvecs = torch.linalg.eigh(Cov)
            v = eigvecs[:, :, -1]                                         # (B,C)

        Z = torch.einsum('bnc,bc->bn', Xz32, v)                           # (B,N)

    Zmin = Z.min(dim=1, keepdim=True).values
    Zmax = Z.max(dim=1, keepdim=True).values
    M = (Z - Zmin) / (Zmax - Zmin + 1e-6)
    return M.reshape(B, H, W).contiguous()

def _rgb_to_luma_Y(bchw: torch.Tensor):
    x = _srgb_to_linear(bchw)
    return (0.2126*x[:,0:1] + 0.7152*x[:,1:2] + 0.0722*x[:,2:3]).clamp(0,1)
# ------------------ 严格等量 top‑q（与原 topk 语义一致） ------------------
@torch.no_grad()
def _top_frac_mask_exact(G: torch.Tensor, q: float, valid: torch.Tensor) -> torch.Tensor:
    import math
    B, H, W = G.shape
    out = torch.zeros_like(valid, dtype=torch.bool)
    for i in range(B):
        vi = valid[i].flatten()
        if vi.sum() == 0: continue
        vals = G[i].flatten()[vi].to(torch.float32)
        if vals.numel() < 4: continue
        vmin, vmax = vals.min(), vals.max()
        if (vmax - vmin) < 1e-6: continue
        k = max(1, int(math.ceil(q * float(vals.numel()))))
        topk = torch.topk(vals, k, sorted=False)
        idxs = torch.nonzero(vi, as_tuple=False).flatten()
        sel = torch.zeros_like(vi)
        sel[idxs[topk.indices]] = True
        out[i] = sel.view(H, W)
    return out

# ------------------ 你已有的 FCD 融合（保持一致） ------------------
@torch.no_grad()
def compute_FCD_from_cached(
    *, G_raw, E_pca, centerline_img, valid, hf_share, SCL_feat,
    image_hw, feat_hw, r1=1, r2=3, scl_ref=64.0
) -> torch.Tensor:
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



@torch.no_grad()
def _prep_srgb01(batch_images: torch.Tensor) -> torch.Tensor:
    """
    返回 sRGB 浮点图 (B,3,H0,W0)，范围 [0,1]，不做线性化。
    """
    x = batch_images
    assert x.ndim == 4 and x.shape[1] == 3, f"expect (B,3,H0,W0), got {x.shape}"
    x = x.to(dtype=torch.float32)
    if x.max() > 1:
        x = x / 255.0
    return x.clamp(0, 1)

@torch.no_grad()
def compute_FCD_from_features_fast(
    feat: torch.Tensor,                      # (B,C,H,W)
    batch_images: torch.Tensor | None = None,# (B,3,H0,W0)，若传了 precomputed_srgb01 可为 None
    *,
    precomputed_srgb01: torch.Tensor | None = None,  # (B,3,H0,W0)，注意：是 sRGB 0~1，不是 Y
    q_top: float = 0.10,
    q_img: float = 0.10,
    dilate_r: int = 2,
    border: int = 1,
    # —— 先用“旧版等价”的配置，确保数值对齐 —— #
    scl_down: int = 1,       # 与旧版一致：不降采样
    scl_dirs: tuple = ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)),  # 8方向
    scl_num_radii: int = 16,
    scl_early_stop: bool = False,
    pca_solver: str = "eigh",      # 与 sklearn 更接近
    pca_power_iters: int = 5,      # 仅在 solver="power" 时有效
) -> torch.Tensor:
    assert feat.ndim == 4
    B, C, H, W = feat.shape
    dev = feat.device
    valid = _valid_mask(B, H, W, border, dev)

    # ---------- PCA 标量图（保持不变） ----------
    M = _pca_first_component_map_torch(
        feat, solver=pca_solver, power_iters=pca_power_iters
    ).clamp(0, 1)   # (B,H,W)

    G_raw, _, _ = _sobel_replicate(M.unsqueeze(1))   # (B,1,H,W)
    G = _minmax_in_mask(G_raw.squeeze(1), valid)     # (B,H,W)
    E_pca = _top_frac_mask_exact(G, q_top, valid)    # 与旧版严格 top-k 一致

    # ---------- 频域能量（保持不变） ----------
    Mz = M.clone()
    flat = Mz.reshape(B, -1)
    mean = flat.mean(dim=1, keepdim=True)
    std  = flat.std(dim=1, keepdim=True).clamp_min(EPS)
    Mz   = ((flat - mean) / std).reshape_as(Mz)
    mu   = Mz.mean(dim=(1,2), keepdim=True)
    Mz   = Mz.clamp(mu - 8.0, mu + 8.0)

    win  = _get_hann_cached(H, W, Mz.device, Mz.dtype)
    Mz_w = Mz * win
    low, mid, high = rfft2_radial_energy_cached(Mz_w, low_thr=0.15, mid_thr=0.45)
    total    = (low + mid + high).clamp_min(1e-8)
    hf_share = ((mid + high) / total).clamp(0, 1)
    # hf_share = _hf_share_from_pca_map(Mz, low_thr=0.15, mid_thr=0.45, fft_cap=128)

    # ---------- SCL（与旧版相同配置） ----------
    F32 = torch.nan_to_num(feat.detach().to(torch.float32), 0.0)
    if scl_down and scl_down > 1:
        Hs, Ws = max(2, H // scl_down), max(2, W // scl_down)
        F_for_scl = F.interpolate(F32, size=(Hs, Ws), mode="bilinear", align_corners=False)
    else:
        F_for_scl = F32
    SCL = compute_SCL(
        F_for_scl, tau=0.5, num_radii=scl_num_radii, r_max_frac=0.6,
        directions=scl_dirs, pad_mode="crop", highpass_ks=3, early_stop=scl_early_stop
    )
    if scl_down and scl_down > 1:
        SCL = SCL * scl_down  # 单位补偿：回到原特征像素

    # ---------- 图像侧（关键修复：先下采样 sRGB，再线性化） ----------
    if precomputed_srgb01 is not None:
        srgb01_full = precomputed_srgb01
        assert srgb01_full.ndim == 4 and srgb01_full.shape[1] == 3 and srgb01_full.shape[0] == B
    else:
        assert batch_images is not None, "must provide batch_images or precomputed_srgb01"
        srgb01_full = _prep_srgb01(batch_images)

    img_srgb_stage = F.interpolate(srgb01_full, size=(H, W), mode="bilinear", align_corners=False)  # 先下采样（sRGB）
    Y = _rgb_to_luma_Y(img_srgb_stage)     # 再 sRGB→linear→luma
    GI, _, _ = _sobel_replicate(Y)
    GI = _minmax_in_mask(GI.squeeze(1), valid)
    E_img_top  = _top_frac_mask_exact(GI, q_img, valid)
    center_img = _centerline_localmax(GI, valid) & E_img_top

    # ---------- FCD（关键修复：image_hw 传 (H,W) 与旧版一致） ----------
    FCD = compute_FCD_from_cached(
        G_raw=G_raw[:, 0],
        E_pca=E_pca,
        centerline_img=center_img,
        valid=valid,
        hf_share=hf_share,
        SCL_feat=SCL,
        image_hw=(H, W),       # ★ 修复：与旧版一致。不要传原图尺寸
        feat_hw=(H, W),
        r1=dilate_r if dilate_r > 0 else 1,
        r2=max(dilate_r * 2, 3) if dilate_r > 0 else 3,
        scl_ref=64.0,
    )

    return FCD.round(decimals=2)



# ------------------ 多 stage 一次性计算（推荐调用入口） ------------------
@torch.no_grad()
def compute_FCD_for_stages_fast(
    features_aux: dict,              # {name: (B,C,H,W)}
    batch_images: torch.Tensor,      # (B,3,H0,W0)
    *,
    order: list[str] | None = None,  # 计算顺序，可传 ["res2","res3","res4","res5"]
    q_top: float = 0.10, q_img: float = 0.10,
    dilate_r: int = 2, border: int = 1,
    scl_down: int = 4, scl_dirs=((1,0),(-1,0),(0,1),(0,-1)),
    scl_num_radii: int = 12, scl_early_stop: bool = True,
    pca_solver: str = "power", pca_power_iters: int = 5
) -> dict:
    """
    返回：{name: FCD (B,)}，所有 stage 的 FCD 一次性算完
    关键：预先算一次 Y_full，stage 内部只做下采样
    """
    names = order if order is not None else list(features_aux.keys())
    # 一次性准备 sRGB 浮点图（0~1）
    pre_srgb = _prep_srgb01(batch_images)   # (B,3,H0,W0)
    out = {}
    for name, feat in features_aux.items():
        fcd = compute_FCD_from_features_fast(
            feat, precomputed_srgb01=pre_srgb,
            q_top=0.10, q_img=0.10, dilate_r=2, border=1,
            scl_down=1, scl_dirs=((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)),
            scl_num_radii=16, scl_early_stop=False,
            pca_solver="eigh"
        )
        out[name] = fcd

    return out


# ------------------------ Quick self-test (optional) ------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    features_aux = {"res2":torch.zeros(3, 8, 32, 32),"res3":torch.zeros(3, 8, 32, 32),"res4":torch.zeros(3, 8, 32, 32),"res5":torch.zeros(3, 8, 32, 32)}
    batch_images = torch.rand(3, 3, 1024, 2048)


    s = time.time()
    fcd_all = compute_FCD_for_stages_fast(
        features_aux, batch_images,
        order=["res2","res3","res4","res5"],   # 可选
        q_top=0.10, q_img=0.10, dilate_r=2, border=1,
        scl_down=4,                            # 关键提速
        scl_dirs=((1,0),(-1,0),(0,1),(0,-1)),  # 4 方向
        scl_num_radii=12, scl_early_stop=True,
        pca_solver="power", pca_power_iters=5  # 更快的 PCA
    )
    print(fcd_all)
    print("FCD 4-stage total time:", time.time() - s)


