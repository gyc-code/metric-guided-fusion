# vfm_semantic_core.py
# ------------------------------------------------------------
# GT-free semantic diagnostics for VFM features (res4 recommended).
# Implements SAL, SFC, CPR with careful edge cases & numerical stability.
# PyTorch >= 1.10; runs on CPU or GPU.
# ------------------------------------------------------------
from __future__ import annotations
from typing import Dict, Iterable, Tuple
import torch
import torch.nn.functional as F

EPS = 1e-8


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


# --------------------------- SAL ---------------------------

@torch.no_grad()
def compute_SAL(
    Fmap: torch.Tensor,
    tau: float = 0.2,
    num_radii: int = 8,
    r_max_frac: float = 0.4,
    directions: Iterable[Tuple[int, int]] = ((1, 0), (-1, 0), (0, 1), (0, -1)),
    pad_mode: str = "replicate",         # "replicate" | "crop"
    return_normalized: bool = False,     # also return SAL / min(H,W)
) -> torch.Tensor:
    """
    Semantic Aggregation Length: first radius r where NCC(M, shift_r(M)) < tau.
    - Fmap: (B,C,H,W)
    - tau:   correlation threshold (0.2 default)
    - num_radii: how many radii to evaluate (log/lin spaced to r_max)
    - r_max_frac: r_max = floor(r_max_frac * min(H,W)), upper bound for search
    - directions: offsets to average NCC over (default: 4 axial directions)
    - pad_mode: "replicate" (default) or "crop" (padding-free overlap)
    - return_normalized: if True, return SAL / min(H,W)
    Returns: (B,) SAL in pixels (float), or normalized if return_normalized=True.
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
    SAL = torch.full((B,), float(radii[-1]), device=device)
    below = (corr < tau)
    for b in range(B):
        hits = torch.where(below[b])[0]
        if hits.numel() > 0:
            SAL[b] = float(radii[int(hits[0])])

    if return_normalized:
        SAL = SAL / float(min(H, W))

    return SAL


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
def compute_CPR(Fmap: torch.Tensor) -> torch.Tensor:
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



# --------------------------- minimal sanity tests ---------------------------

def _quick_sanity():
    """
    Run a few cheap assertions to catch obvious bugs.
    """
    B, C, H, W = 2, 256, 64, 64
    torch.manual_seed(0)
    Fmap = torch.randn(B, C, H, W)

    # 1) Shapes
    sal = compute_SAL(Fmap)
    sfc = compute_SFC(Fmap)
    cpr = compute_CPR(Fmap)
    assert sal.shape == (B,), f"SAL shape {sal.shape}"
    assert sfc.shape == (B,), f"SFC shape {sfc.shape}"
    assert cpr.shape == (B,), f"CPR shape {cpr.shape}"
    print("sal,sfc,cpr---", sal, sfc, cpr)

    # # 2) Smoothing should increase SAL (longer coherence) on average
    # Fsmooth = F.avg_pool2d(Fmap, kernel_size=3, stride=1, padding=1)
    # sal_s = compute_SAL(Fsmooth)
    # # allow ties; just check mean tendency
    # assert sal_s.mean().item() >= sal.mean().item() - 1e-4, "SAL should not decrease after smoothing on average"

    # # 3) CPR within (0,1]; pooling (reducing rank) shouldn't increase CPR
    # cpr_s = compute_CPR(Fsmooth)
    # assert (cpr > 0).all() and (cpr <= 1).all(), "CPR out of (0,1]"
    # assert cpr_s.mean().item() <= cpr.mean().item() + 1e-4, "CPR should not increase after smoothing"

    # # 4) SFC is finite & non-negative
    # assert torch.isfinite(sfc).all() and (sfc >= 0).all(), "SFC invalid values"

    # # 5) SAL crop-mode should be defined and similar scale
    # sal_crop = compute_SAL(Fmap, pad_mode="crop")
    # assert sal_crop.shape == (B,), "SAL(crop) shape mismatch"

    # # 6) CPR N<2 fallback
    # Ftiny = torch.randn(B, C, 1, 1)
    # cpr_tiny = compute_CPR(Ftiny)
    # assert torch.allclose(cpr_tiny, torch.ones_like(cpr_tiny)), "CPR N<2 should be 1.0"

    # # 7) Bundle includes SALn
    # bundle = compute_semantic_bundle(Fmap)
    # print("bundle---", bundle)
    # assert "SALn" in bundle and bundle["SALn"].shape == (B,), "Bundle should include SALn"


if __name__ == "__main__":
    _quick_sanity()
    print("SAL/SFC/CPR sanity checks passed.")
