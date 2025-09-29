# -*- coding: utf-8 -*-
"""
FFT 可视化脚本（RGB/灰度 + 可选频带掩码）
用法：
  python fft_vis.py --image /path/to/img.png --outdir ./out --prefix demo --bands 0.25,0.5
依赖：numpy, pillow, matplotlib
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def ensure_rgb_array(img_input):
    """
    输入：路径 / PIL.Image / ndarray
    输出：float32, [0,1], 形状 (H,W,3)
    """
    if isinstance(img_input, (str, Path)):
        img = Image.open(img_input).convert("RGB")
    elif isinstance(img_input, Image.Image):
        img = img_input.convert("RGB")
    elif isinstance(img_input, np.ndarray):
        arr = img_input
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, :3]
        else:
            raise ValueError("Unsupported ndarray shape for image.")
        img = Image.fromarray(np.uint8(np.clip(arr, 0, 255)))
        img = img.convert("RGB")
    else:
        raise ValueError("Unsupported input type.")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def fft2_mag_phase(channel_2d):
    """
    2D FFT（中心化），返回：log(1+|F|) 幅度谱、相位谱、中心化频谱
    """
    F = np.fft.fft2(channel_2d)
    F_shift = np.fft.fftshift(F)
    mag = np.abs(F_shift)
    mag_log = np.log1p(mag)
    phase = np.angle(F_shift)
    return mag_log, phase, F_shift


def save_fig(np_img, title, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    if np_img.ndim == 2:
        plt.imshow(np_img, cmap=None)
    else:
        plt.imshow(np_img)
    plt.title(title)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close()


def radial_masks(H, W, low_thr=0.25, mid_thr=0.5):
    """
    生成三段径向掩码（低/中/高），半径归一化到 [0,1]
    """
    yy = np.linspace(-1.0, 1.0, H, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, W, dtype=np.float32)[None, :]
    rr = np.sqrt(yy ** 2 + xx ** 2)
    rr = rr / max(rr.max(), 1e-6)
    low_mask = (rr < low_thr)
    mid_mask = (rr >= low_thr) & (rr < mid_thr)
    high_mask = (rr >= mid_thr)
    return low_mask, mid_mask, high_mask


def main(args):
    arr = ensure_rgb_array(args.image)           # (H, W, 3) in [0,1]
    H, W, _ = arr.shape
    outdir = Path(args.outdir)
    prefix = args.prefix

    # 原图
    save_fig(arr, "Original (RGB)", outdir / f"{prefix}_original.png")

    # 灰度
    gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
    save_fig(gray, "Grayscale", outdir / f"{prefix}_gray.png")

    # 灰度谱
    mag_g, phase_g, Fg_shift = fft2_mag_phase(gray)
    save_fig(mag_g, "FFT Magnitude (log1p, gray, shifted)", outdir / f"{prefix}_mag_gray.png")
    save_fig(phase_g, "FFT Phase (gray, shifted)", outdir / f"{prefix}_phase_gray.png")

    # 灰度重建（正确性 sanity check）
    Fg = np.fft.ifftshift(Fg_shift)
    recon_g = np.real(np.fft.ifft2(Fg))
    rg = recon_g - recon_g.min()
    rg = rg / max(rg.max(), 1e-8)
    save_fig(rg, "Reconstruction from gray FFT", outdir / f"{prefix}_recon_gray.png")

    # RGB 三通道
    chan_names = ["R", "G", "B"]
    recons = []
    for ci, cname in enumerate(chan_names):
        ch = arr[:, :, ci]
        mag_c, phase_c, Fc_shift = fft2_mag_phase(ch)
        save_fig(mag_c, f"FFT Magnitude (log1p, {cname}, shifted)", outdir / f"{prefix}_mag_{cname}.png")
        save_fig(phase_c, f"FFT Phase ({cname}, shifted)", outdir / f"{prefix}_phase_{cname}.png")
        # 重建该通道
        Fc = np.fft.ifftshift(Fc_shift)
        recon_c = np.real(np.fft.ifft2(Fc))
        recon_c = np.clip(recon_c, 0.0, 1.0)
        recons.append(recon_c)

    # 合并重建 RGB
    recon_rgb = np.stack(recons, axis=-1)
    recon_rgb = np.clip(recon_rgb, 0.0, 1.0)
    save_fig(recon_rgb, "Reconstruction from RGB FFT", outdir / f"{prefix}_recon_rgb.png")

    # 可选：导出低/中/高频掩码 & 掩码后的谱（和你论文里 rfft2_radial_energy 对齐）
    if args.bands is not None:
        try:
            low_thr, mid_thr = [float(x) for x in args.bands.split(",")]
        except Exception:
            raise ValueError("--bands 需要形如 '0.25,0.5'")
        low_mask, mid_mask, high_mask = radial_masks(H, W, low_thr, mid_thr)

        # 掩码图本身
        save_fig(low_mask.astype(np.float32),  f"Low-band mask (r<{low_thr})",             outdir / f"{prefix}_mask_low.png")
        save_fig(mid_mask.astype(np.float32),  f"Mid-band mask ({low_thr}≤r<{mid_thr})",   outdir / f"{prefix}_mask_mid.png")
        save_fig(high_mask.astype(np.float32), f"High-band mask (r≥{mid_thr})",            outdir / f"{prefix}_mask_high.png")

        # 灰度功率谱 × 掩码
        # 注意：功率谱 = |F|^2；这里为了看清楚，用 log1p 显示。
        Fg_abs2 = np.abs(Fg_shift) ** 2
        # 为了可视化稳定，把每个 masked 谱做 log1p
        spec_low  = np.log1p(Fg_abs2 * low_mask)
        spec_mid  = np.log1p(Fg_abs2 * mid_mask)
        spec_high = np.log1p(Fg_abs2 * high_mask)

        save_fig(spec_low,  "Spectrum × Low mask (log1p, gray)",  outdir / f"{prefix}_spec_low.png")
        save_fig(spec_mid,  "Spectrum × Mid mask (log1p, gray)",  outdir / f"{prefix}_spec_mid.png")
        save_fig(spec_high, "Spectrum × High mask (log1p, gray)", outdir / f"{prefix}_spec_high.png")

        # 计算各带能量（和 rfft2_radial_energy 一致的定义）
        E_low  = float((Fg_abs2 * low_mask).sum())
        E_mid  = float((Fg_abs2 * mid_mask).sum())
        E_high = float((Fg_abs2 * high_mask).sum())
        print(f"[Bands] low={E_low:.4f}, mid={E_mid:.4f}, high={E_high:.4f}  (thr={low_thr},{mid_thr})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default='/home/yguo/Documents/other/UDA4Inst/000000039769.jpg', help="输入图片路径（任意格式可）")
    parser.add_argument("--outdir", type=str, default="./fft_out", help="输出目录")
    parser.add_argument("--prefix", type=str, default="fft", help="输出文件名前缀")
    parser.add_argument("--bands", type=str, default=None, help="可选，形如 '0.25,0.5'，导出低/中/高频掩码与掩码谱")
    args = parser.parse_args()
    main(args)
