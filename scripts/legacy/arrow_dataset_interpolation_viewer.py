"""
AstroPT Interpolation Viewer.

Visual checker for Arrow image interpolation quality.

For each target ID, it creates a dashboard with images on top and spectra below:
- Top-left:  Original RGB image
- Top-right: Interpolated RGB image
- Middle row: Spectrum (blue channel: lower wavelength half)
- Bottom row: Spectrum (red channel: upper wavelength half)

The default target list matches the IDs used in slurm/plot_images_spectra.sh.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from datasets import concatenate_datasets, load_from_disk


logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("AstroPT-InterpolationViewer")

# Plotting Global Configuration (mirrors plot_images_spectra.py)
plt.rcParams['text.latex.preamble'] = r'''
            \usepackage{siunitx}
            \usepackage{bm}
            \usepackage{amsmath}
            \sisetup{
            detect-family,
            separate-uncertainty=true,
            output-decimal-marker={.},
            exponent-product=\cdot,
            inter-unit-product=\cdot,
            }
            \DeclareSIUnit{\cts}{cts}
            '''
plt.rc('text', usetex=True)
plt.rc('font', family='serif', weight='bold')

plt.rcParams.update({
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'font.size': 14,
    'axes.labelsize': 19,
    'axes.titlesize': 21,
    'xtick.labelsize': 19,
    'ytick.labelsize': 19,
    'legend.fontsize': 16,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titlesize': 24,
    'figure.titleweight': 'bold',
})

# Spectral lines (mirrors plot_images_spectra.py)
MAIN_LINES = {
    r"Ly$\alpha$": 1216.0, r"C IV": 1549.0, "C III": 1908.7, r"Mg II": 2798.0, r"[O II]": 3727.3,
    r"[Ne III]": 3868.7, r"Ca K": 3933.7, r"Ca H": 3968.5, r"H$\delta$": 4102.0, r"H$\gamma$": 4341.0,
    r"H$\beta$": 4861.0, r"[O III]": 4959.0, r"[O III]": 5007.0, r"Mg I": 5175.0, r"Na D": 5890.0,
    r"[N II]": 6548.0, r"H$\alpha$": 6563.0, r"[N II]": 6583.5, r"[S II]": 6730.8,
}


# Keep exactly the same IDs from slurm/plot_images_spectra.sh (including repeated IDs)
DEFAULT_TARGET_IDS: List[int] = [
    39627061836389042,
    39627853679036156,
    39633445487378968,
    39627346218590254,
    39633442895301960,
    39633491763136752,
    39633523014894811,
    39633312192397870,
    39633476848190817,
    39633526185788566,
    39633516366922547,
    39633118423944455,
    39633448033322139,
    39633530795330888,
    39089837394909544,
    39633478949537029,
    39633312192397870,
    39633414239814702,
    39633493688322559,
    39627859714640945,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arrow interpolation visual checker")
    parser.add_argument("--data_dir_original", type=str, required=True, help="Original Arrow dataset root")
    parser.add_argument("--data_dir_interpolated", type=str, required=True, help="Interpolated Arrow dataset root")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save generated plots")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "test", "all"],
        help="Dataset split to inspect (train/test/all)",
    )
    parser.add_argument("--target_ids", nargs="+", type=int, default=None, help="Optional custom target IDs")
    parser.add_argument(
        "--wl_range",
        nargs=2,
        type=float,
        default=None,
        help="Optional wavelength range [min max] to zoom spectra",
    )
    parser.add_argument(
        "--num_plot",
        type=int,
        default=25,
        help="Target number of objects to plot. If fewer IDs are provided, random ones are added.",
    )
    return parser.parse_args()


def load_split_dataset(data_root: Path, split: str):
    if split == "all":
        split_dirs = sorted(data_root.glob("train_*")) + sorted(data_root.glob("test_*"))
    else:
        split_dirs = sorted(data_root.glob(f"{split}_*"))

    if not split_dirs:
        raise FileNotFoundError(f"No split directories found for '{split}' in {data_root}")
    ds = concatenate_datasets([load_from_disk(str(p)) for p in split_dirs])
    return ds


def build_target_index_map(ds) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    target_ids = ds["targetid"]
    for i, tid in enumerate(target_ids):
        itid = int(tid)
        if itid not in mapping:
            mapping[itid] = i
    return mapping


def make_rgb_lupton(
    image_tensor: np.ndarray,
    Q: float = 10.0,
    stretch: float = 0.5,
    m: float = 0.0,
) -> np.ndarray:
    """
    Lupton et al. (2004) algorithm implementation for RGB visualization.
    """
    # Compute Intensity (I)
    I = np.mean(image_tensor, axis=0)
    I = I - m
    I = np.maximum(I, 1e-10)
    
    # 3. Apply Lupton transfer function (Asinh scaling)
    f_I = np.arcsinh(Q * stretch * I) / Q
    
    # 4. Preserve Color Ratios
    scale_factor = f_I / I
    rgb_out = image_tensor * scale_factor[np.newaxis, :, :]
    
    # Final Normalization (same strategy used in plot_images_spectra)
    max_rgb = np.percentile(rgb_out, 99.5)
    if max_rgb > 0:
        rgb_out = rgb_out / max_rgb
        
    rgb_out = np.clip(rgb_out, 0, 1)
    return rgb_out.transpose(1, 2, 0)


def make_rgb_from_record(record: dict) -> np.ndarray:
    vis = np.array(record["image_vis"], dtype=np.float32)
    y = np.array(record["image_nisp_y"], dtype=np.float32)
    j = np.array(record["image_nisp_j"], dtype=np.float32)
    h = np.array(record["image_nisp_h"], dtype=np.float32)

    # Match exactly the color preparation used in plot_images_spectra.py
    # stack order: [VIS, H, J, Y]
    raw_stack = np.stack([vis, h, j, y], axis=0)

    # Channel-wise background subtraction with P50
    bg_val = np.percentile(raw_stack, 50, axis=(1, 2), keepdims=True)
    raw_bg = raw_stack - bg_val

    # Channel-wise scaling with P99.5 and clipping to [0, 100]
    scaled_channels = []
    for c in range(4):
        v_max = np.percentile(np.abs(raw_bg[c]), 99.5)
        if v_max <= 0:
            v_max = 1.0
        scaled_channels.append(np.clip(raw_bg[c] / v_max, 0, 100))

    raw_norm = np.stack(scaled_channels)

    # RGB weights and mapping: R=H, G=(J+Y)/2, B=VIS
    r = raw_norm[1] * 1.2
    g = ((raw_norm[2] + raw_norm[3]) / 2.0) * 1.3
    b = raw_norm[0] * 1.0
    rgb_input = np.stack([r, g, b], axis=0)

    return make_rgb_lupton(rgb_input, Q=12.0, stretch=0.5)


def sanitize_spectrum(record: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    flux = record.get("spectrum_flux", None)
    wave = record.get("spectrum_wave", None)

    if flux is None or wave is None:
        return None, None

    flux_arr = np.array(flux, dtype=np.float32).flatten()
    wave_arr = np.array(wave, dtype=np.float32).flatten()

    min_len = min(len(flux_arr), len(wave_arr))
    if min_len == 0:
        return None, None

    flux_arr = flux_arr[:min_len]
    wave_arr = wave_arr[:min_len]

    valid = np.isfinite(flux_arr) & np.isfinite(wave_arr)
    if not np.any(valid):
        return None, None

    return flux_arr[valid], wave_arr[valid]


def split_wave_limits(wave: np.ndarray, wl_range: Optional[List[float]]) -> Tuple[float, float, float]:
    if wl_range is not None:
        w_min = float(wl_range[0])
        w_max = float(wl_range[1])
    else:
        w_min = float(np.min(wave))
        w_max = float(np.max(wave))
    w_mid = (w_min + w_max) / 2.0
    return w_min, w_mid, w_max


def plot_spectral_lines(ax, min_wl: float, max_wl: float, z: float) -> None:
    """Annotates spectral lines with alternating heights."""
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    pos_high = y_min + (y_range * 0.95)
    pos_low = y_min + (y_range * 0.25)

    sorted_lines = sorted(MAIN_LINES.items(), key=lambda x: x[1])
    counter = 0

    for name, rest_wave in sorted_lines:
        obs_wave = rest_wave * (1 + z)
        if min_wl < obs_wave < max_wl:
            y_pos = pos_high if counter % 2 == 0 else pos_low
            ax.axvline(obs_wave, color='royalblue', linestyle='--', alpha=0.6, lw=1)
            ax.text(
                obs_wave,
                y_pos,
                rf"\textbf{{{name}}}",
                rotation=90,
                color='royalblue',
                va='top',
                ha='right',
                fontsize=12,
                alpha=1,
                fontweight='bold',
            )
            counter += 1


def plot_target(
    target_id: int,
    idx_in_list: int,
    rec_orig: dict,
    rec_interp: dict,
    save_dir: Path,
    wl_range: Optional[List[float]],
) -> None:
    rgb_orig = make_rgb_from_record(rec_orig)
    rgb_interp = make_rgb_from_record(rec_interp)
    z_val = float(rec_orig.get("redshift", 0.0))

    flux_o, wave_o = sanitize_spectrum(rec_orig)
    flux_i, wave_i = sanitize_spectrum(rec_interp)

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], wspace=0.1, hspace=0.3)

    fig.suptitle(
        rf"\textbf{{Interpolation Check | ID: {target_id} | z={z_val:.3f}}}",
        fontsize=22,
        y=0.96,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb_orig, origin="lower")
    ax1.set_title(r"\textbf{Original (Log Scale)}")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(rgb_interp, origin="lower")
    ax2.set_title(r"\textbf{Interpolated (Log Scale)}")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, :])
    ax4 = fig.add_subplot(gs[2, :])

    if flux_o is not None and wave_o is not None and flux_i is not None and wave_i is not None:
        min_len = min(len(flux_o), len(wave_o), len(flux_i), len(wave_i))
        flux_o = flux_o[:min_len]
        wave_o = wave_o[:min_len]
        flux_i = flux_i[:min_len]
        wave_i = wave_i[:min_len]

        w_min, w_mid, w_max = split_wave_limits(wave_o, wl_range)

        for idx, (ax, lo, hi, title) in enumerate([
            (ax3, w_min, w_mid, "Spectrum (Blue channel)"),
            (ax4, w_mid, w_max, "Spectrum (Red channel)"),
        ]):
            ax.plot(wave_o, flux_o, color="black", lw=1.2, alpha=0.8, label="Original")
            ax.plot(wave_i, flux_i, color="crimson", lw=1.1, alpha=0.75, label="Interpolated")
            ax.set_xlim(lo, hi)
            ax.set_title(rf"\textbf{{{title}}}")
            ax.set_ylabel(r"Flux")
            plot_spectral_lines(ax, lo, hi, z_val)

            if idx == 1:
                ax.set_xlabel(r"Wavelength [\AA]")

        ax3.legend(loc='lower left')
    else:
        for ax in [ax3, ax4]:
            ax.text(0.5, 0.5, "No spectrum available", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

    out_name = f"Inter_ID_{target_id}.png"
    out_path = save_dir / out_name
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    data_dir_original = Path(args.data_dir_original)
    data_dir_interpolated = Path(args.data_dir_interpolated)
    save_dir = Path(args.save_dir)

    if not data_dir_original.exists():
        logger.error(f"Original dataset path not found: {data_dir_original}")
        sys.exit(1)
    if not data_dir_interpolated.exists():
        logger.error(f"Interpolated dataset path not found: {data_dir_interpolated}")
        sys.exit(1)

    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading datasets...")
    ds_orig = load_split_dataset(data_dir_original, args.split)
    ds_interp = load_split_dataset(data_dir_interpolated, args.split)

    logger.info(f"Original split size: {len(ds_orig)}")
    logger.info(f"Interpolated split size: {len(ds_interp)}")

    map_orig = build_target_index_map(ds_orig)
    map_interp = build_target_index_map(ds_interp)

    requested_ids = args.target_ids if args.target_ids else DEFAULT_TARGET_IDS

    # Keep provided IDs that are present in both datasets (deduplicated, stable order)
    target_ids: List[int] = []
    for tid in requested_ids:
        tid_i = int(tid)
        if tid_i in map_orig and tid_i in map_interp and tid_i not in target_ids:
            target_ids.append(tid_i)

    # Fill randomly until reaching args.num_plot targets
    need = max(0, int(args.num_plot) - len(target_ids))
    if need > 0:
        common_pool = sorted(set(map_orig.keys()) & set(map_interp.keys()))
        pool = [tid for tid in common_pool if tid not in target_ids]
        if pool:
            take = min(need, len(pool))
            sampled = np.random.choice(pool, size=take, replace=False).tolist()
            target_ids.extend(int(x) for x in sampled)
            logger.info(f"Added {take} random target IDs to reach {len(target_ids)} plots.")

    logger.info(f"Generating plots for {len(target_ids)} target IDs...")
    plotted = 0

    for i, tid in enumerate(target_ids, start=1):
        rec_orig = ds_orig[map_orig[tid]]
        rec_interp = ds_interp[map_interp[tid]]

        try:
            plot_target(
                target_id=tid,
                idx_in_list=i,
                rec_orig=rec_orig,
                rec_interp=rec_interp,
                save_dir=save_dir,
                wl_range=args.wl_range,
            )
            plotted += 1
            logger.info(f"[{plotted}] Saved plot for TARGETID {tid}")
        except Exception as e:
            logger.error(f"Failed TARGETID {tid}: {e}")

    logger.info(f"Done. Generated {plotted} plots in {save_dir}")


if __name__ == "__main__":
    main()
