"""
AstroPT Cross-Modal Generative Inspector.

This script performs Zero-Shot cross-reconstruction. It tests the alignment of 
the multimodal latent space by forcing the model to generate one modality 
autoregressively conditioned strictly on the other.

Features:
- Image -> Spectrum Generation.
- Spectrum -> Image Generation.
- Image -> Image Self Generation.
- Spectrum -> Spectrum Self Generation.
- Autoregressive continuous token prediction loop.
- Separate dashboards for cross-modal and self-modal qualitative analysis.

Author: Victor Alonso Rodriguez
Date: March 2026
"""

import argparse
import csv
import einops
import json
import logging
from pathlib import Path
import sys
from typing import Optional, Any, Dict, List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow
from astropt.model_utils import load_local_model

# Logger Configuration
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-CrossRecon")

# Plotting Global Configuration
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
    'axes.labelsize': 17,     
    'axes.titlesize': 19,     
    'xtick.labelsize': 15,   
    'ytick.labelsize': 15,
    'legend.fontsize': 14,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titlesize': 22,
    'figure.titleweight': 'bold',
})

MAIN_LINES = {
    r"Ly$\alpha$": 1216.0, r"C IV": 1549.0, "C III": 1908.7, r"Mg II": 2798.0, r"[O II]": 3727.3, r"[Ne III]": 3868.7,
    r"Ca K": 3933.7, r"Ca H": 3968.5, r"H$\delta$": 4102.0, r"H$\gamma$": 4341.0,
    r"H$\beta$": 4861.0, r"[O III]": 4959.0, r"[O III]": 5007.0, r"Mg I": 5175.0,
    r"Na D": 5890.0, r"[N II]": 6548.0, r"H$\alpha$": 6563.0, r"[N II]": 6583.5, r"[S II]": 6730.8
}

EPS = 1e-12

def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="AstroPT Cross-Modal Generative Inspector")
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing training weights")
    parser.add_argument("--data_dir", type=str, required=True, help="Arrow data root directory")
    parser.add_argument("--save_dir", type=str, default=None, help="Plot Saving Directory (Defaults to weights_dir/../plots)")
    parser.add_argument("--ckpt_name", type=str, default="ckpt_best.pt", help="Checkpoint filename")
    parser.add_argument("--target_ids", nargs="+", type=int, help="Specific Target IDs to plot")
    parser.add_argument("--num_plot", type=int, default=5, help="Number of random plots (Warning: Generation is slow)")
    parser.add_argument("--split", type=str, default="test", help="Data split (train/test)")
    parser.add_argument("--train_name", type=str, default=None, help="Custom title for the plot")
    parser.add_argument("--wl_range", nargs=2, type=float, default=None, help="Zoom wavelength (min max)") # <--- NUEVO
    parser.add_argument("--metrics_csv_name", type=str, default="reconstruction_metrics.csv", help="Per-object reconstruction metrics CSV")
    parser.add_argument("--summary_json_name", type=str, default="reconstruction_metrics_summary.json", help="Aggregated reconstruction metrics summary JSON")
    parser.add_argument("--disable_metrics", action="store_true", help="Disable objective metric export")
    return parser.parse_args()


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < EPS or sy < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < EPS:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def _safe_global_ssim(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64).reshape(-1)
    y = y.astype(np.float64).reshape(-1)
    if x.size == 0 or y.size == 0:
        return float("nan")

    mu_x = float(np.mean(x))
    mu_y = float(np.mean(y))
    var_x = float(np.var(x))
    var_y = float(np.var(y))
    cov_xy = float(np.mean((x - mu_x) * (y - mu_y)))

    data_range = float(max(np.ptp(x), np.ptp(y), 1.0))
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    if abs(den) < EPS:
        return float("nan")
    return float(num / den)


def _quality_score(pearson: float, nrmse: float, smape: float) -> float:
    if not np.isfinite(pearson) or not np.isfinite(nrmse) or not np.isfinite(smape):
        return float("nan")
    pearson_01 = float(np.clip((pearson + 1.0) / 2.0, 0.0, 1.0))
    return float(pearson_01 * np.exp(-max(nrmse, 0.0)) * np.exp(-max(smape, 0.0)))


def _compute_vector_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
    y_true = y_true.astype(np.float64).reshape(-1)
    y_pred = y_pred.astype(np.float64).reshape(-1)

    if y_true.size == 0 or y_pred.size == 0:
        return {
            f"{prefix}_mae": float("nan"),
            f"{prefix}_rmse": float("nan"),
            f"{prefix}_nrmse": float("nan"),
            f"{prefix}_bias": float("nan"),
            f"{prefix}_smape": float("nan"),
            f"{prefix}_pearson": float("nan"),
            f"{prefix}_r2": float("nan"),
            f"{prefix}_cosine": float("nan"),
            f"{prefix}_quality": float("nan"),
        }

    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    nrmse = float(rmse / (float(np.std(y_true)) + EPS))
    bias = float(np.mean(diff))
    smape = float(np.mean((2.0 * np.abs(diff)) / (np.abs(y_true) + np.abs(y_pred) + EPS)))

    pearson = _safe_pearson(y_true, y_pred)
    r2 = _safe_r2(y_true, y_pred)

    cos_den = float(np.linalg.norm(y_true) * np.linalg.norm(y_pred) + EPS)
    cosine = float(np.dot(y_true, y_pred) / cos_den)
    cosine = float(np.clip(cosine, -1.0, 1.0))

    return {
        f"{prefix}_mae": mae,
        f"{prefix}_rmse": rmse,
        f"{prefix}_nrmse": nrmse,
        f"{prefix}_bias": bias,
        f"{prefix}_smape": smape,
        f"{prefix}_pearson": pearson,
        f"{prefix}_r2": r2,
        f"{prefix}_cosine": cosine,
        f"{prefix}_quality": _quality_score(pearson, nrmse, smape),
    }


def compute_spectrum_metrics(spec_gt: np.ndarray, spec_pred: np.ndarray, wave_ang: np.ndarray) -> Dict[str, float]:
    metrics = _compute_vector_metrics(spec_gt, spec_pred, prefix="spec")

    gt = spec_gt.astype(np.float64).reshape(-1)
    pred = spec_pred.astype(np.float64).reshape(-1)
    wave = wave_ang.astype(np.float64).reshape(-1)

    cos_den = float(np.linalg.norm(gt) * np.linalg.norm(pred) + EPS)
    cos_sim = float(np.clip(np.dot(gt, pred) / cos_den, -1.0, 1.0))
    sam_deg = float(np.degrees(np.arccos(cos_sim)))

    flux_gt = float(np.sum(gt))
    flux_pred = float(np.sum(pred))
    flux_rel_error = float(np.abs(flux_pred - flux_gt) / (np.abs(flux_gt) + EPS))

    if wave.size == gt.size:
        int_gt = float(np.trapz(gt, wave))
        int_pred = float(np.trapz(pred, wave))
        integral_rel_error = float(np.abs(int_pred - int_gt) / (np.abs(int_gt) + EPS))
    else:
        integral_rel_error = float("nan")

    metrics.update({
        "spec_sam_deg": sam_deg,
        "spec_flux_rel_error": flux_rel_error,
        "spec_integral_rel_error": integral_rel_error,
    })
    return metrics


def _align_image_shapes(img_gt: np.ndarray, img_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if img_gt.shape == img_pred.shape:
        return img_gt, img_pred
    c = min(img_gt.shape[0], img_pred.shape[0])
    h = min(img_gt.shape[1], img_pred.shape[1])
    w = min(img_gt.shape[2], img_pred.shape[2])
    if c <= 0 or h <= 0 or w <= 0:
        return img_gt, img_pred
    return img_gt[:c, :h, :w], img_pred[:c, :h, :w]


def compute_image_metrics(img_gt: np.ndarray, img_pred: np.ndarray) -> Dict[str, float]:
    img_gt, img_pred = _align_image_shapes(img_gt, img_pred)
    metrics = _compute_vector_metrics(img_gt, img_pred, prefix="img")

    gt_flat = img_gt.astype(np.float64).reshape(-1)
    pred_flat = img_pred.astype(np.float64).reshape(-1)

    rmse = metrics["img_rmse"]
    data_range = float(np.max(gt_flat) - np.min(gt_flat)) if gt_flat.size > 0 else float("nan")
    if np.isfinite(data_range) and data_range > EPS:
        psnr = float(20.0 * np.log10(data_range / (rmse + EPS)))
    else:
        psnr = float("nan")

    flux_gt = float(np.sum(gt_flat)) if gt_flat.size > 0 else float("nan")
    flux_pred = float(np.sum(pred_flat)) if pred_flat.size > 0 else float("nan")
    flux_rel_error = float(np.abs(flux_pred - flux_gt) / (np.abs(flux_gt) + EPS))

    metrics.update({
        "img_psnr": psnr,
        "img_global_ssim": _safe_global_ssim(img_gt, img_pred),
        "img_flux_rel_error": flux_rel_error,
    })

    channel_labels = ["vis", "nisp_h", "nisp_j", "nisp_y"] if img_gt.shape[0] == 4 else [f"c{i}" for i in range(img_gt.shape[0])]
    for i, label in enumerate(channel_labels):
        gt_ch = img_gt[i].reshape(-1)
        pred_ch = img_pred[i].reshape(-1)
        ch_metrics = _compute_vector_metrics(gt_ch, pred_ch, prefix=f"img_{label}")
        metrics.update(ch_metrics)

    return metrics


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_metrics_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    if not rows:
        logger.warning("No metric rows to save.")
        return

    priority_cols = [
        "target_id", "mode", "z", "dashboard_file", "train_name", "weights_dir", "ckpt_name", "split"
    ]
    all_cols = sorted({k for row in rows for k in row.keys()})
    metric_cols = [c for c in all_cols if c not in priority_cols]
    fieldnames = [c for c in priority_cols if c in all_cols] + metric_cols

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean_row = {}
            for col in fieldnames:
                val = row.get(col, "")
                if isinstance(val, np.generic):
                    val = val.item()
                if isinstance(val, float) and not np.isfinite(val):
                    val = ""
                clean_row[col] = val
            writer.writerow(clean_row)


def build_metrics_summary(rows: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    if not rows:
        return {"metadata": metadata, "n_rows": 0, "overall": {}, "by_mode": {}}

    non_metric_keys = {
        "target_id", "mode", "z", "dashboard_file", "train_name", "weights_dir", "ckpt_name", "split"
    }
    metric_keys = sorted({k for row in rows for k in row.keys() if k not in non_metric_keys})

    def summarize(group_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key in metric_keys:
            vals = []
            for row in group_rows:
                val = row.get(key, None)
                if isinstance(val, np.generic):
                    val = val.item()
                if isinstance(val, (int, float)) and np.isfinite(val):
                    vals.append(float(val))
            if not vals:
                continue

            arr = np.array(vals, dtype=np.float64)
            out[key] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "n": int(arr.size),
            }
        return out

    by_mode: Dict[str, Any] = {}
    modes = sorted({str(row.get("mode", "unknown")) for row in rows})
    for mode in modes:
        mode_rows = [r for r in rows if str(r.get("mode", "unknown")) == mode]
        by_mode[mode] = {
            "n_rows": len(mode_rows),
            "metrics": summarize(mode_rows),
        }

    summary = {
        "metadata": metadata,
        "n_rows": len(rows),
        "overall": summarize(rows),
        "by_mode": by_mode,
    }
    return _sanitize_for_json(summary)


def get_spiral_indices(side_len: int) -> np.ndarray:
    layout = np.arange(side_len * side_len).reshape(side_len, side_len)
    spiral_indices = []
    while layout.size > 0:
        spiral_indices.append(layout[0])           
        layout = layout[1:]
        if layout.size == 0: break
        spiral_indices.append(layout[:, -1])       
        layout = layout[:, :-1]
        if layout.size == 0: break
        spiral_indices.append(layout[-1][::-1])    
        layout = layout[:-1]
        if layout.size == 0: break
        spiral_indices.append(layout[:, 0][::-1])  
        layout = layout[:, 1:]
    spiral_order = np.concatenate(spiral_indices)
    flat_indices = np.empty(side_len * side_len, dtype=int)
    flat_indices[spiral_order] = np.arange(side_len * side_len)
    return (side_len * side_len - 1) - flat_indices


def reconstruct_image_from_patches(
    patch_sequence: np.ndarray,
    mod_config: Optional[Any] = None,
    apply_antispiral: bool = True,
) -> np.ndarray:
    seq_len, patch_dim = patch_sequence.shape
    channels = mod_config.input_size // (mod_config.patch_size ** 2) if mod_config else 4
    p_size = mod_config.patch_size if mod_config else int(np.sqrt(patch_dim // channels))
    grid_side = int(np.sqrt(seq_len))
    
    if grid_side * grid_side != seq_len:
        return np.zeros((channels, grid_side*p_size, grid_side*p_size))

    # 1. Deshacemos la espiral para volver a orden raster (solo si aplica)
    if apply_antispiral:
        spiral_indices = get_spiral_indices(grid_side)
        raster_patches = patch_sequence[spiral_indices]
    else:
        raster_patches = patch_sequence
    
    # 2. Reconstrucción con el orden EXACTO del dataloader: (p1 p2 c)
    image = einops.rearrange(
        raster_patches, 
        '(h w) (p1 p2 c) -> c (h p1) (w p2)', 
        h=grid_side, w=grid_side, p1=p_size, p2=p_size, c=channels
    )
    return image

def denormalize(data: np.ndarray, method: str, scaler: float, const: float) -> np.ndarray:
    if method == "asinh": 
        # Inversa matemática estricta: si y = asinh(x * const) / scaler 
        # entonces x = sinh(y * scaler) / const
        return np.sinh(data * scaler) / const
    elif method == "constant": 
        return data * const
    return data


def make_rgb_lupton(image_tensor: np.ndarray, Q: float = 10.0, stretch: float = 0.5, m: float = 0.0) -> np.ndarray:
    I = np.maximum(np.mean(image_tensor, axis=0) - m, 1e-10) 
    f_I = np.arcsinh(Q * stretch * I) / Q
    rgb_out = image_tensor * (f_I / I)[np.newaxis, :, :]
    max_rgb = np.percentile(rgb_out, 99.5)
    if max_rgb > 0: rgb_out /= max_rgb
    return np.clip(rgb_out, 0, 1).transpose(1, 2, 0)

def plot_spectral_lines(ax, min_wl, max_wl, z):
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    pos_high = y_min + (y_range * 0.95)
    pos_low  = y_min + (y_range * 0.25)
    
    sorted_lines = sorted(MAIN_LINES.items(), key=lambda x: x[1])
    counter = 0
    
    for name, rest_wave in sorted_lines:
        obs_wave = rest_wave * (1 + z)
        if min_wl < obs_wave < max_wl:
            y_pos = pos_high if counter % 2 == 0 else pos_low
            ax.axvline(obs_wave, color='royalblue', linestyle='--', alpha=0.6, lw=1)
            ax.text(obs_wave, y_pos, rf"\textbf{{{name}}}", rotation=90, 
                    color='royalblue', va='top', ha='right', fontsize=12, alpha=1, fontweight='bold')
            counter += 1


def generate_autoregressive(
    model,
    X_full: Dict[str, torch.Tensor],
    target_mod: str,
    max_len: int,
    device: torch.device,
    cond_mod: Optional[str] = None,
) -> np.ndarray:
    """
    Generates a modality autoregressively from a single seed token.
    If cond_mod is provided, generation is cross-modal; otherwise it is self-modal.
    """
    
    # [CRITICAL FIX FOR TOKEN MIXING]: 
    # If the model was trained with 'use_token_mixing=True', we MUST disable it 
    # during zero-shot generation. Token mixing interleaves the full condition 
    # with the 1-token target, causing the target to be completely isolated from 
    # the condition due to the causal mask.
    # By forcing sequential processing (token_mixing = False), we assemble the sequence as:
    # [Condition_0 ... Condition_N, Target_0 ... Target_K]
    # This allows Target_K to cleanly attend to the ENTIRE condition modality.
    original_token_mixing = getattr(model.config, 'use_token_mixing', False)
    if original_token_mixing:
        model.config.use_token_mixing = False

    model.eval()
    with torch.no_grad():
        if cond_mod is not None:
            # True Zero-Shot Cross-Generation:
            # We predict the very first target token (Target_0) from the last condition token (Cond_N).
            # We don't cheat by using the ground-truth first token.
            X_init = {cond_mod: X_full[cond_mod], f"{cond_mod}_positions": X_full[f"{cond_mod}_positions"]}
            outputs, _ = model(X_init, target_modality=target_mod)
            Target_0 = outputs[target_mod][:, -1:, :]
            generated_tokens = [Target_0]
        else:
            # Self-Generation:
            # Since there is no conditioning context and no discrete <BOS> token for continuous patches, 
            # we MUST seed the sequence with the real first token (typically the top-left sky patch).
            generated_tokens = [X_full[target_mod][:, 0:1, :]]

        for _ in tqdm(range(max_len - 1), desc=f"Generating {target_mod.upper()}", leave=False):
            X_current = {}

            # ORDERING IS CRITICAL HERE:
            # We insert cond_mod first into the dictionary. Since Python 3.7+ preserves 
            # insertion order, model._forward_native will place cond_mod at the beginning 
            # of the sequence. This forms our Causal Bridge!
            if cond_mod is not None:
                X_current[cond_mod] = X_full[cond_mod]
                X_current[f"{cond_mod}_positions"] = X_full[f"{cond_mod}_positions"]

            # Target history built autoregressively
            curr_tokens = torch.cat(generated_tokens, dim=1)
            real_len = curr_tokens.shape[1]

            full_pos_vec = X_full[f"{target_mod}_positions"]
            input_pos = full_pos_vec[:, :real_len]

            if input_pos.shape[1] < real_len:
                extra = torch.arange(input_pos.shape[1], real_len, device=device).unsqueeze(0)
                input_pos = torch.cat([input_pos, extra], dim=1)

            X_current[target_mod] = curr_tokens
            X_current[f"{target_mod}_positions"] = input_pos

            # Forward pass
            outputs, _ = model(X_current)

            # Next token is always read from the latest output slot.
            next_token = outputs[target_mod][:, -1:, :]
            generated_tokens.append(next_token)
            
    # Restore the model's original configuration
    if original_token_mixing:
        model.config.use_token_mixing = True

    full_sequence = torch.cat(generated_tokens, dim=1)
    return full_sequence.float().cpu().numpy()[0]


def postprocess_spectrum(
    pred_spec_seq: np.ndarray,
    wave_ang: np.ndarray,
    inverse_spec: bool,
    norm_type_spec: str,
    norm_scaler_spec: float,
    norm_const_spec: float,
) -> np.ndarray:
    spec_pred = denormalize(pred_spec_seq.flatten(), norm_type_spec, norm_scaler_spec, norm_const_spec)
    spec_pred = spec_pred[: len(wave_ang)]
    if inverse_spec:
        spec_pred = spec_pred[::-1]
    return spec_pred


def postprocess_image_prediction(
    pred_img_seq: np.ndarray,
    img_config: Any,
    apply_antispiral: bool,
    norm_type_img: str,
    norm_scaler_img: float,
    norm_const_img: float,
    img_gt_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    img_pred_phys = denormalize(
        reconstruct_image_from_patches(
            pred_img_seq,
            img_config,
            apply_antispiral=apply_antispiral,
        ),
        norm_type_img,
        norm_scaler_img,
        norm_const_img,
    )

    RGB_WEIGHTS = [1.2, 1.3, 1.0]
    bg_val = np.percentile(img_gt_raw, 50, axis=(1, 2), keepdims=True)

    raw_bg = img_gt_raw - bg_val
    pred_bg = img_pred_phys - bg_val

    raw_rgb_stack = []
    pred_rgb_stack = []
    for c in range(4):
        v_max = np.percentile(np.abs(raw_bg[c]), 99.5)
        if v_max <= 0:
            v_max = 1.0
        r_ch = np.clip(raw_bg[c] / v_max, 0, 100)
        p_ch = np.clip(pred_bg[c] / v_max, 0, 100)
        raw_rgb_stack.append(r_ch)
        pred_rgb_stack.append(p_ch)

    raw_norm = np.stack(raw_rgb_stack)
    pred_norm = np.stack(pred_rgb_stack)

    def stack_to_rgb_weighted(stack):
        vis, h, j, y = stack[0], stack[1], stack[2], stack[3]
        r = h * RGB_WEIGHTS[0]
        g = ((j + y) / 2.0) * RGB_WEIGHTS[1]
        b = vis * RGB_WEIGHTS[2]
        return np.stack([r, g, b], axis=0)

    rgb_input_gt = stack_to_rgb_weighted(raw_norm)
    rgb_input_pred = stack_to_rgb_weighted(pred_norm)

    rgb_gt = make_rgb_lupton(rgb_input_gt, Q=12.0, stretch=0.5)
    rgb_pred = make_rgb_lupton(rgb_input_pred, Q=12.0, stretch=0.5)

    if img_gt_raw.shape == img_pred_phys.shape:
        res_map = np.mean(img_gt_raw - img_pred_phys, axis=0)
    else:
        logger.warning(
            f"Shape mismatch: GT {img_gt_raw.shape} vs Pred {img_pred_phys.shape}. Using zero residual map."
        )
        res_map = np.zeros((img_gt_raw.shape[1], img_gt_raw.shape[2]))

    return rgb_gt, rgb_pred, res_map, img_pred_phys


def plot_reconstruction_dashboard(
    target_id: int, z_val: float, train_name: str,
    rgb_gt: np.ndarray, rgb_pred: np.ndarray, res_map: np.ndarray,
    wave_ang: np.ndarray, spec_gt: np.ndarray, spec_pred: np.ndarray,
    save_dir: Path, filename: str,
    dashboard_title: str,
    image_pred_title: str,
    spectrum_pred_legend: str,
    spectrum_panel_prefix: str,
):
    """Plots a unified reconstruction dashboard in a 3-row layout."""
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], wspace=0.1, hspace=0.3)
    
    fig.suptitle(rf"\textbf{{{dashboard_title} | ID: {target_id} | z={z_val:.3f}}}"
                 + f"\n[{train_name}]", fontsize=22, y=0.96)

    # ROW 0: IMAGES
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb_gt, origin='lower')
    ax1.set_title(r"\textbf{Target: Real Image}")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(rgb_pred, origin='lower')
    ax2.set_title(rf"\textbf{{{image_pred_title}}}")
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    vlim = np.percentile(np.abs(res_map), 98)
    im = ax3.imshow(res_map, origin='lower', cmap='seismic', vmin=-vlim, vmax=vlim)
    ax3.set_title(r"\textbf{Residuals (Physical)}")
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="Flux Diff")

    # ROW 1 & 2: SPECTRA
    w_min, w_max = wave_ang.min(), wave_ang.max()
    w_mid = (w_min + w_max) / 2
    
    for i, (start, end, title, loc) in enumerate([
        (w_min, w_mid, "Blue Channel", gs[1, :]), 
        (w_mid, w_max, "Red Channel", gs[2, :])
    ]):
        ax = fig.add_subplot(loc)
        ax.plot(wave_ang, spec_gt, 'k-', lw=1, alpha=0.6, label='Input: Real Spectrum')
        ax.plot(wave_ang, spec_pred, 'r-', lw=1.5, alpha=0.8, label=spectrum_pred_legend)
        ax.set_xlim(start, end)
        ax.set_title(rf"\textbf{{{spectrum_panel_prefix} ({title})}}")
        ax.set_ylabel(r"Flux")
        if i==1: ax.set_xlabel(r"Wavelength [\AA]")
        if i==0: ax.legend(loc='lower left')
        plot_spectral_lines(ax, start, end, z_val)

    # Save
    save_path = save_dir / filename
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f" --> Saved dashboard: {save_path.name}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    weights_dir = Path(args.weights_dir)
    if args.save_dir is not None:
        save_root = Path(args.save_dir) / "cross_reconstructions"
    else:
        save_root = weights_dir.parent / "plots" / "cross_reconstructions"
        
    cross_save_dir = save_root / "cross_modal"
    self_save_dir = save_root / "self_modal"
    metrics_save_dir = save_root / "metrics"
    cross_save_dir.mkdir(parents=True, exist_ok=True)
    self_save_dir.mkdir(parents=True, exist_ok=True)
    metrics_save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing Cross-Modal Generative Analysis...")
    ckpt_filename = weights_dir / args.ckpt_name
    name_no_ext = ckpt_filename.stem
    if '_' in name_no_ext:
        run_suffix = f"_{name_no_ext.split('_')[-1]}" 
    else:
        run_suffix = f"_{name_no_ext}"
        
    # Extract training run name for plot titles
    config_path = weights_dir / "config.json"
    json_name = None
    if config_path.is_file():
        try:
            with open(config_path, 'r') as f:
                json_name = json.load(f).get("train_name", None)
        except Exception: pass 
    train_name = args.train_name or json_name or weights_dir.parent.name
    
    # Loading weights
    ckpt_path = weights_dir / args.ckpt_name
    
    if not ckpt_path.is_file():
        ckpt_path = sorted(weights_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)[0]

    try:
        model, config, registry, raw_config_dict = load_local_model(ckpt_path, device)
        model.eval()
    except Exception as e:
        logger.critical(f"Model load failed: {e}")
        sys.exit(1)
        
    img_config = registry.get_config("images")
    data_dir = Path(raw_config_dict.get('data_dir', args.data_dir))
    
    # Transforms & Dataloader setup
    # Recuperamos los parámetros exactos del config.json del .pt
    norm_type_img = raw_config_dict.get('images_norm_type', 'asinh')
    norm_scaler_img = raw_config_dict.get('images_norm_scaler', 1.0)
    norm_const_img = raw_config_dict.get('images_norm_const', 1.0)
    
    inverse_spec = raw_config_dict.get('spectra_inverse', False)
    norm_type_spec = raw_config_dict.get('spectra_norm_type', 'asinh')
    norm_scaler_spec = raw_config_dict.get('spectra_norm_scaler', 1.0)
    norm_const_spec = raw_config_dict.get('spectra_norm_const', 1.0)

    # Actualizamos el transform para que el dataloader use la misma escala
    data_tf = EuclidDESIDatasetArrow.data_transforms(
        norm_type_img=norm_type_img, norm_scaler_img=norm_scaler_img, norm_const_img=norm_const_img,
        norm_type_spec=norm_type_spec, norm_scaler_spec=norm_scaler_spec, norm_const_spec=norm_const_spec,
    )

    ds = EuclidDESIDatasetArrow(
        arrow_folder_root=data_dir, 
        split=args.split, 
        modality_registry=registry,
        spiral=raw_config_dict.get('spiral', True), 
        transform=data_tf,
        spectra_inverse=inverse_spec,
    )
    
    # Sample Selection
    indices_to_plot = []
    specific_tids = set()
    
    if args.target_ids:
        all_ids = ds.ds['targetid']
        id_map = {int(tid): idx for idx, tid in enumerate(all_ids)}
        for tid in args.target_ids:
            if tid in id_map: 
                indices_to_plot.append(id_map[tid])
                specific_tids.add(int(tid))
    
    if len(indices_to_plot) < args.num_plot:
        pool = list(set(range(len(ds))) - set(indices_to_plot))
        needed = args.num_plot - len(indices_to_plot)
        indices_to_plot.extend(np.random.choice(pool, min(needed, len(pool)), replace=False))
        
    # Ensuring correct type
    indices_to_plot = [int(i) for i in indices_to_plot]
    loader = DataLoader(Subset(ds, indices_to_plot), batch_size=1, shuffle=False)

    apply_antispiral = bool(raw_config_dict.get('spiral', True))

    if device.type == 'cuda':
        ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype) # type: ignore
    else:
        import contextlib
        ctx = contextlib.nullcontext()
        model = model.to(torch.float32)

    metric_rows: List[Dict[str, Any]] = []

    # Generation Loop
    for batch_idx, batch in enumerate(loader):
        
        try:
            arrow_idx = int(batch['idx'].item())
        except:
            arrow_idx = int(batch['idx'][0])

        raw_record = ds.ds[arrow_idx]
        target_id = int(raw_record['targetid'])
        z_val = float(raw_record.get('redshift', 0.0))
        
        logger.info(f"Processing Target ID {target_id} ({batch_idx+1}/{len(indices_to_plot)})")
        
        X = EuclidDESIDatasetArrow.prepare_batch(batch, registry, device)
        
        if device.type == 'cpu':
            for k, v in X.items():
                if isinstance(v, torch.Tensor):
                    if 'positions' in k:
                        X[k] = v.to(torch.long)
                    else:
                        X[k] = v.to(torch.float32)

        if 'images' not in X or 'spectra' not in X:
            logger.warning("Missing modalities in batch. Skipping.")
            continue
            
        # Filtro de seguridad por si alguna galaxia viene vacía desde el catálogo
        if X['images'].shape[1] == 0 or X['spectra'].shape[1] == 0:
            logger.warning(f"Target ID {target_id} is missing data. Skipping.")
            continue

        # Sequence lengths
        max_img_len = X['images'].shape[1]
        max_spec_len = X['spectra'].shape[1]

        with ctx:
            # Cross-modal generation
            logger.info(" --> Generating Spectrum from Image...")
            pred_spec_seq_cross = generate_autoregressive(
                model=model,
                X_full=X,
                target_mod='spectra',
                max_len=max_spec_len,
                device=device,
                cond_mod='images',
            )
            
            logger.info(" --> Generating Image from Spectrum...")
            pred_img_seq_cross = generate_autoregressive(
                model=model,
                X_full=X,
                target_mod='images',
                max_len=max_img_len,
                device=device,
                cond_mod='spectra',
            )

            # Self-modal generation
            logger.info(" --> Generating Spectrum from itself...")
            pred_spec_seq_self = generate_autoregressive(
                model=model,
                X_full=X,
                target_mod='spectra',
                max_len=max_spec_len,
                device=device,
                cond_mod=None,
            )

            logger.info(" --> Generating Image from itself...")
            pred_img_seq_self = generate_autoregressive(
                model=model,
                X_full=X,
                target_mod='images',
                max_len=max_img_len,
                device=device,
                cond_mod=None,
            )
            
        # --- DATA POST-PROCESSING (De-normalization & Formatting) ---
        raw_record = ds.ds[int(batch['idx'][0])]
        wave_ang = np.array(raw_record['spectrum_wave']).flatten()
        spec_gt_raw = np.array(raw_record['spectrum_flux']).flatten()
        
        spec_pred_cross = postprocess_spectrum(
            pred_spec_seq_cross,
            wave_ang,
            inverse_spec,
            norm_type_spec,
            norm_scaler_spec,
            norm_const_spec,
        )
        spec_pred_self = postprocess_spectrum(
            pred_spec_seq_self,
            wave_ang,
            inverse_spec,
            norm_type_spec,
            norm_scaler_spec,
            norm_const_spec,
        )
        
        min_l = min(len(spec_gt_raw), len(spec_pred_cross), len(spec_pred_self), len(wave_ang))
        
        # Image Post-processing
        img_gt_raw = np.stack([raw_record['image_vis'], raw_record['image_nisp_h'], 
                               raw_record['image_nisp_j'], raw_record['image_nisp_y']])
        rgb_gt_cross, rgb_pred_cross, res_map_cross, img_pred_cross_phys = postprocess_image_prediction(
            pred_img_seq_cross,
            img_config,
            apply_antispiral,
            norm_type_img,
            norm_scaler_img,
            norm_const_img,
            img_gt_raw,
        )
        rgb_gt_self, rgb_pred_self, res_map_self, img_pred_self_phys = postprocess_image_prediction(
            pred_img_seq_self,
            img_config,
            apply_antispiral,
            norm_type_img,
            norm_scaler_img,
            norm_const_img,
            img_gt_raw,
        )

        # FILENAME AND SUFFIX LOGIC
        random_suffix = "_R" if int(target_id) not in specific_tids else ""
        zoom_suffix = "_zoom" if args.wl_range else ""
        filename_cross = f"CrossRecon_ID_{target_id}{run_suffix}{zoom_suffix}{random_suffix}.png"
        filename_self = f"SelfRecon_ID_{target_id}{run_suffix}{zoom_suffix}{random_suffix}.png"

        # Cross-modal dashboard
        plot_reconstruction_dashboard(
            target_id=target_id, z_val=z_val, 
            train_name=train_name,
            rgb_gt=rgb_gt_cross,
            rgb_pred=rgb_pred_cross,
            res_map=res_map_cross,
            wave_ang=wave_ang[:min_l],
            spec_gt=spec_gt_raw[:min_l],
            spec_pred=spec_pred_cross[:min_l],
            save_dir=cross_save_dir,
            filename=filename_cross,
            dashboard_title="Cross-Modal Generation",
            image_pred_title="Output: Gen. from Spectrum",
            spectrum_pred_legend="Output: Gen. from Image",
            spectrum_panel_prefix="Cross-Generated Spectrum",
        )

        # Self-modal dashboard
        plot_reconstruction_dashboard(
            target_id=target_id, z_val=z_val,
            train_name=train_name,
            rgb_gt=rgb_gt_self,
            rgb_pred=rgb_pred_self,
            res_map=res_map_self,
            wave_ang=wave_ang[:min_l],
            spec_gt=spec_gt_raw[:min_l],
            spec_pred=spec_pred_self[:min_l],
            save_dir=self_save_dir,
            filename=filename_self,
            dashboard_title="Self-Modal Generation",
            image_pred_title="Output: Self-Generated Image",
            spectrum_pred_legend="Output: Self-Generated Spectrum",
            spectrum_panel_prefix="Self-Generated Spectrum",
        )

        if not args.disable_metrics:
            spec_gt_eval = spec_gt_raw[:min_l]
            wave_eval = wave_ang[:min_l]

            spec_metrics_cross = compute_spectrum_metrics(spec_gt_eval, spec_pred_cross[:min_l], wave_eval)
            spec_metrics_self = compute_spectrum_metrics(spec_gt_eval, spec_pred_self[:min_l], wave_eval)
            img_metrics_cross = compute_image_metrics(img_gt_raw, img_pred_cross_phys)
            img_metrics_self = compute_image_metrics(img_gt_raw, img_pred_self_phys)

            cross_row: Dict[str, Any] = {
                "target_id": target_id,
                "mode": "cross",
                "z": z_val,
                "dashboard_file": filename_cross,
                "train_name": train_name,
                "weights_dir": str(weights_dir),
                "ckpt_name": ckpt_path.name,
                "split": args.split,
            }
            cross_row.update(spec_metrics_cross)
            cross_row.update(img_metrics_cross)
            cross_row["joint_quality"] = float(np.sqrt(cross_row.get("spec_quality", np.nan) * cross_row.get("img_quality", np.nan)))
            metric_rows.append(cross_row)

            self_row: Dict[str, Any] = {
                "target_id": target_id,
                "mode": "self",
                "z": z_val,
                "dashboard_file": filename_self,
                "train_name": train_name,
                "weights_dir": str(weights_dir),
                "ckpt_name": ckpt_path.name,
                "split": args.split,
            }
            self_row.update(spec_metrics_self)
            self_row.update(img_metrics_self)
            self_row["joint_quality"] = float(np.sqrt(self_row.get("spec_quality", np.nan) * self_row.get("img_quality", np.nan)))
            metric_rows.append(self_row)

    if not args.disable_metrics:
        csv_path = metrics_save_dir / args.metrics_csv_name
        summary_path = metrics_save_dir / args.summary_json_name

        write_metrics_csv(metric_rows, csv_path)

        summary_metadata = {
            "train_name": train_name,
            "weights_dir": str(weights_dir),
            "ckpt_name": ckpt_path.name,
            "split": args.split,
            "n_targets": len(indices_to_plot),
            "run_suffix": run_suffix,
            "generated_at": str(np.datetime64("now")),
        }
        summary_payload = build_metrics_summary(metric_rows, summary_metadata)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        logger.info(f" --> Saved metrics CSV: {csv_path}")
        logger.info(f" --> Saved metrics summary JSON: {summary_path}")

        if metric_rows:
            cross_joint = [r.get("joint_quality", np.nan) for r in metric_rows if r.get("mode") == "cross"]
            self_joint = [r.get("joint_quality", np.nan) for r in metric_rows if r.get("mode") == "self"]

            cross_joint = [float(v) for v in cross_joint if isinstance(v, (float, int)) and np.isfinite(v)]
            self_joint = [float(v) for v in self_joint if isinstance(v, (float, int)) and np.isfinite(v)]

            if cross_joint:
                logger.info(f"[Summary] Cross joint_quality mean: {np.mean(cross_joint):.4f}")
            if self_joint:
                logger.info(f"[Summary] Self joint_quality mean: {np.mean(self_joint):.4f}")
    else:
        logger.info("Objective metric export disabled by --disable_metrics")
        
if __name__ == "__main__":
    main()