"""
AstroPT Zero-Shot Dataset Corruption Audit Tool.

This script uses a pre-trained AstroPT checkpoint to automatically audit galaxy images
and detect observational corruption, glitches, coordinate misalignments, satellite trails, 
and camera artifacts.

It computes:
1. AstroPT Visual Reconstruction MAE Loss (via unimodal visual forward pass)
2. Zero Pixel Fraction (detects sensor masking, coordinate offsets, large black holes)
3. High-Frequency Edge Score (detects sharp satellite trails, cosmic rays, diagonal saturation)

It outputs a calibrated 'Corruption Probability (%)' and generates a PDF dashboard
of the most flagged corrupted galaxies for visual validation.

Author: Victor Alonso Rodriguez
Date: June 2026
"""

import argparse
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.table import Table
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import fields

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from astropt.dataloader_multimodal import MultimodalDatasetArrow
from astropt.training_utils import create_dataloaders
from astropt.config import TrainingConfig
from astropt.model_utils import load_local_model

# --- Configure logging ---
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-DatasetAuditor")

def parse_args():
    parser = argparse.ArgumentParser(description="AstroPT Dataset Corruption Auditor")
    parser.add_argument("--ckpt_path", type=str, required=True, 
                        help="Path to pre-trained AstroPT checkpoint (.pt)")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Root folder of raw Arrow processed data")
    parser.add_argument("--metadata_path", type=str, required=True, 
                        help="Path to FITS metadata catalog")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Saving directory for catalogs and plots")
    parser.add_argument("--n_samples", type=int, default=5000, 
                        help="Number of samples to audit (set -1 for full dataset)")
    parser.add_argument("--split", type=str, default="both", 
                        choices=["both", "train", "val"],
                        help="Which dataset split to audit: 'train', 'val', or 'both' (default)")
    return parser.parse_args()

def make_rgb_lupton(image_tensor: np.ndarray, Q: float = 12.0, stretch: float = 0.5) -> np.ndarray:
    """Lupton et al. (2004) algorithm implementation for false-color RGB."""
    I = np.mean(image_tensor, axis=0)
    I = np.maximum(I, 1e-10)
    f_I = np.arcsinh(Q * stretch * I) / Q
    scale_factor = f_I / I
    rgb_out = image_tensor * scale_factor[np.newaxis, :, :]
    max_rgb = np.percentile(rgb_out, 99.5)
    if max_rgb > 0:
        rgb_out = rgb_out / max_rgb
    rgb_out = np.clip(rgb_out, 0, 1)
    return rgb_out.transpose(1, 2, 0)

def extract_raw_rgb(raw_record) -> np.ndarray:
    """Safely extracts channels and returns false-color RGB."""
    try:
        def get_ch(k):
            val = raw_record.get(k)
            return np.array(val if val is not None else [], dtype=np.float32)
        
        vis = get_ch('image_vis')
        y = get_ch('image_nisp_y')
        j = get_ch('image_nisp_j')
        h = get_ch('image_nisp_h')
        
        if vis.size > 0:
            h = h if h.size > 0 else np.zeros_like(vis)
            j = j if j.size > 0 else np.zeros_like(vis)
            y = y if y.size > 0 else np.zeros_like(vis)
            
            raw_stack = np.stack([vis, h, j, y], axis=0)
            bg_val = np.percentile(raw_stack, 50, axis=(1,2), keepdims=True)
            raw_bg = raw_stack - bg_val
            
            raw_rgb_stack = []
            for c in range(raw_bg.shape[0]):
                v_max = np.percentile(np.abs(raw_bg[c]), 99.5)
                if v_max <= 0: v_max = 1.0
                raw_rgb_stack.append(np.clip(raw_bg[c] / v_max, 0, 100))
            raw_norm = np.stack(raw_rgb_stack)
            
            RGB_WEIGHTS = [1.2, 1.3, 1.0]
            r = raw_norm[1] * RGB_WEIGHTS[0]
            g = ((raw_norm[2] + raw_norm[3]) / 2.0) * RGB_WEIGHTS[1]
            b = raw_norm[0] * RGB_WEIGHTS[2]
            
            return make_rgb_lupton(np.stack([r, g, b], axis=0), Q=12.0, stretch=0.5)
    except Exception as e:
        logger.error(f"Error extracting RGB image: {e}")
    return None

def compute_spatial_priors(raw_record) -> tuple:
    """Computes zero-pixel fraction and Laplace high-frequency edge scores from raw visual data."""
    try:
        vis_val = raw_record.get('image_vis')
        if vis_val is None:
            return 0.0, 0.0
        vis = np.array(vis_val, dtype=np.float32)
        if vis.size == 0:
            return 0.0, 0.0
        
        # 1. Zero pixel fraction (masking/dead pixels)
        # We consider pixels exactly equal to 0.0 or below a very tiny threshold
        zero_fraction = float(np.mean(vis <= 1e-7))
        
        # 2. Laplace high-frequency edge score (gradient variance)
        # Unphysical sharp columns, streaks, or boundaries generate massive localized gradients.
        if vis.ndim == 2:
            # Simple 3x3 Laplace kernel convolution
            lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
            from scipy.signal import convolve2d
            laplacian = convolve2d(vis, lap_kernel, mode='same')
            edge_score = float(np.std(laplacian))
        else:
            edge_score = float(np.std(np.diff(vis)))
            
        return zero_fraction, edge_score
    except Exception as e:
        logger.error(f"Error computing spatial priors: {e}")
        return 0.0, 0.0

def main():
    args = parse_args()
    
    ckpt_path = Path(args.ckpt_path)
    embeddings_dir = ckpt_path.parent.parent / "embeddings"
    save_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent.parent / "anomalies"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load pre-trained model and configuration
    logger.info(f"Loading pre-trained AstroPT checkpoint from {ckpt_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, registry, raw_config_dict = load_local_model(ckpt_path, device)
    
    # Set to evaluation mode
    model.eval()
    
    # Configure dataloader parameters
    raw_config_dict['data_dir'] = args.data_dir
    raw_config_dict['metadata_path'] = args.metadata_path
    raw_config_dict['batch_size'] = 1  # batch_size=1 ensures we get exact per-galaxy losses
    raw_config_dict['shuffle_modality_train'] = False
    
    valid_keys = {f.name for f in fields(TrainingConfig)}
    clean_config_dict = {k: v for k, v in raw_config_dict.items() if k in valid_keys}
    training_config = TrainingConfig(**clean_config_dict)
    
    train_loader, val_loader, registry = create_dataloaders(training_config, ddp=False)
    ds_train = train_loader.dataset
    ds_val = val_loader.dataset
    
    # Resolve split selection
    split = getattr(args, "split", "both").lower().strip()
    if split == "train":
        datasets_to_audit = [ds_train]
        logger.info("Selected split: training dataset only.")
    elif split == "val":
        datasets_to_audit = [ds_val]
        logger.info("Selected split: validation dataset only.")
    else:
        datasets_to_audit = [ds_train, ds_val]
        logger.info("Selected split: combined training + validation datasets.")

    def get_sample_and_record(i):
        current_idx = i
        for ds_subset in datasets_to_audit:
            if current_idx < len(ds_subset):
                return ds_subset[current_idx], ds_subset.ds[current_idx]
            current_idx -= len(ds_subset)
        raise IndexError(f"Index {i} out of bounds for audited datasets.")
        
    # Select subset or full dataset
    total_galaxies = sum(len(ds_subset) for ds_subset in datasets_to_audit)
    n_audit = total_galaxies if args.n_samples <= 0 else min(args.n_samples, total_galaxies)
    logger.info(f"Starting zero-shot audit of {n_audit} galaxies (out of {total_galaxies} available)...")
    
    audit_records = []
    
    # Context manager for AMP
    ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
    
    # Loop over dataset
    for i in range(n_audit):
        try:
            raw_sample, raw_record = get_sample_and_record(i)
        except Exception as e:
            logger.warning(f"Skipping index {i}: Critical loading error: {e}")
            continue

        tid = int(raw_record.get('targetid', 0))
        
        # Determine the expected index within the current subset to check for dataloader resampling
        expected_subset_idx = i
        for ds_subset in datasets_to_audit:
            if expected_subset_idx < len(ds_subset):
                break
            expected_subset_idx -= len(ds_subset)
            
        # If the returned sample index does not match the expected index, the dataloader resampled due to an error
        if raw_sample.get("idx") != expected_subset_idx or int(raw_sample.get("targetid", -1)) != tid:
            logger.warning(f"Skipping corrupt galaxy at index {i} (ID: {tid}) due to internal dataloader loading/reshaping error.")
            continue
        
        # Load and collate single sample into a batch dictionary of stacked tensors
        batch_data = {}
        for k, v in raw_sample.items():
            if isinstance(v, torch.Tensor):
                batch_data[k] = v.unsqueeze(0)
            elif isinstance(v, (list, np.ndarray)):
                batch_data[k] = torch.tensor(v).unsqueeze(0)
            else:
                batch_data[k] = [v]
        
        # 1. Spatial priors directly from raw visual array
        zero_frac, edge_score = compute_spatial_priors(raw_record)
        
        # 2. AstroPT Visual Reconstruction MAE Loss
        mae_loss = 0.0
        try:
            B = MultimodalDatasetArrow.process_modes(
                batch_data=batch_data,
                modality_registry=registry,
                device=device,
                shuf=False,
                use_token_mixing=config.use_token_mixing,
                token_mixing_seed=config.token_mixing_seed,
                use_cls_token=config.use_cls_token,
                cls_position=config.cls_position
            )
            
            with torch.no_grad():
                with ctx:
                    # Drop spectra completely to evaluate ONLY visual reconstruction capability
                    outputs, loss = model(B["X"], targets=B["Y"].copy(), dropped_modality="DESISpectrum")
                    mae_loss = float(loss.item())
        except Exception as e:
            logger.error(f"Error evaluating AstroPT loss on galaxy Index {i} (ID: {tid}): {e}")
            mae_loss = 999.0  # Fallback for completely corrupt/unprocessable entries
            
        audit_records.append({
            "TargetID": tid,
            "Index": i,
            "AstroPT_Visual_Loss": mae_loss,
            "Zero_Pixel_Fraction": zero_frac,
            "Sharp_Edge_Score": edge_score
        })
        
        if (i + 1) % 1000 == 0:
            logger.info(f"  Processed {i+1} / {n_audit} galaxies...")
            
    df_audit = pd.DataFrame(audit_records)
    
    # 3. Calibrate metrics into a robust Corruption Probability
    # Percentile standardize each metric
    # Higher MAE loss, higher zero pixel fraction, and extreme/unusual edge scores mean highly corrupted.
    logger.info("Calibrating metrics into unified observational quality scores...")
    
    # Handle NaNs / Inf safely
    df_audit = df_audit.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    def get_percentiles(col_vals):
        return np.argsort(np.argsort(col_vals)) / (len(col_vals) - 1.0)
    
    pct_loss = get_percentiles(df_audit["AstroPT_Visual_Loss"].values)
    pct_zeros = get_percentiles(df_audit["Zero_Pixel_Fraction"].values)
    pct_edges = get_percentiles(df_audit["Sharp_Edge_Score"].values)
    
    # Combined Corruption Index (CCI)
    # An image is flagged as highly corrupted if it is extreme in ANY of these indicators
    # We apply a smooth scaling to map to a nice percentage [0 - 100%]
    cci_scores = np.maximum(pct_loss * 0.85, np.maximum(pct_zeros, pct_edges))
    
    df_audit["Corruption_Probability (%)"] = np.clip(cci_scores * 100.0, 0.0, 100.0)
    
    # Sort by corruption probability descending
    df_audit = df_audit.sort_values(by="Corruption_Probability (%)", ascending=False).reset_index(drop=True)
    
    # Save Catalog
    csv_path = save_dir / "dataset_corruption_audit.csv"
    df_audit.to_csv(csv_path, index=False)
    logger.info(f"Saved complete corruption audit catalog to {csv_path}")
    
    # 4. Generate visual validation PDF of top 5 flagged corrupted galaxies
    pdf_path = save_dir / "top_flagged_corrupt_galaxies.pdf"
    logger.info(f"Generating visual validation report for top flagged galaxies at {pdf_path}...")
    
    # Set up LaTeX serif font styling and include amsmath and xcolor packages
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{xcolor}'
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', weight='bold')
    
    n_plot = min(30, len(df_audit))
    fig, axes = plt.subplots(n_plot, 1, figsize=(10, 4 * n_plot), dpi=150)
    fig.subplots_adjust(hspace=0.45, top=0.92, bottom=0.04)
    
    if n_plot == 1:
        axes = [axes]
        
    for r in range(n_plot):
        row = df_audit.iloc[r]
        tid = int(row.TargetID)
        prob = row["Corruption_Probability (%)"]
        loss_val = row.AstroPT_Visual_Loss
        z_frac = row.Zero_Pixel_Fraction
        edge_val = row.Sharp_Edge_Score
        idx = int(row.Index)
        
        ax = axes[r]
        ax.axis('off')
        
        _, raw_record = get_sample_and_record(idx)
        rgb = extract_raw_rgb(raw_record)
        
        # Subplot inside subplot layout: Image on left, metadata block on right
        sub_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=ax.get_subplotspec(), width_ratios=[1, 1], wspace=0.1)
        
        ax_img = fig.add_subplot(sub_gs[0, 0])
        ax_txt = fig.add_subplot(sub_gs[0, 1])
        
        ax_img.axis('off')
        if rgb is not None:
            ax_img.imshow(rgb, origin='lower')
        else:
            ax_img.text(0.5, 0.5, "EuclidImage missing", ha='center', va='center', fontsize=12)
            
        ax_txt.axis('off')
        meta_text = (
            rf"\textbf{{TargetID}}: {tid}\n"
            rf"\textbf{{Audit Index}}: {idx}\n\n"
            rf"\textbf{{AstroPT Visual MAE Loss}}: {loss_val:.4f}\n"
            rf"\textbf{{Zero Pixel Fraction}}: {z_frac:.2%}\n"
            rf"\textbf{{Laplacian Edge Score}}: {edge_val:.1f}\n\n"
            rf"\textbf{{Corruption Confidence}}: \textcolor{{red}}{{{prob:.2f}\%}}"
        )
        # Handle string lines
        meta_text = meta_text.replace(r"\n", "\n").replace("_", "\\_")
        
        ax_txt.text(
            0.05, 0.95, meta_text,
            transform=ax_txt.transAxes,
            fontsize=12,
            linespacing=1.6,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.7", fc="mistyrose", alpha=0.95, ec="red", lw=1.5)
        )
        
        ax.set_title(rf"\textbf{{Rank \#{r+1} - Flagged Corruption ({prob:.2f}\%)}}", fontsize=14, fontweight='bold', color='darkred', pad=10)
        
    plt.suptitle(r"\textbf{AstroPT Zero-Shot Observational Quality Audit (Top Flags)}", fontsize=18, fontweight='bold', y=0.97)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Successfully saved validation PDF dashboard to {pdf_path}")
    logger.info("Dataset quality audit completed successfully.")

if __name__ == "__main__":
    main()
