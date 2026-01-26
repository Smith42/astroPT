"""
AstroPT Visual Inspector: Images & Spectra.

This script generates a comprehensive dashboard for qualitative analysis of the model.
It reconstructs images from the token sequences (unpatchify + antispiral) and 
displays them alongside their spectra, including spectral line identification.

Features:
- RGB Reconstruction from Euclid bands (H/J/VIS).
- Spectral analysis with Redshift (z) correction.
- Annotation of common emission/absorption lines (H-alpha, OIII, etc.).
- Split-view spectral plotting (Blue/Red sides) with optional zooming.

Layout:
- Row 1: Real Image (RGB) | Reconstructed Image (RGB) | Residuals (Heatmap)
- Row 2: Spectrum Part 1 (Blue/Left side of range)
- Row 3: Spectrum Part 2 (Red/Right side of range)

Author: Victor Alonso Rodriguez
Date: January 2026
"""

import argparse
import logging
import os
import sys
from typing import Optional, Any, Dict, Tuple, List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow
from astropt.model_utils import load_local_model

# --- Logging Configuration ---
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-Inspect")

# --- Matplotlib / LaTeX Configuration ---
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
    'lines.linewidth': 1.5,
    'font.size': 14,          
    'axes.labelsize': 18,     
    'axes.titlesize': 20,     
    'xtick.labelsize': 16,   
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titlesize': 24,
    'figure.titleweight': 'bold',
})

# --- Constants ---
# Common emission and absorption lines in Angstroms (Rest Frame)
# Source: Common lines in Galaxy Spectra (DESI/Euclid context)
MAIN_LINES = {
    r"Ly$\alpha$": 1216.0,
    r"C IV": 1549.0,
    r"Mg II": 2798.0,
    r"[O II]": 3727.0,
    r"Ca K": 3933.7,
    r"Ca H": 3968.5,
    r"H$\delta$": 4102.0,
    r"H$\gamma$": 4341.0,
    r"H$\beta$": 4861.0,
    r"[O III]": 4959.0,
    r"[O III]": 5007.0,
    r"Mg I": 5175.0,
    r"Na D": 5890.0,
    r"[N II]": 6548.0,
    r"H$\alpha$": 6563.0,
    r"[N II]": 6584.0,
    r"[S II]": 6717.0,
    r"[S II]": 6731.0
}

def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="AstroPT Visual Inspection Dashboard")
    
    # Paths
    parser.add_argument("--out_dir", type=str, required=True, help="Directory containing the checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory of Arrow data")
    parser.add_argument("--ckpt_name", type=str, default="ckpt_best.pt", help="Checkpoint filename")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory for plots")
    
    # Selection
    parser.add_argument("--target_ids", nargs="+", type=int, help="List of specific Target IDs to plot")
    parser.add_argument("--num_plot", type=int, default=10, help="Total number of plots to generate")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (train/test)")
    
    # Visualization options
    parser.add_argument("--wl_range", nargs=2, type=float, default=None, 
                        help="Wavelength range in Angstroms to zoom in (e.g., --wl_range 4000 7000)")

    return parser.parse_args()


def unpatchify_image(
    patch_sequence: np.ndarray, 
    mod_config: Optional[Any] = None
) -> np.ndarray:
    """
    Reconstructs a 2D image from a sequence of flattened patches.
    Assumes patches are in SPIRAL order (center-to-outskirts).

    Args:
        patch_sequence (np.ndarray): Flattened sequence of shape (Seq_Len, Patch_Dim).
        mod_config (Optional[Any]): ModalityConfig object containing 'input_size' and 'patch_size'.
                                    If None, defaults to AstroPT standard (4 channels, 16px patch).

    Returns:
        np.ndarray: Reconstructed image of shape (Channels, Height, Width).
    """
    
    # 1. Infer dimensions from sequence
    seq_len, patch_dim = patch_sequence.shape
    
    # 2. Determine Configuration (Dynamic or Fallback)
    if mod_config:
        config_patch_dim = mod_config.input_size
        patch_size = mod_config.patch_size
        # Calculate channels: input_size = patch_size^2 * channels
        channels = config_patch_dim // (patch_size ** 2)
        
        if config_patch_dim != patch_dim:
            logger.warning(f"Config input_size ({config_patch_dim}) mismatches sequence dim ({patch_dim}).")
    else:
        # Fallback
        logger.warning("Modality config not provided. Falling back to defaults (C=4, P=16).")
        channels = 4
        patch_size = int(np.sqrt(patch_dim // channels))

    # Grid side (e.g., sqrt(196) = 14)
    grid_side = int(np.sqrt(seq_len))
    
    # 3. Anti-Spiralise (Reorder to Raster)
    def get_spiral_indices(n: int) -> np.ndarray:
        layout = np.arange(n * n).reshape(n, n)
        spiral_indices = []
        while layout.size > 0:
            spiral_indices.append(layout[0])           # Top
            layout = layout[1:]
            if layout.size == 0: break
            spiral_indices.append(layout[:, -1])       # Right
            layout = layout[:, :-1]
            if layout.size == 0: break
            spiral_indices.append(layout[-1][::-1])    # Bottom
            layout = layout[:-1]
            if layout.size == 0: break
            spiral_indices.append(layout[:, 0][::-1])  # Left
            layout = layout[:, 1:]
        
        spiral_order = np.concatenate(spiral_indices)
        result = np.empty(n * n, dtype=int)
        result[spiral_order] = np.arange(n * n)
        return (n * n - 1) - result 
    
    # Apply reordering
    spiral_idx = get_spiral_indices(grid_side)
    raster_patches = patch_sequence[spiral_idx]
    
    # 4. Reshape and Rearrange
    h = w = grid_side
    p1 = p2 = patch_size
    c = channels
    
    # (N_patches, p1, p2, c)
    raster_patches_3d = raster_patches.reshape(h*w, p1, p2, c)
    # (h, w, p1, p2, c)
    grid = raster_patches_3d.reshape(h, w, p1, p2, c)
    # (c, h*p1, w*p2)
    image = np.einsum('hwpqc->chpwq', grid)
    image = image.reshape(c, h*p1, w*p2)
    
    return image


def make_rgb(image_tensor: np.ndarray) -> np.ndarray:
    """
    Creates a visualization-friendly RGB image from 4-channel Euclid data.
    Mapping: R=H, G=J, B=VIS.
    """
    vis = image_tensor[0]
    h = image_tensor[1]
    j = image_tensor[2]
    # y = image_tensor[3] # Y band is available but usually ignored for 3-channel RGB
    
    rgb = np.stack([h, j, vis], axis=-1)
    
    # Robust Normalization per channel
    for i in range(3):
        ch = rgb[:, :, i]
        vmin, vmax = np.percentile(ch, 1), np.percentile(ch, 99)
        denom = vmax - vmin + 1e-8
        ch = (ch - vmin) / denom
        rgb[:, :, i] = np.clip(ch, 0, 1)
        
    return rgb


def plot_spectral_lines(ax, min_wl: float, max_wl: float, z: float):
    """Adds vertical lines for common emission/absorption features at redshift z."""
    for name, rest_wave in MAIN_LINES.items():
        # Shift to observed frame
        obs_wave = rest_wave * (1 + z)
        
        if min_wl < obs_wave < max_wl:
            ax.axvline(obs_wave, color='teal', linestyle='--', alpha=0.5, lw=1)
            # Annotate near top
            ylim = ax.get_ylim()
            y_pos = ylim[1] * 0.90
            ax.text(obs_wave, y_pos, name, rotation=90, color='teal', 
                    va='top', ha='right', fontsize=10, alpha=0.8)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup Output
    save_dir = args.save_dir if args.save_dir else os.path.join(args.out_dir, "plots_inspection")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Load Model (Universal Loader)
    ckpt_path = os.path.join(args.out_dir, args.ckpt_name)
    try:
        model, config, registry, raw_config = load_local_model(ckpt_path, device)
        logger.info(f"Model loaded successfully from {ckpt_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # 2. Load Dataset
    logger.info(f"Loading {args.split} dataset...")
    transforms_dict = EuclidDESIDatasetArrow.data_transforms(
        norm_type_img=raw_config.get('img_norm_type', 'constant'),
        norm_const_img=raw_config.get('img_norm_const', 1.0),
        norm_type_spec=raw_config.get('spectra_norm_type', 'constant'),
        norm_const_spec=raw_config.get('spectra_norm_const', 1.0)
    )
    
    ds = EuclidDESIDatasetArrow(
        arrow_folder_root=args.data_dir,
        split=args.split,
        modality_registry=registry,
        spiral=True, 
        transform=transforms_dict
    )
    
    # 3. Selection Logic
    indices_to_plot = []
    
    if args.target_ids:
        logger.info(f"Searching for {len(args.target_ids)} specific IDs...")
        all_ids = np.array(ds.ds['targetid'])
        for tid in args.target_ids:
            matches = np.where(all_ids == tid)[0]
            if len(matches) > 0:
                indices_to_plot.append(matches[0])
            else:
                logger.warning(f"TargetID {tid} not found.")
    
    num_needed = args.num_plot - len(indices_to_plot)
    if num_needed > 0:
        logger.info(f"Selecting {num_needed} random samples...")
        available_indices = list(set(range(len(ds))) - set(indices_to_plot))
        if len(available_indices) >= num_needed:
            random_indices = np.random.choice(available_indices, num_needed, replace=False)
            indices_to_plot.extend(random_indices)
        else:
            indices_to_plot.extend(available_indices)
            
    logger.info(f"Total samples to plot: {len(indices_to_plot)}")
    
    # 4. Main Loop
    subset = Subset(ds, indices_to_plot)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=2)
    
    img_config = None
    try:
        img_config = registry.get_config("images")
    except KeyError:
        logger.warning("Images modality not found in registry.")

    for batch_idx, batch in enumerate(loader):
        logger.info(f"Processing plot {batch_idx+1}/{len(indices_to_plot)}...")
        
        # Prepare Data
        X_raw = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                X_raw[k] = v.to(device)
            else:
                X_raw[k] = v 
        
        # Get Redshift for this target (Assuming 'z' or 'Z' is in metadata or Arrow columns)
        # Note: EuclidDESIDatasetArrow returns 'targetid' in batch, but Z might need retrieval
        # We try to get it from the original dataset using the idx
        current_idx = batch['idx'].item()
        try:
            # Direct access to arrow record
            arrow_record = ds.ds[current_idx]
            z_val = float(arrow_record.get('z', arrow_record.get('Z', 0.0)))
        except Exception:
            z_val = 0.0
            
        # Inference
        processed = EuclidDESIDatasetArrow.process_modes(X_raw, registry, device)
        X = processed['X']
        Y = processed['Y']
        
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs, _ = model(X)
                
        target_id = X_raw['targetid'].item()
        
        # --- DATA PROCESSING ---
        
        # Images
        has_image = False
        rgb_gt = rgb_pred = res_map = None
        if 'images' in Y and 'images' in outputs:
            gt_seq = Y['images'][0].float().cpu().numpy()
            pred_seq = outputs['images'][0].float().cpu().numpy()
            img_gt = unpatchify_image(gt_seq, mod_config=img_config)
            img_pred = unpatchify_image(pred_seq, mod_config=img_config)
            
            rgb_gt = make_rgb(img_gt)
            rgb_pred = make_rgb(img_pred)
            res_map = np.mean(img_gt - img_pred, axis=0)
            has_image = True
        
        # Spectra
        has_spectra = False
        spec_gt = spec_pred = wave_ang = None
        if 'spectra' in Y and 'spectra' in outputs:
            spec_gt = Y['spectra'][0].cpu().float().numpy().flatten()
            spec_pred = outputs['spectra'][0].cpu().float().numpy().flatten()
            
            # Reconstruct Wavelengths (Angstroms)
            # Model uses normalized positions. Data loader logic:
            # norm = (wave - 3000) / 7000.  => wave = norm * 7000 + 3000
            # Retrieve positions from Input (X) or Target (Y)
            if 'spectra_positions' in Y:
                pos_seq = Y['spectra_positions'][0].cpu().float().numpy().flatten()
            else:
                # Fallback: linear range 0-1
                pos_seq = np.linspace(0, 1, len(spec_gt))
                
            wave_ang = pos_seq * 7000.0 + 3000.0
            has_spectra = True
            
        # --- PLOTTING ---
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 3, height_ratios=[1, 0.8, 0.8])
        
        fig.suptitle(r"\textbf{AstroPT Reconstruction Analysis}" + f"\nTarget ID: {target_id} | z = {z_val:.3f}", 
                     fontsize=20, y=0.96)
        
        # ROW 1: IMAGES
        if has_image:
            ax_img1 = fig.add_subplot(gs[0, 0])
            ax_img1.imshow(rgb_gt, origin='lower')
            ax_img1.set_title(r"\textbf{Real (RGB)}")
            ax_img1.axis('off')
            
            ax_img2 = fig.add_subplot(gs[0, 1])
            ax_img2.imshow(rgb_pred, origin='lower')
            ax_img2.set_title(r"\textbf{Reconstructed}")
            ax_img2.axis('off')
            
            ax_img3 = fig.add_subplot(gs[0, 2])
            limit = np.max(np.abs(res_map)) * 0.8
            im = ax_img3.imshow(res_map, origin='lower', cmap='seismic', vmin=-limit, vmax=limit)
            ax_img3.set_title(r"\textbf{Residuals}")
            ax_img3.axis('off')
            plt.colorbar(im, ax=ax_img3, fraction=0.046, pad=0.04)
        else:
            ax_msg = fig.add_subplot(gs[0, :])
            ax_msg.text(0.5, 0.5, "No Image Data", ha='center')
            ax_msg.axis('off')

        # ROWS 2 & 3: SPECTRA
        if has_spectra:
            # Define Wavelength Range
            if args.wl_range:
                min_w, max_w = args.wl_range
            else:
                min_w, max_w = wave_ang.min(), wave_ang.max()
                
            mid_w = (min_w + max_w) / 2
            
            # --- Row 2: Blue Side ---
            ax_spec1 = fig.add_subplot(gs[1, :])
            ax_spec1.plot(wave_ang, spec_gt, color='black', alpha=0.5, lw=1, label='Real')
            ax_spec1.plot(wave_ang, spec_pred, color='red', alpha=0.8, lw=1.2, label='AstroPT')
            
            ax_spec1.set_xlim(min_w, mid_w)
            ax_spec1.set_title(r"\textbf{Spectrum (Blue Side)}")
            ax_spec1.set_ylabel(r"Flux ($\sigma$-norm)")
            
            plot_spectral_lines(ax_spec1, min_w, mid_w, z_val)
            ax_spec1.legend(loc='upper right')
            
            # --- Row 3: Red Side ---
            ax_spec2 = fig.add_subplot(gs[2, :])
            ax_spec2.plot(wave_ang, spec_gt, color='black', alpha=0.5, lw=1)
            ax_spec2.plot(wave_ang, spec_pred, color='red', alpha=0.8, lw=1.2)
            
            ax_spec2.set_xlim(mid_w, max_w)
            ax_spec2.set_title(r"\textbf{Spectrum (Red Side)}")
            ax_spec2.set_ylabel(r"Flux ($\sigma$-norm)")
            ax_spec2.set_xlabel(r"Wavelength [\AA]")
            
            plot_spectral_lines(ax_spec2, mid_w, max_w, z_val)
            
        else:
            ax_msg2 = fig.add_subplot(gs[1:, :])
            ax_msg2.text(0.5, 0.5, "No Spectra Data", ha='center')
            ax_msg2.axis('off')
            
        # Save
        plt.tight_layout(rect=[0, 0.03, 1, 0.93], h_pad=2.0)
        suffix = f"_zoom_{int(min_w)}-{int(max_w)}" if args.wl_range else ""
        save_path = os.path.join(save_dir, f"inspect_{target_id}{suffix}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {save_path}")

    logger.info("Inspection process completed.")

if __name__ == "__main__":
    main()