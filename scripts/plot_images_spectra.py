"""
AstroPT Visual Inspector.

This script generates a comprehensive dashboard for qualitative analysis of the AstroPT model.
It handles the reconstruction of images from flattened patch sequences, correcting for
autoregressive shifts and spiral ordering.

Features:
- Robust RGB Reconstruction (VIS/H/J mapping) with Asinh Scaling.
- Support for multiple normalization schemes (Linear, Asinh).
- Correct handling of Spiral -> Raster unpatchification.
- Autoregressive sequence reconstruction.
- Spectral analysis with Redshift correction.

Author: Victor Alonso Rodriguez
Date: January 2026
"""

import argparse
import logging
import os
import sys
from typing import Optional, Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow
from astropt.model_utils import load_local_model

# Logger Configuration
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-Inspect")

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

# Spectral Lines
MAIN_LINES = {
    r"Ly$\alpha$": 1216.0, r"C IV": 1549.0, "C III": 1908.7, r"Mg II": 2798.0, r"[O II]": 3727.3, r"[Ne III]": 3868.7,
    r"Ca K": 3933.7, r"Ca H": 3968.5, r"H$\delta$": 4102.0, r"H$\gamma$": 4341.0,
    r"H$\beta$": 4861.0, r"[O III]": 4959.0, r"[O III]": 5007.0, r"Mg I": 5175.0,
    r"Na D": 5890.0, r"[N II]": 6548.0, r"H$\alpha$": 6563.0, r"[N II]": 6583.5, r"[S II]": 6730.8
}


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="AstroPT Visual Inspector")
    parser.add_argument("--out_dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--data_dir", type=str, required=True, help="Arrow data root directory")
    parser.add_argument("--ckpt_name", type=str, default="ckpt_best.pt", help="Checkpoint filename")
    parser.add_argument("--target_ids", nargs="+", type=int, help="Specific Target IDs to plot")
    parser.add_argument("--num_plot", type=int, default=10, help="Number of random plots")
    parser.add_argument("--split", type=str, default="test", help="Data split (train/test)")
    parser.add_argument("--wl_range", nargs=2, type=float, default=None, help="Zoom wavelength (min max)")
    parser.add_argument("--run_name", type=str, default=None, help="Custom title for the plot (defaults to folder name)")
    return parser.parse_args()


def get_spiral_indices(side_len: int) -> np.ndarray:
    """Generates indices to map Raster -> Spiral (and inverse logic)."""
    layout = np.arange(side_len * side_len).reshape(side_len, side_len)
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
    
    # Invert mapping: Spiral -> Raster
    flat_indices = np.empty(side_len * side_len, dtype=int)
    flat_indices[spiral_order] = np.arange(side_len * side_len)
    final_indices = (side_len * side_len - 1) - flat_indices
    
    return final_indices


def reconstruct_image_from_patches(
    patch_sequence: np.ndarray, 
    mod_config: Optional[Any] = None
) -> np.ndarray:
    """Reconstructs (C, H, W) image from flattened patches."""
    seq_len, patch_dim = patch_sequence.shape
    
    if mod_config:
        p_size = mod_config.patch_size
        channels = mod_config.input_size // (p_size ** 2)
    else:
        channels = 4
        p_size = int(np.sqrt(patch_dim // channels))
        
    grid_side = int(np.sqrt(seq_len))
    
    if grid_side * grid_side != seq_len:
        logger.error(f"Cannot reconstruct: Sequence length {seq_len} is not a perfect square.")
        return np.zeros((channels, grid_side*p_size, grid_side*p_size))

    # Anti-Spiral
    spiral_indices = get_spiral_indices(grid_side)
    raster_patches = patch_sequence[spiral_indices]
    
    # Un-Patchify
    grid = raster_patches.reshape(grid_side, grid_side, p_size, p_size, channels)
    grid = grid.transpose(4, 0, 2, 1, 3) # (C, Grid_H, P_H, Grid_W, P_W)
    image = grid.reshape(channels, grid_side * p_size, grid_side * p_size)
    
    return image


def make_rgb_lupton(image_tensor: np.ndarray, Q: float = 10.0, stretch: float = 0.5, m: float = 0.0) -> np.ndarray:
    """
    Lupton et al. (2004) algorithm implementation.
    Reference: 'Preparing Red-Green-Blue Images from CCD Data'.
    
    Args:
        image_tensor: (C, H, W) array. Assumes values are roughly scaled to 0-1 range.
        Q: Softening parameter (controls linear-to-log transition).
        stretch: Linear scale factor.
        m: Minimum value to subtract (background).
    """
    # Compute Intensity (I)
    I = np.mean(image_tensor, axis=0)
    I = I - m
    I = np.maximum(I, 1e-10) # Avoid division by zero
    
    # Apply Lupton transfer function (Asinh)
    f_I = np.arcsinh(Q * stretch * I) / Q
    
    # Preserve Color Ratios
    scale_factor = f_I / I
    rgb_out = image_tensor * scale_factor[np.newaxis, :, :]
    
    # Final Normalization
    max_rgb = np.percentile(rgb_out, 99.5)
    if max_rgb > 0:
        rgb_out = rgb_out / max_rgb
        
    rgb_out = np.clip(rgb_out, 0, 1)
    
    return rgb_out.transpose(1, 2, 0)


def denormalize(data: np.ndarray, method: str, const: float) -> np.ndarray:
    """
    Reverts normalization to return Physical Units.
    
    Args:
        data: Normalized data from model.
        method: 'constant', 'asinh', or 'z_score'.
        const: Normalization constant (or softening parameter 'a').
    """
    if method == "asinh":
        # x_norm = asinh(x_phys / a)  -> x_phys = a * sinh(x_norm)
        return const * np.sinh(data)
    
    elif method == "constant":
        # x_norm = x_phys / const     -> x_phys = x_norm * const
        return data * const
    
    elif method == "z_score":
        # Cannot reverse Z-score without mean/std
        return data
    
    return data


def plot_spectral_lines(ax, min_wl, max_wl, z):
    """Annotates spectral lines with alternating heights."""
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


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract suffix from checkpoint for naming files
    ckpt_filename = os.path.basename(args.ckpt_name)
    name_no_ext = os.path.splitext(ckpt_filename)[0]
    
    if '_' in name_no_ext:
        run_suffix = f"_{name_no_ext.split('_')[-1]}" 
    else:
        run_suffix = f"_{name_no_ext}"
        
    logger.info(f"Run Suffix identified: {run_suffix}")
    
    # Extract training run name for plot titles
    train_name = args.run_name if args.run_name else os.path.basename(os.path.normpath(args.out_dir))
    
    # Load Model
    ckpt_path = os.path.join(args.out_dir, args.ckpt_name)
    try:
        model, config, registry, raw_config = load_local_model(ckpt_path, device)
        logger.info(f"Loaded model from {ckpt_path}")
    except Exception as e:
        logger.critical(f"Model load failed: {e}")
        sys.exit(1)
        
    img_config = registry.get_config("images")
    
    # Retrieve Normalization Constants
    norm_type_img = raw_config.get('img_norm_type', 'constant')
    norm_const_img = raw_config.get('img_norm_const', 1.0)
    norm_type_spec = raw_config.get('spectra_norm_type', 'constant')
    norm_const_spec = raw_config.get('spectra_norm_const', 1.0)
    
    transforms_dict = EuclidDESIDatasetArrow.data_transforms(
        norm_type_img=norm_type_img, norm_const_img=norm_const_img,
        norm_type_spec=norm_type_spec, norm_const_spec=norm_const_spec
    )
    
    ds = EuclidDESIDatasetArrow(
        arrow_folder_root=args.data_dir, split=args.split,
        modality_registry=registry, spiral=True, transform=transforms_dict
    )
    
    # Sample Selection
    indices_to_plot = []
    if args.target_ids:
        all_ids = ds.ds['targetid']
        id_map = {int(tid): idx for idx, tid in enumerate(all_ids)}
        for tid in args.target_ids:
            if tid in id_map: indices_to_plot.append(id_map[tid])
    
    if len(indices_to_plot) < args.num_plot:
        pool = list(set(range(len(ds))) - set(indices_to_plot))
        needed = args.num_plot - len(indices_to_plot)
        indices_to_plot.extend(np.random.choice(pool, min(needed, len(pool)), replace=False))
        
    subset = Subset(ds, indices_to_plot)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=2)
    
    # Plotting Loop
    for batch_idx, batch in enumerate(loader):
        logger.info(f"Plotting {batch_idx+1}/{len(indices_to_plot)}...")
        
        processed = EuclidDESIDatasetArrow.process_modes(batch, registry, device)
        X = processed['X']
        
        # Inference
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs, _ = model(X)
        
        # Raw data
        try:
            arrow_idx = int(batch['idx'].item())
        except:
            arrow_idx = int(batch['idx'][0])

        raw_record = ds.ds[arrow_idx]
        target_id = raw_record['targetid']
        z_val = float(raw_record.get('redshift', 0.0))

        # IMAGES
        has_image = False
        rgb_gt = rgb_pred = res_map = None
        
        if 'images' in outputs:
            def get_raw(k): 
                val = raw_record[k]
                return np.array(val if val is not None else [], dtype=np.float32)

            vis, y, j, h = get_raw('image_vis'), get_raw('image_nisp_y'), get_raw('image_nisp_j'), get_raw('image_nisp_h')
            
            if vis.size > 0:
                raw_stack = np.stack([vis, h, j, y], axis=0)
                
                start_token = X['images'][:, 0:1, :] 
                pred_tokens = outputs['images'] 
                
                full_seq = torch.cat([start_token, pred_tokens], dim=1).float().cpu().numpy()[0]
                
                img_pred_model = reconstruct_image_from_patches(full_seq, img_config)
                
                # Denormalize
                img_pred_phys = denormalize(img_pred_model, norm_type_img, norm_const_img)
                
                # Channel Weights [R, G, B]
                RGB_WEIGHTS = [1.2, 1.3, 1.0]
                
                # Pre process
                bg_val = np.percentile(raw_stack, 50, axis=(1,2), keepdims=True)
                
                raw_bg = raw_stack - bg_val
                pred_bg = img_pred_phys - bg_val
                
                # Image Information
                print(f"[ID {target_id}] Channel Stats (Max Flux):")
                bands = ['VIS (Blue)', 'H (Red)', 'J (Green)', 'Y (Green)']
                for c_idx, name in enumerate(bands):
                    v_max = np.percentile(raw_bg[c_idx], 99.5)
                    print(f"  {name}: {v_max:.4e}")

                # Normalize per channel
                raw_rgb_stack = []
                pred_rgb_stack = []
                
                for c in range(4):
                    
                    # Compute percetile 99
                    v_max = np.percentile(np.abs(raw_bg[c]), 99.5)
                    if v_max <= 0: v_max = 1.0
                    
                    # Clip to zero values
                    r_ch = np.clip(raw_bg[c] / v_max, 0, 100) 
                    p_ch = np.clip(pred_bg[c] / v_max, 0, 100)
                    
                    raw_rgb_stack.append(r_ch)
                    pred_rgb_stack.append(p_ch)
                
                # Stack
                raw_norm = np.stack(raw_rgb_stack)
                pred_norm = np.stack(pred_rgb_stack)

                # Compute channels
                def stack_to_rgb_weighted(stack):
                    vis, h, j, y = stack[0], stack[1], stack[2], stack[3]
                    
                    r = h * RGB_WEIGHTS[0]
                    g = ((j + y) / 2.0) * RGB_WEIGHTS[1]
                    b = vis * RGB_WEIGHTS[2]
                    
                    return np.stack([r, g, b], axis=0) # (3, H, W)

                rgb_input_gt = stack_to_rgb_weighted(raw_norm)
                rgb_input_pred = stack_to_rgb_weighted(pred_norm)

                # Lupton function
                rgb_gt = make_rgb_lupton(rgb_input_gt, Q=12.0, stretch=0.5)
                rgb_pred = make_rgb_lupton(rgb_input_pred, Q=12.0, stretch=0.5)
                
                # Residuals
                if raw_stack.shape == img_pred_phys.shape:
                    diff = raw_stack - img_pred_phys
                    res_map = np.mean(diff, axis=0)
                else:
                    res_map = np.zeros_like(rgb_gt[:,:,0])
                
                has_image = True

        # SPECTRA
        has_spectra = False
        spec_gt = spec_pred = wave_ang = None
        
        if 'spectra' in outputs and raw_record['spectrum_flux'] is not None:
            spec_gt = np.array(raw_record['spectrum_flux']).flatten()
            
            start_s = X['spectra'][:, 0:1, :]
            pred_s = outputs['spectra']
            full_s = torch.cat([start_s, pred_s], dim=1).float().cpu().numpy().flatten()
            
            spec_pred = denormalize(full_s, norm_type_spec, norm_const_spec)
            wave_ang = np.array(raw_record['spectrum_wave']).flatten()
            
            min_len = min(len(spec_gt), len(spec_pred), len(wave_ang))
            spec_gt = spec_gt[:min_len]
            spec_pred = spec_pred[:min_len]
            wave_ang = wave_ang[:min_len]
            
            has_spectra = True

        # PLOTTING
        if not has_image and not has_spectra: continue

        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], wspace=0.1, hspace=0.3)
        fig.suptitle(rf"\textbf{{AstroPT Reconstruction | ID: {target_id} | z={z_val:.3f}}}"
                     + f"\n[{train_name}]", fontsize=22, y=0.96)
        
        if has_image:
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(rgb_gt, origin='lower')
            ax1.set_title(r"\textbf{Real (Log Scale)}")
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(rgb_pred, origin='lower')
            ax2.set_title(r"\textbf{Reconstructed (Log Scale)}")
            ax2.axis('off')
            
            ax3 = fig.add_subplot(gs[0, 2])
            vlim = np.percentile(np.abs(res_map), 98) if res_map is not None else 1
            im = ax3.imshow(res_map, origin='lower', cmap='seismic', vmin=-vlim, vmax=vlim)
            ax3.set_title(r"\textbf{Residuals (Physical)}")
            ax3.axis('off')
            plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="Flux Diff")

        if has_spectra:
            if args.wl_range: w_min, w_max = args.wl_range
            else: w_min, w_max = wave_ang.min(), wave_ang.max()
            w_mid = (w_min + w_max) / 2
            
            for i, (start, end, title, loc) in enumerate([
                (w_min, w_mid, "Blue Channel", gs[1, :]), 
                (w_mid, w_max, "Red Channel", gs[2, :])
            ]):
                ax = fig.add_subplot(loc)
                ax.plot(wave_ang, spec_gt, 'k-', lw=1, alpha=0.6, label='Real')
                ax.plot(wave_ang, spec_pred, 'r-', lw=1.5, alpha=0.8, label='AstroPT')
                ax.set_xlim(start, end)
                ax.set_title(rf"\textbf{{Spectrum ({title})}}")
                ax.set_ylabel(r"Flux")
                if i==1: ax.set_xlabel(r"Wavelength [\AA]")
                if i==0: ax.legend(loc='lower left')
                plot_spectral_lines(ax, start, end, z_val)


        # Saving
        zoom_suffix = f"_zoom" if args.wl_range else ""
        
        # Final name
        filename = f"ID_{target_id}{zoom_suffix}{run_suffix}.png"
        save_path = os.path.join(args.out_dir, filename)
        
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"-> Saved plot: {save_path}\n")

    logger.info("Done.")

if __name__ == "__main__":
    main()