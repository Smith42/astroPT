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
Date: March 2026
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Optional, Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow
from astropt.model_utils import load_local_model

# Workspace Paths for AION and FMB Dependencies
WORKSPACE_ROOT = "/home/valonso/iac18_mhuertas_shared/valonso"
sys.path.append(os.path.join(WORKSPACE_ROOT, "AION"))
sys.path.append(os.path.join(WORKSPACE_ROOT, "foundation-models-benchmark/src"))

from astropt.aion_tokeniser import MultiprocessCodecManager
from aion.modalities import DESISpectrum
from astropt.resnet_adapter import EuclidImage

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
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing training weights")
    parser.add_argument("--data_dir", type=str, required=True, help="Arrow data root directory")
    parser.add_argument("--save_dir", type=str, default=None, help="Plot Saving Directory (Defaults to weights_dir/../plots)")
    parser.add_argument("--ckpt_name", type=str, default="ckpt_best.pt", help="Checkpoint filename")
    parser.add_argument("--target_ids", nargs="+", type=int, help="Specific Target IDs to plot")
    parser.add_argument("--num_plot", type=int, default=10, help="Number of random plots")
    parser.add_argument("--split", type=str, default="test", help="Data split (train/test)")
    parser.add_argument("--wl_range", nargs=2, type=float, default=None, help="Zoom wavelength (min max)")
    parser.add_argument("--train_name", type=str, default=None, help="Custom title for the plot (defaults to folder name)")
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
    mod_config: Optional[Any] = None,
    apply_antispiral: bool = True,
) -> np.ndarray:
    """Reconstructs (C, H, W) image from flattened patches."""
    seq_len, patch_dim = patch_sequence.shape
    
    if mod_config:
        p_size = mod_config.patch_size
        channels = mod_config.input_size // (p_size ** 2)
    else:
        channels = 4
        p_size = int(np.sqrt(patch_dim // channels))
        
    if grid_side * grid_side != seq_len:
        logger.error(f"Cannot reconstruct: Sequence length {seq_len} is not a perfect square.")
        return np.zeros((channels, grid_side*p_size, grid_side*p_size))

    # Handle Discrete Tokens (AION)
    # If the patch_dim is the vocab_size (e.g. 64000), we cannot reshape into pixels.
    if patch_dim > 1024: # Heuristic for discrete vocab
        logger.warning(f"Detected discrete tokens (dim={patch_dim}). Reshaping as a token map instead of pixels.")
        # We return a 1-channel "image" where each pixel is the token ID
        token_map = patch_sequence.reshape(grid_side, grid_side)
        return token_map[np.newaxis, :, :] # (1, H, W)

    # Anti-Spiral (only if data was generated in spiral order)
    if apply_antispiral:
        spiral_indices = get_spiral_indices(grid_side)
        raster_patches = patch_sequence[spiral_indices]
    else:
        raster_patches = patch_sequence
    
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


def denormalize(data: np.ndarray, method: str, scaler: float, const: float) -> np.ndarray:
    """
    Reverts normalization to return Physical Units.
    
    Args:
        data: Normalized data from model.
        method: 'constant', 'asinh', or 'z_score'.
        const: Normalization constant (or softening parameter 'a').
    """
    if method == "asinh":
        # x_norm = asinh(x_phys / a) / C  -> x_phys = a * sinh(data * C)
        return scaler * np.sinh(data * const)
    
    elif method == "constant":
        # x_norm = x_phys / const     -> x_phys = x_norm * const
        return data * const
    
    elif method == "z_score":
        # Cannot reverse Z-score without mean/std
        return data
    
    return data


def rebuild_full_sequence_from_teacher_forcing(
    input_tokens: torch.Tensor,
    pred_tokens: torch.Tensor,
    mode_name: str,
) -> torch.Tensor:
    """
    Rebuilds a full-length predicted sequence from model outputs under process_modes().

    process_modes + model forward can produce three valid length relations:
    - pred_len == input_len + 1: prediction is already full sequence (no prepend)
    - pred_len == input_len    : prepend first input token
    """
    # Ensure input and pred have same ndim (e.g. if one is (B,L) and other (B,L,1))
    if input_tokens.ndim < pred_tokens.ndim:
        input_tokens = input_tokens.unsqueeze(-1)
    elif pred_tokens.ndim < input_tokens.ndim:
        pred_tokens = pred_tokens.unsqueeze(-1)

    input_len = int(input_tokens.shape[1])
    pred_len = int(pred_tokens.shape[1])

    if pred_len == input_len + 1:
        return pred_tokens

    if pred_len >= input_len - 1:
        # Prepend first token if necessary (for causal shifted predictions)
        diff = input_len - pred_len
        if diff > 0:
            start_token = input_tokens[:, 0:diff, ...]
            return torch.cat([start_token, pred_tokens], dim=1)
        else:
            return pred_tokens[:, :input_len, ...]

    # Fallback for Token Mixing (Interleaved Sparse Outputs)
    logger.warning(f"Unexpected lengths: {mode_name} input={input_len}, pred={pred_len}. Padding with zeros.")
    
    # Preserve dimensionality (2D for tokens, 3D for continuous patches)
    full_shape = list(pred_tokens.shape)
    full_shape[1] = input_len
    padded = torch.zeros(tuple(full_shape), device=pred_tokens.device, dtype=pred_tokens.dtype)
    
    actual_len = min(input_len, pred_len)
    padded[:, :actual_len, ...] = pred_tokens[:, :actual_len, ...]
    return padded


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


def plot_dashboard(
    target_id: int,
    z_val: float,
    train_name: str,
    rgb_gt: Optional[np.ndarray],
    rgb_pred: Optional[np.ndarray],
    res_map: Optional[np.ndarray],
    has_image: bool,
    wave_ang: Optional[np.ndarray],
    spec_gt: Optional[np.ndarray],
    spec_pred: Optional[np.ndarray],
    has_spectra: bool,
    wl_range: Optional[tuple],
    save_dir: str | Path,
    filename: str
):
    """Encapsulates the dashboard visualization logic."""
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], wspace=0.1, hspace=0.3)
    fig.suptitle(rf"\textbf{{AstroPT Reconstruction | ID: {target_id} | z={z_val:.3f}}}"
                 + f"\n[{train_name}]", fontsize=22, y=0.96)
    
    if has_image and rgb_gt is not None and rgb_pred is not None:
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
        if vlim <= 0: vlim = 1.0
        im = ax3.imshow(res_map, origin='lower', cmap='seismic', vmin=-vlim, vmax=vlim) # type: ignore
        ax3.set_title(r"\textbf{Residuals (Physical)}")
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="Flux Diff")

    if has_spectra and wave_ang is not None and spec_gt is not None and spec_pred is not None:
        if wl_range: w_min, w_max = wl_range
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

    # Save logic
    save_path = Path(save_dir) / filename
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f" --> Saved dashboard: {save_path}\n")


def get_modality_keys(registry_or_inputs):
    """Helper to detect modality names dynamically (handles 'images' vs 'aion_images')."""
    if hasattr(registry_or_inputs, 'modalities'):
        keys = registry_or_inputs.modalities.keys()
    else:
        keys = registry_or_inputs.keys()
    img_key = next((k for k in keys if "image" in k and not k.endswith("_positions")), "images")
    spec_key = next((k for k in keys if "spec" in k and not k.endswith("_positions")), "spectra")
    return img_key, spec_key


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Required paths
    weights_dir = Path(args.weights_dir)
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = weights_dir.parent / "plots" / "images_spectra_reconstructions"
        
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Reconstructions will be saved to: {save_dir}")
    
    logger.info("Analysis Directories:")
    logger.info(f" --> [Weights]:   {weights_dir}")
    logger.info(f" --> [Saving]:    {save_dir}")
    
    # Extract suffix from checkpoint for naming files
    ckpt_filename = weights_dir / args.ckpt_name
    name_no_ext = ckpt_filename.stem
    
    if '_' in name_no_ext:
        run_suffix = f"_{name_no_ext.split('_')[-1]}" 
    else:
        run_suffix = f"_{name_no_ext}"
        
    logger.info(f"Run Suffix identified: {run_suffix}")
    
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
        logger.error(f"Checkpoint not found: {ckpt_path}. Starting smart search.")

        all_ckpts = list(weights_dir.glob("*.pt"))
        if not all_ckpts:
            logger.error(f"FATAL: No .pt file found in {weights_dir}")
            sys.exit(1)

        best_matches = [c for c in all_ckpts if "best" in c.name]
        last_matches = [c for c in all_ckpts if "last" in c.name]
        
        if best_matches:
            ckpt_path = sorted(best_matches, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            logger.info(f"Selected by priority [BEST]: {ckpt_path.name}")
            
        elif last_matches:
            ckpt_path = sorted(last_matches, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            logger.info(f"Selected by priority [LAST]: {ckpt_path.name}")
        
        else:
            ckpt_path = sorted(all_ckpts, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            logger.info(f"Selected by date [RECENT]: {ckpt_path.name}")

    logger.info(f"Loading checkpoint: {ckpt_path}")
    
    
    # Load Model
    try:
        model, config, registry, raw_config_dict = load_local_model(ckpt_path, device)
    except Exception as e:
        logger.critical(f"Model load failed: {e}")
        sys.exit(1)
        
    # Modality detection (from registry)
    img_key, spec_key = get_modality_keys(registry)
    img_config = registry.get_config(img_key)
    
    # Arrow data directory
    data_dir = args.data_dir if args.data_dir else raw_config_dict.get('data_dir')
    data_dir = Path(data_dir)
    assert data_dir is not None, "data_dir cannot be None. Check your config.json or --data_dir argument."
    
    # Retrieve Normalization Constants
    norm_type_img = raw_config_dict.get(f'{img_key}_norm_type', raw_config_dict.get('img_norm_type', 'asinh'))
    norm_scaler_img=raw_config_dict.get(f'{img_key}_norm_scaler',raw_config_dict.get('img_norm_scaler',1.0))
    norm_const_img = raw_config_dict.get(f'{img_key}_norm_const',raw_config_dict.get('img_norm_const',1.0))
    
    inverse_spec = raw_config_dict.get(f'{spec_key}_inverse', False)
    norm_type_spec = raw_config_dict.get(f'{spec_key}_norm_type', 'constant')
    norm_scaler_spec = raw_config_dict.get(f'{spec_key}_norm_scaler',1.0)
    norm_const_spec = raw_config_dict.get(f'{spec_key}_norm_const', 1.0)
    
    # Aplying tranformations
    data_tf = EuclidDESIDatasetArrow.data_transforms(
        norm_type_img=norm_type_img,
        norm_scaler_img=norm_scaler_img,
        norm_const_img=norm_const_img,
        norm_type_spec=norm_type_spec,
        norm_scaler_spec=norm_scaler_spec,
        norm_const_spec=norm_const_spec,
    )
    
    apply_antispiral = bool(raw_config_dict.get('spiral', True))
    
    # Initialize AION Codec Manager for discrete reconstruction
    codec_manager = MultiprocessCodecManager(
        device=device,
        resnet_weights_path=raw_config_dict.get("images_resnet_weights_path"),
        aion_image_size=raw_config_dict.get("images_aion_image_size", 112),
        aion_image_transform=raw_config_dict.get("images_aion_image_transform", "resize"),
    )

    ds = EuclidDESIDatasetArrow(
        arrow_folder_root=data_dir,
        split=args.split,
        modality_registry=registry,
        spiral=apply_antispiral,
        stochastic=False,  
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
    subset = Subset(ds, indices_to_plot)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=2)
    
    # Modality detection
    has_cls = getattr(config, 'use_cls_token', False)
    logger.info(f"Detected Modalities: Img={img_key}, Spec={spec_key} | CLS={has_cls}")

    # Plotting Loop
    for batch_idx, batch in enumerate(loader):
        logger.info(f"Plotting {batch_idx+1}/{len(indices_to_plot)}...")
        
        processed = EuclidDESIDatasetArrow.process_modes(
            batch, registry, device, 
            use_token_mixing=config.use_token_mixing,
            use_cls_token=has_cls,
            cls_position=getattr(config, 'cls_position', 'last')
        )
        X = processed['X']
        
        # Inference
        with torch.no_grad():
            if device.type == 'cuda':
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16): # type: ignore
                    outputs, _ = model(X, targets=X.copy())
            else:
                model.to(torch.float32)
                X = {}
                for k, v in processed['X'].items():
                    if isinstance(v, torch.Tensor):
                        # Positions and discrete tokens must be Long
                        is_discrete_mod = any(dk in k for dk in ["aion_images", "aion_spectra"])
                        if k.endswith('_positions') or is_discrete_mod:
                            X[k] = v.to(torch.long)
                        else:
                            X[k] = v.to(torch.float32)
                    else:
                        X[k] = v
                outputs, _ = model(X, targets=X.copy())

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
        
        if img_key in outputs:
            def get_raw(k): 
                val = raw_record[k]
                return np.array(val if val is not None else [], dtype=np.float32)

            vis, y, j, h = get_raw('image_vis'), get_raw('image_nisp_y'), get_raw('image_nisp_j'), get_raw('image_nisp_h')
            
            if vis.size > 0:
                raw_stack = np.stack([vis, h, j, y], axis=0)
                
                pred_tokens = outputs[img_key]
                if pred_tokens.shape[-1] > 1024:
                    # Discrete case: take argmax
                    pred_tokens = pred_tokens.argmax(dim=-1, keepdim=True)

                full_seq_tensor = rebuild_full_sequence_from_teacher_forcing(
                    input_tokens=X[img_key],
                    pred_tokens=pred_tokens,
                    mode_name=img_key,
                )
                
                is_discrete = (full_seq_tensor.shape[-1] == 1)
                
                if is_discrete:
                    # Discrete AION Reconstruction using Decoder
                    tokens_dict = {EuclidImage.token_key: full_seq_tensor.squeeze(-1)} # (B, T)
                    try:
                        decoded_img = codec_manager.decode(tokens_dict, EuclidImage)
                        # flux is (B, C, H, W)
                        img_pred_phys = decoded_img.flux.float().detach().cpu().numpy()
                        if img_pred_phys.ndim == 4: img_pred_phys = img_pred_phys[0]
                        
                        # Match channel count with GT (Euclid 4 bands: VIS, Y, J, H)
                        # AION decoder may return all registered survey bands (e.g. 9 or 5)
                        if img_pred_phys.shape[0] != raw_stack.shape[0]:
                            logger.warning(f"Channel mismatch: AION={img_pred_phys.shape[0]}, GT={raw_stack.shape[0]}. Slicing/Padding.")
                            if img_pred_phys.shape[0] > raw_stack.shape[0]:
                                img_pred_phys = img_pred_phys[:raw_stack.shape[0]]
                            else:
                                pad = np.zeros((raw_stack.shape[0] - img_pred_phys.shape[0], *img_pred_phys.shape[1:]))
                                img_pred_phys = np.concatenate([img_pred_phys, pad], axis=0)
                        
                        # AION decoder output is now in Euclid linear physical units
                        logger.info(f"  [DIAG] Decoded img shape: {img_pred_phys.shape} | "
                                    f"range: [{img_pred_phys.min():.4e}, {img_pred_phys.max():.4e}] | "
                                    f"mean: {img_pred_phys.mean():.4e}")
                        logger.info(f"  [DIAG] GT raw_stack shape: {raw_stack.shape} | "
                                    f"range: [{raw_stack.min():.4e}, {raw_stack.max():.4e}] | "
                                    f"mean: {raw_stack.mean():.4e}")
                    except Exception as e:
                        logger.error(f"AION Image decoding failed: {e}")
                        import traceback; traceback.print_exc()
                        img_pred_phys = np.zeros((4, 112, 112)) # Fallback
                else:
                    # Continuous Regression Reconstruction
                    full_seq = full_seq_tensor.float().detach().cpu().numpy()[0]
                    img_pred_model = reconstruct_image_from_patches(
                        full_seq,
                        img_config,
                        apply_antispiral=apply_antispiral,
                    )
                    # Denormalize to Physical Units
                    img_pred_phys = denormalize(
                        img_pred_model, 
                        norm_type_img, 
                        norm_scaler_img,
                        norm_const_img
                    )
                
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
                if raw_stack.shape != img_pred_phys.shape:
                    try:
                        import torch.nn.functional as F
                        # Resize raw_stack to match img_pred_phys spatial dimensions
                        tmp_raw = torch.from_numpy(raw_stack).float().unsqueeze(0) # (1, C, H_raw, W_raw)
                        tmp_raw = F.interpolate(tmp_raw, size=img_pred_phys.shape[1:], mode='bilinear', align_corners=False)
                        raw_stack_resized = tmp_raw.squeeze(0).numpy()
                        diff = raw_stack_resized - img_pred_phys
                        res_map = np.mean(diff, axis=0)
                    except Exception as e:
                        logger.warning(f"Could not compute residuals due to shape mismatch and resize error: {e}")
                        res_map = np.zeros_like(rgb_gt[:,:,0])
                else:
                    diff = raw_stack - img_pred_phys
                    res_map = np.mean(diff, axis=0)
                
                has_image = True

        # SPECTRA
        has_spectra = False
        spec_gt = spec_pred = wave_ang = None
        
        if spec_key in outputs and raw_record['spectrum_flux'] is not None:
            spec_gt = np.array(raw_record['spectrum_flux']).flatten()
            
            pred_s = outputs[spec_key]
            if pred_s.shape[-1] > 1024:
                # Discrete case: take argmax
                pred_s = pred_s.argmax(dim=-1, keepdim=True)

            full_s_tensor = rebuild_full_sequence_from_teacher_forcing(
                input_tokens=X[spec_key],
                pred_tokens=pred_s,
                mode_name=spec_key,
            )

            is_discrete_spec = (full_s_tensor.shape[-1] == 1)

            if is_discrete_spec:
                # Discrete AION Reconstruction
                tokens_dict = {DESISpectrum.token_key: full_s_tensor.squeeze(-1)}
                try:
                    decoded_spec = codec_manager.decode(tokens_dict, DESISpectrum)
                    spec_pred = decoded_spec.flux.float().detach().cpu().numpy().flatten()
                    # AION decodes to normalized space, so we must denormalize
                    spec_pred = denormalize(spec_pred, norm_type_spec, norm_scaler_spec, norm_const_spec)
                except Exception as e:
                    logger.error(f"AION Spectrum decoding failed: {e}")
                    spec_pred = np.zeros_like(spec_gt)
            else:
                full_s = full_s_tensor.float().detach().cpu().numpy().flatten()
                spec_pred = denormalize(
                    full_s, 
                    norm_type_spec, 
                    norm_scaler_spec,
                    norm_const_spec
                )
            
            wave_ang = np.array(raw_record['spectrum_wave']).flatten()
            true_len = len(wave_ang)
            spec_pred = spec_pred[:true_len]
            
            if inverse_spec:
                spec_pred = spec_pred[::-1]
                
            min_len = min(len(spec_gt), len(spec_pred), len(wave_ang))
            spec_gt = spec_gt[:min_len]
            spec_pred = spec_pred[:min_len]
            wave_ang = wave_ang[:min_len]
            
            has_spectra = True

        # FILENAME AND SUFFIX LOGIC
        random_suffix = "_R" if int(target_id) not in specific_tids else ""
        zoom_suffix = "_zoom" if args.wl_range else ""
        filename = f"ID_{target_id}{run_suffix}{zoom_suffix}{random_suffix}.png"
        
        if not has_image and not has_spectra: continue

        plot_dashboard(
            target_id=target_id, z_val=z_val,
            train_name=train_name, rgb_gt=rgb_gt, rgb_pred=rgb_pred, res_map=res_map,
            has_image=has_image, wave_ang=wave_ang, spec_gt=spec_gt, spec_pred=spec_pred,
            has_spectra=has_spectra, wl_range=args.wl_range, save_dir=save_dir, filename=filename
        )

    logger.info("Done.")

if __name__ == "__main__":
    main()