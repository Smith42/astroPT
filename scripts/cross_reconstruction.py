"""
AstroPT Cross-Modal Generator (The "Translation" Test).

This script evaluates the model's ability to generate one modality given ONLY the other.
It performs "Blind Inference" by masking the target modality in the input, forcing
the model to rely on cross-modal attention to reconstruct the missing data.

Modes:
- img2spec: Input Image -> (Masked Spectrum) -> Output Predicted Spectrum.
- spec2img: Input Spectrum -> (Masked Image) -> Output Predicted Image.

Author: Victor Alonso Rodriguez
Date: January 2026
"""

import argparse
import logging
import os
import sys
from typing import Optional, Any, Dict, List

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
logger = logging.getLogger("AstroPT-CrossGen")

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


# --- Constants (Spectral Lines) ---
MAIN_LINES = {
    r"H$\alpha$": 6563.0,
    r"[O III]": 5007.0,
    r"Mg II": 2798.0,
    r"Ly$\alpha$": 1216.0
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AstroPT Cross-Modal Generation Test")
    
    parser.add_argument("--out_dir", type=str, required=True, help="Directory containing the checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory of Arrow data")
    parser.add_argument("--ckpt_name", type=str, default="ckpt_best.pt", help="Checkpoint filename")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory for plots")
    
    parser.add_argument("--mode", type=str, choices=['img2spec', 'spec2img'], required=True,
                        help="Direction of generation: Image->Spectrum or Spectrum->Image")
    parser.add_argument("--num_plot", type=int, default=5, help="Number of samples to test")
    parser.add_argument("--target_ids", nargs="+", type=int, help="Specific IDs to test")
    
    return parser.parse_args()


# --- Helper Functions (Reused from plot_images_spectra to maintain consistency) ---

def unpatchify_image(patch_sequence: np.ndarray, mod_config: Optional[Any] = None) -> np.ndarray:
    """Reconstructs 2D image from patches (Antispiral + Reshape)."""
    seq_len, patch_dim = patch_sequence.shape
    
    if mod_config:
        config_patch_dim = mod_config.input_size
        patch_size = mod_config.patch_size
        channels = config_patch_dim // (patch_size ** 2)
    else:
        channels = 4
        patch_size = int(np.sqrt(patch_dim // channels))

    grid_side = int(np.sqrt(seq_len))
    
    def get_spiral_indices(n):
        layout = np.arange(n * n).reshape(n, n)
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
        result = np.empty(n * n, dtype=int)
        result[spiral_order] = np.arange(n * n)
        return (n * n - 1) - result 
    
    spiral_idx = get_spiral_indices(grid_side)
    raster_patches = patch_sequence[spiral_idx]
    
    h = w = grid_side
    p1 = p2 = patch_size
    c = channels
    
    return raster_patches.reshape(h*w, p1, p2, c).reshape(h, w, p1, p2, c).transpose(0, 2, 1, 3, 4).reshape(h*p1, w*p2, c).transpose(2, 0, 1)

def make_rgb(image_tensor: np.ndarray) -> np.ndarray:
    """Creates RGB from (4, H, W) tensor using H, (J+Y)/2, VIS bands."""
    vis, h, j = image_tensor[0], image_tensor[1], image_tensor[2]
    y = image_tensor[3] if image_tensor.shape[0] > 3 else j
    
    green = (j + y) / 2.0
    rgb = np.stack([h, green, vis], axis=-1)
    
    for i in range(3):
        ch = rgb[:, :, i]
        vmin, vmax = np.percentile(ch, 1), np.percentile(ch, 99)
        rgb[:, :, i] = np.clip((ch - vmin) / (vmax - vmin + 1e-8), 0, 1)
    return rgb


def plot_cross_img2spec(fig, gs, rgb_in, spec_gt, spec_pred, wave_ang, z, target_id):
    """Layout for Image -> Spectrum."""
    # Input: Image
    ax_in = fig.add_subplot(gs[:, 0])
    ax_in.imshow(rgb_in, origin='lower')
    ax_in.set_title(r"\textbf{INPUT: Image (RGB)}", color='darkgreen')
    ax_in.axis('off')
    
    # Output: Spectrum
    ax_out = fig.add_subplot(gs[:, 1])
    ax_out.plot(wave_ang, spec_gt, color='black', alpha=0.4, lw=1, label='Real (Ground Truth)')
    ax_out.plot(wave_ang, spec_pred, color='crimson', alpha=0.9, lw=1.5, label='Predicted (from Image)')
    
    ax_out.set_title(rf"\textbf{{OUTPUT: Predicted Spectrum}} ($z={z:.3f}$)", color='darkblue')
    ax_out.set_xlabel(r"Wavelength [\AA]")
    ax_out.set_ylabel(r"Flux ($\sigma$-norm)")
    ax_out.legend(loc='upper right')
    
    # Lines
    for name, rest_w in MAIN_LINES.items():
        obs_w = rest_w * (1 + z)
        if wave_ang.min() < obs_w < wave_ang.max():
            ax_out.axvline(obs_w, color='teal', ls='--', alpha=0.5)
            ax_out.text(obs_w, ax_out.get_ylim()[1]*0.9, name, rotation=90, color='teal', fontsize=10)


def plot_cross_spec2img(fig, gs, spec_in, rgb_gt, rgb_pred, wave_ang, z, target_id):
    """Layout for Spectrum -> Image."""
    # Input: Spectrum
    ax_in = fig.add_subplot(gs[0, :])
    ax_in.plot(wave_ang, spec_in, color='black', lw=1)
    ax_in.set_title(rf"\textbf{{INPUT: Spectrum}} ($z={z:.3f}$)", color='darkgreen')
    ax_in.set_xlim(wave_ang.min(), wave_ang.max())
    ax_in.set_ylabel("Flux")
    
    # Output: Real vs Pred Image
    ax_gt = fig.add_subplot(gs[1, 0])
    ax_gt.imshow(rgb_gt, origin='lower')
    ax_gt.set_title(r"\textbf{Ground Truth Image}", color='gray')
    ax_gt.axis('off')
    
    ax_pred = fig.add_subplot(gs[1, 1])
    ax_pred.imshow(rgb_pred, origin='lower')
    ax_pred.set_title(r"\textbf{GENERATED Image (from Spec)}", color='darkblue')
    ax_pred.axis('off')


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    save_dir = args.save_dir if args.save_dir else os.path.join(args.out_dir, "plots_cross_gen")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Load Model
    ckpt_path = os.path.join(args.out_dir, args.ckpt_name)
    model, config, registry, raw_config = load_local_model(ckpt_path, device)
    logger.info(f"Model loaded. Testing Mode: {args.mode.upper()}")

    # 2. Dataset
    transforms_dict = EuclidDESIDatasetArrow.data_transforms(
        norm_type_img=raw_config.get('img_norm_type', 'constant'),
        norm_const_img=raw_config.get('img_norm_const', 1.0),
        norm_type_spec=raw_config.get('spectra_norm_type', 'constant'),
        norm_const_spec=raw_config.get('spectra_norm_const', 1.0)
    )
    
    ds = EuclidDESIDatasetArrow(
        arrow_folder_root=args.data_dir,
        split="test",
        modality_registry=registry,
        spiral=True,
        transform=transforms_dict
    )
    
    # 3. Selection
    indices = []
    if args.target_ids:
        all_ids = np.array(ds.ds['targetid'])
        for tid in args.target_ids:
            matches = np.where(all_ids == tid)[0]
            if len(matches) > 0: indices.append(matches[0])
    
    if len(indices) < args.num_plot:
        avail = list(set(range(len(ds))) - set(indices))
        indices.extend(np.random.choice(avail, args.num_plot - len(indices), replace=False))
        
    loader = DataLoader(Subset(ds, indices), batch_size=1, shuffle=False)
    
    img_config = registry.get_config("images")
    
    # 4. Main Loop
    for i, batch in enumerate(loader):
        logger.info(f"Processing sample {i+1}/{len(indices)}...")
        
        # Prepare Data
        X = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        # --- BLINDING STEP (The Core of Cross-Gen) ---
        # We zero-out the content of the target modality to force the model 
        # to generate it from the context of the source modality.
        
        # Note: We keep positions if embedded, but content is zeroed.
        if args.mode == 'img2spec':
            if 'spectra' in X:
                X['spectra'] = torch.zeros_like(X['spectra']) # BLIND!
        elif args.mode == 'spec2img':
            if 'images' in X:
                X['images'] = torch.zeros_like(X['images']) # BLIND!
        
        # Inference
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16): # type: ignore
                outputs, _ = model(X)

        # Get Ground Truth (Unmasked) from batch (CPU)
        gt_spec = batch['spectra'][0].numpy().flatten()
        gt_img_seq = batch['images'][0].numpy()
        
        # Get Prediction
        pred_spec = outputs['spectra'][0].float().cpu().numpy().flatten()
        pred_img_seq = outputs['images'][0].float().cpu().numpy()
        
        # Metadata
        tid = batch['targetid'].item()
        z = batch.get('z', [0])[0]
        if isinstance(z, torch.Tensor): z = z.item()
        
        # Reconstruct Wavelengths
        if 'spectra_positions' in batch:
            pos = batch['spectra_positions'][0].numpy().flatten()
            wave_ang = pos * 7000.0 + 3000.0
        else:
            wave_ang = np.linspace(3000, 10000, len(gt_spec))

        # --- PLOTTING ---
        fig = plt.figure(figsize=(16, 8))
        
        if args.mode == 'img2spec':
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
            
            # Reconstruct Input Image (We use Ground Truth because we fed it)
            img_in = unpatchify_image(gt_img_seq, img_config)
            rgb_in = make_rgb(img_in)
            
            plot_cross_img2spec(fig, gs, rgb_in, gt_spec, pred_spec, wave_ang, z, tid)
            
        elif args.mode == 'spec2img':
            gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2])
            
            # Reconstruct Images
            gt_img = unpatchify_image(gt_img_seq, img_config)
            pred_img = unpatchify_image(pred_img_seq, img_config)
            
            rgb_gt = make_rgb(gt_img)
            rgb_pred = make_rgb(pred_img)
            
            plot_cross_spec2img(fig, gs, gt_spec, rgb_gt, rgb_pred, wave_ang, z, tid)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"cross_{args.mode}_{tid}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Saved: {save_path}")

if __name__ == "__main__":
    main()