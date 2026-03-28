"""
AstroPT Cross-Modal Generative Inspector.

This script performs Zero-Shot cross-reconstruction. It tests the alignment of 
the multimodal latent space by forcing the model to generate one modality 
autoregressively conditioned strictly on the other.

Features:
- Image -> Spectrum Generation.
- Spectrum -> Image Generation.
- Autoregressive continuous token prediction loop.
- Unified dashboard plotting for cross-modal qualitative analysis.

Author: Victor Alonso Rodriguez
Date: March 2026
"""

import argparse
import einops
import json
import logging
from pathlib import Path
import sys
from typing import Optional, Any, Dict

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

def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="AstroPT Cross-Modal Generative Inspector")
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing training weights")
    parser.add_argument("--data_dir", type=str, required=True, help="Arrow data root directory")
    parser.add_argument("--save_dir", type=str, required=True, help="Plot Saving Directory")
    parser.add_argument("--ckpt_name", type=str, default="ckpt_best.pt", help="Checkpoint filename")
    parser.add_argument("--target_ids", nargs="+", type=int, help="Specific Target IDs to plot")
    parser.add_argument("--num_plot", type=int, default=5, help="Number of random plots (Warning: Generation is slow)")
    parser.add_argument("--split", type=str, default="test", help="Data split (train/test)")
    parser.add_argument("--train_name", type=str, default=None, help="Custom title for the plot")
    parser.add_argument("--wl_range", nargs=2, type=float, default=None, help="Zoom wavelength (min max)") # <--- NUEVO
    return parser.parse_args()


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


def reconstruct_image_from_patches(patch_sequence: np.ndarray, mod_config: Optional[Any] = None) -> np.ndarray:
    seq_len, patch_dim = patch_sequence.shape
    channels = mod_config.input_size // (mod_config.patch_size ** 2) if mod_config else 4
    p_size = mod_config.patch_size if mod_config else int(np.sqrt(patch_dim // channels))
    grid_side = int(np.sqrt(seq_len))
    
    if grid_side * grid_side != seq_len:
        return np.zeros((channels, grid_side*p_size, grid_side*p_size))

    # 1. Deshacemos la espiral para volver a orden raster
    spiral_indices = get_spiral_indices(grid_side)
    raster_patches = patch_sequence[spiral_indices]
    
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


def generate_autoregressive(model, X_full: Dict, cond_mod: str, target_mod: str, max_len: int, device) -> np.ndarray:
    """
    Generates a modality autoregressively using dummy tokens to bypass model.py shifts,
    but strictly extracting the prediction aligned with the last REAL token.
    """
    all_modes = sorted(model.modality_registry.names())
    generated_tokens = [X_full[target_mod][:, 0:1, :]]
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(max_len - 1), desc=f"Generating {target_mod.upper()}", leave=False):
            X_current = {}
            
            # 1. Contexto íntegro
            X_current[cond_mod] = X_full[cond_mod]
            X_current[f"{cond_mod}_positions"] = X_full[f"{cond_mod}_positions"]
            
            # 2. Target actual
            curr_tokens = torch.cat(generated_tokens, dim=1)
            real_len = curr_tokens.shape[1]  # Guardamos la longitud real ANTES de los dummies
            
            # 3. Añadimos dummies para absorber el recorte interno
            dummy_count = 2 if target_mod == all_modes[0] else 1
            dummy_toks = torch.zeros((1, dummy_count, curr_tokens.shape[-1]), device=device, dtype=curr_tokens.dtype)
            input_tokens = torch.cat([curr_tokens, dummy_toks], dim=1)
            
            full_pos_vec = X_full[f"{target_mod}_positions"]
            idx_end = real_len + dummy_count
            input_pos = full_pos_vec[:, :idx_end]
            
            if input_pos.shape[1] < idx_end:
                extra = torch.arange(input_pos.shape[1], idx_end, device=device).unsqueeze(0)
                input_pos = torch.cat([input_pos, extra], dim=1)

            X_current[target_mod] = input_tokens
            X_current[f"{target_mod}_positions"] = input_pos
            
            # Forward pass
            outputs, _ = model(X_current)
            
            # --- LA CLAVE QUE FALTABA ---
            # Extraemos la predicción en el índice (real_len - 1). 
            # Esto ignora la basura generada por los dummies y extrae el futuro físico real.
            next_token = outputs[target_mod][:, real_len - 1 : real_len, :]
            generated_tokens.append(next_token)
            
    full_sequence = torch.cat(generated_tokens, dim=1)
    return full_sequence.float().cpu().numpy()[0]


def plot_cross_dashboard(
    target_id: int, z_val: float, train_name: str,
    rgb_gt: np.ndarray, rgb_pred: np.ndarray, res_map: np.ndarray,
    wave_ang: np.ndarray, spec_gt: np.ndarray, spec_pred: np.ndarray,
    save_dir: Path, filename: str
):
    """Plots the unified cross-reconstruction dashboard in a 3-row layout."""
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], wspace=0.1, hspace=0.3)
    
    fig.suptitle(rf"\textbf{{Cross-Modal Generation | ID: {target_id} | z={z_val:.3f}}}"
                 + f"\n[{train_name}]", fontsize=22, y=0.96)

    # ROW 0: IMAGES
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb_gt, origin='lower')
    ax1.set_title(r"\textbf{Target: Real Image}")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(rgb_pred, origin='lower')
    ax2.set_title(r"\textbf{Output: Gen. from Spectrum}")
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
        ax.plot(wave_ang, spec_pred, 'r-', lw=1.5, alpha=0.8, label='Output: Gen. from Image')
        ax.set_xlim(start, end)
        ax.set_title(rf"\textbf{{Cross-Generated Spectrum ({title})}}")
        ax.set_ylabel(r"Flux")
        if i==1: ax.set_xlabel(r"Wavelength [\AA]")
        if i==0: ax.legend(loc='lower left')
        plot_spectral_lines(ax, start, end, z_val)

    # Save
    save_path = save_dir / filename
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f" --> Saved Cross-Reconstruction: {save_path.name}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    weights_dir = Path(args.weights_dir)
    save_dir = Path(args.save_dir) / "cross_reconstructions"
    save_dir.mkdir(parents=True, exist_ok=True)
    
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
        transform=data_tf
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

    if device.type == 'cuda':
        ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype) # type: ignore
    else:
        import contextlib
        ctx = contextlib.nullcontext()
        model = model.to(torch.float32)

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

        # Sequences
        full_img = X['images']
        full_spec = X['spectra']
        start_img = full_img[:, 0:1, :]
        start_spec = full_spec[:, 0:1, :]
        
        max_img_len = full_img.shape[1]
        max_spec_len = full_spec.shape[1]

        with ctx:
            # 1. Image -> Spectrum
            logger.info(" --> Generating Spectrum from Image...")
            pred_spec_seq = generate_autoregressive(
                model=model, X_full=X, cond_mod='images', target_mod='spectra', 
                max_len=max_spec_len, device=device
            )
            
            # 2. Spectrum -> Image
            logger.info(" --> Generating Image from Spectrum...")
            pred_img_seq = generate_autoregressive(
                model=model, X_full=X, cond_mod='spectra', target_mod='images', 
                max_len=max_img_len, device=device
            )
            
        # --- DATA POST-PROCESSING (De-normalization & Formatting) ---
        raw_record = ds.ds[int(batch['idx'][0])]
        wave_ang = np.array(raw_record['spectrum_wave']).flatten()
        spec_gt_raw = np.array(raw_record['spectrum_flux']).flatten()
        
        # Spec Post-processing
        is_spec_pred = denormalize(pred_spec_seq.flatten(), norm_type_spec, norm_scaler_spec, norm_const_spec)
        min_l = min(len(spec_gt_raw), len(is_spec_pred), len(wave_ang))
        
        # Image Post-processing
        img_gt_raw = np.stack([raw_record['image_vis'], raw_record['image_nisp_h'], 
                               raw_record['image_nisp_j'], raw_record['image_nisp_y']])
        
        si_img_pred_phys = denormalize(
            reconstruct_image_from_patches(pred_img_seq, img_config), 
            norm_type_img, norm_scaler_img, norm_const_img
        )
        
        # --- NUEVO PROCESAMIENTO DE COLOR (Copiado de plot_images_spectra) ---
        RGB_WEIGHTS = [1.2, 1.3, 1.0]
        bg_val = np.percentile(img_gt_raw, 50, axis=(1,2), keepdims=True)
        
        raw_bg = img_gt_raw - bg_val
        pred_bg = si_img_pred_phys - bg_val
        
        raw_rgb_stack = []
        pred_rgb_stack = []
        for c in range(4):
            v_max = np.percentile(np.abs(raw_bg[c]), 99.5)
            if v_max <= 0: v_max = 1.0
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
            return np.stack([r, g, b], axis=0) # (3, H, W)

        rgb_input_gt = stack_to_rgb_weighted(raw_norm)
        rgb_input_pred = stack_to_rgb_weighted(pred_norm)

        is_img_real = make_rgb_lupton(rgb_input_gt, Q=12.0, stretch=0.5)
        si_img_pred = make_rgb_lupton(rgb_input_pred, Q=12.0, stretch=0.5)
        
        if img_gt_raw.shape == si_img_pred_phys.shape:
            res_map = np.mean(img_gt_raw - si_img_pred_phys, axis=0)
        else:
            logger.warning(f"Shape mismatch: GT {img_gt_raw.shape} vs Pred {si_img_pred_phys.shape}. Skipping residuals.")
            res_map = np.zeros((img_gt_raw.shape[1], img_gt_raw.shape[2]))

        # FILENAME AND SUFFIX LOGIC
        random_suffix = "_R" if int(target_id) not in specific_tids else ""
        zoom_suffix = "_zoom" if args.wl_range else ""
        filename = f"CrossRecon_ID_{target_id}{run_suffix}{zoom_suffix}{random_suffix}.png"

        # Plotting
        plot_cross_dashboard(
            target_id=target_id, z_val=z_val, 
            train_name=train_name, rgb_gt=is_img_real, rgb_pred=si_img_pred, res_map=res_map,
            wave_ang=wave_ang[:min_l], spec_gt=spec_gt_raw[:min_l], spec_pred=is_spec_pred[:min_l],
            save_dir=save_dir, filename=filename
        )
        
if __name__ == "__main__":
    main()