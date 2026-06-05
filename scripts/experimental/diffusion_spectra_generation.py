import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import glob

from astropt.experimental.model_diffusion import DiffusionModelConfig, SpectrumDiffusionModel

def make_rgb_lupton(image_tensor: np.ndarray, Q: float = 12.0, stretch: float = 0.5, m: float = 0.0) -> np.ndarray:
    """Lupton et al. (2004) algorithm to combine multichannels into an RGB image."""
    I = np.mean(image_tensor, axis=0)
    I = I - m
    I = np.maximum(I, 1e-10)
    f_I = np.arcsinh(Q * stretch * I) / Q
    scale_factor = f_I / I
    rgb_out = image_tensor * scale_factor[np.newaxis, :, :]
    max_rgb = np.percentile(rgb_out, 99.5)
    if max_rgb > 0:
        rgb_out = rgb_out / max_rgb
    rgb_out = np.clip(rgb_out, 0, 1)
    return rgb_out.transpose(1, 2, 0)

def find_arrow_record(data_dir: Path, target_id: int) -> Optional[dict]:
    """Finds a record with the matching target_id in the Arrow files inside data_dir."""
    from datasets import load_from_disk, load_dataset
    data_dir = Path(data_dir)
    folders = sorted([str(p) for p in data_dir.iterdir() if p.is_dir()])
    print(f"[DEBUG] Scanning {len(folders)} folders for target ID {target_id}: {folders}")
    for folder in folders:
        try:
            ds = load_from_disk(folder)
        except Exception:
            files = sorted(glob.glob(os.path.join(folder, "*.arrow")))
            if not files:
                continue
            ds = load_dataset("arrow", data_files=files, split="train")
        
        available_cols = ds.column_names
        tid_col = "targetid" if "targetid" in available_cols else "target_id" if "target_id" in available_cols else None
        if tid_col is None:
            continue
        
        # Convert to numpy format for ultra-fast filtering
        ds_np = ds.select_columns([tid_col])
        target_ids = np.array(ds_np[tid_col])
        
        try:
            matches = np.where(target_ids.astype(np.int64) == int(target_id))[0]
        except Exception:
            try:
                matches = np.where(target_ids.astype(str) == str(target_id))[0]
            except Exception:
                matches = np.where(target_ids == target_id)[0]
                
        if len(matches) > 0:
            idx = int(matches[0])
            print(f"[DEBUG] Found match in folder: {folder} at index {idx}")
            return ds[idx]
    return None

# Plotting Global LaTeX Configuration (matching dash_internal_reconstruction_samples.py)
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
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
})

# Spectral Lines and rest wavelengths in Angstroms
MAIN_LINES = [
    (r"Ly$\alpha$", 1216.0), (r"C IV", 1549.0), ("C III", 1908.7), (r"Mg II", 2798.0), 
    (r"[O II]", 3727.3), (r"[Ne III]", 3868.7), (r"Ca K", 3933.7), (r"Ca H", 3968.5), 
    (r"H$\delta$", 4102.0), (r"H$\gamma$", 4341.0), (r"H$\beta$", 4861.0), 
    (r"[O III]", 4959.0), (r"[O III]", 5007.0), (r"Mg I", 5175.0), (r"Na D", 5890.0), 
    (r"[N II]", 6548.0), (r"H$\alpha$", 6563.0), (r"[N II]", 6583.5), (r"[S II]", 6730.8)
]

def plot_spectral_lines(ax, start_wl, end_wl, z):
    """Annotates key spectral lines on the panel if their redshifted wavelength falls in the wavelength range."""
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    pos_high = y_max - (y_range * 0.15) # 15% from the top
    pos_low  = y_min + (y_range * 0.15) # 15% from the bottom
    
    sorted_lines = sorted(MAIN_LINES, key=lambda x: x[1])
    counter = 0
    
    for name, rest_wave in sorted_lines:
        obs_wave = rest_wave * (1 + z)
        if start_wl <= obs_wave < end_wl:
            y_pos = pos_high if counter % 2 == 0 else pos_low
            ax.axvline(obs_wave, color='royalblue', linestyle='--', alpha=0.45, lw=0.95)
            ax.text(obs_wave, y_pos, rf"\textbf{{{name}}}", rotation=90, 
                    color='royalblue', va='bottom', ha='right', fontsize=8, alpha=0.8)
            counter += 1

# Define AdaptiveMLP structure to match probing_downstream_benchmark
class AdaptiveMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, output_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def get_morphological_meta_str(tid: int, metadata) -> str:
    import pandas as pd
    if metadata is not None and tid in metadata.index:
        meta = metadata.loc[tid]
        
        # Coordinates
        ra_val = meta.get('RA', None)
        dec_val = meta.get('DEC', None)
        coords_str = f"${ra_val:.6f}^\\circ, {dec_val:.6f}^\\circ$" if (ra_val is not None and not pd.isna(ra_val)) else "N/A"
        
        # Redshift
        z_val = meta.get('Z', None)
        z_str = f"{z_val:.4f}" if (z_val is not None and not pd.isna(z_val)) else "N/A"
        
        # Stellar Mass
        logmstar_val = meta.get('LOGMSTAR', None)
        mstar = f"$10^{{{logmstar_val:.2f}}} \\text{{ M}}_\\odot$" if (logmstar_val is not None and not pd.isna(logmstar_val)) else "N/A"
        
        # Star Formation Rate
        logsfr_val = meta.get('LOGSFR', None)
        sfr = f"${10**logsfr_val:.3f} \\text{{ M}}_\\odot/\\text{{yr}}$" if (logsfr_val is not None and not pd.isna(logsfr_val)) else "N/A"
        
        # Spectype
        spectype = str(meta.get('SPECTYPE', 'N/A'))
        
        # Euclid Sersic Index
        sersic_n_val = meta.get('sersic_sersic_vis_index', None)
        n_str = f"{sersic_n_val:.2f}" if (sersic_n_val is not None and not pd.isna(sersic_n_val)) else "N/A"
        
        # Euclid Effective Radius
        sersic_re_val = meta.get('sersic_sersic_vis_radius', None)
        re_str = f"{sersic_re_val:.3f}''" if (sersic_re_val is not None and not pd.isna(sersic_re_val)) else "N/A"
        
        # Axis Ratio
        ba_val = meta.get('sersic_sersic_vis_axis_ratio', None)
        ba_str = f"{ba_val:.3f}" if (ba_val is not None and not pd.isna(ba_val)) else "N/A"
        
        # VIS Aperture Flux
        vis_flux_val = meta.get('flux_vis_1fwhm_aper', None)
        vis_flux_str = f"{vis_flux_val:.2f}" if (vis_flux_val is not None and not pd.isna(vis_flux_val)) else "N/A"
        
        meta_str = (
            f"\\textbf{{Coords}}: {coords_str}\n"
            f"\\textbf{{Z}}: {z_str}\n"
            f"\\textbf{{$M_*$}}: {mstar}\n"
            f"\\textbf{{SFR}}: {sfr}\n"
            f"\\textbf{{Spectype}}: {spectype}\n"
            f"\\textbf{{VIS Flux}}: {vis_flux_str}\n"
            f"\\textbf{{Sersic n}}: {n_str}\n"
            f"\\textbf{{Radius $R_{{eff}}$}}: {re_str}\n"
            f"\\textbf{{Axis Ratio}}: {ba_str}"
        )
        return meta_str.replace("_", "\\_")
    return f"\\textbf{{ID}}: {tid}\n(No metadata)"

def get_spectroscopic_meta_str(tid: int, metadata) -> str:
    import pandas as pd
    if metadata is not None and tid in metadata.index:
        meta = metadata.loc[tid]
        
        ha_f = meta.get('HALPHA_FLUX', None)
        hb_f = meta.get('HBETA_FLUX', None)
        oiii_f = meta.get('OIII_5007_FLUX', None)
        oii_f = meta.get('OII_3726_FLUX', None)
        
        ha_ew = meta.get('HALPHA_EW', None)
        hb_ew = meta.get('HBETA_EW', None)
        oii_ew = meta.get('OII_3726_EW', None)
        
        ha_sig = meta.get('HALPHA_SIGMA', None)
        snr_r = meta.get('SNR_SPEC_R', None)
        snr_z = meta.get('SNR_SPEC_Z', None)
        
        def fmt_flux(v): return f"{v:.2f}" if (v is not None and not pd.isna(v)) else "N/A"
        def fmt_ew(v): return f"{v:.2f} \\AA" if (v is not None and not pd.isna(v)) else "N/A"
        def fmt_val(v, unit=""): return f"{v:.1f}{unit}" if (v is not None and not pd.isna(v)) else "N/A"
        
        stats_text = (
            f"\\textbf{{Spectral Line Fluxes}}:\n"
            f"  - H$\\alpha$ Flux: {fmt_flux(ha_f)}\n"
            f"  - H$\\beta$ Flux: {fmt_flux(hb_f)}\n"
            f"  - [O III] 5007 Flux: {fmt_flux(oiii_f)}\n"
            f"  - [O II] 3726 Flux: {fmt_flux(oii_f)}\n\n"
            f"\\textbf{{Equivalent Widths}}:\n"
            f"  - H$\\alpha$ EW: {fmt_ew(ha_ew)}\n"
            f"  - H$\\beta$ EW: {fmt_ew(hb_ew)}\n"
            f"  - [O II] 3726 EW: {fmt_ew(oii_ew)}\n\n"
            f"\\textbf{{Spectra Quality}}:\n"
            f"  - H$\\alpha$ Width $\\sigma$: {fmt_val(ha_sig, ' km/s')}\n"
            f"  - SNR Spec R: {fmt_val(snr_r)} | SNR Spec Z: {fmt_val(snr_z)}"
        )
        return stats_text.replace("_", "\\_")
    return "\\textbf{Spectra data missing}"

def parse_args():
    parser = argparse.ArgumentParser(description="User-Interactive Multi-Seed Zero-Label Spectrum Diffusion Synthesizer")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to DDPM model ckpt_best.pt")
    parser.add_argument("--embeddings_path", type=str, required=True, help="Path to embeddings_all.npz")
    parser.add_argument("--probing_weights_path", type=str, required=True, help="Path to trained probing_MLP.pt weights dict")
    parser.add_argument("--galaxy_id", type=int, default=None, help="Galaxy target ID to analyze (overrides index)")
    parser.add_argument("--index", type=int, default=0, help="Sequential index fallback")
    parser.add_argument("--all_modalities", action="store_true", help="Estimate redshift and generate spectrum for all available image modalities instead of just the most precise one")
    parser.add_argument("--ensemble", action="store_true", help="Use the weighted average of all estimators for spectrum conditioning instead of the single best estimator")
    parser.add_argument("--best_by", type=str, default="balanced", choices=["r2", "bias", "nmad", "rmse", "outliers", "balanced"], help="Metric used to select the single best model or weight the ensemble (default: balanced combination of R2, Bias, and NMAD)")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save visual multi-panel plot (default: ID_<galaxy_id>_diffusion.png)")
    parser.add_argument("--layout", type=str, default="dashboard", choices=["stacked", "dashboard"], help="Visual layout to use: 'stacked' for 7-row zoom-in details, 'dashboard' for Euclid RGB galaxy image + full spectrum comparison.")
    parser.add_argument("--plot_original", action="store_true", help="Plot the original galaxy image and ground-truth spectrum from the raw Arrow dataset")
    parser.add_argument("--data_dir", type=str, default="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated", help="Path to raw processed Arrow dataset root folder")
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"], help="Sampling method to use: ddpm or ddim")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of steps for DDIM sampler (default: 100)")
    parser.add_argument("--guidance_scale", type=float, default=1.5, help="Classifier-Free Guidance scale. 0.0 means no guidance. (default: 1.5)")
    parser.add_argument("--embeddings_key", type=str, default=None, help="Key in NPZ file to use for embeddings (e.g. EuclidImage_phase2)")
    parser.add_argument("--metadata_path", type=str, default=None, help="Path to FITS metadata catalog for physical properties display")
    return parser.parse_args()

def load_redshift_prober(weights_path: Path, target_key: str, device: torch.device) -> tuple:
    """Loads state dict and returns (model, scaler_mean, scaler_scale, scaler_type, score, metrics)."""
    weights = torch.load(weights_path, map_location=device, weights_only=False)
    if target_key not in weights:
        raise KeyError(f"Target/Modality key '{target_key}' not found in unified weights file.")
        
    entry = weights[target_key]
    model = AdaptiveMLP(input_dim=entry["input_dim"], output_dim=entry["output_dim"])
    model.load_state_dict(entry["model_state_dict"])
    model.to(device)
    model.eval()
    
    mean = entry.get("scaler_y_mean", 0.0)
    scale = entry.get("scaler_y_scale", 1.0)
    scaler_type = entry.get("scaler_type", "StandardScaler")
    score = entry.get("score", 0.0)
    metrics = entry.get("metrics", {})
    
    return model, mean, scale, scaler_type, score, metrics

def sample_with_seed(
    model: SpectrumDiffusionModel, 
    cond_tensor: torch.Tensor, 
    redshift_tensor: Optional[torch.Tensor], 
    device: torch.device, 
    seed_val: int,
    sampler: str = "ddpm",
    num_steps: int = 50,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """Samples from diffusion model setting specific seed for exact reproducibility."""
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
    np.random.seed(seed_val)
    
    with torch.no_grad():
        if sampler == "ddim":
            return model.sample_ddim(
                cond_tensor, 
                redshift=redshift_tensor, 
                num_steps=num_steps, 
                guidance_scale=guidance_scale,
                device=device
            )
        else:
            return model.sample(
                cond_tensor, 
                redshift=redshift_tensor, 
                guidance_scale=guidance_scale,
                device=device
            )

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Running on device: {device}")

    # Load FITS metadata catalog if provided
    metadata = None
    if getattr(args, "metadata_path", None) is not None:
        metadata_path = Path(args.metadata_path)
        if metadata_path.exists():
            try:
                from astropy.table import Table
                import pandas as pd
                print(f"[INFO] Loading FITS metadata catalog from: {metadata_path}")
                catalog = Table.read(metadata_path)
                target_col = 'TARGETID' if 'TARGETID' in catalog.colnames else 'targetid'
                df_meta = catalog.to_pandas()
                df_meta.set_index(target_col, inplace=True)
                metadata = df_meta
                print(f"Successfully loaded metadata catalog for {len(metadata)} targets.")
            except Exception as e:
                print(f"[WARNING] Failed to load metadata catalog: {e}")
    
    # 1. Load diffusion model
    print(f"Loading Spectrum Diffusion Model from: {args.checkpoint_path}")
    diff_ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model_config = DiffusionModelConfig(**diff_ckpt["model_config"])
    diff_model = SpectrumDiffusionModel(model_config)
    diff_model.load_state_dict(diff_ckpt["model"])
    diff_model.to(device)
    diff_model.eval()
    
    diff_scaler_mean = diff_ckpt.get("scaler_mean", 0.0)
    diff_scaler_std = diff_ckpt.get("scaler_std", 1.0)
    print(f"Diffusion inverse scalers loaded (mean={diff_scaler_mean:.5f}, std={diff_scaler_std:.5f})")
    
    # Load physical norm parameters (scaler, const) from training config.json if available
    ckpt_dir = Path(args.checkpoint_path).parent
    config_json_path = ckpt_dir / "config.json"
    norm_scaler = 1.0
    norm_const = 1.0
    if config_json_path.exists():
        try:
            import json
            with open(config_json_path, 'r') as f:
                pt_cfg = json.load(f)
                norm_scaler = pt_cfg.get("spectra_norm_scaler", 1.0)
                norm_const = pt_cfg.get("spectra_norm_const", 1.0)
                print(f"[INFO] Loaded normalisation parameters from config: scaler={norm_scaler}, const={norm_const}")
        except Exception as e:
            print(f"[WARNING] Could not read config.json for normalization parameters: {e}")
    
    # 2. Load embeddings
    print(f"Loading embeddings from: {args.embeddings_path}")
    emb_data = np.load(args.embeddings_path)
    
    emb_key = None
    if getattr(args, "embeddings_key", None) is not None:
        emb_key = args.embeddings_key
        if emb_key not in emb_data:
            raise KeyError(f"Requested embeddings key '{emb_key}' not found in NPZ. Available keys: {list(emb_data.keys())}")
    else:
        for key in ['images', 'EuclidImage', 'best_img-mean']:
            if key in emb_data:
                emb_key = key
                break
        if emb_key is None:
            raise KeyError(f"No valid image embedding key found. Available keys: {list(emb_data.keys())}")
        
    embeddings = emb_data[emb_key]
    ids = emb_data['ids'] if 'ids' in emb_data else emb_data['targetid'] if 'targetid' in emb_data else np.arange(len(embeddings))
    
    idx = args.index
    if args.galaxy_id is not None:
        matched = np.where(ids == args.galaxy_id)[0]
        if len(matched) == 0:
            print(f"[ERROR] Target ID {args.galaxy_id} not found. Available samples: {list(ids[:5])}...")
            return
        idx = int(matched[0])
        print(f"Matched Target ID {args.galaxy_id} to index {idx}.")
    else:
        if idx < 0 or idx >= len(embeddings):
            raise IndexError(f"Index {idx} is out of bounds (0 to {len(embeddings)-1})")
            
    galaxy_id = ids[idx]
    image_embedding = embeddings[idx]
    
    # 3. Retrieve modalities and estimate redshift Z
    if not args.probing_weights_path:
        raise ValueError("[ERROR] Probing weights path is empty. Please provide a valid file path using --probing_weights_path.")
        
    prob_path = Path(args.probing_weights_path)
    if not prob_path.exists() or prob_path.is_dir():
        raise FileNotFoundError(f"Unified probing weights dict not found or is a directory at: {prob_path}")
        
    probing_weights = torch.load(prob_path, map_location=device, weights_only=False)
    # Filter to only allow image-based modalities (EuclidImage, EuclidImage_phase1, etc.) since we perform inference from images
    z_keys = [k for k in probing_weights.keys() if k.startswith("Z/") and ("image" in k.lower() or "euclid" in k.lower())]
    
    if not z_keys:
        raise ValueError(f"No redshift probing models ('Z/*') found in {prob_path}")
        
    print(f"\n[INFO] Found {len(z_keys)} redshift estimators:")
    entries = []
    for k in z_keys:
        score = probing_weights[k].get("score", 0.0)
        modality_name = k.split("/", 1)[1]
        entries.append((k, modality_name, score))
        print(f"  - {modality_name} | Validation R^2 = {score:.4f}")
        
    # 4. Selection and Redshift Estimation Logic (with Euclid Image-based Ensembling by default)
    # Calculate advanced astronomical metrics and weights for all available models
    weighted_entries = []
    best_by = args.best_by.lower()
    
    print(f"\n[INFO] Evaluating metadata weights using metric selection: '{best_by}'")
    for target_key, modality_name, r2_score in entries:
        prober, p_mean, p_scale, p_scaler_type, p_score, p_metrics = load_redshift_prober(prob_path, target_key, device)
        
        # Safe metric extraction (check both capitalized and uppercase forms for robustness)
        rmse = p_metrics.get("RMSE", p_metrics.get("rmse", 0.04))
        bias = abs(p_metrics.get("Bias", p_metrics.get("BIAS", p_metrics.get("bias", 0.001))))
        nmad = p_metrics.get("NMAD", p_metrics.get("nmad", 0.03))
        outliers = p_metrics.get("Outliers", p_metrics.get("outliers", 0.1))
        
        # Calculate custom sorting scores and ensembling weights
        if best_by == "r2":
            sort_val = r2_score
            weight = max(0.0, r2_score)
        elif best_by == "bias":
            sort_val = -bias  # smaller absolute bias is better
            weight = 1.0 / (bias + 1e-6)
        elif best_by == "nmad":
            sort_val = -nmad  # smaller NMAD is better
            weight = 1.0 / (nmad + 1e-6)
        elif best_by == "rmse":
            sort_val = -rmse  # smaller RMSE is better
            weight = 1.0 / (rmse + 1e-6)
        elif best_by == "outliers":
            sort_val = -outliers  # smaller outliers is better
            weight = 1.0 / (outliers + 1e-6)
        else:  # balanced (default)
            # Combines R^2, Bias, and NMAD to penalize positive systematic shifts and reward core precision
            sort_val = r2_score / (nmad * (bias + 0.001))
            weight = sort_val
            
        weighted_entries.append({
            "target_key": target_key,
            "modality_name": modality_name,
            "r2_score": r2_score,
            "sort_val": sort_val,
            "weight": weight,
            "metrics": p_metrics
        })
        
    # Sort models descending based on sort_val
    weighted_entries = sorted(weighted_entries, key=lambda x: x["sort_val"], reverse=True)
    
    # Pre-normalize image embedding using L2 norm
    emb_norm = image_embedding / np.linalg.norm(image_embedding).clip(min=1e-10)
    emb_tensor = torch.tensor(emb_norm, dtype=torch.float32).unsqueeze(0).to(device)
    cond_tensor = torch.tensor(image_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    
    selected_runs = []
    if args.all_modalities:
        for item in weighted_entries:
            target_key = item["target_key"]
            modality_name = item["modality_name"]
            prober, p_mean, p_scale, p_scaler_type, p_score, p_metrics = load_redshift_prober(prob_path, target_key, device)
            with torch.no_grad():
                pred_z_scaled = prober(emb_tensor).cpu().numpy().flatten()[0]
            pred_z = float(pred_z_scaled * p_scale + p_mean)
            
            metrics_str = ""
            if p_metrics:
                bias_val = p_metrics.get("Bias", p_metrics.get("BIAS", 0.0))
                metrics_str = f" | NMAD = {p_metrics.get('NMAD', 0.0):.4f} | Outliers = {p_metrics.get('Outliers', 0.0):.4f}% | BIAS = {bias_val:.4f}"
            print(f"\n--> Evaluating {modality_name}: Predicted Z: {pred_z:.5f}{metrics_str}")
            selected_runs.append((modality_name, pred_z))
    else:
        # Evaluate all available image estimators and print ensembling details
        print(f"\n[INFO] Evaluating {len(weighted_entries)} Euclid image-based redshift estimators (Weights by '{best_by}'):")
        preds = []
        weights = []
        for item in weighted_entries:
            target_key = item["target_key"]
            modality_name = item["modality_name"]
            prober, p_mean, p_scale, p_scaler_type, p_score, p_metrics = load_redshift_prober(prob_path, target_key, device)
            with torch.no_grad():
                pred_z_scaled = prober(emb_tensor).cpu().numpy().flatten()[0]
            pred_z_val = float(pred_z_scaled * p_scale + p_mean)
            
            metrics_str = ""
            if p_metrics:
                bias_val = p_metrics.get("Bias", p_metrics.get("BIAS", 0.0))
                metrics_str = f" | NMAD = {p_metrics.get('NMAD', 0.0):.4f} | Outliers = {p_metrics.get('Outliers', 0.0):.4f}% | BIAS = {bias_val:.4f}"
            print(f"  - {modality_name}: Z = {pred_z_val:.5f} (Weight = {item['weight']:.4f}{metrics_str})")
            
            preds.append(pred_z_val)
            weights.append(item["weight"])
            
        simple_avg = float(np.mean(preds))
        sum_weights = sum(weights)
        weighted_avg = float(sum(p * w for p, w in zip(preds, weights)) / sum_weights) if sum_weights > 0 else simple_avg
        
        print(f"\n[ENSEMBLE RESULTS]")
        print(f"  --> Simple Average Z: {simple_avg:.5f}")
        print(f"  --> Weighted Ensemble Z: {weighted_avg:.5f}")
        
        if args.ensemble:
            print(f"Selecting Weighted Ensemble Z = {weighted_avg:.5f} for joint conditional synthesis.")
            selected_runs = [("EuclidEnsemble", weighted_avg)]
        else:
            best_item = weighted_entries[0]
            best_modality = best_item["modality_name"]
            best_z = preds[0]
            print(f"Selecting Best Estimator (by default) | {best_modality}: Z = {best_z:.5f} for conditional synthesis.")
            selected_runs = [(best_modality, best_z)]

    # 5. Generate spectra for selected modalities/ensembles
    for modality_name, pred_z in selected_runs:
        # Generate spectra for seeds [61, 21, 278]
        z_tensor = torch.tensor([pred_z], dtype=torch.float32).to(device) if model_config.use_redshift else None
        
        seeds = [61, 21, 278]
        generated_spectra = {}
        
        for s in seeds:
            print(f"Sampling spectrum with Seed {s:03d}...")
            spec_tensor = sample_with_seed(
                diff_model, 
                cond_tensor, 
                z_tensor, 
                device, 
                s,
                sampler=args.sampler,
                num_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
            )
            spec_np = spec_tensor.cpu().numpy().squeeze()
            # Inverse standard scale to get the normalized asinh-flux
            spec_norm = spec_np * diff_scaler_std + diff_scaler_mean
            # Inverse asinh to get physical flux
            generated_spectra[s] = norm_scaler * np.sinh(spec_norm * norm_const)
            
        # Try to load Arrow record if data_dir exists
        arrow_record = None
        data_dir = Path(args.data_dir)
        if data_dir.exists():
            print(f"Searching for Galaxy ID {galaxy_id} in Arrow files at {data_dir}...")
            try:
                arrow_record = find_arrow_record(data_dir, galaxy_id)
                if arrow_record is not None:
                    print(f"[SUCCESS] Arrow record found for Galaxy ID {galaxy_id}!")
                else:
                    print(f"[WARNING] Galaxy ID {galaxy_id} was not found in the Arrow dataset.")
            except Exception as e:
                print(f"[WARNING] Error searching Arrow dataset: {e}")
        else:
            print(f"[WARNING] data_dir does not exist at {data_dir}. Skipping original data plotting.")

        # Extract ground truth spectrum if present
        real_spec = None
        real_spec_key = None
        for key in ['spectra', 'DESISpectrum', 'best_spec-rank']:
            if key in emb_data:
                real_spec_key = key
                break
        if real_spec_key is not None:
            try:
                raw_real = emb_data[real_spec_key][idx].squeeze()
                if raw_real.ndim == 1 and raw_real.shape[0] == 7781:
                    real_spec = raw_real
                    print("Successfully loaded real spectrum Ground Truth reference from NPZ.")
            except Exception as e:
                print(f"[WARNING] Error reading spectrum from NPZ: {e}")
                
        # Fallback to Arrow record if NPZ ground truth not found
        if real_spec is None and arrow_record is not None:
            try:
                if 'spectrum_flux' in arrow_record and arrow_record['spectrum_flux'] is not None:
                    real_flux = np.array(arrow_record['spectrum_flux'], dtype=np.float32)
                    if real_flux.ndim == 1 and real_flux.shape[0] == 7781:
                        real_spec = real_flux
                        print("Successfully loaded real spectrum Ground Truth reference from Arrow dataset.")
            except Exception as e:
                print(f"[WARNING] Error reading spectrum from Arrow record: {e}")

        # Choose visual layout
        if args.layout == "dashboard":
            print("\nRendering publication-quality side-by-side dashboard...")
            fig = plt.figure(figsize=(20, 14), dpi=150)
            gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], width_ratios=[1, 1.1, 0.9], hspace=0.4, wspace=0.3)
            
            ax_meta = fig.add_subplot(gs[0, 0])
            ax_img = fig.add_subplot(gs[0, 1])
            ax_stats = fig.add_subplot(gs[0, 2])
            ax_spec_blue = fig.add_subplot(gs[1, :])
            ax_spec_red = fig.add_subplot(gs[2, :])
            
            # --- Load & Render Meta data ---
            query_meta = get_morphological_meta_str(galaxy_id, metadata)
            ax_meta.axis('off')
            ax_meta.text(
                0.5, 0.95, query_meta, 
                transform=ax_meta.transAxes, 
                fontsize=12,
                linespacing=1.8,
                horizontalalignment='center',
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.8", fc="ivory", alpha=0.95, ec="darkgrey", lw=1.0)
            )
            ax_meta.set_title(r"\textbf{Physical and Morphological properties}", fontsize=14, fontweight='bold', color='navy', pad=15)
            
            # --- Load & Render Spectroscopic stats ---
            query_spec_meta = get_spectroscopic_meta_str(galaxy_id, metadata)
            ax_stats.axis('off')
            ax_stats.text(
                0.5, 0.95, query_spec_meta, 
                transform=ax_stats.transAxes, 
                fontsize=12,
                linespacing=1.6,
                horizontalalignment='center',
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.8", fc="aliceblue", alpha=0.95, ec="steelblue", lw=1.0)
            )
            ax_stats.set_title(r"\textbf{Spectroscopic properties}", fontsize=14, fontweight='bold', color='steelblue', pad=15)
            
            # --- Reconstruct and plot image ---
            rgb_img_plotted = False
            if arrow_record is not None:
                try:
                    def get_band(k):
                        val = arrow_record.get(k)
                        return np.array(val if val is not None else [], dtype=np.float32)
                        
                    vis = get_band('image_vis')
                    y = get_band('image_nisp_y')
                    j = get_band('image_nisp_j')
                    h = get_band('image_nisp_h')
                    
                    h = h if h.size > 0 else np.zeros_like(vis)
                    j = j if j.size > 0 else np.zeros_like(vis)
                    y = y if y.size > 0 else np.zeros_like(vis)
                    
                    if vis.size > 0:
                        raw_stack = np.stack([vis, h, j, y], axis=0)
                        bg_val = np.percentile(raw_stack, 50, axis=(1,2), keepdims=True)
                        raw_bg = raw_stack - bg_val
                        
                        RGB_WEIGHTS = [1.2, 1.3, 1.0]
                        raw_rgb_stack = []
                        for c in range(4):
                            v_max = np.percentile(np.abs(raw_bg[c]), 99.5)
                            if v_max <= 0: v_max = 1.0
                            r_ch = np.clip(raw_bg[c] / v_max, 0, 100)
                            raw_rgb_stack.append(r_ch)
                            
                        raw_norm = np.stack(raw_rgb_stack)
                        vis_norm, h_norm, j_norm, y_norm = raw_norm[0], raw_norm[1], raw_norm[2], raw_norm[3]
                        r = h_norm * RGB_WEIGHTS[0]
                        g = ((j_norm + y_norm) / 2.0) * RGB_WEIGHTS[1]
                        b = vis_norm * RGB_WEIGHTS[2]
                        rgb_input = np.stack([r, g, b], axis=0)
                        
                        rgb_img = make_rgb_lupton(rgb_input, Q=12.0, stretch=0.5)
                        ax_img.imshow(rgb_img, origin='lower')
                        rgb_img_plotted = True
                except Exception as e:
                    print(f"[WARNING] Error rendering galaxy image: {e}")
                    
            if not rgb_img_plotted:
                ax_img.text(0.5, 0.5, "EuclidImage missing", ha='center', va='center', fontsize=12)
            ax_img.axis("off")
            
            # --- Plot generated and original spectra split in two channels ---
            x_wavelengths = 3600.0 + np.arange(7781) * 0.8
            start_wl, end_wl = 3600.0, 3600.0 + 7780 * 0.8
            w_mid = (start_wl + end_wl) / 2
            
            # Blue Channel Plot
            ax_spec_blue.plot(x_wavelengths, generated_spectra[61], color="#e74c3c", alpha=0.85, linewidth=0.75, label="DDPM Seed 61")
            ax_spec_blue.plot(x_wavelengths, generated_spectra[21], color="#3498db", alpha=0.85, linewidth=0.75, label="DDPM Seed 21")
            ax_spec_blue.plot(x_wavelengths, generated_spectra[278], color="#9b59b6", alpha=0.85, linewidth=0.75, label="DDPM Seed 278")
            
            if real_spec is not None:
                ax_spec_blue.plot(x_wavelengths, real_spec, color="#2c3e50", alpha=0.8, linewidth=1, label="Observed Spectrum (DESI)")
                
            ax_spec_blue.set_xlim(start_wl, w_mid)
            ax_spec_blue.set_title(r"\textbf{Spectrum (Blue Channel)}", fontsize=12, fontweight='bold', pad=8)
            ax_spec_blue.set_ylabel(r"\textbf{Flux}", fontsize=11)
            ax_spec_blue.grid(True, linestyle=':', alpha=0.5)
            ax_spec_blue.legend(loc="upper right", frameon=True, facecolor='white', framealpha=0.9)
            plot_spectral_lines(ax_spec_blue, start_wl, w_mid, pred_z)
            
            # Red Channel Plot
            ax_spec_red.plot(x_wavelengths, generated_spectra[61], color="#e74c3c", alpha=0.85, linewidth=0.75, label="DDPM Seed 61")
            ax_spec_red.plot(x_wavelengths, generated_spectra[21], color="#3498db", alpha=0.85, linewidth=0.75, label="DDPM Seed 21")
            ax_spec_red.plot(x_wavelengths, generated_spectra[278], color="#9b59b6", alpha=0.85, linewidth=0.75, label="DDPM Seed 278")
            
            if real_spec is not None:
                ax_spec_red.plot(x_wavelengths, real_spec, color="#2c3e50", alpha=0.8, linewidth=0.75, label="Observed Spectrum (DESI)")
                
            ax_spec_red.set_xlim(w_mid, end_wl)
            ax_spec_red.set_title(r"\textbf{Spectrum (Red Channel)}", fontsize=12, fontweight='bold', pad=8)
            ax_spec_red.set_ylabel(r"\textbf{Flux}", fontsize=11)
            ax_spec_red.set_xlabel(r"\textbf{Observed Wavelength [\AA]}", fontsize=11)
            ax_spec_red.grid(True, linestyle=':', alpha=0.5)
            plot_spectral_lines(ax_spec_red, w_mid, end_wl, pred_z)
            
            plt.suptitle(rf"\textbf{{Cross-Modal Synthesis Dashboard | Galaxy ID: {galaxy_id} | Modality: {modality_name} | Est. Redshift }} $z = {pred_z:.5f}$", fontsize=16, fontweight='bold', y=0.98)
            
        else:
            # Default stacked layout
            print("\nRendering publication-quality 7-row stacked visualization...")
            fig, axes = plt.subplots(7, 1, figsize=(15, 22), sharey=True)
            
            # Wavelength grid divisions
            channels_per_row = 7781 // 7
            
            # Compute pointwise standard deviations across the three seeds to evaluate local variance
            all_seeds_arr = np.stack([generated_spectra[61], generated_spectra[21], generated_spectra[278]], axis=0)
            pointwise_stds = np.std(all_seeds_arr, axis=0)
            
            # Compute global 15th and 85th percentiles of standard deviation for adaptive background color mapping
            q15 = np.percentile(pointwise_stds, 15)
            q85 = np.percentile(pointwise_stds, 85)
            
            for i in range(7):
                start_ch = i * channels_per_row
                end_ch = (i + 1) * channels_per_row if i < 6 else 7781
                
                x_channels = np.arange(start_ch, end_ch)
                x_wavelengths = 3600.0 + x_channels * 0.8
                
                start_wl = 3600.0 + start_ch * 0.8
                end_wl = 3600.0 + (end_ch - 1) * 0.8
                
                ax = axes[i]
                
                # Plot Real Spectrum (Ground Truth reference)
                if real_spec is not None:
                    ax.plot(x_wavelengths, real_spec[start_ch:end_ch], color="#2c3e50", alpha=0.9, linewidth=1.5, label="Ground Truth Spectrum")
                    
                # Plot three generated seeds
                ax.plot(x_wavelengths, generated_spectra[61][start_ch:end_ch], color="#e74c3c", alpha=0.85, linewidth=1.1, label="DDPM Seed 61")
                ax.plot(x_wavelengths, generated_spectra[21][start_ch:end_ch], color="#3498db", alpha=0.85, linewidth=1.1, label="DDPM Seed 21")
                ax.plot(x_wavelengths, generated_spectra[278][start_ch:end_ch], color="#9b59b6", alpha=0.85, linewidth=1.1, label="DDPM Seed 278")
                
                # Dynamic color coding of panel background
                local_mean_std = float(np.mean(pointwise_stds[start_ch:end_ch]))
                
                if local_mean_std <= q15:
                    ax.set_facecolor((0.83, 0.93, 0.86, 0.25))
                    conv_status = "High Convergence"
                elif local_mean_std >= q85:
                    ax.set_facecolor((0.97, 0.84, 0.85, 0.25))
                    conv_status = "Low Convergence / High Variance"
                else:
                    ax.set_facecolor((1.0, 0.95, 0.8, 0.25))
                    conv_status = "Intermediate Convergence"
                    
                # Titles, labels, and grid details per panel
                ax.set_title(f"Panel {i+1}: Wavelengths {start_wl:.1f} to {end_wl:.1f} Å | {conv_status} (Local Mean Std: {local_mean_std:.5f})", 
                             loc="left", fontsize=10, fontweight="bold", pad=4)
                ax.set_xlim(start_wl, end_wl)
                ax.grid(True, linestyle="--", alpha=0.35)
                ax.tick_params(axis='both', which='major', labelsize=9)
                
                # Plot redshifted spectral lines
                plot_spectral_lines(ax, start_wl, end_wl, pred_z)
                
                if i == 3:
                    ax.set_ylabel("Normalized Flux Scale", fontsize=11, fontweight="bold")
                if i == 6:
                    ax.set_xlabel("Observed Wavelength [Å]", fontsize=11, fontweight="bold")
                    
                if i == 0:
                    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
                    
            # Main figure title and layout adjusting
            fig.suptitle(f"Multi-Seed Image-to-Spectrum Synthesis Comparison\nTarget ID: {galaxy_id} | Modality: {modality_name} | Est. Redshift z = {pred_z:.5f}", 
                         fontsize=14, fontweight="bold", y=0.99)
            plt.tight_layout()
            
        if args.output_path is None:
            output_path = Path(f"ID_{galaxy_id}_diffusion.png")
        else:
            op = Path(args.output_path)
            # If path ends with / or represents an existing directory, treat as folder and append file name
            if args.output_path.endswith("/") or op.is_dir():
                output_path = op / f"ID_{galaxy_id}_{args.layout}.png"
            else:
                output_path = op
            
        if args.all_modalities:
            # Suffix if generating plots for multiple modalities
            output_path = output_path.parent / f"{output_path.stem}_{modality_name}{output_path.suffix}"
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[SUCCESS] Publication-ready comparison plot saved to: {output_path}")

if __name__ == "__main__":
    main()
