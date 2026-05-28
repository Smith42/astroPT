"""
AstroPT Multimodal Latent Projections (UMAP).

This script computes UMAP projections ONCE per modality and generates multiple
layers of visualizations (Standard, Visual Mosaic, Spectral Mosaic).
It is designed to be modality-agnostic, supporting any number of input modalities
from the multimodal extraction pipeline.

Available Plotting Modules:
    1. Standard (--plot_standard): Plots physical, chemical, and morphological properties.
    2. Visual (--plot_visual): Dense mosaic of galaxy RGB thumbnails mapped to latent space.
    3. Spectral (--plot_spectral): Dense mosaic of 1D spectra mapped to latent space.

Author: Victor Alonso Rodriguez
Date: May 2026
"""

import argparse
import collections
import json
import logging
import sys
from tqdm import tqdm
import warnings
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from datasets import concatenate_datasets
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Logger Configuration
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-Latent")

warnings.simplefilter('ignore', category=AstropyWarning)

# Plotting Style (PRESERVED - DO NOT MODIFY)
plt.rcParams['text.latex.preamble'] = r'''
            \usepackage{siunitx}
            \usepackage{bm}
            \usepackage{amsmath} 
            \sisetup{detect-family, separate-uncertainty=true, output-decimal-marker={.}}
            '''
plt.rc('text', usetex=True)
plt.rc('font', family='serif', weight='bold') 

# Constants
UMAP_PARAMS = {
    "n_neighbors": 30,
    "min_dist": 0.1,
    "n_components": 2,
    "metric": "cosine",
    "random_state": 61,
    "verbose": True,
    "n_jobs": -1 # Use all available cores (Note: random_state might override this in some umap-learn versions)
}

SURVEY_ORDER = ['main', 'sv1', 'sv3', 'special']
CAT_PALETTE = {"GALAXY": "#006eff", 
               "QSO": "#fbd500", 
               "DR1_R1": "#1b9e77", 
               "DR1_R2": "#e7298a", 
               "Other": "#666666"}

def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="AstroPT Multimodal Latent Projections")
    
    # Core Data Arguments
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to .fits catalog")
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing training weights")
    parser.add_argument("--emb_dir", type=str, required=True, help="Directory containing .npy embedding files")
    parser.add_argument("--save_dir", type=str, default=None, help="Plot Saving Directory")
    parser.add_argument("--subsample", type=int, default=None, help="Global limit of points for speed")
    parser.add_argument("--train_name", type=str, default=None, help="Custom title for plots")
    
    # Activation Flags (Modular Execution)
    parser.add_argument("--plot_standard", action="store_true", help="Generate physical/chemical property grids")
    parser.add_argument("--plot_visual", action="store_true", help="Generate dense RGB image mosaic")
    parser.add_argument("--plot_spectral", action="store_true", help="Generate dense 1D spectra mosaic")
    
    # Arrow Data (Required for Visual/Spectral)
    parser.add_argument("--data_dir", type=str, default=None, help="Arrow data root directory")
    parser.add_argument("--grid_size", type=int, default=100, help="Grid density for mosaic plots")
    
    # Spectral Plotting specific arguments
    parser.add_argument("--spec_wl", type=float, default=6562.8, help="Central wl (e.g. H-alpha)")
    parser.add_argument("--spec_range", type=float, default=100, help="Wl range around central wl")
    
    return parser.parse_args()

def load_data(emb_dir: Path, metadata_path: Path) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Loads embeddings and metadata with flexible modality detection."""
    logger.info(f"Scanning embeddings directory: {emb_dir}...")
    data_dict = {}
    
    # Find IDs
    id_candidates = ['targetid.npy', 'ids.npy', 'target_ids.npy', 'object_ids.npy']
    id_path = next((emb_dir / c for c in id_candidates if (emb_dir / c).exists()), None)
    if not id_path: 
        raise FileNotFoundError(f"Could not find Target IDs in {emb_dir}")
    data_dict['targetid'] = np.load(id_path)

    # Detect ALL modalities in the directory
    # Exclude files that are known metadata or too small
    excluded = set(id_candidates) | {'metadata.npy', 'indices.npy'}
    for f in emb_dir.glob("*.npy"):
        if f.name not in excluded:
            logger.info(f"Detected Modality: {f.stem}")
            data_dict[f.stem] = np.load(f, mmap_mode='r')

    # Load Metadata
    catalog = Table.read(metadata_path)
    df = catalog.to_pandas()
    
    for col in df.columns:
        if df[col].dtype == object and isinstance(df[col].iloc[0], bytes):
            try: df[col] = df[col].str.decode('utf-8')
            except: pass
    if 'SURVEY' in df.columns: 
        df['SURVEY'] = df['SURVEY'].str.lower().str.strip()
        
    return data_dict, df

def align_data(embeddings_dict: Dict[str, np.ndarray], catalog_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Aligns metadata with embeddings using strict TARGETID intersection."""
    emb_ids = embeddings_dict['targetid']
    target_col = 'TARGETID' if 'TARGETID' in catalog_df.columns else 'targetid'
    
    catalog_df[target_col] = catalog_df[target_col].astype('int64')
    catalog_indexed = catalog_df.drop_duplicates(subset=[target_col]).set_index(target_col)
    
    common_ids = np.intersect1d(emb_ids, catalog_indexed.index.values)
    matched_catalog = catalog_indexed.loc[common_ids].reset_index()
    
    id_to_idx = {id_val: i for i, id_val in enumerate(emb_ids)}
    valid_indices = np.array([id_to_idx[uid] for uid in matched_catalog[target_col].values])
    
    logger.info(f"Final aligned samples: {len(matched_catalog)}")
    return matched_catalog, valid_indices

def plot_standard_grid(umap_2d: np.ndarray, df: pd.DataFrame, mod_name: str, save_dir: Path, suffix: str, target_set: Optional[str] = None):
    """Generates the grid plots (Physical, Chemical, Morphological). STYLED PRESERVED."""
    logger.info(f"Generating Standard Grids for {mod_name}...")
    
    def log_transform(x): return np.log10(np.where(x <= 0, 1e-5, x)) 

    # Plot Configurations (Preserved)
    CONFIGS = {
        "Physics": [
            {"col": "Z", "label": r"Redshift (z)", "cat": False, "trans": None},
            {"col": "LOGMSTAR", "label": r"Stellar Mass ($\log M_*$)", "cat": False, "trans": None},
            {"col": "LOGSFR", "label": r"SFR ($\log \psi$)", "cat": False, "trans": None},
            {"col": "GR", "label": r"Color ($g-r$)", "cat": False, "trans": None},
            {"col": "SPECTYPE", "label": "Spectral Type", "cat": True, "trans": None},
            {"col": "data_set_release", "label": "Data Set Release", "cat": True, "trans": None},
        ],
        "Chemistry": [
            {"col": "flux_detection_total", "label": r"Total Flux ($\log$)", "cat": False, "trans": log_transform},
            {"col": "HALPHA_EW", "label": r"H$\alpha$ EW ($\log$)", "cat": False, "trans": log_transform},
            {"col": "NII_6584_FLUX", "label": r"[NII] Flux ($\log$)", "cat": False, "trans": log_transform},
            {"col": "HALPHA_FLUX", "label": r"H$\alpha$ Flux ($\log$)", "cat": False, "trans": log_transform},
            {"col": "OIII_5007_FLUX", "label": r"[OIII] Flux ($\log$)", "cat": False, "trans": log_transform},
            {"col": "HBETA_FLUX", "label": r"H$\beta$ Flux ($\log$)", "cat": False, "trans": log_transform},
        ],
        "Morphology": [
            {"col": "sersic_sersic_vis_radius", "label": r"Sersic VIS Radius ($\log$)", "cat": False, "trans": log_transform},
            {"col": "sersic_sersic_vis_index", "label": r"Sersic VIS Index", "cat": False, "trans": None},
            {"col": "sersic_sersic_vis_axis_ratio", "label": r"Sersic VIS Axis Ratio", "cat": False, "trans": None},
            {"col": "has_spiral_arms_yes", "label": r"Spiral Arms Prob", "cat": False, "trans": None},
            {"col": "smoothness", "label": r"Smoothness", "cat": False, "trans": None},
            {"col": "gini", "label": r"Gini Coefficient", "cat": False, "trans": None},
        ]
    }
    
    def plot_single_set(set_name, plot_configs):
        logger.info(f"Generating plot set: {set_name} for {mod_name}...")
        
        # Compute Global Limits for Colorbars (only for this set)
        global_limits = {}
        for config in plot_configs:
            col = config["col"]
            if not config["cat"] and col in df.columns:
                vals = df[col].values.astype(float)
                if config["trans"]: vals = config["trans"](vals)
                valid_vals = vals[np.isfinite(vals)]
                if len(valid_vals) > 0:
                    global_limits[col] = (np.percentile(valid_vals, 1), np.percentile(valid_vals, 99))

        # Spatial Limits
        x_min, x_max = np.percentile(umap_2d[:, 0], 1), np.percentile(umap_2d[:, 0], 99)
        y_min, y_max = np.percentile(umap_2d[:, 1], 1), np.percentile(umap_2d[:, 1], 99)
        margin_x, margin_y = (x_max - x_min) * 0.1, (y_max - y_min) * 0.1
        xlims, ylims = (x_min - margin_x, x_max + margin_x), (y_min - margin_y, y_max + margin_y)

        n_rows = len(SURVEY_ORDER)
        n_cols = len(plot_configs)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), constrained_layout=True)
        if n_rows == 1: axes = axes[np.newaxis, :] 
        
        for row_idx, survey_name in enumerate(SURVEY_ORDER):
            mask = df['SURVEY'] == survey_name
            subset_umap, subset_df = umap_2d[mask], df[mask]
            
            if len(subset_df) == 0:
                for ax in axes[row_idx]: ax.axis('off')
                axes[row_idx, 0].set_ylabel(f"{survey_name.upper()}", fontsize=16, fontweight='bold')
                continue

            for col_idx, config in enumerate(plot_configs):
                ax = axes[row_idx, col_idx]
                col_name = config["col"]
                if col_name not in subset_df.columns:
                    ax.axis('off'); continue
                
                vals = subset_df[col_name].values
                if config["cat"]:
                    global_cats = df[col_name].dropna().astype(str)
                    sorted_cats = sorted(np.unique(global_cats), key=lambda x: np.sum(global_cats == x), reverse=True)
                    for cat in sorted_cats:
                        cat_mask = subset_df[col_name].astype(str) == cat
                        if np.sum(cat_mask) > 0:
                            ax.scatter(subset_umap[cat_mask, 0], subset_umap[cat_mask, 1],
                                       s=3, alpha=0.8, label=cat, c=CAT_PALETTE.get(cat, "#333333"), rasterized=True)
                    if row_idx == n_rows - 1 and len(sorted_cats) < 8:
                        legend_els = [Line2D([0], [0], marker='o', color='w', label=c, markerfacecolor=CAT_PALETTE.get(c, "#333333"), markersize=10) for c in sorted_cats]
                        ax.legend(handles=legend_els, fontsize=14, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=min(4, len(sorted_cats)), frameon=True)
                else:
                    vals = vals.astype(float)
                    if config["trans"]: vals = config["trans"](vals)
                    vmin, vmax = global_limits.get(col_name, (0, 1))
                    sc = ax.scatter(subset_umap[:, 0], subset_umap[:, 1], c=vals, cmap='turbo', s=1, alpha=0.4, vmin=vmin, vmax=vmax, rasterized=True)
                    if row_idx == n_rows - 1:
                        cbar = fig.colorbar(sc, cax=ax.inset_axes([0.025, -0.12, 0.95, 0.05]), orientation='horizontal')
                        cbar.ax.tick_params(labelsize=14)
                
                ax.set_xlim(xlims); ax.set_ylim(ylims)
                ax.set_xticks([]); ax.set_yticks([])
                if col_idx == 0: ax.set_ylabel(rf"\textbf{{{survey_name.upper()}}}" + "\n" + rf"\textbf{{(N={len(subset_df)})}}", fontsize=18)
                if row_idx == 0: ax.set_title(rf"\boldmath\textbf{{{config['label']}}}", fontsize=20, pad=15)

        fig.suptitle(rf"\textbf{{AstroPT Latent Space - {mod_name.upper()} - {set_name}}}" + f"\n{suffix}", fontsize=24, y=1.05)
        plt.savefig(save_dir / f"umap_{mod_name}_{set_name.lower()}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Dispatch sets
    for s_name, s_config in CONFIGS.items():
        if target_set and s_name != target_set: continue
        plot_single_set(s_name, s_config)

def make_rgb_lupton(rgb_array: np.ndarray) -> np.ndarray:
    """Applies Lupton stretch for RGB generation."""
    I = np.maximum(np.mean(rgb_array, axis=0), 1e-10)
    f_I = np.arcsinh(10.0 * 0.4 * I) / 10.0
    out = rgb_array * (f_I / I)[np.newaxis, :, :]
    max_val = np.percentile(out, 99.5)
    return np.clip(out / (max_val if max_val > 0 else 1.0), 0, 1).transpose(1, 2, 0)

def render_mini_spectrum(flux: np.ndarray, idx_min: int, idx_max: int, fig: plt.Figure, ax: plt.Axes) -> np.ndarray:
    """Renders a 1D array to an RGBA buffer rapidly."""
    ax.clear()
    segment = flux[idx_min:idx_max]
    if len(segment) > 1: ax.set_xlim(0, len(segment) - 1)
    if len(segment) > 0:
        vmin, vmax = np.nanpercentile(segment, 1), np.nanpercentile(segment, 99.5)
        if vmin == vmax: vmax = vmin + 1e-5 
        ax.set_ylim(vmin, vmax)
    ax.plot(segment, color='black', linewidth=0.5)
    ax.patch.set_alpha(1.0); ax.set_xticks([]); ax.set_yticks([])
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = np.roll(img, 3, axis=2) # ARGB -> RGBA
    img[:1,:,:] = img[-1:,:,:] = img[:,:1,:] = img[:,-1:,:] = [220, 220, 220, 255] # Border
    return img

def plot_mosaic_grid(umap_2d: np.ndarray, df: pd.DataFrame, ds: Any, args: argparse.Namespace, mod_name: str, mode: str, suffix: str, save_dir: Path):
    """Generates dense mosaics for Visual or Spectral data. STYLE PRESERVED."""
    logger.info(f"Generating {mode.upper()} Mosaic Grid for {mod_name}...")
    target_col = 'TARGETID' if 'TARGETID' in df.columns else 'targetid'
    id_to_idx = {tid: i for i, tid in enumerate(np.array(ds.ds['targetid']))}
    
    fig_main, ax_main = plt.subplots(figsize=(24, 24))
    grid_size = args.grid_size
    auto_zoom = 6.5 / grid_size
    x_min, x_max, y_min, y_max = umap_2d[:, 0].min(), umap_2d[:, 0].max(), umap_2d[:, 1].min(), umap_2d[:, 1].max()
    step_x, step_y = (x_max - x_min) / grid_size, (y_max - y_min) / grid_size

    occupied_cells = collections.defaultdict(list)
    for idx, (px, py) in enumerate(umap_2d):
        ix, iy = np.clip(int((px-x_min)/step_x), 0, grid_size-1), np.clip(int((py-y_min)/step_y), 0, grid_size-1)
        cx, cy = x_min + (ix+0.5)*step_x, y_min + (iy+0.5)*step_y
        occupied_cells[(ix, iy)].append(((px-cx)**2 + (py-cy)**2, idx, cx, cy))

    for k in occupied_cells: occupied_cells[k].sort(key=lambda x: x[0])

    spec_fig, spec_ax = None, None
    if mode == 'spectral':
        spec_fig, spec_ax = plt.subplots(figsize=(1.5, 1.5), dpi=85)
        spec_fig.subplots_adjust(0,0,1,1); spec_fig.patch.set_alpha(0.0); spec_ax.patch.set_alpha(0.0)

    # 1. Identify all needed indices for the grid (ONE pass)
    needed_ids = []
    cell_metadata = []
    for (ix, iy), candidates in occupied_cells.items():
        for _, idx, cx, cy in candidates:
            tid = df.iloc[idx][target_col]
            if tid in id_to_idx:
                needed_ids.append(tid)
                cell_metadata.append((cx, cy, idx))
                break # Only one per cell
    
    # 2. Batch load from Arrow (MUCH faster than individual access)
    logger.info(f"Batch loading {len(needed_ids)} samples from Arrow...")
    arrow_indices = [id_to_idx[tid] for tid in needed_ids]
    # Use .select() for batch fetching
    subset_ds = ds.ds.select(arrow_indices)
    
    # 3. Render
    images_placed = 0
    for i, rec in enumerate(tqdm(subset_ds, desc=f"Rendering {mode}")):
        cx, cy, idx = cell_metadata[i]
        try:
            if mode == 'visual':
                vis, h, j, y = [np.array(rec[k], dtype=np.float32) for k in ['image_vis', 'image_nisp_h', 'image_nisp_j', 'image_nisp_y']]
                if any(v.size == 0 for v in [vis, h, j, y]): continue
                raw_bg = np.stack([vis, h, j, y], 0) - np.nanpercentile(np.stack([vis, h, j, y], 0), 60, (1,2), keepdims=True)
                vmax = np.percentile(np.abs(raw_bg), 99.5, (1,2), keepdims=True); vmax[vmax==0]=1.0
                raw_norm = np.clip(raw_bg / vmax, 0, 100)
                rendered_img = make_rgb_lupton(np.stack([raw_norm[1]*1.2, (raw_norm[2]+raw_norm[3])*0.65, raw_norm[0]*1.0], 0))
                ax_main.add_artist(AnnotationBbox(OffsetImage(rendered_img, zoom=auto_zoom), (cx, cy), frameon=False, pad=0.0))
            else:
                flux = np.array(rec['spectrum_flux'], dtype=np.float32)
                if flux.size == 0: continue
                z_val = df.iloc[idx]['Z']
                l_obs = args.spec_wl * (1.0 + z_val)
                i_min, i_max = [np.clip(int((l_obs + offset - 3600.0)/0.8), 0, 7780) for offset in [-args.spec_range, args.spec_range]]
                if i_min >= i_max: continue
                rendered_img = render_mini_spectrum(flux, i_min, i_max, spec_fig, spec_ax)
                ax_main.imshow(rendered_img, extent=(cx-step_x/2, cx+step_x/2, cy-step_y/2, cy+step_y/2), aspect='auto', zorder=2)
            images_placed += 1
        except: continue

    if spec_fig: plt.close(spec_fig)
    ax_main.set_xlim(x_min, x_max); ax_main.set_ylim(y_min, y_max); ax_main.axis('off')
    title_text = rf"\textbf{{AstroPT Latent Space - {mod_name.upper()} - {mode.capitalize()}}}" + f"\n{suffix}"
    if mode == "spectral": title_text += rf" [${args.spec_wl}$ \AA]"
    fig_main.suptitle(title_text, fontsize=24, y=0.98)
    plt.savefig(save_dir / f"umap_{mod_name}_{mode}.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_main)

def main():
    args = parse_args()
    weights_dir = Path(args.weights_dir)
    emb_dir = Path(args.emb_dir)
    metadata_path = Path(args.metadata_path)
    save_dir = Path(args.save_dir) / "umaps" if args.save_dir else emb_dir / "umaps"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if (args.plot_visual or args.plot_spectral) and not args.data_dir:
        logger.error("--data_dir required for visual/spectral plots."); sys.exit(1)
        
    raw_data, df = load_data(emb_dir, metadata_path)
    df_aligned, valid_idx = align_data(raw_data, df)
    
    if args.subsample and len(df_aligned) > args.subsample:
        subset_idx = np.sort(np.random.choice(len(df_aligned), args.subsample, replace=False))
        df_aligned, valid_idx = df_aligned.iloc[subset_idx].reset_index(drop=True), valid_idx[subset_idx]

    arrow_ds = None
    if args.plot_visual or args.plot_spectral:
        from datasets import load_from_disk, Dataset
        data_path = Path(args.data_dir)
        test_dirs = sorted(data_path.glob("test_*")) or [data_path / "test"]
        loaded = [Dataset.from_file(str(f)) for d in test_dirs for f in d.glob("*.arrow")]
        if not loaded: logger.error(f"No .arrow in {data_path}"); sys.exit(1)
        class Wrapper: 
            def __init__(self, ds): self.ds = ds
        arrow_ds = Wrapper(concatenate_datasets(loaded))
        
    # Title logic
    train_name = weights_dir.parent.name
    try:
        with open(weights_dir / "config.json", 'r') as f:
            train_name = json.load(f).get("train_name", train_name)
    except: pass
    embedding_method = " + ".join([p.upper() for p in emb_dir.name.split('_')])
    title_suffix = f"[{train_name} - {embedding_method}]".replace('_', r'\_')

    # 1. PHASE 1: Pre-compute ALL UMAPs (most expensive part)
    umap_storage = {}
    for mod, emb_full in raw_data.items():
        if mod == 'targetid': continue
        logger.info(f"\n--- PRE-COMPUTING UMAP: {mod.upper()} ---")
        cache_path = save_dir / f"cache_umap_{mod}.npy"
        if cache_path.exists():
            logger.info(f"Loading cached coordinates...")
            umap_coords = np.load(cache_path)
        else:
            logger.info(f"Computing manifold (this may take 1-5 minutes depending on CPU)...")
            umap_coords = umap.UMAP(**UMAP_PARAMS).fit_transform(emb_full[valid_idx])
            np.save(cache_path, umap_coords)
            logger.info(f"UMAP coordinates cached to {cache_path.name}")
        umap_storage[mod] = umap_coords

    # 2. PHASE 2: Plotting by DASHBOARD TYPE (Comparative workflow)
    # Order: Morphology -> Physics -> Chemistry -> Visual -> Spectral
    
    # A. Standard Grids
    if args.plot_standard:
        logger.info("\n" + "="*40 + "\nPHASE 2A: STANDARD GRIDS (Comparative)\n" + "="*40)
        # We need to re-initialize plot_standard_grid helper logic but grouped
        def log_transform(x): return np.log10(np.where(x <= 0, 1e-5, x)) 
        ALL_SETS = ["Morphology", "Physics", "Chemistry"]
        
        for set_name in ALL_SETS:
            logger.info(f"\n>>> PLOTTING ALL MODALITIES FOR: {set_name} <<<")
            for mod, umap_coords in umap_storage.items():
                # Call refined plot_standard_grid to plot ONLY this set
                # To do this cleanly, we can temporarily modify the internal set loop 
                # or just call it with a single set.
                
                # Let's adjust plot_standard_grid signature to allow specific sets
                # For now, I'll pass a 'target_set' argument
                plot_standard_grid(umap_coords, df_aligned, mod, save_dir, title_suffix, target_set=set_name)

    # B. Visual Mosaics
    if args.plot_visual:
        logger.info("\n" + "="*40 + "\nPHASE 2B: VISUAL MOSAICS\n" + "="*40)
        for mod, umap_coords in umap_storage.items():
            m_mask = np.ones(len(df_aligned), dtype=bool)
            if 'sersic_sersic_vis_radius' in df_aligned.columns:
                m_mask = df_aligned['sersic_sersic_vis_radius'].values > np.nanpercentile(df_aligned['sersic_sersic_vis_radius'].values, 75)
            plot_mosaic_grid(umap_coords[m_mask], df_aligned[m_mask].reset_index(drop=True), arrow_ds, args, mod, "visual", title_suffix, save_dir)

    # C. Spectral Mosaics
    if args.plot_spectral:
        logger.info("\n" + "="*40 + "\nPHASE 2C: SPECTRAL MOSAICS\n" + "="*40)
        for mod, umap_coords in umap_storage.items():
            m_mask = np.ones(len(df_aligned), dtype=bool)
            if 'sersic_sersic_vis_radius' in df_aligned.columns:
                m_mask = df_aligned['sersic_sersic_vis_radius'].values > np.nanpercentile(df_aligned['sersic_sersic_vis_radius'].values, 75)
            plot_mosaic_grid(umap_coords[m_mask], df_aligned[m_mask].reset_index(drop=True), arrow_ds, args, mod, "spectral", title_suffix, save_dir)

if __name__ == "__main__":
    main()
