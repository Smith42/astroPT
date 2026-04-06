"""
AstroPT Unified UMAP Visualizer.

This script computes UMAP projections ONCE per modality and generates multiple
layers of visualizations based on user flags. It ensures absolute spatial consistency
across all plots and significantly reduces compute time.

Available Plotting Modules:
    1. Standard (--plot_standard): Plots physical, chemical, and morphological properties.
    2. Visual (--plot_visual): Dense mosaic of galaxy RGB thumbnails mapped to latent space.
    3. Spectral (--plot_spectral): Dense mosaic of 1D spectra mapped to latent space.

Author: Victor Alonso Rodriguez
Date: March 2026
"""

import argparse
import collections
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from datasets import load_from_disk, concatenate_datasets
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Logger Configuration
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-Unified-UMAP")

warnings.simplefilter('ignore', category=AstropyWarning)

# Plotting Style
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
    "random_state": 61
}

SURVEY_ORDER = ['main', 'sv1', 'sv3', 'special']
CAT_PALETTE = {"GALAXY": "#006eff", 
               "QSO": "#fbd500", 
               "DR1_R1": "#1b9e77", 
               "DR1_R2": "#e7298a", 
               "Other": "#666666"}

class DummyRegistry:
    """Mock registry for Arrow DataLoader to bypass strict modality checks during plotting."""
    def get_config(self, name: str) -> Any: return None


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for the unified pipeline."""
    parser = argparse.ArgumentParser(description="AstroPT Unified UMAP Generator")
    
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
    parser.add_argument("--data_dir", type=str, default=None, help="Arrow data root directory (Required if visual/spectral)")
    parser.add_argument("--grid_size", type=int, default=100, help="Grid density for visual/spectral plots")
    
    # Spectral Plotting specific arguments
    parser.add_argument("--spec_wl", type=float, default=6562.8, help="Central wl")
    parser.add_argument("--spec_range", type=float, default=100, help="Wl range around central wl")
    
    return parser.parse_args()

# UMAPS Data Loader
def load_data(
        emb_dir: str | Path, 
        metadata_path: str | Path
    ) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Loads embeddings and FITS metadata."""
    logger.info(f"Scanning embeddings directory: {emb_dir}...")
    data_dict = {}
    
    emb_dir = Path(emb_dir)
    metadata_path = Path(metadata_path)
    
    # Scan for IDs
    if not emb_dir.exists:
        raise FileNotFoundError(f"Embeddings directory not found: {emb_dir}")
    
    # Find IDs
    id_candidates = ['targetid.npy', 'ids.npy', 'target_ids.npy', 'object_ids.npy']
    id_path = next((emb_dir / c for c in id_candidates if (emb_dir / c).exists()), None)
    if not id_path: raise FileNotFoundError(f"Could not find Target IDs in {emb_dir}")
    data_dict['targetid'] = np.load(id_path)

    # Find Modalities
    for mod in ['images', 'spectra', 'joint']:
        candidates = list(emb_dir.glob(f"{mod}*.npy"))
        if candidates:
            best = next((c for c in candidates if c.name == f"{mod}.npy"), candidates[0])
            logger.info(f"Loading {mod} from: {best.name}")
            data_dict[mod] = np.load(best)

    # Load Metadata
    catalog = Table.read(metadata_path)
    df = catalog.to_pandas()
    
    for col in df.columns:
        if df[col].dtype == object and isinstance(df[col].iloc[0], bytes):
            try: df[col] = df[col].str.decode('utf-8')
            except: pass
    if 'SURVEY' in df.columns: df['SURVEY'] = df['SURVEY'].str.lower().str.strip()
        
    return data_dict, df

def align_data(embeddings_dict: Dict[str, np.ndarray], catalog_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Aligns metadata with embeddings using strict TARGETID intersection."""
    logger.info("Aligning embeddings with metadata...")
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


# STANDARD UMAPS
def plot_standard_grid(
        umap_2d: np.ndarray, 
        df: pd.DataFrame, 
        mod_name: str, 
        save_dir: str | Path, 
        suffix: str
    ) -> None:
    """
    Generates the grid plots with DYNAMIC percentiles, SMART Z-ORDERING and CUSTOM PALETTES.
    (Restored from original full version)
    """
    logger.info(f"Generating Standard Grids for {mod_name}...")
    
    # Logaritmic transform
    def log_transform(x): 
        return np.log10(np.where(x <= 0, 1e-5, x)) 

    # Column Configurations
    set_1_config = [
        {"col": "Z", "label": r"Redshift (z)", "cat": False, "trans": None},
        {"col": "LOGMSTAR", "label": r"Stellar Mass ($\log M_*$)", "cat": False, "trans": None},
        {"col": "LOGSFR", "label": r"SFR ($\log \psi$)", "cat": False, "trans": None},
        {"col": "GR", "label": r"Color ($g-r$)", "cat": False, "trans": None},
        {"col": "SPECTYPE", "label": "Spectral Type", "cat": True, "trans": None},
        {"col": "data_set_release", "label": "Data Set Release", "cat": True, "trans": None},
    ]
    
    set_2_config = [
        {"col": "flux_detection_total", "label": r"Total Flux ($\log$)", "cat": False, "trans": log_transform},
        {"col": "HALPHA_EW", "label": r"H$\alpha$ EW ($\log$)", "cat": False, "trans": log_transform},
        {"col": "NII_6584_FLUX", "label": r"[NII] Flux ($\log$)", "cat": False, "trans": log_transform},
        {"col": "HALPHA_FLUX", "label": r"H$\alpha$ Flux ($\log$)", "cat": False, "trans": log_transform},
        {"col": "OIII_5007_FLUX", "label": r"[OIII] Flux ($\log$)", "cat": False, "trans": log_transform},
        {"col": "HBETA_FLUX", "label": r"H$\beta$ Flux ($\log$)", "cat": False, "trans": log_transform},
    ]
    
    set_3_config = [
        {"col": "sersic_sersic_vis_radius", "label": r"Sersic VIS Radius ($\log$)", "cat": False, "trans": log_transform},
        {"col": "sersic_sersic_vis_index", "label": r"Sersic VIS Index", "cat": False, "trans": None},
        {"col": "sersic_sersic_vis_axis_ratio", "label": r"Sersic VIS Axis Ratio", "cat": False, "trans": None},
        {"col": "has_spiral_arms_yes", "label": r"Spiral Arms Prob", "cat": False, "trans": None},
        {"col": "smoothness", "label": r"Smoothness", "cat": False, "trans": None},
        {"col": "gini", "label": r"Gini Coefficient", "cat": False, "trans": None},
    ]
    
    sets_to_plot = [("Physics", set_1_config), ("Chemistry", set_2_config), ("Morphology", set_3_config)]
    
    # Dynamic Limits (P1 - P99)
    global_limits = {}
    for config in set_1_config + set_2_config + set_3_config:
        col = config["col"]
        if not config["cat"] and col in df.columns:
            vals = df[col].values.astype(float)
            if config["trans"]: vals = config["trans"](vals)
            valid_vals = vals[np.isfinite(vals)]
            
            if len(valid_vals) > 0:
                vmin = np.percentile(valid_vals, 1)
                vmax = np.percentile(valid_vals, 99)
                global_limits[col] = (vmin, vmax)
            else:
                global_limits[col] = (0, 1)

    # Spatial Limits
    x_min, x_max = np.percentile(umap_2d[:, 0], 1), np.percentile(umap_2d[:, 0], 99)
    y_min, y_max = np.percentile(umap_2d[:, 1], 1), np.percentile(umap_2d[:, 1], 99)
    margin_x, margin_y = (x_max - x_min) * 0.1, (y_max - y_min) * 0.1
    xlims, ylims = (x_min - margin_x, x_max + margin_x), (y_min - margin_y, y_max + margin_y)

    for set_name, plot_configs in sets_to_plot:
        logger.info(f"Generating plot set: {set_name}...")
        n_rows = len(SURVEY_ORDER)
        n_cols = len(plot_configs)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), constrained_layout=True)
        if n_rows == 1: axes = axes[np.newaxis, :] 
        
        for row_idx, survey_name in enumerate(SURVEY_ORDER):
            mask = df['SURVEY'] == survey_name
            subset_umap = umap_2d[mask]
            subset_df = df[mask]
            
            if len(subset_df) == 0:
                for ax in axes[row_idx]: ax.axis('off')
                axes[row_idx, 0].set_ylabel(f"{survey_name.upper()}", fontsize=16, fontweight='bold')
                continue

            for col_idx, config in enumerate(plot_configs):
                ax = axes[row_idx, col_idx]
                col_name = config["col"]
                
                if col_name not in subset_df.columns:
                    ax.axis('off')
                    continue
                
                # Prepare values for plotting
                vals = subset_df[col_name].values
                if not config["cat"]:
                    vals = vals.astype(float)
                    if config["trans"]: vals = config["trans"](vals)

                #PLOTTING LOGIC
                if config["cat"]:
                    # Categorical Logic
                    global_cat_data = df[col_name].dropna().astype(str)
                    unique_cats_global = np.unique(global_cat_data)
                    global_counts = {cat: np.sum(global_cat_data == cat) for cat in unique_cats_global}
                    sorted_cats_global = sorted(unique_cats_global, key=lambda x: global_counts[x], reverse=True)
                    
                    for cat in sorted_cats_global:
                        cat_mask = subset_df[col_name].astype(str) == cat
                        if np.sum(cat_mask) > 0:
                            c_hex = CAT_PALETTE.get(cat, "#333333")
                            ax.scatter(subset_umap[cat_mask, 0], subset_umap[cat_mask, 1],
                                       s=3, alpha=0.8, label=cat, c=c_hex, rasterized=True)
                    
                    # LEGEND
                    if row_idx == n_rows - 1 and len(sorted_cats_global) < 8:
                        legend_elements = []
                        for cat in sorted_cats_global:
                            c_hex = CAT_PALETTE.get(cat, "#333333")
                            legend_elements.append(
                                Line2D([0], [0], marker='o', color='w', label=cat, 
                                       markerfacecolor=c_hex, markersize=10)
                            )
                        
                        ax.legend(
                            handles=legend_elements,      
                            markerscale=1.0,              
                            fontsize=14, 
                            loc='upper center',           
                            bbox_to_anchor=(0.5, 0.0), 
                            ncol=min(4, len(sorted_cats_global)),
                            frameon=True, 
                            framealpha=0.9
                        )
                        
                else:
                    
                    vmin, vmax = global_limits.get(col_name, (0, 1))
                    sc = ax.scatter(subset_umap[:, 0], subset_umap[:, 1],
                                    c=vals, cmap='turbo', s=1, alpha=0.4,
                                    vmin=vmin, vmax=vmax, rasterized=True)
                    
                    # COLORBAR
                    if row_idx == n_rows - 1:
                        # Create fixed ghost axis below plot
                        cax = ax.inset_axes([0.025, -0.12, 0.95, 0.05])
                        
                        cbar = fig.colorbar(sc, cax=cax, orientation='horizontal')
                        cbar.ax.tick_params(labelsize=14)
                
                # Clean Layout
                ax.set_xlim(xlims)
                ax.set_ylim(ylims)
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Axes and Titles
                if col_idx == 0:
                    ylabel_text = rf"\textbf{{{survey_name.upper()}}}" + "\n" + rf"\textbf{{(N={len(subset_df)})}}"
                    ax.set_ylabel(ylabel_text, fontsize=18)
                
                if row_idx == 0:
                    raw_label = config["label"]
                    formatted_label = rf"\boldmath\textbf{{{raw_label}}}"
                    ax.set_title(formatted_label, fontsize=20, pad=15)

        # Plot titlle
        title_text = (
            rf"\textbf{{AstroPT Latent Space - {mod_name.upper()} - {set_name}}}" + f"\n{suffix}"
        ) 

        fig.suptitle(title_text, fontsize=24, y=1.05)
        
        save_name = f"umap_{mod_name}_{set_name.lower()}.png"
        save_path = Path(save_dir) / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f" --> Saved: {save_path}")
        plt.close()


# VISUAL & SPECTRAL UMAPS
def make_rgb_lupton(rgb_array: np.ndarray) -> np.ndarray:
    """Applies Lupton stretch for RGB generation."""
    I = np.maximum(np.mean(rgb_array, axis=0), 1e-10)
    f_I = np.arcsinh(10.0 * 0.4 * I) / 10.0
    out = rgb_array * (f_I / I)[np.newaxis, :, :]
    max_val = np.percentile(out, 99.5)
    return np.clip(out / (max_val if max_val > 0 else 1.0), 0, 1).transpose(1, 2, 0)

def render_mini_spectrum(flux: np.ndarray, idx_min: int, idx_max: int, fig: plt.Figure, ax: plt.Axes) -> np.ndarray:
    """Renders a 1D array to an RGBA buffer rapidly without re-instantiating figures."""
    ax.clear()
    segment = flux[idx_min:idx_max]
    
    # Touching edges
    if len(segment) > 1:
        ax.set_xlim(0, len(segment) - 1)
        
    if len(segment) > 0:
        vmin, vmax = np.nanpercentile(segment, 1), np.nanpercentile(segment, 99.5)
        # Seguro contra espectros planos (ruido cero)
        if vmin == vmax: vmax = vmin + 1e-5 
        ax.set_ylim(vmin, vmax)
        
    ax.plot(segment, color='black', linewidth=0.5)
    
    # White background
    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    
    # Convert ARGB toa RGBA
    img = np.roll(img, 3, axis=2)
    
    # Edge color
    color_borde = [220, 220, 220, 255] 
    
    img[:1, :, :] = color_borde  # Up 
    img[-1:, :, :] = color_borde # Down 
    img[:, :1, :] = color_borde  # Left 
    img[:, -1:, :] = color_borde # Right
    
    return img

def plot_mosaic_grid(
        umap_2d: np.ndarray, 
        df: pd.DataFrame, 
        ds: Any, 
        args: argparse.Namespace, 
        mod_name: str, 
        mode: str, 
        suffix: str,
        save_dir: str | Path
    ) -> None:
    """
    Generates dense mosaics for either Visual (RGB) or Spectral (1D) data.
    Uses an occupancy queue to guarantee 1:1 pairing and avoid missing data holes.
    """
    logger.info(f"Generating {mode.upper()} Mosaic Grid for {mod_name}...")
    
    target_col = 'TARGETID' if 'TARGETID' in df.columns else 'targetid'
    id_to_idx = {tid: i for i, tid in enumerate(np.array(ds.ds['targetid']))}
    
    fig_main, ax_main = plt.subplots(figsize=(24, 24))
    grid_size = args.grid_size
    auto_zoom = 6.5 / grid_size
    
    x_min, x_max = umap_2d[:, 0].min(), umap_2d[:, 0].max()
    y_min, y_max = umap_2d[:, 1].min(), umap_2d[:, 1].max()
    step_x, step_y = (x_max - x_min) / grid_size, (y_max - y_min) / grid_size

    # Occupation priority list
    occupied_cells = collections.defaultdict(list)
    
    for idx, (px, py) in enumerate(umap_2d):
        ix = np.clip(int((px - x_min) / step_x), 0, grid_size - 1)
        iy = np.clip(int((py - y_min) / step_y), 0, grid_size - 1)
        cx, cy = x_min + (ix + 0.5) * step_x, y_min + (iy + 0.5) * step_y
        dist_sq = (px - cx)**2 + (py - cy)**2
        
        # Saving all cells candidates
        occupied_cells[(ix, iy)].append((dist_sq, idx, cx, cy))

    # Closer candidates first
    for cell_key in occupied_cells:
        occupied_cells[cell_key].sort(key=lambda item: item[0])

    spec_fig, spec_ax = None, None
    if mode == 'spectral':
        spec_fig, spec_ax = plt.subplots(figsize=(1.5, 1.5), dpi=85)
        spec_fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        spec_fig.patch.set_alpha(0.0)
        spec_ax.patch.set_alpha(0.0)

    # Images - Spectra pair object validation
    # If object has no pair skip to the next
    images_placed = 0
    error_count = 0
    for (ix, iy), candidates in occupied_cells.items():
        for dist_sq, idx, cx, cy in candidates:
            try:
                target_id = df.iloc[idx][target_col]
                
                if target_id not in id_to_idx:
                    if str(target_id) in id_to_idx: target_id = str(target_id)
                    elif int(target_id) in id_to_idx: target_id = int(target_id)
                    else: continue
                
                rec = ds.ds[id_to_idx[target_id]]
                
                vis = np.array(rec['image_vis'], dtype=np.float32)
                h = np.array(rec['image_nisp_h'], dtype=np.float32)
                j = np.array(rec['image_nisp_j'], dtype=np.float32)
                y = np.array(rec['image_nisp_y'], dtype=np.float32)
                flux = np.array(rec['spectrum_flux'], dtype=np.float32) 

                if vis.size == 0 or h.size == 0 or j.size == 0 or y.size == 0 or flux.size == 0:
                    continue
                
                raw_stack = np.stack([vis, h, j, y], axis=0)
                if np.isnan(raw_stack).all() or np.nanmax(np.abs(raw_stack)) == 0:
                    continue
                
                z_val = df.iloc[idx]['Z']
                lambda_obs = args.spec_wl * (1.0 + z_val)
                step_wl = (9824.0 - 3600.0) / (7781 - 1)
                idx_min = np.clip(int((lambda_obs - args.spec_range - 3600.0) / step_wl), 0, 7780)
                idx_max = np.clip(int((lambda_obs + args.spec_range - 3600.0) / step_wl), 0, 7780)
                if idx_min >= idx_max: 
                    continue
                
                if mode == 'visual':
                    raw_bg = raw_stack - np.nanpercentile(raw_stack, 60, axis=(1,2), keepdims=True)
                    
                    vmax = np.percentile(np.abs(raw_bg), 99.5, axis=(1,2), keepdims=True)
                    vmax[vmax == 0] = 1.0  
                    
                    raw_norm = np.clip(raw_bg / vmax, 0, 100)
                    rgb_input = np.stack([raw_norm[1]*1.2, ((raw_norm[2]+raw_norm[3])/2.0)*1.3, raw_norm[0]*1.0], axis=0)
                    rendered_img = make_rgb_lupton(rgb_input)
                    
                    ab = AnnotationBbox(OffsetImage(rendered_img, zoom=auto_zoom), (cx, cy), frameon=False, pad=0.0)
                    ax_main.add_artist(ab)
                    
                elif mode == 'spectral':
                    rendered_img = render_mini_spectrum(flux, idx_min, idx_max, spec_fig, spec_ax)
                    ax_main.imshow(rendered_img, 
                                   extent=(cx - step_x/2.0, cx + step_x/2.0, cy - step_y/2.0, cy + step_y/2.0), 
                                   aspect='auto', 
                                   zorder=2)
                
                images_placed += 1
                
                # Next cell
                break
                
            except Exception as e:
                # Logging first 3 errors
                if error_count < 3:
                    logger.warning(f"Skiping object {target_id} beacuse error: {e}")
                    error_count += 1
                continue

    if spec_fig: plt.close(spec_fig)

    logger.info(f"Placed {images_placed} {mode} thumbnails.")
    ax_main.set_xlim(x_min, x_max); ax_main.set_ylim(y_min, y_max)
    ax_main.axis('off')
    
    # Plot title
    title_text = (
        rf"\textbf{{AstroPT Latent Space - {mod_name.upper()} - {mode.capitalize()}}}" + f"\n{suffix}"
    )
    
    if mode == "spectral":
        title_text += rf" [${args.spec_wl}$ \AA]"

    fig_main.suptitle(title_text, fontsize=24, y=0.98)
    
    save_path = Path(save_dir) / f"umap_{mod_name}_{mode}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    logger.info(f" --> Saved: {save_path}")
    plt.close(fig_main)



def main():
    args = parse_args()
    
    # Required paths
    weights_dir = Path(args.weights_dir)
    emb_dir = Path(args.emb_dir)
    data_dir = Path(args.data_dir) if args.data_dir else None
    metadata_path = Path(args.metadata_path)
    save_dir = Path(args.save_dir) / "umaps" if args.save_dir else emb_dir / "umaps"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Validation
    if (args.plot_visual or args.plot_spectral) and data_dir is None:
        logger.error("--data_dir is required if --plot_visual or --plot_spectral is enabled.")
        sys.exit(1)
        
    # Load Data
    raw_data, df = load_data(emb_dir, metadata_path)
    df_aligned, valid_idx = align_data(raw_data, df)
    
    if args.subsample and len(df_aligned) > args.subsample:
        logger.info(f"Subsampling globally to {args.subsample} points...")
        subset_idx = np.sort(np.random.choice(len(df_aligned), args.subsample, replace=False))
        df_aligned = df_aligned.iloc[subset_idx].reset_index(drop=True)
        valid_idx = valid_idx[subset_idx]

    # Initialize Arrow Dataset only if needed
    arrow_ds = None
    if args.plot_visual or args.plot_spectral:
        logger.info("Bypassing DataLoader logic and loading raw Arrow files directly...")
        
        test_dirs = sorted(data_dir.glob("test_*"))
        if not test_dirs:
            test_dirs = [data_dir / "test"] 
            
        logger.info(f"Loading {len(test_dirs)} Arrow parts directly to avoid unimodal filters...")
        
        loaded_parts = [load_from_disk(str(d)) for d in test_dirs]
        raw_hf_ds = concatenate_datasets(loaded_parts)
        
        class RawArrowWrapper:
            def __init__(self, hf_dataset):
                self.ds = hf_dataset
                
        arrow_ds = RawArrowWrapper(raw_hf_ds)
        logger.info(f"Raw dataset loaded. Total samples: {len(arrow_ds.ds)}")
        
    # TITLE LOGIC
    config_path = weights_dir / "config.json"
    json_train_name = None
    
    # Try reading config.json
    if config_path.is_file():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                json_train_name = config.get("train_name", None)
        except Exception:
            pass 

    # Select ID: CLI > JSON > Folder
    train_name = args.train_name or json_train_name or weights_dir.parent.name
    raw_emb_name = emb_dir.name
    emb_parts = [p.upper() for p in raw_emb_name.split('_')]
    embedding_method = " + ".join(emb_parts)
    
    title_suffix = f"[{train_name} - {embedding_method}]"
    title_suffix = title_suffix.replace('_', r'\_')

    # Core Execution Loop
    for mod in ['images', 'spectra', 'joint']:
        if mod not in raw_data: continue
        
        logger.info(f"\n{'-'*40}\nPROCESSING MODALITY: {mod.upper()}")
        emb = raw_data[mod][valid_idx]
        
        logger.info("Computing Universal UMAP Coordinates...")
        reducer = umap.UMAP(**UMAP_PARAMS)
        umap_coords = reducer.fit_transform(emb)
        
        if args.plot_standard: 
            plot_standard_grid(umap_coords, df_aligned, mod, save_dir, suffix=title_suffix)
        
        if args.plot_visual or args.plot_spectral:

            if 'sersic_sersic_vis_radius' in df_aligned.columns:
                logger.info("Filtering large objects for Mosaic Grids...")
                radius_vals = df_aligned['sersic_sersic_vis_radius'].values
                
                size_threshold = np.nanpercentile(radius_vals, 75) 
                large_mask = radius_vals > size_threshold
                
                mosaic_umap = umap_coords[large_mask]
                mosaic_df = df_aligned[large_mask].reset_index(drop=True)
                logger.info(f"Mosaic reduced from {len(df_aligned)} to {len(mosaic_df)} large objects.")
            else:
                logger.warning("Size column not found. Plotting all objects in mosaic.")
                mosaic_umap = umap_coords
                mosaic_df = df_aligned

            if args.plot_visual: 
                plot_mosaic_grid(mosaic_umap, mosaic_df, arrow_ds, args, mod, "visual", 
                                 suffix=title_suffix, save_dir=save_dir)
            if args.plot_spectral: 
                plot_mosaic_grid(mosaic_umap, mosaic_df, arrow_ds, args, mod, "spectral", 
                                 suffix=title_suffix, save_dir=save_dir)

    logger.info("Unified UMAP Pipeline completed successfully.")

if __name__ == "__main__":
    main()