"""
AstroPT UMAP Visualizer (Dissected by Survey).

This script generates UMAP visualizations split into 4 rows (one per Survey: Main, SV1, SV3, Special)
to allow detailed inspection of domain shifts and batch effects.

It plots two sets of properties to diagnose both physical learning and instrumental biases:
- Set 1: Physical Properties & Data Release (Is the model learning physics or instrument modes?)
- Set 2: Spectral Lines & Quality (Is the model sensitive to chemistry and noise?)

Author: Victor Alonso Rodriguez
Date: January 2026
"""

import argparse
import logging
import os
import sys
import warnings
from typing import Dict, Any, Tuple, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning

# --- Logging Configuration ---
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-UMAP")

# Suppress Astropy warnings
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.filterwarnings('ignore', module='astropy.io.fits')
warnings.filterwarnings('ignore', module='erfa')

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
UMAP_PARAMS = {
    "n_neighbors": 30,
    "min_dist": 0.1,
    "n_components": 2,
    "metric": "cosine",
    "random_state": 42
}

# Explicit order for rows to ensure consistency
SURVEY_ORDER = ['main', 'sv1', 'sv3', 'special']

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AstroPT UMAP Generator")
    parser.add_argument("--embeddings_path", type=str, required=True, help="Path to .npz embeddings")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to .fits catalog")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--subsample", type=int, default=None, help="Limit points for speed")
    return parser.parse_args()


def load_data(embeddings_path: str, metadata_path: str) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Loads .npz embeddings and .fits metadata catalog."""
    logger.info(f"Loading embeddings from {embeddings_path}...")
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    raw_data = np.load(embeddings_path)
    
    logger.info(f"Loading metadata from {metadata_path}...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    catalog = Table.read(metadata_path)
    df = catalog.to_pandas()
    
    # Decode bytes to strings
    for col in df.columns:
        if df[col].dtype == object and isinstance(df[col].iloc[0], bytes):
            try:
                df[col] = df[col].str.decode('utf-8')
            except Exception:
                pass
                
    # Normalize SURVEY column to lowercase for matching
    if 'SURVEY' in df.columns:
        df['SURVEY'] = df['SURVEY'].str.lower().str.strip()
        
    return dict(raw_data), df


def align_data(embeddings_dict: Dict[str, np.ndarray], catalog_df: pd.DataFrame) -> pd.DataFrame:
    """Aligns metadata with embeddings using TARGETID."""
    logger.info("Aligning embeddings with metadata...")
    if 'targetid' not in embeddings_dict:
        raise KeyError("'targetid' key missing in embeddings file.")
    
    emb_ids = embeddings_dict['targetid']
    catalog_df['TARGETID'] = catalog_df['TARGETID'].astype('int64')
    catalog_indexed = catalog_df.set_index('TARGETID')
    
    try:
        matched_catalog = catalog_indexed.loc[emb_ids].reset_index()
    except KeyError as e:
        logger.warning(f"ID mismatch. Intersecting available IDs...")
        available_ids = catalog_indexed.index.intersection(emb_ids)
        matched_catalog = catalog_indexed.loc[available_ids].reset_index()
        
    logger.info(f"Final aligned samples: {len(matched_catalog)}")
    return matched_catalog


def compute_umap(embeddings: np.ndarray, subsample: Optional[int] = None) -> np.ndarray:
    """Computes UMAP reduction."""
    n_samples = embeddings.shape[0]
    if subsample and n_samples > subsample:
        logger.info(f"Subsampling to {subsample} points for UMAP calculation...")
        indices = np.random.choice(n_samples, subsample, replace=False)
        data_to_fit = embeddings[indices]
    else:
        data_to_fit = embeddings
        
    logger.info(f"Running UMAP on {data_to_fit.shape} matrix...")
    reducer = umap.UMAP(**UMAP_PARAMS)
    embedding_2d = reducer.fit_transform(data_to_fit)
    return embedding_2d


def plot_dissected_grid(umap_2d: np.ndarray, df: pd.DataFrame, modality_name: str, save_dir: str):
    """Generates the 4-row grid (Surveys) x 6-col grid (Properties)."""
    
    # --- Cleaning Functions ---
    def clip_z(x): return np.where((x < 0) | (x > 4), np.nan, x)
    def clip_mass(x): return np.where((x < 6) | (x > 13), np.nan, x)
    def clip_sfr(x): return np.where((x < -15) | (x > 5), np.nan, x)
    def clip_color(x): return np.where((x < -1) | (x > 3), np.nan, x)
    def clip_flux(x): return np.where((x > 1000) | (x < 0), np.nan, np.log10(x + 1))
    
    # --- Property Configuration (Max 6 per set) ---
    
    # SET 1: Physics & Metadata (The "Identity" set)
    set_1_config = [
        {"col": "Z", "label": r"Redshift ($z$)", "cat": False, "process": clip_z},
        {"col": "LOGMSTAR", "label": r"Stellar Mass ($\log M_*$)", "cat": False, "process": clip_mass},
        {"col": "LOGSFR", "label": r"SFR ($\log \psi$)", "cat": False, "process": clip_sfr},
        {"col": "SPECTYPE", "label": "Spectral Type", "cat": True, "process": None},
        {"col": "GR", "label": r"Color ($g-r$)", "cat": False, "process": clip_color},
        {"col": "data_set_release", "label": "Data Release", "cat": True, "process": None}, # Critical for diagnosing "globs"
    ]
    
    # SET 2: Lines & Observables (The "Appearance" set)
    set_2_config = [
        {"col": "flux_detection_total", "label": r"Total Flux ($\log$)", "cat": False, "process": clip_flux},
        {"col": "SNR_SPEC_R", "label": r"SNR (R-Band)", "cat": False, "process": lambda x: np.where(x>20, 20, x)},
        {"col": "HALPHA_FLUX", "label": r"H$\alpha$ Flux", "cat": False, "process": clip_flux},
        {"col": "OIII_5007_FLUX", "label": r"[OIII] Flux", "cat": False, "process": clip_flux},
        {"col": "OII_3726_FLUX", "label": r"[OII] Flux", "cat": False, "process": clip_flux},
        {"col": "NII_6584_FLUX", "label": r"[NII] Flux", "cat": False, "process": clip_flux},
    ]
    
    sets_to_plot = [("Physics_Identity", set_1_config), ("Chemistry_Quality", set_2_config)]
    
    # --- Global Color Scaling ---
    # Compute global vmin/vmax across ALL data to ensure comparable colors
    global_limits = {}
    for config in set_1_config + set_2_config:
        col = config["col"]
        if not config["cat"] and col in df.columns:
            vals = df[col].values
            if config["process"]: vals = config["process"](vals)
            valid = vals[np.isfinite(vals)]
            if len(valid) > 0:
                global_limits[col] = (np.percentile(valid, 2), np.percentile(valid, 98))
            else:
                global_limits[col] = (0, 1)

    # --- Spatial Limits ---
    # Fix the view so all rows are zoomed into the same region
    x_min, x_max = np.percentile(umap_2d[:, 0], 1), np.percentile(umap_2d[:, 0], 99)
    y_min, y_max = np.percentile(umap_2d[:, 1], 1), np.percentile(umap_2d[:, 1], 99)
    margin_x = (x_max - x_min) * 0.1
    margin_y = (y_max - y_min) * 0.1
    xlims = (x_min - margin_x, x_max + margin_x)
    ylims = (y_min - margin_y, y_max + margin_y)

    # --- Plotting Loop ---
    for set_name, plot_configs in sets_to_plot:
        logger.info(f"   Generating plot set: {set_name}...")
        
        n_rows = len(SURVEY_ORDER)
        n_cols = len(plot_configs)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), constrained_layout=True)
        if n_rows == 1: axes = axes[np.newaxis, :] 
        
        for row_idx, survey_name in enumerate(SURVEY_ORDER):
            # Filter Data
            mask = df['SURVEY'] == survey_name
            subset_umap = umap_2d[mask]
            subset_df = df[mask]
            
            # Row Label handling if empty
            if len(subset_df) == 0:
                axes[row_idx, 0].set_ylabel(f"{survey_name.upper()}", fontsize=14, fontweight='bold')
                for ax in axes[row_idx]: ax.axis('off')
                continue

            for col_idx, config in enumerate(plot_configs):
                ax = axes[row_idx, col_idx]
                col_name = config["col"]
                
                if col_name not in subset_df.columns:
                    ax.text(0.5, 0.5, "N/A", ha='center')
                    ax.axis('off')
                    continue
                
                vals = subset_df[col_name].values
                if config["process"]: vals = config["process"](vals)
                
                # PLOT
                if config["cat"]:
                    # Categorical (e.g., Spectype, Release)
                    unique_cats = np.unique(subset_df[col_name].dropna().astype(str))
                    cmap = plt.get_cmap("tab10")
                    for i, cat in enumerate(unique_cats):
                        cat_mask = subset_df[col_name].astype(str) == cat
                        ax.scatter(subset_umap[cat_mask, 0], subset_umap[cat_mask, 1],
                                   s=2, alpha=0.6, label=cat, c=[cmap(i % 10)])
                    
                    if len(unique_cats) < 8: 
                        ax.legend(markerscale=4, fontsize=8, loc='upper right')
                        
                else:
                    # Continuous (Physics/Fluxes)
                    vmin, vmax = global_limits.get(col_name, (0, 1))
                    sc = ax.scatter(subset_umap[:, 0], subset_umap[:, 1],
                                    c=vals, cmap='turbo', s=1, alpha=0.4,
                                    vmin=vmin, vmax=vmax, rasterized=True)
                    
                    if row_idx == 0:
                        cbar = fig.colorbar(sc, ax=ax, location='top', shrink=0.9, pad=0.01)
                        cbar.ax.tick_params(labelsize=8)
                
                # Layout
                ax.set_xlim(xlims)
                ax.set_ylim(ylims)
                ax.set_xticks([])
                ax.set_yticks([])
                
                if col_idx == 0:
                    ax.set_ylabel(f"{survey_name.upper()}\n(N={len(subset_df)})", fontsize=12, fontweight='bold')
                
                if row_idx == 0:
                    ax.set_title(config["label"], fontsize=14, fontweight='bold', pad=25)

        fig.suptitle(f"AstroPT Latent Space - {modality_name.upper()} - {set_name}", fontsize=20, y=1.02)
        out_name = f"umap_{modality_name}_{set_name.lower()}.png"
        out_path = os.path.join(save_dir, out_name)
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        logger.info(f"Saved: {out_path}")
        plt.close()


def main():
    args = parse_args()
    
    # 1. Output Setup
    save_dir = args.output_dir if args.output_dir else os.path.dirname(args.embeddings_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. Load
    raw_data, df = load_data(args.embeddings_path, args.metadata_path)
    df_aligned = align_data(raw_data, df)
    
    # Subsample
    if args.subsample and len(df_aligned) > args.subsample:
        logger.info(f"Aligning dataframe to subsample size {args.subsample}...")
        df_aligned = df_aligned.iloc[:args.subsample]
    
    # 3. Process Modalities
    for mod in ['images', 'spectra', 'joint']:
        if mod not in raw_data: continue
        
        logger.info(f"=== Processing {mod.upper()} ===")
        emb = raw_data[mod]
        
        if len(emb) > len(df_aligned):
            emb = emb[:len(df_aligned)]
            
        umap_2d = compute_umap(emb, subsample=None)
        
        plot_dissected_grid(umap_2d, df_aligned, mod, save_dir)
        
    logger.info("Done.")

if __name__ == "__main__":
    main()