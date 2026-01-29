"""
AstroPT UMAP Visualizer.

This script generates UMAP visualizations split into 4 rows (one per Survey: Main, SV1, SV3, Special)
to allow detailed inspection of domain shifts and batch effects.

It plots two sets of properties to diagnose both physical learning and instrumental biases:
- Set 1: Physical Properties & Data Release (Is the model learning physics or instrument modes?)
- Set 2: Spectral Lines & Quality (Is the model sensitive to chemistry and noise?)

Author: Victor Alonso Rodriguez
Date: January 2026
"""

import argparse
import glob
import json
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
from matplotlib.lines import Line2D

# Logger Configuration
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-UMAP")

# Suppress warnings
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.filterwarnings('ignore', module='astropy.io.fits')

# Plotting Global Configuration
plt.rcParams['text.latex.preamble'] = r'''
            \usepackage{siunitx}
            \usepackage{bm}
            \usepackage{amsmath} 
            \sisetup{detect-family, separate-uncertainty=true, output-decimal-marker={.}}
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

# PLOT Constants
UMAP_PARAMS = {
    "n_neighbors": 30,
    "min_dist": 0.1,
    "n_components": 2,
    "metric": "cosine",
    "random_state": 42
}

SURVEY_ORDER = ['main', 'sv1', 'sv3', 'special']

CAT_PALETTE = {
    "GALAXY": "#006eff",
    "QSO": "#fbd500",
    "DR1_R1": "#1b9e77",
    "DR1_R2": "#e7298a",
    "Other":  "#666666"
}


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="AstroPT UMAP Generator")
    
    parser.add_argument("--out_dir", type=str, default=".", help="Output directory")
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Directory containing .npy embedding files")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to .fits catalog")
    parser.add_argument("--subsample", type=int, default=None, help="Limit points for speed")
    parser.add_argument("--train_name", type=str, default=None, help="Custom title for the plot (defaults to folder name)")
    
    return parser.parse_args()


def load_data(embeddings_dir: str, metadata_path: str) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """
    Loads .npy embeddings from a directory and .fits metadata catalog.
    Scans for 'images.npy', 'spectra.npy', 'joint.npy' and 'targetid.npy'.

    Args:
        embeddings_dir: Path to the directory containing embedding files.
        metadata_path: Path to the FITS catalog file.

    Returns:
        Tuple containing a dictionary of embeddings and the metadata DataFrame.
    """
    logger.info(f"Scanning embeddings directory: {embeddings_dir}...")
    if not os.path.exists(embeddings_dir):
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")
    
    data_dict = {}
    
    # Load Target IDs
    id_candidates = ['targetid.npy', 'ids.npy', 'target_ids.npy', 'object_ids.npy']
    id_path = None
    for cand in id_candidates:
        p = os.path.join(embeddings_dir, cand)
        if os.path.exists(p):
            id_path = p
            break
            
    if id_path:
        logger.info(f"Loading Target IDs from {os.path.basename(id_path)}...")
        data_dict['targetid'] = np.load(id_path)
    else:
        raise FileNotFoundError(f"Could not find IDs ({id_candidates}) in {embeddings_dir}")

    # Load Modalities (images, spectra, joint)
    modalities = ['images', 'spectra', 'joint']
    for mod in modalities:
        # Search for pattern mod*.npy
        candidates = glob.glob(os.path.join(embeddings_dir, f"{mod}*.npy"))
        
        if candidates:
            # Prefer exact name "images.npy"
            best_cand = candidates[0] 
            for c in candidates:
                if os.path.basename(c) == f"{mod}.npy":
                    best_cand = c
                    break
            
            logger.info(f"Loading {mod} embeddings from {os.path.basename(best_cand)}...")
            data_dict[mod] = np.load(best_cand)
        else:
            logger.warning(f"No embeddings found for modality: {mod}")

    # Load Metadata
    logger.info(f"Loading metadata from {metadata_path}...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    catalog = Table.read(metadata_path)
    df = catalog.to_pandas()
    
    # Basic string cleaning
    for col in df.columns:
        if df[col].dtype == object and isinstance(df[col].iloc[0], bytes):
            try: df[col] = df[col].str.decode('utf-8')
            except: pass
    if 'SURVEY' in df.columns:
        df['SURVEY'] = df['SURVEY'].str.lower().str.strip()
        
    return data_dict, df


def align_data(embeddings_dict: Dict[str, np.ndarray], catalog_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Aligns metadata with embeddings using TARGETID intersection.

    Args:
        embeddings_dict: Dictionary containing embeddings and target IDs.
        catalog_df: Pandas DataFrame containing metadata.

    Returns:
        Tuple containing the aligned DataFrame and valid embedding indices.
    """
    logger.info("Aligning embeddings with metadata...")
    emb_ids = embeddings_dict['targetid']
    
    # Prepare catalog
    catalog_df['TARGETID'] = catalog_df['TARGETID'].astype('int64')
    catalog_indexed = catalog_df.drop_duplicates(subset=['TARGETID']).set_index('TARGETID')
    
    # Intersection
    common_ids = np.intersect1d(emb_ids, catalog_indexed.index.values)
    
    if len(common_ids) < len(emb_ids):
        logger.warning(f"Dropping {len(emb_ids) - len(common_ids)} objects (missing in metadata).")
    
    # Align Catalog
    matched_catalog = catalog_indexed.loc[common_ids].reset_index()
    
    # Align Embeddings (Recover original indices)
    id_to_idx = {id_val: i for i, id_val in enumerate(emb_ids)}
    valid_indices = np.array([id_to_idx[uid] for uid in matched_catalog['TARGETID'].values])
    
    logger.info(f"Final aligned samples: {len(matched_catalog)}")
    return matched_catalog, valid_indices


def compute_umap(embeddings: np.ndarray, subsample: Optional[int] = None) -> np.ndarray:
    """
    Computes UMAP reduction.

    Args:
        embeddings: Input high-dimensional data.
        subsample: Optional limit on number of points for speed.

    Returns:
        2D UMAP projection.
    """
    if subsample and len(embeddings) > subsample:
        logger.info(f"Subsampling to {subsample} for UMAP calc...")
        idx = np.random.choice(len(embeddings), subsample, replace=False)
        data = embeddings[idx]
    else:
        data = embeddings
        
    logger.info(f"Running UMAP on {data.shape}...")
    reducer = umap.UMAP(**UMAP_PARAMS)
    return reducer.fit_transform(data)


def plot_umaps_grid(
    umap_2d: np.ndarray, 
    df: pd.DataFrame, 
    modality_name: str, 
    out_dir: str, 
    custom_title: Optional[str] = None
):
    """
    Generates the grid plots with DYNAMIC percentiles, SMART Z-ORDERING and CUSTOM PALETTES.

    Args:
        umap_2d: 2D embedding coordinates.
        df: Aligned metadata DataFrame.
        modality_name: Name of the modality (e.g., 'images').
        out_dir: Directory to save plots.
        custom_title: Optional override for the plot title.
    """
    
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
        {"col": "SNR_SPEC_R", "label": r"SNR (R-Band)", "cat": False, "trans": None},
        {"col": "HALPHA_FLUX", "label": r"H$\alpha$ Flux ($\log$)", "cat": False, "trans": log_transform},
        {"col": "OIII_5007_FLUX", "label": r"[OIII] Flux ($\log$)", "cat": False, "trans": log_transform},
        {"col": "OII_3726_FLUX", "label": r"[OII] Flux ($\log$)", "cat": False, "trans": log_transform},
        {"col": "NII_6584_FLUX", "label": r"[NII] Flux ($\log$)", "cat": False, "trans": log_transform},
    ]
    
    sets_to_plot = [("Physics", set_1_config), ("Chemistry", set_2_config)]
    
    # Dynamic Limits (P1 - P99) ---
    global_limits = {}
    for config in set_1_config + set_2_config:
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
        
        # TITLE LOGIC
        config_path = os.path.join(out_dir, "config.json")
        json_name = None
        
        # Try reading config.json
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    json_name = config.get("train_name", None)
            except Exception:
                pass 

        # Select ID: CLI > JSON > Folder
        if custom_title:
            run_id = custom_title
        elif json_name:
            run_id = json_name
        else:
            run_id = os.path.basename(os.path.normpath(out_dir))

        title_text = (
            rf"\textbf{{AstroPT Latent Space - {modality_name.upper()} - {set_name}}}" + "\n" +
            f"[{run_id}]"
        ) 

        fig.suptitle(title_text, fontsize=24, y=1.05)
        
        out_name = f"umap_{modality_name}_{set_name.lower()}.png"
        out_path = os.path.join(out_dir, out_name)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        logger.info(f"-> Saved: {out_path}")
        plt.close()


def main():
    args = parse_args()
    if args.out_dir and not os.path.exists(args.out_dir): os.makedirs(args.out_dir)
    
    raw_data, df = load_data(args.embeddings_dir, args.metadata_path)
    df_aligned, valid_emb_indices = align_data(raw_data, df)
    
    # Global Subsampling
    if args.subsample and len(df_aligned) > args.subsample:
        logger.info(f"Randomly subsampling to {args.subsample} points...")
        rng = np.random.default_rng(42)
        subset_idx = rng.choice(len(df_aligned), args.subsample, replace=False)
        subset_idx.sort()
        df_aligned = df_aligned.iloc[subset_idx].reset_index(drop=True)
        final_indices = valid_emb_indices[subset_idx]
    else:
        final_indices = valid_emb_indices
    
    found_modalities = [k for k in raw_data.keys() if k in ['images', 'spectra', 'joint']]
    for mod in found_modalities:
        logger.info("-" * 40)
        logger.info(f"Processing {mod.upper()}")
        emb = raw_data[mod]
        if len(emb) > len(final_indices): emb = emb[final_indices]
        
        umap_2d = compute_umap(emb, subsample=None)
        
        plot_umaps_grid(
            umap_2d, 
            df_aligned, 
            mod, 
            args.out_dir if args.out_dir else ".", 
            custom_title=args.train_name
        )
        
    logger.info("Done.")

if __name__ == "__main__":
    main()