"""
AstroPT Anomaly Hunter.

This script uses the pre-computed latent space (embeddings) to identify
astrophysical anomalies. It applies an Isolation Forest algorithm to score
every galaxy by its 'weirdness' and visualizes the top outliers.

This is a key step for 'Scientific Discovery': finding objects that do not
conform to the standard distribution (e.g., mergers, lenses, artifacts).

Author: Victor Alonso Rodriguez
Date: January 2026
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader, Subset

# Importar tus utilidades (Asegúrate de que astropt está en el path)
from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow

# --- Logging Configuration ---
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-Anomalies")

# --- Plotting Config ---
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rc('text', usetex=True)
plt.rc('font', family='serif', weight='bold')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AstroPT Anomaly Detection")
    
    parser.add_argument("--embeddings_path", type=str, required=True, help="Path to .npz embeddings")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory of Arrow data")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to .fits catalog (for info)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    
    parser.add_argument("--n_anomalies", type=int, default=10, help="Number of outliers to plot")
    parser.add_argument("--contamination", type=float, default=0.01, help="Expected fraction of anomalies (approx)")
    parser.add_argument("--modality", type=str, default="joint", choices=['images', 'spectra', 'joint'],
                        help="Which embedding space to use for outlier detection")
    
    return parser.parse_args()


def load_embeddings(path: str, modality: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads embeddings and returns the selected modality + TargetIDs."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings not found: {path}")
    
    data = np.load(path)
    if modality not in data:
        raise KeyError(f"Modality '{modality}' not found in {path}")
        
    embeddings = data[modality]
    target_ids = data['targetid']
    
    logger.info(f"Loaded {len(embeddings)} embeddings (Modality: {modality})")
    return embeddings, target_ids


def find_outliers(embeddings: np.ndarray, contamination: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trains Isolation Forest and returns anomaly scores and indices of outliers.
    Score: Lower is more anomalous.
    """
    logger.info(f"Training Isolation Forest (Contamination: {contamination})...")
    
    # n_jobs=-1 uses all CPUs
    clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    preds = clf.fit_predict(embeddings) # 1 for inliers, -1 for outliers
    scores = clf.decision_function(embeddings) # Lower scores = more anomalous
    
    # Get indices of the most anomalous objects (lowest scores)
    # Sort indices by score ascending
    sorted_indices = np.argsort(scores)
    
    return sorted_indices, scores


def make_rgb(image_tensor: np.ndarray) -> np.ndarray:
    """Helper to create RGB from tensor (4, H, W) -> R=H, G=(J+Y)/2, B=VIS"""
    # Assuming standard order: 0=VIS, 1=H, 2=J, 3=Y (Check your dataloader!)
    # Based on previous scripts: 
    vis = image_tensor[0]
    h = image_tensor[1]
    j = image_tensor[2]
    y = image_tensor[3] if image_tensor.shape[0] > 3 else j
    
    green = (j + y) / 2.0
    rgb = np.stack([h, green, vis], axis=-1)
    
    # Robust Norm
    for i in range(3):
        ch = rgb[:, :, i]
        vmin, vmax = np.percentile(ch, 1), np.percentile(ch, 99)
        rgb[:, :, i] = np.clip((ch - vmin) / (vmax - vmin + 1e-8), 0, 1)
        
    return rgb


def plot_anomaly(ax_img, ax_spec, batch_data, score, rank):
    """Plots a single anomaly entry."""
    target_id = batch_data['targetid'].item()
    
    # 1. Image
    if 'images' in batch_data:
        img_tensor = batch_data['images'][0].numpy() # (Seq, Patch)
        # We need to unpatchify. Since we don't have the model config here easily,
        # we will use a simplified unpatchify assuming standard AstroPT (4 channels, 14x14 grid)
        # If this fails, we skip image plotting or need to copy unpatchify logic.
        
        # --- QUICK UNPATCHIFY (Copy-Paste from previous logic for robustness) ---
        seq_len, patch_dim = img_tensor.shape
        channels = 4
        patch_size = int(np.sqrt(patch_dim // channels))
        grid_side = int(np.sqrt(seq_len))
        
        # Spiral Layout Logic
        layout = np.arange(grid_side**2).reshape(grid_side, grid_side)
        spiral_indices = []
        while layout.size > 0:
            spiral_indices.append(layout[0]); layout = layout[1:]
            if layout.size==0: break
            spiral_indices.append(layout[:, -1]); layout = layout[:, :-1]
            if layout.size==0: break
            spiral_indices.append(layout[-1][::-1]); layout = layout[:-1]
            if layout.size==0: break
            spiral_indices.append(layout[:, 0][::-1]); layout = layout[:, 1:]
        
        spiral_order = np.concatenate(spiral_indices)
        raster_idx = np.empty(grid_side**2, dtype=int)
        raster_idx[spiral_order] = np.arange(grid_side**2)
        raster_idx = (grid_side**2 - 1) - raster_idx # Center 0 fix
        
        raster = img_tensor[raster_idx]
        grid = raster.reshape(grid_side, grid_side, patch_size, patch_size, channels)
        img = np.einsum('hwpqc->chpwq', grid).reshape(channels, grid_side*patch_size, grid_side*patch_size)
        
        rgb = make_rgb(img)
        ax_img.imshow(rgb, origin='lower')
        ax_img.set_title(f"Rank #{rank} (ID: {target_id})", fontsize=14, color='darkred')
        ax_img.text(0.05, 0.9, f"Score: {score:.3f}", transform=ax_img.transAxes, color='white', fontweight='bold')
    
    ax_img.axis('off')
    
    # 2. Spectrum
    if 'spectra' in batch_data:
        spec = batch_data['spectra'][0].numpy().flatten()
        
        # Reconstruct wavelengths roughly
        if 'spectra_positions' in batch_data:
            pos = batch_data['spectra_positions'][0].numpy().flatten()
            wave = pos * 7000.0 + 3000.0
        else:
            wave = np.linspace(3000, 10000, len(spec))
            
        ax_spec.plot(wave, spec, color='black', lw=0.8)
        ax_spec.set_ylabel("Flux")
        ax_spec.set_yticks([]) # Clean look
        ax_spec.grid(True, alpha=0.3)
        if rank == 1:
            ax_spec.set_title("Spectrum", fontsize=10)


def main():
    args = parse_args()
    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.embeddings_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Embeddings
    X, all_ids = load_embeddings(args.embeddings_path, args.modality)
    
    # 2. Detect Anomalies
    sorted_idx, scores = find_outliers(X, args.contamination)
    
    # Get top N anomalies
    top_indices_local = sorted_idx[:args.n_anomalies]
    top_scores = scores[top_indices_local]
    top_target_ids = all_ids[top_indices_local]
    
    logger.info(f"Top 3 Anomalous IDs: {top_target_ids[:3]}")
    
    # 3. Load Real Data for Visualization
    # We need the dataloader to fetch the actual images/spectra of these IDs
    # Note: We use 'test' split assuming embeddings came from test. 
    # If embeddings are mixed, this might miss if ID is in train. 
    # Ideally, dataset should look at all files or match the embedding source.
    
    logger.info("Initializing Dataset to fetch real data...")
    # Minimal transform just to get tensors
    ds = EuclidDESIDatasetArrow(
        arrow_folder_root=args.data_dir,
        split="test", # Assuming embeddings were extracted from test
        modality_registry=None, # Not needed for raw fetch if logic handles it, but safer to pass None or minimal
        spiral=True
    )
    
    # Map IDs to Dataset Indices
    # This can be slow for huge datasets. We optimize by checking only top N.
    # Note: ds.ds['targetid'] loads column. 
    logger.info("Matching IDs to Dataset Indices...")
    ds_ids = np.array(ds.ds['targetid'])
    
    plot_indices = []
    found_scores = []
    found_ranks = []
    
    for rank, (tid, score) in enumerate(zip(top_target_ids, top_scores)):
        matches = np.where(ds_ids == tid)[0]
        if len(matches) > 0:
            plot_indices.append(matches[0])
            found_scores.append(score)
            found_ranks.append(rank + 1)
        else:
            logger.warning(f"ID {tid} (Rank {rank+1}) not found in Test Split. Skipping plot.")

    if not plot_indices:
        logger.error("No anomalous IDs found in the dataset files provided.")
        return

    # 4. Plotting
    logger.info(f"Generating summary plot for {len(plot_indices)} anomalies...")
    
    # Layout: One column per anomaly (Image on top, Spectrum below)
    # We split into pages if too many
    
    n_plot = len(plot_indices)
    fig = plt.figure(figsize=(n_plot * 4, 6))
    gs = gridspec.GridSpec(2, n_plot, height_ratios=[1, 0.6])
    
    loader = DataLoader(Subset(ds, plot_indices), batch_size=1, shuffle=False)
    
    for i, batch in enumerate(loader):
        ax_img = fig.add_subplot(gs[0, i])
        ax_spec = fig.add_subplot(gs[1, i])
        
        plot_anomaly(ax_img, ax_spec, batch, found_scores[i], found_ranks[i])
        
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"anomalies_top{args.n_anomalies}_{args.modality}.pdf")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved Anomaly Report to: {save_path}")
    
    # Save CSV list
    df_out = pd.DataFrame({
        "Rank": found_ranks,
        "TargetID": [ds_ids[i] for i in plot_indices],
        "Score": found_scores
    })
    csv_path = os.path.join(output_dir, "anomaly_list.csv")
    df_out.to_csv(csv_path, index=False)
    logger.info(f"Saved list to {csv_path}")

if __name__ == "__main__":
    main()