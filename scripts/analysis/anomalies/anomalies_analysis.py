"""
AstroPT Multimodal Anomaly Hunter (Consolidated Rigorous Framework).

This script performs advanced astrophysical anomaly detection on pre-computed
AstroPT latent spaces. It combines classical geometric, density-based, and 
deep reconstruction methods to calculate a unified weirdness rank for each galaxy.

Available Anomaly Methods:
1. Cross-Modal Cosine Gap: Discrepancy between image and spectal expert tokens in CLIP space (perfect for dual AGNs, mergers).
2. Isolation Forest: Density isolation on the joint representation.
3. Local Outlier Factor (LOF): Local density clustering deviation.
4. Deep SVDD (Hypersphere Distance): L2 distance from robust distribution centroid.
5. Mahalanobis Distance: Outlier scoring normalized by robust covariance structure.
6. Bottleneck Autoencoder (MLP): PyTorch autoencoder reconstruction error.
7. KNN Distance: Average distance to k-nearest neighbors in latent space.

Author: Victor Alonso Rodriguez
Date: May 2026
"""

import argparse
import json
import logging
import os
import sys
import gc
import re
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import fields

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from astropy.table import Table
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.manifold import TSNE

from astropt.dataloader_multimodal import MultimodalDatasetArrow
from astropt.training_utils import create_dataloaders
from astropt.config import TrainingConfig
from astropt.model_utils import load_local_model

# --- Configure logging ---
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-AnomalyHunter")

def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="AstroPT Multimodal Anomaly Hunter")
    
    parser.add_argument("--embeddings_dir", type=str, required=True, 
                        help="Directory containing pre-computed embeddings (.npy)")
    parser.add_argument("--ckpt_path", type=str, required=True, 
                        help="Path to the training checkpoint (.pt)")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Root folder of raw Arrow processed data")
    parser.add_argument("--metadata_path", type=str, required=True, 
                        help="Path to FITS metadata catalog")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Saving directory for anomaly catalogs and plots")
    
    parser.add_argument("--n_anomalies", type=int, default=10, 
                        help="Number of top anomalies to extract and plot")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for dataloader")
    parser.add_argument("--contamination", type=float, default=0.01,  
                        help="Expected fraction of outliers for IF/LOF algorithms")
    parser.add_argument("--knn_neighbors", type=int, default=20, 
                        help="Neighbors count for KNN and LOF models")
    parser.add_argument("--ae_epochs", type=int, default=25, 
                        help="Training epochs for PyTorch Bottleneck Autoencoder")
    parser.add_argument("--ae_latent_dim", type=int, default=16, 
                        help="Latent bottleneck dimension of the Autoencoder")
    
    parser.add_argument("--methods", nargs="+", default=["gap", "iforest", "lof", "svdd", "mahalanobis", "autoencoder", "knn"],
                        help="Anomaly detection methods to include")
    parser.add_argument("--base_modality", type=str, default="joint", 
                        choices=["joint", "cls", "EuclidImage", "DESISpectrum"],
                        help="Base embedding modality for general outlier detectors")
    parser.add_argument("--plot_projection", action="store_true", default=True,
                        help="Generate a 2D t-SNE plot highlighting anomaly locations")
    
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Deep Bottleneck Autoencoder (PyTorch)
# ---------------------------------------------------------------------------

class BottleneckAutoencoder(nn.Module):
    """Simple Bottleneck Autoencoder for reconstruction-based anomaly scoring."""
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

def train_autoencoder(X: np.ndarray, latent_dim: int, epochs: int, batch_size: int = 128) -> np.ndarray:
    """Trains the Autoencoder and returns MSE reconstruction loss per sample."""
    logger.info(f"Training Bottleneck Autoencoder on {X.shape[0]} samples (dim: {X.shape[1]} -> {latent_dim})...")
    
    # Scale input data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BottleneckAutoencoder(X.shape[1], latent_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss(reduction="none")
    
    tensor_X = torch.FloatTensor(X_scaled)
    dataset = torch.utils.data.TensorDataset(tensor_X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            xb = batch[0].to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, xb).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  [AE Epoch {epoch+1:02d}/{epochs:02d}] Reconstruction Loss: {epoch_loss / len(dataset):.4f}")
            
    # Compute reconstruction error per sample
    model.eval()
    all_errors = []
    with torch.no_grad():
        loader_eval = DataLoader(dataset, batch_size=batch_size * 2, shuffle=False)
        for batch in loader_eval:
            xb = batch[0].to(device)
            out = model(xb)
            # MSE error per sample (mean across features)
            errors = criterion(out, xb).mean(dim=1).cpu().numpy()
            all_errors.append(errors)
            
    return np.concatenate(all_errors)


# ---------------------------------------------------------------------------
# Spiral Image Unpatchification Helper
# ---------------------------------------------------------------------------

def unpatchify_spiral_image(img_tensor: np.ndarray, channels: int = 4) -> np.ndarray:
    """
    Restores the original 2D image from a 1D spiralized patch sequence.
    Handles general AstroPT image patch shapes.
    
    Args:
        img_tensor: Shape (seq_len, patch_dim)
        channels: Expected filters count (e.g., VIS, Y, J, H = 4)
        
    Returns:
        RGB-compatible array of shape (height, width, 3) normalized in [0, 1]
    """
    seq_len, patch_dim = img_tensor.shape
    patch_size = int(np.sqrt(patch_dim // channels))
    grid_side = int(np.sqrt(seq_len))
    
    # 1. Reconstruct the center-outwards spiral indexing mapping
    layout = np.arange(grid_side * grid_side).reshape(grid_side, grid_side)
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
    
    # Invert the spiral indices
    inv_spiral_indices = np.argsort(spiral_order)
    
    # Rearrange patches to standard raster order
    raster_patches = img_tensor[inv_spiral_indices]
    
    # Rearrange shape: (grid_side, grid_side, patch_size, patch_size, channels)
    grid = raster_patches.reshape(grid_side, grid_side, patch_size, patch_size, channels)
    # Transpose and reshape back to (channels, height, width)
    img_reconstructed = np.einsum('hwpqc->chpwq', grid).reshape(channels, grid_side*patch_size, grid_side*patch_size)
    
    # 2. Extract Channels for False RGB
    # Mapping VIS -> blue, Average(J, Y) -> green, H -> red
    vis = img_reconstructed[0]
    h = img_reconstructed[1]
    j = img_reconstructed[2]
    y = img_reconstructed[3] if img_reconstructed.shape[0] > 3 else j
    
    green = (j + y) / 2.0
    rgb = np.stack([h, green, vis], axis=-1)
    
    # Apply robust standardization (percentile clipping) to wow the observer
    for i in range(3):
        ch = rgb[:, :, i]
        vmin, vmax = np.percentile(ch, 1), np.percentile(ch, 99)
        diff = vmax - vmin
        rgb[:, :, i] = np.clip((ch - vmin) / (diff if diff > 1e-8 else 1.0), 0, 1)
        
    return rgb


# ---------------------------------------------------------------------------
# Anomaly Scoring & Analysis Engine
# ---------------------------------------------------------------------------

def get_allowed_ids_from_filters(fits_df: pd.DataFrame, applied_filters: List[str]) -> set:
    """Evaluates applied_filters on FITS dataframe and returns a set of allowed TargetIDs."""
    if not applied_filters:
        return set(fits_df['TARGETID'].unique())
        
    mask = np.ones(len(fits_df), dtype=bool)
    for expr in applied_filters:
        py_expr = expr.replace('&&', '&').replace('||', '|')
        found_words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr)
        eval_namespace = {}
        for word in found_words:
            if word in fits_df.columns:
                eval_namespace[word] = fits_df[word].values
        
        if 'Z' not in eval_namespace and 'redshift' in fits_df.columns:
            eval_namespace['Z'] = fits_df['redshift'].values
        elif 'redshift' not in eval_namespace and 'Z' in fits_df.columns:
            eval_namespace['redshift'] = fits_df['Z'].values
            
        try:
            expr_mask = eval(py_expr, {"np": np}, eval_namespace)
            mask &= np.array(expr_mask, dtype=bool)
        except Exception as e:
            logger.error(f"Error evaluating filter '{expr}' on FITS catalog: {e}")
            
    # Resolve the ID column
    id_col = None
    for candidate in ['TARGETID', 'targetid', 'ID', 'id', 'OBJID', 'objid']:
        if candidate in fits_df.columns:
            id_col = candidate
            break
            
    if id_col is None:
        raise KeyError(f"Could not find ID column in fits catalog. Columns: {fits_df.columns}")
        
    allowed_ids = fits_df.loc[mask, id_col].values.astype(int)
    logger.info(f"FITS filter matched {len(allowed_ids)} / {len(fits_df)} records.")
    return set(allowed_ids)


def calculate_anomaly_scores(
    embeddings_dir: Path, 
    methods: List[str], 
    base_modality: str,
    contamination: float,
    knn_neighbors: int,
    ae_epochs: int,
    ae_latent_dim: int,
    allowed_ids_filter: set = None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Loads all embeddings, runs anomaly detectors, and returns standardized scores."""
    logger.info("Loading pre-computed embeddings...")
    
    emb_files = {}
    for f in embeddings_dir.glob("*.npy"):
        emb_files[f.stem] = f
        
    if "ids" not in emb_files:
        raise FileNotFoundError(f"ids.npy not found in {embeddings_dir}")
        
    ids = np.load(embeddings_dir / "ids.npy")
    
    keep_mask = None
    if allowed_ids_filter is not None:
        keep_mask = np.isin(ids.astype(int), list(allowed_ids_filter))
        num_kept = np.sum(keep_mask)
        logger.info(f"Filtering embeddings: keeping {num_kept} / {len(ids)} sources matching dynamic metadata filters.")
        if num_kept == 0:
            logger.warning("Zero sources match dynamic filters! Disabling filter to prevent empty array errors.")
            keep_mask = None
            
    if keep_mask is not None:
        ids = ids[keep_mask]
        
    df_scores = pd.DataFrame({"TargetID": ids})
    
    # Dictionary to keep active 2D matrices for t-SNE / further analysis
    feature_matrices = {}
    
    # --- 1. Cross-Modal Cosine Gap ---
    if "gap" in methods:
        # Resolve Phase 1 expert tokens (pure unimodal boundary)
        img_key = "EuclidImage_phase1"
        spec_key = "DESISpectrum_phase1"
        
        # Fallback to general phase 2 or unpooled if phase 1 is missing
        if img_key not in emb_files:
            img_key = "EuclidImage"
        if spec_key not in emb_files:
            spec_key = "DESISpectrum"
            
        if img_key in emb_files and spec_key in emb_files:
            logger.info(f"Computing Cross-Modal Gap using '{img_key}' vs '{spec_key}'...")
            z_img = np.load(embeddings_dir / f"{img_key}.npy")
            z_spec = np.load(embeddings_dir / f"{spec_key}.npy")
            
            if keep_mask is not None:
                z_img = z_img[keep_mask]
                z_spec = z_spec[keep_mask]
            
            # Squeeze if they contain sequence dimensions (e.g. mean pooling on sequence if needed)
            if z_img.ndim == 3: z_img = z_img.mean(axis=1)
            if z_spec.ndim == 3: z_spec = z_spec.mean(axis=1)
            
            # L2 Normalization
            z_img_norm = z_img / (np.linalg.norm(z_img, axis=1, keepdims=True) + 1e-8)
            z_spec_norm = z_spec / (np.linalg.norm(z_spec, axis=1, keepdims=True) + 1e-8)
            
            # Cosine gap = 1 - cosine similarity
            cosine_gap = 1.0 - np.sum(z_img_norm * z_spec_norm, axis=1)
            df_scores["score_gap"] = cosine_gap
        else:
            logger.warning(f"Could not compute Cross-Modal Gap. Missing EuclidImage or DESISpectrum.")
            
    # Load base modality for other algorithms
    if base_modality not in emb_files:
        raise KeyError(f"Selected base modality '{base_modality}' not found in embeddings directory.")
        
    X_base = np.load(embeddings_dir / f"{base_modality}.npy")
    if keep_mask is not None:
        X_base = X_base[keep_mask]
        
    if X_base.ndim == 3:
        logger.info(f"Base modality '{base_modality}' contains sequence tokens. Applying mean pooling for anomaly algorithms.")
        X_base = X_base.mean(axis=1)
        
    feature_matrices[base_modality] = X_base
    
    # --- 2. Isolation Forest ---
    if "iforest" in methods:
        logger.info("Running Isolation Forest...")
        clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        clf.fit(X_base)
        # Decision function: lower values mean more anomalous. We invert it so higher = anomalous.
        df_scores["score_iforest"] = -clf.decision_function(X_base)
        
    # --- 3. Local Outlier Factor ---
    if "lof" in methods:
        logger.info(f"Running Local Outlier Factor (k={knn_neighbors})...")
        clf = LocalOutlierFactor(n_neighbors=knn_neighbors, contamination=contamination, novelty=True, n_jobs=-1)
        clf.fit(X_base)
        # Invert score so higher = anomalous
        df_scores["score_lof"] = -clf.decision_function(X_base)
        
    # --- 4. Deep SVDD / Centroid Distance ---
    if "svdd" in methods:
        logger.info("Running Deep SVDD Hypersphere distance...")
        # Compute robust centroid of distribution
        c = np.median(X_base, axis=0)
        # Score is L2 distance to centroid
        svdd_dist = np.linalg.norm(X_base - c, axis=1)**2
        df_scores["score_svdd"] = svdd_dist
        
    # --- 5. Mahalanobis Distance ---
    if "mahalanobis" in methods:
        logger.info("Running Mahalanobis Distance (with Ledoit-Wolf Shrinkage)...")
        try:
            cov = LedoitWolf().fit(X_base)
            df_scores["score_mahalanobis"] = cov.mahalanobis(X_base)
        except Exception as e:
            logger.error(f"Mahalanobis computation failed: {e}. Falling back to standard covariance.")
            cov_emp = np.cov(X_base.T)
            mean_emp = np.mean(X_base, axis=0)
            inv_cov = np.linalg.pinv(cov_emp)
            dists = []
            for xi in X_base:
                diff = xi - mean_emp
                dists.append(diff @ inv_cov @ diff.T)
            df_scores["score_mahalanobis"] = np.sqrt(np.array(dists))
            
    # --- 6. Bottleneck Autoencoder (Deep Reconstruction) ---
    if "autoencoder" in methods:
        df_scores["score_autoencoder"] = train_autoencoder(
            X_base, latent_dim=ae_latent_dim, epochs=ae_epochs
        )
        
    # --- 7. KNN Distance ---
    if "knn" in methods:
        logger.info(f"Running KNN Outlier detector (k={knn_neighbors})...")
        neigh = NearestNeighbors(n_neighbors=knn_neighbors, n_jobs=-1)
        neigh.fit(X_base)
        distances, _ = neigh.kneighbors(X_base)
        # Outlier score is average distance to neighbors
        df_scores["score_knn"] = distances.mean(axis=1)
        
    # -----------------------------------------------------------------------
    # Score Standardization & Unified Weirdness Index (UWI)
    # -----------------------------------------------------------------------
    score_cols = [c for c in df_scores.columns if c.startswith("score_")]
    if not score_cols:
        raise RuntimeError("No anomaly scoring methods completed successfully.")
        
    # For each method, convert scores to percentile ranks [0.0 to 1.0] (1.0 = most anomalous)
    logger.info("Standardizing scores and calculating Unified Weirdness Index (UWI)...")
    for col in score_cols:
        raw_vals = df_scores[col].values
        # Sort and map rank
        ranks = np.argsort(np.argsort(raw_vals)) / (len(raw_vals) - 1.0)
        df_scores[f"percentile_{col.replace('score_', '')}"] = ranks
        
    percentile_cols = [c for c in df_scores.columns if c.startswith("percentile_")]
    # Compute UWI as the consensus average percentile across all enabled methods
    df_scores["Unified_Weirdness_Index"] = df_scores[percentile_cols].mean(axis=1)
    
    # Sort dataset by weirdness index descending
    df_scores = df_scores.sort_values(by="Unified_Weirdness_Index", ascending=False).reset_index(drop=True)
    
    return df_scores, feature_matrices


# ---------------------------------------------------------------------------
# Visualization & Scientific Dashboarding
# ---------------------------------------------------------------------------

def generate_tSNE_plot(
    X: np.ndarray, 
    top_indices: np.ndarray, 
    uwi_scores: np.ndarray,
    save_path: Path
):
    """Generates a 2D t-SNE plot highlighting where top anomalies lie in latent space."""
    logger.info("Generating t-SNE projection for visualization...")
    
    # Downsample background points for plotting speed if dataset is huge
    max_bg = 15000
    n_samples = X.shape[0]
    
    if n_samples > max_bg:
        indices_bg = np.random.choice(n_samples, max_bg, replace=False)
        # Ensure top anomalies are in the plotted indices
        all_plot_idx = list(set(indices_bg) | set(top_indices))
    else:
        all_plot_idx = np.arange(n_samples)
        
    X_plot = X[all_plot_idx]
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, n_jobs=-1)
    X_embedded = tsne.fit_transform(X_plot)
    
    # Map raw indices to plot indices
    plot_idx_map = {idx: i for i, idx in enumerate(all_plot_idx)}
    top_plot_idx = [plot_idx_map[idx] for idx in top_indices if idx in plot_idx_map]
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    # Plot background distribution
    sc = ax.scatter(
        X_embedded[:, 0], X_embedded[:, 1], 
        c="lightgray", s=10, alpha=0.5, label="Normal Distribution", rasterized=True
    )
    
    # Plot top anomalies highlighted by their UWI score
    sc_anom = ax.scatter(
        X_embedded[top_plot_idx, 0], X_embedded[top_plot_idx, 1],
        c=uwi_scores[:len(top_plot_idx)], cmap="YlOrRd", s=100, edgecolors="black", 
        linewidths=1.5, label="AstroPT Anomalies"
    )
    
    cb = plt.colorbar(sc_anom, ax=ax, label="Unified Weirdness Index (UWI)")
    cb.ax.tick_params(labelsize=11)
    
    # Label top 5 anomalies
    for i, idx in enumerate(top_plot_idx[:5]):
        ax.annotate(
            f"Rank #{i+1}", 
            (X_embedded[idx, 0], X_embedded[idx, 1]),
            textcoords="offset points", 
            xytext=(10, 10), 
            fontsize=10, 
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="black", lw=0.5)
        )
        
    ax.set_title("AstroPT Joint Latent Space Projection (t-SNE)", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.grid(True, alpha=0.15)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=11)
    
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved t-SNE plot to {save_path}")


MAIN_LINES = {
    r"Ly$\alpha$": 1216.0, r"C IV": 1549.0, "C III": 1908.7, r"Mg II": 2798.0, r"[O II]": 3727.3, r"[Ne III]": 3868.7,
    r"Ca K": 3933.7, r"Ca H": 3968.5, r"H$\delta$": 4102.0, r"H$\gamma$": 4341.0,
    r"H$\beta$": 4861.0, r"[O III]": 4959.0, r"[O III]": 5007.0, r"Mg I": 5175.0,
    r"Na D": 5890.0, r"[N II]": 6548.0, r"H$\alpha$": 6563.0, r"[N II]": 6583.5, r"[S II]": 6730.8
}

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

def make_rgb_lupton(image_tensor: np.ndarray, Q: float = 12.0, stretch: float = 0.5, m: float = 0.0) -> np.ndarray:
    """Lupton et al. (2004) algorithm implementation."""
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

def generate_anomaly_report_dashboard(
    ds: Any, 
    df_top: pd.DataFrame, 
    fits_catalog: pd.DataFrame, 
    save_path: Path
):
    """Generates a highly rigorous multi-page dashboard plotting images, spectra, and z-score radar-like bars."""
    logger.info("Generating multipanel anomaly report...")
    
    # Set up global LaTeX configuration once
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', weight='bold')
    
    n_plot = len(df_top)
    
    # Prepare FITS catalogs index for quick metadata search
    target_id_col = 'TARGETID' if 'TARGETID' in fits_catalog.columns else 'targetid'
    fits_indexed = fits_catalog.drop_duplicates(subset=[target_id_col]).set_index(target_id_col)
    
    import matplotlib
    matplotlib.use('Agg')
    
    from matplotlib.backends.backend_pdf import PdfPages
    
    with PdfPages(save_path) as pdf:
        for rank, row in enumerate(df_top.itertuples()):
            tid = row.TargetID
            uwi = row.Unified_Weirdness_Index
            
            # Retrieve processed batch entry from dataloader
            ds_ids = np.array(ds.ds['targetid'])
            matches = np.where(ds_ids == tid)[0]
            if len(matches) == 0:
                logger.warning(f"ID {tid} not found in active dataset split. Skipping dashboard sheet.")
                continue
                
            sample = ds[int(matches[0])]
            
            # Fetch FITS metadata
            meta_str = ""
            if tid in fits_indexed.index:
                meta = fits_indexed.loc[tid]
                
                # Redshift
                z_val = meta.get('Z', meta.get('z', None))
                z_str = f"{z_val:.4f}" if (z_val is not None and not pd.isna(z_val)) else "N/A"
                
                # Stellar Mass
                logmstar_val = meta.get('LOGMSTAR', meta.get('logmstar', None))
                mstar = f"$10^{{{logmstar_val:.2f}}} \\text{{ M}}_\\odot$" if (logmstar_val is not None and not pd.isna(logmstar_val)) else "N/A"
                
                # SFR
                logsfr_val = meta.get('LOGSFR', meta.get('logsfr', None))
                sfr = f"${10**logsfr_val:.3f} \\text{{ M}}_\\odot/\\text{{yr}}$" if (logsfr_val is not None and not pd.isna(logsfr_val)) else "N/A"
                
                # Sersic Radius
                sersic_r_val = meta.get('sersic_sersic_vis_radius', None)
                sersic_r = f"{sersic_r_val:.3f} arcsec" if (sersic_r_val is not None and not pd.isna(sersic_r_val)) else "N/A"
                
                # Sersic Index
                sersic_n_val = meta.get('sersic_sersic_vis_index', None)
                sersic_n = f"{sersic_n_val:.3f}" if (sersic_n_val is not None and not pd.isna(sersic_n_val)) else "N/A"
                
                # Gini
                gini_val = meta.get('gini', meta.get('GINI', None))
                gini = f"{gini_val:.3f}" if (gini_val is not None and not pd.isna(gini_val)) else "N/A"
                
                # Spectype & Release
                spectype = str(meta.get('SPECTYPE', meta.get('spectype', 'N/A')))
                release = str(meta.get('data_set_release', meta.get('release', 'N/A')))
                
                meta_str = (
                    f"\\textbf{{Spectroscopic Redshift (Z)}}: {z_str}\n"
                    f"\\textbf{{Stellar Mass ($M_*$)}}: {mstar}\n"
                    f"\\textbf{{Star Formation Rate (SFR)}}: {sfr}\n"
                    f"\\textbf{{Sersic VIS Radius}}: {sersic_r}\n"
                    f"\\textbf{{Sersic VIS Index}}: {sersic_n}\n"
                    f"\\textbf{{Gini Coefficient}}: {gini}\n"
                    f"\\textbf{{Galaxy Spectype}}: {spectype}\n"
                    f"\\textbf{{Survey Catalog Release}}: {release}"
                )
                # Escape underscores for LaTeX safety
                meta_str = meta_str.replace("_", "\\_")
            else:
                meta_str = f"\\textbf{{TargetID}}: {tid} (No metadata in catalog)"
                
            # Create a 4-panel figure for this anomaly with metadata, image, scores, and spectrum
            fig = plt.figure(figsize=(22, 14), dpi=150)
            
            gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], width_ratios=[1, 1.1, 0.9], hspace=0.4, wspace=0.3)
            
            ax_meta = fig.add_subplot(gs[0, 0])
            ax_img = fig.add_subplot(gs[0, 1])
            ax_stats = fig.add_subplot(gs[0, 2])
            ax_spec_blue = fig.add_subplot(gs[1, :])
            ax_spec_red = fig.add_subplot(gs[2, :])
            
            # Load the raw Arrow record from the dataset to ensure original physical scaling
            raw_record = ds.ds[int(matches[0])]
            
            # 1. Optical RGB Euclid Image using Lupton asinh scaling (VIS+NISP)
            has_image = False
            
            def get_raw(k):
                """Safely extract raw image channel from Arrow record."""
                try:
                    val = raw_record[k]
                    return np.array(val if val is not None else [], dtype=np.float32)
                except (KeyError, TypeError):
                    return np.array([], dtype=np.float32)
            
            try:
                vis = get_raw('image_vis')
                y = get_raw('image_nisp_y')
                j = get_raw('image_nisp_j')
                h = get_raw('image_nisp_h')
                
                # Fallback for missing bands
                if vis.size > 0:
                    h = h if h.size > 0 else np.zeros_like(vis)
                    j = j if j.size > 0 else np.zeros_like(vis)
                    y = y if y.size > 0 else np.zeros_like(vis)
                    
                    raw_stack = np.stack([vis, h, j, y], axis=0)  # (4, H, W)
                    
                    # Background subtraction
                    bg_val = np.percentile(raw_stack, 50, axis=(1,2), keepdims=True)
                    raw_bg = raw_stack - bg_val
                    
                    # Normalize channels
                    raw_rgb_stack = []
                    for c in range(raw_bg.shape[0]):
                        v_max = np.percentile(np.abs(raw_bg[c]), 99.5)
                        if v_max <= 0: v_max = 1.0
                        r_ch = np.clip(raw_bg[c] / v_max, 0, 100)
                        raw_rgb_stack.append(r_ch)
                    raw_norm = np.stack(raw_rgb_stack)
                    
                    # Weighted RGB: H->R, (J+Y)/2->G, VIS->B
                    RGB_WEIGHTS = [1.2, 1.3, 1.0]
                    vis_ch, h_ch, j_ch, y_ch = raw_norm[0], raw_norm[1], raw_norm[2], raw_norm[3]
                    r = h_ch * RGB_WEIGHTS[0]
                    g = ((j_ch + y_ch) / 2.0) * RGB_WEIGHTS[1]
                    b = vis_ch * RGB_WEIGHTS[2]
                    
                    rgb_input = np.stack([r, g, b], axis=0)
                    rgb_gt = make_rgb_lupton(rgb_input, Q=12.0, stretch=0.5)
                    
                    ax_img.imshow(rgb_gt, origin='lower')
                    ax_img.set_title(f"Euclid VIS+NISP False Color (Rank \\#{rank+1})", fontsize=15, fontweight='bold', color='darkred')
                    has_image = True
                else:
                    ax_img.text(0.5, 0.5, "EuclidImage not in raw database", ha='center', va='center', fontsize=12)
            except Exception as e:
                logger.error(f"Error rendering Euclid RGB image: {e}")
                logger.error(traceback.format_exc())
                ax_img.text(0.5, 0.5, "EuclidImage rendering failed", ha='center', va='center', fontsize=12)
            ax_img.axis('off')
            
            # 2. Anomaly Scores Profile Bar
            percentiles = {}
            for col in df_top.columns:
                if col.startswith("percentile_"):
                    method_name = col.replace("percentile_", "").upper()
                    percentiles[method_name] = getattr(row, col) * 100.0 # Convert to percent
                    
            y_pos = np.arange(len(percentiles))
            colors = plt.cm.plasma(np.array(list(percentiles.values())) / 100.0)
            
            ax_stats.barh(y_pos, list(percentiles.values()), align='center', color=colors, edgecolor='black', height=0.6)
            ax_stats.set_yticks(y_pos)
            ax_stats.set_yticklabels(list(percentiles.keys()), fontsize=10, fontweight='bold')
            ax_stats.invert_yaxis()  # top-down
            ax_stats.set_xlabel('Anomaly Score Percentile (%)', fontsize=11, fontweight='bold')
            ax_stats.set_xlim(0, 100)
            ax_stats.axvline(99.0, color='red', linestyle='--', alpha=0.7, label='Top 1% Threshold')
            ax_stats.set_title(f"UWI: {uwi:.4f} (Consensus Outlier Profile)", fontsize=13, fontweight='bold')
            ax_stats.grid(True, alpha=0.2)
            
            # Print metadata inside the dedicated ax_meta panel
            ax_meta.axis('off')
            ax_meta.text(
                0.05, 0.95, meta_str, 
                transform=ax_meta.transAxes, 
                fontsize=11.5,
                linespacing=1.6,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.6", fc="ivory", alpha=0.95, ec="darkgrey", lw=1.0)
            )
            ax_meta.set_title(r"\textbf{Galaxy Physical Properties}", fontsize=14, fontweight='bold', color='navy', pad=10)
            
            # 3. DESI Spectrum divided into Blue and Red Channels
            has_spectra = False
            try:
                if raw_record.get('spectrum_flux') is not None:
                    spec_gt = np.array(raw_record['spectrum_flux']).flatten()
                    wave_ang = np.array(raw_record['spectrum_wave']).flatten()
                    
                    w_min, w_max = wave_ang.min(), wave_ang.max()
                    w_mid = (w_min + w_max) / 2
                    
                    # Plot Blue Channel
                    ax_spec_blue.plot(wave_ang, spec_gt, 'k-', lw=1, alpha=0.8, label=f"DESISpectrum (TargetID: {tid})")
                    ax_spec_blue.set_xlim(w_min, w_mid)
                    ax_spec_blue.set_title(r"\textbf{Spectrum (Blue Channel)}", fontsize=14, fontweight='bold')
                    ax_spec_blue.set_ylabel(r"Flux", fontsize=12)
                    ax_spec_blue.legend(loc="upper right", framealpha=0.9)
                    plot_spectral_lines(ax_spec_blue, w_min, w_mid, z_val)
                    
                    # Plot Red Channel
                    ax_spec_red.plot(wave_ang, spec_gt, 'k-', lw=1, alpha=0.8)
                    ax_spec_red.set_xlim(w_mid, w_max)
                    ax_spec_red.set_title(r"\textbf{Spectrum (Red Channel)}", fontsize=14, fontweight='bold')
                    ax_spec_red.set_ylabel(r"Flux", fontsize=12)
                    ax_spec_red.set_xlabel(r"Observed Wavelength [\AA]", fontsize=13, fontweight='bold')
                    plot_spectral_lines(ax_spec_red, w_mid, w_max, z_val)
                    
                    has_spectra = True
                else:
                    ax_spec_blue.text(0.5, 0.5, "DESISpectrum not in raw database", ha='center', va='center', fontsize=12)
                    ax_spec_red.axis('off')
            except Exception as e:
                logger.error(f"Error rendering DESI spectrum: {e}")
                ax_spec_blue.text(0.5, 0.5, "DESISpectrum rendering failed", ha='center', va='center', fontsize=12)
                ax_spec_red.axis('off')
                
            plt.suptitle(r"\textbf{AstroPT Multimodal Outlier Profiler Dashboard}", fontsize=22, fontweight='bold', y=0.98)
            pdf.savefig()
            plt.close()
            
    logger.info(f"Saved Anomaly Report PDF to: {save_path}")


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    
    embeddings_dir = Path(args.embeddings_dir)
    save_dir = Path(args.output_dir) if args.output_dir else embeddings_dir / "anomalies"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load checkpoint model first to retrieve training configuration and applied filters
    logger.info("Initializing active AstroPT Datasets via config checkpoint...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, registry, raw_config_dict = load_local_model(Path(args.ckpt_path), device)
    
    # Reconstruct data dataloader parameters
    raw_config_dict['data_dir'] = args.data_dir
    raw_config_dict['batch_size'] = args.batch_size
    
    valid_keys = {f.name for f in fields(TrainingConfig)}
    clean_config_dict = {k: v for k, v in raw_config_dict.items() if k in valid_keys}
    training_config = TrainingConfig(**clean_config_dict)
    
    _, loader, _ = create_dataloaders(training_config, ddp=False)
    ds = loader.dataset
    
    # 2. Read fits metadata catalog
    logger.info(f"Reading FITS metadata catalog from {args.metadata_path}...")
    fits_table = Table.read(args.metadata_path)
    fits_df = fits_table.to_pandas()
    
    # Decode string bytes
    for col in fits_df.columns:
        if fits_df[col].dtype == object and isinstance(fits_df[col].iloc[0], bytes):
            try: fits_df[col] = fits_df[col].str.decode('utf-8')
            except: pass
            
    # 3. Dynamic dataset filtering from model configuration
    allowed_ids = None
    applied_filters = raw_config_dict.get('applied_filters', getattr(config, 'applied_filters', None))
    if applied_filters:
        logger.info(f"Detected applied dataset filters in model configuration: {applied_filters}")
        allowed_ids = get_allowed_ids_from_filters(fits_df, applied_filters)
    else:
        logger.info("No applied filters found in model configuration. Analyzing full dataset.")
        
    # 4. Calculate scores across the 7 methods
    df_scores, feature_matrices = calculate_anomaly_scores(
        embeddings_dir=embeddings_dir,
        methods=args.methods,
        base_modality=args.base_modality,
        contamination=args.contamination,
        knn_neighbors=args.knn_neighbors,
        ae_epochs=args.ae_epochs,
        ae_latent_dim=args.ae_latent_dim,
        allowed_ids_filter=allowed_ids
    )
    
    # Save the consolidated CSV catalog of anomaly rankings
    csv_path = save_dir / "consolidated_anomalies.csv"
    df_scores.to_csv(csv_path, index=False)
    logger.info(f"Saved complete anomaly scores catalog to {csv_path}")
    
    # Extract top N anomalies for deep report profiling
    df_top = df_scores.head(args.n_anomalies)
    
    # 5. Generate Reports and Plots
    if args.plot_projection:
        X_base = feature_matrices[args.base_modality]
        # Match Top anomalies IDs back to their index in the active ids array
        all_ids = df_scores.TargetID.values
        top_tids = df_top.TargetID.values
        top_indices = np.array([np.where(all_ids == tid)[0][0] for tid in top_tids])
        
        tsne_path = save_dir / f"latent_tsne_{args.base_modality}.png"
        generate_tSNE_plot(
            X_base, 
            top_indices=top_indices, 
            uwi_scores=df_top.Unified_Weirdness_Index.values, 
            save_path=tsne_path
        )
        
    dashboard_pdf_path = save_dir / f"anomaly_hunter_report_{args.base_modality}.pdf"
    generate_anomaly_report_dashboard(
        ds=ds,
        df_top=df_top,
        fits_catalog=fits_df,
        save_path=dashboard_pdf_path
    )
    
    logger.info("Done. Anomaly hunter completes successfully.")

if __name__ == "__main__":
    main()
