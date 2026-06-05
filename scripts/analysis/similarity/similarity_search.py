#!/usr/bin/env python3
"""
AstroPT Multimodal Similarity Search Tool
==================================================
Description:
    Takes a Query TargetID and retrieves the Top K nearest neighbors in three 
    independent latent spaces:
    1. Image space (EuclidImage) - morphological matches
    2. Spectrum space (DESISpectrum) - spectral/physical matches
    3. Joint space (joint) - morpho-spectroscopic matches
    
    Generates an premium multi-page PDF report demonstrating AstroPT's 
    representation learning capabilities.
"""

import argparse
import sys
import os
import re
import logging
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
from torch.utils.data import DataLoader

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
logger = logging.getLogger("AstroPT-SimilaritySearch")


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="AstroPT Multimodal Similarity Search")
    
    parser.add_argument("--embeddings_dir", type=str, required=True, 
                        help="Directory containing pre-computed embeddings (.npy)")
    parser.add_argument("--ckpt_path", type=str, required=True, 
                        help="Path to the training checkpoint (.pt)")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Root folder of raw Arrow processed data")
    parser.add_argument("--metadata_path", type=str, required=True, 
                        help="Path to FITS metadata catalog")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Saving directory for similarity search dashboards")
    
    parser.add_argument("--query_id", type=int, required=True, 
                        help="TargetID of the query galaxy")
    parser.add_argument("--k", type=int, default=5, 
                        help="Number of nearest neighbors to retrieve")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for dataloader")
    
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper functions for RGB and Spectral plotting (LaTeX safe)
# ---------------------------------------------------------------------------

MAIN_LINES = {
    r"Ly$\alpha$": 1216.0, r"C IV": 1549.0, "C III": 1908.7, r"Mg II": 2798.0, r"[O II]": 3727.3, r"[Ne III]": 3868.7,
    r"Ca K": 3933.7, r"Ca H": 3968.5, r"H$\delta$": 4102.0, r"H$\gamma$": 4341.0,
    r"H$\beta$": 4861.0, r"[O III]": 4959.0, r"[O III]": 5007.0, r"Mg I": 5175.0,
    r"Na D": 5890.0, r"[N II]": 6548.0, r"H$\alpha$": 6563.0, r"[N II]": 6583.5, r"[S II]": 6730.8
}

def plot_spectral_lines(ax, min_wl, max_wl, z):
    """Annotates spectral lines with alternating heights."""
    if z is None or pd.isna(z):
        return
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
                    color='royalblue', va='top', ha='right', fontsize=10, alpha=1, fontweight='bold')
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


def extract_raw_rgb(raw_record: dict) -> Optional[np.ndarray]:
    """Safely extract and reconstruct VIS+NISP false color RGB using Lupton."""
    def get_raw(k):
        try:
            val = raw_record[k]
            return np.array(val if val is not None else [], dtype=np.float32)
        except (KeyError, TypeError):
            return np.array([], dtype=np.float32)

    vis = get_raw('image_vis')
    y = get_raw('image_nisp_y')
    j = get_raw('image_nisp_j')
    h = get_raw('image_nisp_h')
    
    if vis.size == 0:
        return None
        
    h = h if h.size > 0 else np.zeros_like(vis)
    j = j if j.size > 0 else np.zeros_like(vis)
    y = y if y.size > 0 else np.zeros_like(vis)
    
    raw_stack = np.stack([vis, h, j, y], axis=0)  # (4, H, W)
    bg_val = np.percentile(raw_stack, 50, axis=(1,2), keepdims=True)
    raw_bg = raw_stack - bg_val
    
    raw_rgb_stack = []
    for c in range(raw_bg.shape[0]):
        v_max = np.percentile(np.abs(raw_bg[c]), 99.5)
        if v_max <= 0: v_max = 1.0
        r_ch = np.clip(raw_bg[c] / v_max, 0, 100)
        raw_rgb_stack.append(r_ch)
    raw_norm = np.stack(raw_rgb_stack)
    
    RGB_WEIGHTS = [1.2, 1.3, 1.0]
    vis_ch, h_ch, j_ch, y_ch = raw_norm[0], raw_norm[1], raw_norm[2], raw_norm[3]
    r = h_ch * RGB_WEIGHTS[0]
    g = ((j_ch + y_ch) / 2.0) * RGB_WEIGHTS[1]
    b = vis_ch * RGB_WEIGHTS[2]
    
    rgb_input = np.stack([r, g, b], axis=0)
    return make_rgb_lupton(rgb_input, Q=12.0, stretch=0.5)


def plot_spectrum_into_axes(ax_blue, ax_red, raw_record: dict, tid: int, z_val: Optional[float] = None):
    """Plots DESI spectrum split in Blue and Red channels into the provided axes."""
    if raw_record.get('spectrum_flux') is not None:
        spec_gt = np.array(raw_record['spectrum_flux']).flatten()
        wave_ang = np.array(raw_record['spectrum_wave']).flatten()
        
        w_min, w_max = wave_ang.min(), wave_ang.max()
        w_mid = (w_min + w_max) / 2
        
        # Blue Channel
        ax_blue.plot(wave_ang, spec_gt, 'k-', lw=1, alpha=0.8)
        ax_blue.set_xlim(w_min, w_mid)
        ax_blue.set_title(r"\textbf{Spectrum (Blue Channel)}", fontsize=11, fontweight='bold')
        ax_blue.set_ylabel(r"Flux", fontsize=9)
        if z_val is not None:
            plot_spectral_lines(ax_blue, w_min, w_mid, z_val)
        
        # Red Channel
        ax_red.plot(wave_ang, spec_gt, 'k-', lw=1, alpha=0.8)
        ax_red.set_xlim(w_mid, w_max)
        ax_red.set_title(r"\textbf{Spectrum (Red Channel)}", fontsize=11, fontweight='bold')
        ax_red.set_ylabel(r"Flux", fontsize=9)
        ax_red.set_xlabel(r"Observed Wavelength [\AA]", fontsize=10, fontweight='bold')
        if z_val is not None:
            plot_spectral_lines(ax_red, w_mid, w_max, z_val)
    else:
        ax_blue.text(0.5, 0.5, "DESISpectrum not found", ha='center', va='center', fontsize=12)
        ax_red.axis('off')


# ---------------------------------------------------------------------------
# Search Math Engine
# ---------------------------------------------------------------------------

def compute_similarity(query_idx: int, X: np.ndarray, ids: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Computes cosine similarity between query vec and all matrix rows."""
    query_vec = X[query_idx]
    
    # L2 normalize
    norm_X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    norm_query = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    
    # Cosine similarities
    sims = np.dot(norm_X, norm_query)
    
    # Exclude query itself by setting similarity to -2.0
    sims[query_idx] = -2.0
    
    # Get top k indices in descending order
    top_indices = np.argsort(sims)[::-1][:k]
    return ids[top_indices], sims[top_indices]


# ---------------------------------------------------------------------------
# Main similarity pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    
    embeddings_dir = Path(args.embeddings_dir)
    save_dir = Path(args.output_dir) if args.output_dir else embeddings_dir / "similarity"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load active embeddings
    logger.info("Loading pre-computed embeddings...")
    emb_files = {f.stem: f for f in embeddings_dir.glob("*.npy")}
    
    if "ids" not in emb_files:
        raise FileNotFoundError(f"ids.npy not found in {embeddings_dir}")
        
    ids = np.load(embeddings_dir / "ids.npy").astype(np.int64)
    
    # Find query index
    query_id = args.query_id
    matches = np.where(ids == query_id)[0]
    if len(matches) == 0:
        logger.error(f"Query TargetID {query_id} not found in the latent database!")
        # Suggest some IDs
        logger.info(f"Available sample IDs in dataset: {list(ids[:10])}...")
        sys.exit(1)
    query_idx = int(matches[0])
    logger.info(f"Resolved Query TargetID {query_id} at index {query_idx}")
    
    # Compute similarity for Image, Spectrum, and Joint spaces
    spaces = {}
    for space_name in ["EuclidImage", "DESISpectrum", "joint"]:
        if space_name not in emb_files:
            logger.warning(f"Embedding space '{space_name}.npy' not found! Skipping similarity computation for it.")
            continue
        X = np.load(embeddings_dir / f"{space_name}.npy")
        if X.ndim == 3:
            X = X.mean(axis=1) # Mean pooling sequence tokens
            
        top_ids, top_sims = compute_similarity(query_idx, X, ids, k=args.k)
        spaces[space_name] = {"ids": top_ids, "sims": top_sims}
        
        logger.info(f"\nTop {args.k} Neighbors in '{space_name}' Space:")
        for idx, (tid, sim) in enumerate(zip(top_ids, top_sims)):
            logger.info(f"  Rank #{idx+1}: TargetID={tid} | Cosine Similarity={sim:.4%}")

    # 2. Setup Dataloader and metadata
    logger.info("\nInitializing active AstroPT Datasets via checkpoint...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, registry, raw_config_dict = load_local_model(Path(args.ckpt_path), device)
    
    raw_config_dict['data_dir'] = args.data_dir
    raw_config_dict['batch_size'] = args.batch_size
    
    valid_keys = {f.name for f in fields(TrainingConfig)}
    clean_config_dict = {k: v for k, v in raw_config_dict.items() if k in valid_keys}
    training_config = TrainingConfig(**clean_config_dict)
    
    _, loader, _ = create_dataloaders(training_config, ddp=False)
    ds = loader.dataset
    
    # Read FITS metadata catalog
    logger.info(f"Reading FITS metadata catalog from {args.metadata_path}...")
    fits_table = Table.read(args.metadata_path)
    fits_df = fits_table.to_pandas()
    
    # Decode string bytes
    for col in fits_df.columns:
        if fits_df[col].dtype == object and isinstance(fits_df[col].iloc[0], bytes):
            try: fits_df[col] = fits_df[col].str.decode('utf-8')
            except: pass
            
    target_id_col = 'TARGETID' if 'TARGETID' in fits_df.columns else 'targetid'
    fits_indexed = fits_df.drop_duplicates(subset=[target_id_col]).set_index(target_id_col)
    
    def find_col_val(df_row, patterns):
        for pat in patterns:
            for col in df_row.index:
                if pat.lower() in col.lower():
                    val = df_row[col]
                    if val is not None and not pd.isna(val):
                        return float(val)
        return None

    def get_morphological_meta_str(tid):
        if tid in fits_indexed.index:
            meta = fits_indexed.loc[tid]
            
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
            return meta_str.replace("_", "\\_"), z_val
        return f"\\textbf{{ID}}: {tid}\n(No metadata)", None

    def get_spectroscopic_meta_str(tid):
        if tid in fits_indexed.index:
            meta = fits_indexed.loc[tid]
            
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

    # --- Helper to retrieve metadata block ---
    def get_meta_str(tid):
        if tid in fits_indexed.index:
            meta = fits_indexed.loc[tid]
            z_val = meta.get('Z', meta.get('z', None))
            z_str = f"{z_val:.4f}" if (z_val is not None and not pd.isna(z_val)) else "N/A"
            logmstar_val = meta.get('LOGMSTAR', meta.get('logmstar', None))
            mstar = f"$10^{{{logmstar_val:.2f}}} \\text{{ M}}_{{\\odot}}$" if (logmstar_val is not None and not pd.isna(logmstar_val)) else "N/A"
            logsfr_val = meta.get('LOGSFR', meta.get('logsfr', None))
            sfr = f"${10**logsfr_val:.3f} \\text{{ M}}_{{\\odot}}/\\text{{yr}}$" if (logsfr_val is not None and not pd.isna(logsfr_val)) else "N/A"
            spectype = str(meta.get('SPECTYPE', meta.get('spectype', 'N/A')))
            
            meta_str = (
                f"\\textbf{{Z}}: {z_str}\n"
                f"\\textbf{{$M_*$}}: {mstar}\n"
                f"\\textbf{{SFR}}: {sfr}\n"
                f"\\textbf{{Spectype}}: {spectype}"
            )
            return meta_str.replace("_", "\\_"), z_val
        return f"\\textbf{{ID}}: {tid}\n(No metadata)", None

    # Retrieve query record
    ds_ids = np.array(ds.ds['targetid'])
    query_matches = np.where(ds_ids == query_id)[0]
    if len(query_matches) == 0:
        logger.error(f"Query TargetID {query_id} not found in the Arrow raw catalog!")
        sys.exit(1)
    query_record = ds.ds[int(query_matches[0])]
    query_rgb = extract_raw_rgb(query_record)
    query_meta, query_z = get_morphological_meta_str(query_id)
    query_spec_meta = get_spectroscopic_meta_str(query_id)
    
    # ---------------------------------------------------------------------------
    # Plot Generation (Premium Dashboard PDF)
    # ---------------------------------------------------------------------------
    matplotlib_use_agg = True
    import matplotlib
    if matplotlib_use_agg:
        matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages
    
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', weight='bold')
    
    pdf_path = save_dir / f"similarity_search_{query_id}.pdf"
    logger.info(f"Generating PDF Similarity Search Report at: {pdf_path}")
    
    with PdfPages(pdf_path) as pdf:
        
        # ==========================================
        # PAGE 1: QUERY GALAXY PROFILE
        # ==========================================
        fig = plt.figure(figsize=(20, 14), dpi=150)
        gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], width_ratios=[1, 1.1, 0.9], hspace=0.4, wspace=0.3)
        
        ax_meta = fig.add_subplot(gs[0, 0])
        ax_img = fig.add_subplot(gs[0, 1])
        ax_stats = fig.add_subplot(gs[0, 2])
        ax_spec_blue = fig.add_subplot(gs[1, :])
        ax_spec_red = fig.add_subplot(gs[2, :])
        
        # Meta
        ax_meta.axis('off')
        ax_meta.text(
            0.5, 0.95, query_meta, 
            transform=ax_meta.transAxes, 
            fontsize=14,
            linespacing=1.8,
            horizontalalignment='center',
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.8", fc="ivory", alpha=0.95, ec="darkgrey", lw=1.0)
        )
        ax_meta.set_title(r"\textbf{Physical and Morphological properties}", fontsize=16, fontweight='bold', color='navy', pad=15)
        
        # Image
        if query_rgb is not None:
            ax_img.imshow(query_rgb, origin='lower')
            # Title is removed per user request
        else:
            ax_img.text(0.5, 0.5, "EuclidImage missing", ha='center', va='center')
        ax_img.axis('off')
        
        # Stats info box (Spectroscopic properties)
        ax_stats.axis('off')
        ax_stats.text(
            0.5, 0.95, query_spec_meta, 
            transform=ax_stats.transAxes, 
            fontsize=14,
            linespacing=1.6,
            horizontalalignment='center',
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.8", fc="aliceblue", alpha=0.95, ec="steelblue", lw=1.0)
        )
        ax_stats.set_title(r"\textbf{Spectroscopic properties}", fontsize=16, fontweight='bold', color='steelblue', pad=15)
        
        # Spectrum
        plot_spectrum_into_axes(ax_spec_blue, ax_spec_red, query_record, query_id, query_z)
        ax_spec_blue.set_title(rf"\textbf{{Query Spectrum (Blue Channel)}}", fontsize=11, fontweight='bold')
        
        plt.suptitle(rf"\textbf{{AstroPT similarity search: ID: {query_id}}}", fontsize=22, fontweight='bold', y=0.98)
        pdf.savefig()
        plt.close()
        
        # ==========================================
        # PAGE 2: EUCLID MORPHOLOGICAL NEIGHBORS (SUMMARY GRID)
        # ==========================================
        if "EuclidImage" in spaces:
            logger.info("Plotting Euclid Morphological Neighbors summary grid...")
            fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=150)
            fig.subplots_adjust(hspace=0.3, wspace=0.2)
            
            # Query Image (Top Left)
            ax_q = axes[0, 0]
            if query_rgb is not None:
                ax_q.imshow(query_rgb, origin='lower')
            # Title is completely removed per user request
            ax_q.axis('off')
            ax_q.patch.set_edgecolor('navy')
            ax_q.patch.set_linewidth(3)
            
            for idx, (tid, sim) in enumerate(zip(spaces["EuclidImage"]["ids"][:5], spaces["EuclidImage"]["sims"][:5])):
                row_ax = (idx + 1) // 3
                col_ax = (idx + 1) % 3
                ax_n = axes[row_ax, col_ax]
                
                n_matches = np.where(ds_ids == tid)[0]
                n_rgb = None
                if len(n_matches) > 0:
                    n_rec = ds.ds[int(n_matches[0])]
                    n_rgb = extract_raw_rgb(n_rec)
                    
                if n_rgb is not None:
                    ax_n.imshow(n_rgb, origin='lower')
                else:
                    ax_n.text(0.5, 0.5, f"Image not found | ID: {tid}", ha='center', va='center')
                # Title is completely removed per user request
                ax_n.axis('off')
                
            plt.suptitle(r"\textbf{Euclid Morphological Neighbors}", fontsize=22, fontweight='bold', y=0.98)
            pdf.savefig()
            plt.close()
            
            # ==========================================
            # PAGES 3-7: DETAILED PROFILE FOR EACH NEIGHBOR
            # ==========================================
            for idx, (tid, sim) in enumerate(zip(spaces["EuclidImage"]["ids"], spaces["EuclidImage"]["sims"])):
                logger.info(f"Plotting detailed profile page for Euclid Morphological Rank #{idx+1} (ID: {tid})...")
                
                n_matches = np.where(ds_ids == tid)[0]
                if len(n_matches) == 0:
                    logger.warning(f"Neighbor TargetID {tid} not found in Arrow datasets, skipping detailed profile.")
                    continue
                    
                n_rec = ds.ds[int(n_matches[0])]
                n_rgb = extract_raw_rgb(n_rec)
                n_meta, n_z = get_morphological_meta_str(tid)
                n_spec_meta = get_spectroscopic_meta_str(tid)
                
                # Render Page
                fig = plt.figure(figsize=(20, 14), dpi=150)
                gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], width_ratios=[1, 1.1, 0.9], hspace=0.4, wspace=0.3)
                
                ax_meta = fig.add_subplot(gs[0, 0])
                ax_img = fig.add_subplot(gs[0, 1])
                ax_stats = fig.add_subplot(gs[0, 2])
                ax_spec_blue = fig.add_subplot(gs[1, :])
                ax_spec_red = fig.add_subplot(gs[2, :])
                
                # Meta box
                ax_meta.axis('off')
                ax_meta.text(
                    0.5, 0.95, n_meta, 
                    transform=ax_meta.transAxes, 
                    fontsize=14,
                    linespacing=1.8,
                    horizontalalignment='center',
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.8", fc="ivory", alpha=0.95, ec="darkgrey", lw=1.0)
                )
                ax_meta.set_title(r"\textbf{Physical and Morphological properties}", fontsize=16, fontweight='bold', color='navy', pad=15)
                
                # Image
                if n_rgb is not None:
                    ax_img.imshow(n_rgb, origin='lower')
                    # No title above the neighbor image per user request
                else:
                    ax_img.text(0.5, 0.5, "EuclidImage missing", ha='center', va='center')
                ax_img.axis('off')
                
                # Stats box (Spectroscopic properties)
                ax_stats.axis('off')
                ax_stats.text(
                    0.5, 0.95, n_spec_meta, 
                    transform=ax_stats.transAxes, 
                    fontsize=14,
                    linespacing=1.6,
                    horizontalalignment='center',
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.8", fc="aliceblue", alpha=0.95, ec="steelblue", lw=1.0)
                )
                ax_stats.set_title(r"\textbf{Spectroscopic properties}", fontsize=16, fontweight='bold', color='steelblue', pad=15)
                
                # Spectrum
                plot_spectrum_into_axes(ax_spec_blue, ax_spec_red, n_rec, tid, n_z)
                ax_spec_blue.set_title(rf"\textbf{{Spectrum (Blue Channel)}}", fontsize=11, fontweight='bold')
                
                # Page super title
                plt.suptitle(rf"\textbf{{Euclid Morphological Rank \#{idx+1} ({sim*100:.2f}\%): ID: {tid}}}", fontsize=22, fontweight='bold', y=0.98)
                pdf.savefig()
                plt.close()
                
        # ==========================================
        # PAGE 8: DESI SPECTROSCOPIC NEIGHBORS (SUMMARY GRID/STACK)
        # ==========================================
        if "DESISpectrum" in spaces:
            logger.info("Plotting DESI Spectroscopic Neighbors summary grid...")
            fig, axes = plt.subplots(6, 1, figsize=(20, 16), dpi=150, sharex=True)
            fig.subplots_adjust(hspace=0.55, left=0.08, right=0.95, top=0.92, bottom=0.06)
            
            # Subplot 0: Query Spectrum
            ax_q = axes[0]
            if query_record.get('spectrum_flux') is not None:
                spec_gt = np.array(query_record['spectrum_flux']).flatten()
                wave_ang = np.array(query_record['spectrum_wave']).flatten()
                ax_q.plot(wave_ang, spec_gt, 'r-', lw=1, alpha=0.8)
                ax_q.set_xlim(wave_ang.min(), wave_ang.max())
                ax_q.set_title(rf"\textbf{{Query (ID: {query_id})}}", fontsize=11, fontweight='bold')
                if query_z is not None:
                    for name, rest_wave in MAIN_LINES.items():
                        obs_wave = rest_wave * (1 + query_z)
                        if wave_ang.min() < obs_wave < wave_ang.max():
                            ax_q.axvline(obs_wave, color='royalblue', linestyle='--', alpha=0.3, lw=0.7)
            else:
                ax_q.text(0.5, 0.5, "Query Spectrum not found", ha='center', va='center', fontsize=12)
            
            ax_q.set_ylabel(r"Flux", fontsize=9)
            
            # Subplots 1-5: Spectroscopic Neighbors
            for idx, (tid, sim) in enumerate(zip(spaces["DESISpectrum"]["ids"][:5], spaces["DESISpectrum"]["sims"][:5])):
                ax_n = axes[idx + 1]
                n_matches = np.where(ds_ids == tid)[0]
                
                n_rec = None
                n_z = None
                if len(n_matches) > 0:
                    n_rec = ds.ds[int(n_matches[0])]
                    if tid in fits_indexed.index:
                        n_z = fits_indexed.loc[tid].get('Z', None)
                
                if n_rec is not None and n_rec.get('spectrum_flux') is not None:
                    spec_gt = np.array(n_rec['spectrum_flux']).flatten()
                    wave_ang = np.array(n_rec['spectrum_wave']).flatten()
                    ax_n.plot(wave_ang, spec_gt, 'k-', lw=1, alpha=0.8)
                    ax_n.set_xlim(wave_ang.min(), wave_ang.max())
                    ax_n.set_title(rf"\textbf{{Rank \#{idx+1} ({sim*100:.2f}\%) | ID: {tid}}}", fontsize=11, fontweight='bold')
                    if n_z is not None and not pd.isna(n_z):
                        for name, rest_wave in MAIN_LINES.items():
                            obs_wave = rest_wave * (1 + n_z)
                            if wave_ang.min() < obs_wave < wave_ang.max():
                                ax_n.axvline(obs_wave, color='royalblue', linestyle='--', alpha=0.3, lw=0.7)
                else:
                    ax_n.text(0.5, 0.5, f"Spectrum not found | ID: {tid}", ha='center', va='center', fontsize=12)
                
                ax_n.set_ylabel(r"Flux", fontsize=9)
            
            axes[-1].set_xlabel(r"Observed Wavelength [\AA]", fontsize=12, fontweight='bold')
            
            plt.suptitle(r"\textbf{DESI Spectroscopic Neighbors}", fontsize=22, fontweight='bold', y=0.98)
            pdf.savefig()
            plt.close()
            
            # ==========================================
            # PAGES 9-13: DETAILED PROFILE FOR EACH SPECTROSCOPIC NEIGHBOR
            # ==========================================
            for idx, (tid, sim) in enumerate(zip(spaces["DESISpectrum"]["ids"], spaces["DESISpectrum"]["sims"])):
                logger.info(f"Plotting detailed profile page for DESI Spectroscopic Rank #{idx+1} (ID: {tid})...")
                
                n_matches = np.where(ds_ids == tid)[0]
                if len(n_matches) == 0:
                    logger.warning(f"Neighbor TargetID {tid} not found in Arrow datasets, skipping detailed profile.")
                    continue
                    
                n_rec = ds.ds[int(n_matches[0])]
                n_rgb = extract_raw_rgb(n_rec)
                n_meta, n_z = get_morphological_meta_str(tid)
                n_spec_meta = get_spectroscopic_meta_str(tid)
                
                # Render Page
                fig = plt.figure(figsize=(20, 14), dpi=150)
                gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], width_ratios=[1, 1.1, 0.9], hspace=0.4, wspace=0.3)
                
                ax_meta = fig.add_subplot(gs[0, 0])
                ax_img = fig.add_subplot(gs[0, 1])
                ax_stats = fig.add_subplot(gs[0, 2])
                ax_spec_blue = fig.add_subplot(gs[1, :])
                ax_spec_red = fig.add_subplot(gs[2, :])
                
                # Meta box
                ax_meta.axis('off')
                ax_meta.text(
                    0.5, 0.95, n_meta, 
                    transform=ax_meta.transAxes, 
                    fontsize=14,
                    linespacing=1.8,
                    horizontalalignment='center',
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.8", fc="ivory", alpha=0.95, ec="darkgrey", lw=1.0)
                )
                ax_meta.set_title(r"\textbf{Physical and Morphological properties}", fontsize=16, fontweight='bold', color='navy', pad=15)
                
                # Image
                if n_rgb is not None:
                    ax_img.imshow(n_rgb, origin='lower')
                else:
                    ax_img.text(0.5, 0.5, "EuclidImage missing", ha='center', va='center')
                ax_img.axis('off')
                
                # Stats box (Spectroscopic properties)
                ax_stats.axis('off')
                ax_stats.text(
                    0.5, 0.95, n_spec_meta, 
                    transform=ax_stats.transAxes, 
                    fontsize=14,
                    linespacing=1.6,
                    horizontalalignment='center',
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.8", fc="aliceblue", alpha=0.95, ec="steelblue", lw=1.0)
                )
                ax_stats.set_title(r"\textbf{Spectroscopic properties}", fontsize=16, fontweight='bold', color='steelblue', pad=15)
                
                # Spectrum
                plot_spectrum_into_axes(ax_spec_blue, ax_spec_red, n_rec, tid, n_z)
                ax_spec_blue.set_title(rf"\textbf{{Spectrum (Blue Channel)}}", fontsize=11, fontweight='bold')
                
                # Page super title
                plt.suptitle(rf"\textbf{{DESI Spectroscopic Rank \#{idx+1} ({sim*100:.2f}\%): ID: {tid}}}", fontsize=22, fontweight='bold', y=0.98)
                pdf.savefig()
                plt.close()
                
        # ==========================================
        # PAGES 14-18: DETAILED PROFILE FOR EACH JOINT NEIGHBOR
        # ==========================================
        if "joint" in spaces:
            for idx, (tid, sim) in enumerate(zip(spaces["joint"]["ids"], spaces["joint"]["sims"])):
                logger.info(f"Plotting detailed profile page for Joint Rank #{idx+1} (ID: {tid})...")
                
                n_matches = np.where(ds_ids == tid)[0]
                if len(n_matches) == 0:
                    logger.warning(f"Neighbor TargetID {tid} not found in Arrow datasets, skipping detailed profile.")
                    continue
                    
                n_rec = ds.ds[int(n_matches[0])]
                n_rgb = extract_raw_rgb(n_rec)
                n_meta, n_z = get_morphological_meta_str(tid)
                n_spec_meta = get_spectroscopic_meta_str(tid)
                
                # Render Page
                fig = plt.figure(figsize=(20, 14), dpi=150)
                gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], width_ratios=[1, 1.1, 0.9], hspace=0.4, wspace=0.3)
                
                ax_meta = fig.add_subplot(gs[0, 0])
                ax_img = fig.add_subplot(gs[0, 1])
                ax_stats = fig.add_subplot(gs[0, 2])
                ax_spec_blue = fig.add_subplot(gs[1, :])
                ax_spec_red = fig.add_subplot(gs[2, :])
                
                # Meta box
                ax_meta.axis('off')
                ax_meta.text(
                    0.5, 0.95, n_meta, 
                    transform=ax_meta.transAxes, 
                    fontsize=14,
                    linespacing=1.8,
                    horizontalalignment='center',
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.8", fc="ivory", alpha=0.95, ec="darkgrey", lw=1.0)
                )
                ax_meta.set_title(r"\textbf{Physical and Morphological properties}", fontsize=16, fontweight='bold', color='navy', pad=15)
                
                # Image
                if n_rgb is not None:
                    ax_img.imshow(n_rgb, origin='lower')
                else:
                    ax_img.text(0.5, 0.5, "EuclidImage missing", ha='center', va='center')
                ax_img.axis('off')
                
                # Stats box (Spectroscopic properties)
                ax_stats.axis('off')
                ax_stats.text(
                    0.5, 0.95, n_spec_meta, 
                    transform=ax_stats.transAxes, 
                    fontsize=14,
                    linespacing=1.6,
                    horizontalalignment='center',
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.8", fc="aliceblue", alpha=0.95, ec="steelblue", lw=1.0)
                )
                ax_stats.set_title(r"\textbf{Spectroscopic properties}", fontsize=16, fontweight='bold', color='steelblue', pad=15)
                
                # Spectrum
                plot_spectrum_into_axes(ax_spec_blue, ax_spec_red, n_rec, tid, n_z)
                ax_spec_blue.set_title(rf"\textbf{{Spectrum (Blue Channel)}}", fontsize=11, fontweight='bold')
                
                # Page super title
                plt.suptitle(rf"\textbf{{Joint Rank \#{idx+1} ({sim*100:.2f}\%): ID: {tid}}}", fontsize=22, fontweight='bold', y=0.98)
                pdf.savefig()
                plt.close()
                
    logger.info(f"Done. Multimodal similarity search complete. Saved dashboard PDF to {pdf_path}")


if __name__ == "__main__":
    main()
