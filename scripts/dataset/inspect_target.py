#!/usr/bin/env python3
"""
AstroPT Target Visual Inspector & Diagnostics
=============================================
Author: Senior Astro-Data Scientist & PyTorch/HPC Infrastructure Engineer
Description:
    This script performs targeted qualitative visual audits of Euclid image stamps 
    and DESI spectra for individual objects in the AstroPT dataset.
    It takes a list of TargetIDs, retrieves them from the Arrow splits, 
    cross-references their catalog flags, and generates premium publication-quality 
    diagnostic dashboards featuring false-color Lupton RGB composite plots 
    and redshift-corrected multi-channel spectra plots.
"""

import os
import sys
import glob
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.table import Table
from datasets import load_from_disk

# Configure premium logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[sys.stdout]
)
logger = logging.getLogger("astropt_target_inspector")

# Apply premium LaTeX visual aesthetics
plt.rc('text', usetex=True)
plt.rc('font', family='serif', weight='bold')

plt.rcParams.update({
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titlesize': 18,
    'figure.titleweight': 'bold',
    'figure.dpi': 200,
    'savefig.bbox': 'tight'
})

# Rest-frame wavelength markers for major spectral lines
MAIN_LINES = {
    r"Ly$\alpha$": 1216.0, r"C IV": 1549.0, r"C III]": 1908.7, r"Mg II": 2798.0, 
    r"[O II]": 3727.3, r"[Ne III]": 3868.7, r"Ca K": 3933.7, r"Ca H": 3968.5, 
    r"H$\delta$": 4102.0, r"H$\gamma$": 4341.0, r"H$\beta$": 4861.0, 
    r"[O III]": 5007.0, r"Mg I": 5175.0, r"Na D": 5890.0, r"[N II]": 6548.0, 
    r"H$\alpha$": 6563.0, r"[N II]": 6583.5, r"[S II]": 6730.8
}

def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="AstroPT Target Inspector & Diagnostic Dashboard Generator")
    parser.add_argument(
        "--target_ids", "-t",
        type=int,
        nargs="+",
        required=True,
        help="List of TargetIDs to inspect."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated",
        help="Root directory containing processed Arrow datasets splits."
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits",
        help="Absolute path to the FITS catalog containing all survey metadata."
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/dataset_images_flux_analysis/inspections",
        help="Directory to save generated diagnostic dashboard PDF or PNG plots."
    )
    parser.add_argument(
        "--dark_registry",
        type=str,
        default="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/dataset_images_flux_analysis/dark_galaxies_registry.csv",
        help="Path to the dark galaxies CSV registry to check anomaly classifications."
    )
    return parser.parse_args()

def make_rgb_lupton(vis: np.ndarray, y: np.ndarray, j: np.ndarray, h: np.ndarray, Q: float = 12.0, stretch: float = 0.5) -> np.ndarray:
    """
    Combines Euclid band images using Lupton et al. (2004) asinh scaling.
    VIS -> Blue, average(J, Y) -> Green, H -> Red.
    """
    green = (j + y) / 2.0
    rgb = np.stack([h, green, vis], axis=0) # Shape: (3, H, W)
    
    # Calculate intensity
    intensity = np.mean(rgb, axis=0)
    intensity = np.maximum(intensity, 1e-10) # avoid division by zero
    
    # Lupton scaling
    f_I = np.arcsinh(Q * stretch * intensity) / Q
    scale_factor = f_I / intensity
    
    rgb_scaled = rgb * scale_factor[np.newaxis, :, :]
    
    # Clip and normalize
    max_rgb = np.percentile(rgb_scaled, 99.5)
    if max_rgb > 0:
        rgb_scaled = rgb_scaled / max_rgb
    rgb_scaled = np.clip(rgb_scaled, 0, 1)
    
    return rgb_scaled.transpose(1, 2, 0) # Return shape (H, W, 3)

def plot_spectral_lines(ax, min_wl, max_wl, z):
    """Annotates shifted spectral lines onto the spectrum axes."""
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    pos_high = y_min + (y_range * 0.92)
    pos_low  = y_min + (y_range * 0.20)
    
    sorted_lines = sorted(MAIN_LINES.items(), key=lambda x: x[1])
    counter = 0
    
    for name, rest_wave in sorted_lines:
        obs_wave = rest_wave * (1 + z)
        if min_wl < obs_wave < max_wl:
            y_pos = pos_high if counter % 2 == 0 else pos_low
            ax.axvline(obs_wave, color='royalblue', linestyle='--', alpha=0.5, lw=1)
            # LaTeX formatting
            ax.text(
                obs_wave, y_pos, rf"\textbf{{{name}}}", rotation=90, 
                color='royalblue', va='top', ha='right', fontsize=9, alpha=0.9, fontweight='bold'
            )
            counter += 1

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("ASTROPT TARGET VISUAL INSPECTOR & DATA QUALITY DIAGNOSTICS")
    logger.info("=" * 80)
    logger.info(f"TargetIDs to inspect: {args.target_ids}")
    logger.info(f"Data directory:      {args.data_dir}")
    logger.info(f"Metadata path:       {args.metadata_path}")
    logger.info(f"Output directory:    {args.output_dir}")
    
    # -------------------------------------------------------------------------
    # STEP 1: Search and Ingest Target records from Arrow Dataset
    # -------------------------------------------------------------------------
    targets_data = {}
    remaining_ids = set(args.target_ids)
    
    for split in ["train", "test"]:
        if not remaining_ids:
            break
            
        arrow_pattern = os.path.join(args.data_dir, f"{split}_*")
        arrow_folders = sorted(glob.glob(arrow_pattern))
        
        if not arrow_folders:
            continue
            
        for path in arrow_folders:
            if not remaining_ids:
                break
            try:
                logger.info(f"Scanning partition: {os.path.basename(path)} for remaining targets...")
                ds = load_from_disk(path)
                
                # Check for target presence
                targetids_in_partition = np.array(ds["targetid"], dtype=np.int64)
                for tid in list(remaining_ids):
                    row_idx = np.where(targetids_in_partition == tid)[0]
                    if len(row_idx) > 0:
                        idx = int(row_idx[0])
                        logger.info(f" -> Found TargetID {tid} in {os.path.basename(path)} split '{split}' at row index {idx}!")
                        
                        # Load record
                        row = ds[idx]
                        targets_data[tid] = {
                            "split": split,
                            "partition": os.path.basename(path),
                            "targetid": tid,
                            "redshift": row.get("redshift", 0.0),
                            "image_vis": np.array(row["image_vis"]) if row.get("image_vis") is not None else None,
                            "image_nisp_y": np.array(row["image_nisp_y"]) if row.get("image_nisp_y") is not None else None,
                            "image_nisp_j": np.array(row["image_nisp_j"]) if row.get("image_nisp_j") is not None else None,
                            "image_nisp_h": np.array(row["image_nisp_h"]) if row.get("image_nisp_h") is not None else None,
                            "spectrum_flux": np.array(row["spectrum_flux"]).flatten() if row.get("spectrum_flux") is not None else None,
                            "spectrum_wave": np.array(row["spectrum_wave"]).flatten() if row.get("spectrum_wave") is not None else None,
                        }
                        remaining_ids.remove(tid)
            except Exception as e:
                logger.error(f"Error checking partition {path}: {e}")

    # Check which targets were missing
    for tid in remaining_ids:
        logger.warning(f"TargetID {tid} could NOT be found in any partition of the processed Arrow datasets.")
        
    if not targets_data:
        logger.error("No target records loaded successfully. Inspector aborted.")
        sys.exit(1)
        
    # -------------------------------------------------------------------------
    # STEP 2: Load FITS Metadata & Quality Registries
    # -------------------------------------------------------------------------
    df_meta = None
    if os.path.exists(args.metadata_path):
        try:
            logger.info("Loading FITS binary table metadata for cross-referencing flags...")
            t = Table.read(args.metadata_path)
            # Find ID column
            id_col = None
            for candidate in ['TARGETID', 'targetid', 'ID', 'id', 'OBJID', 'objid']:
                if candidate in t.colnames:
                    id_col = candidate
                    break
            if id_col is not None:
                # Keep relevant flags
                cols_to_keep = [id_col]
                for flag in ['det_quality_flag', 'spurious_flag', 'flag_vis', 'flag_h', 'flag_j', 'flag_y', 'ra', 'dec']:
                    if flag in t.colnames:
                        cols_to_keep.append(flag)
                
                df_meta = t[cols_to_keep].to_pandas()
                df_meta.columns = [c.strip() for c in df_meta.columns]
                df_meta = df_meta.rename(columns={id_col: "TargetID"}).drop_duplicates(subset=["TargetID"]).set_index("TargetID")
                logger.info("FITS metadata loaded and indexed.")
        except Exception as e:
            logger.error(f"Failed to load FITS metadata: {e}")
            
    # Load Anomaly Classification Registry
    df_anomalies = None
    if os.path.exists(args.dark_registry):
        try:
            logger.info(f"Loading anomaly blacklist registry from: {args.dark_registry}")
            df_anomalies = pd.read_csv(args.dark_registry).drop_duplicates(subset=["TargetID"]).set_index("TargetID")
        except Exception as e:
            logger.error(f"Failed to load anomaly registry: {e}")

    # -------------------------------------------------------------------------
    # STEP 3: Generate Visual Inspector Dashboard for each found target
    # -------------------------------------------------------------------------
    for tid, data in targets_data.items():
        logger.info(f"Rendering diagnostic dashboard for TargetID {tid}...")
        
        # Pull metadata flags
        flags_str = ""
        ra_dec_str = "RA/Dec: N/A"
        if df_meta is not None and tid in df_meta.index:
            meta_row = df_meta.loc[tid]
            ra = meta_row.get("ra", 0.0)
            dec = meta_row.get("dec", 0.0)
            ra_dec_str = rf"RA/Dec: {ra:.6f}$^\circ$, {dec:.6f}$^\circ$"
            
            flags = []
            for f in ['det_quality_flag', 'spurious_flag', 'flag_vis', 'flag_y', 'flag_j', 'flag_h']:
                if f in meta_row:
                    flags.append(f"{f}={int(meta_row[f])}")
            flags_str = ", ".join(flags)
            
        # Pull anomaly class
        q_class = "Good (Passed Audit)"
        q_color = "darkgreen"
        if df_anomalies is not None and tid in df_anomalies.index:
            q_class = str(df_anomalies.loc[tid]["quality_class"])
            q_color = "darkred"
            
        z = data["redshift"]
        
        # Setup Figure & GridSpec (18x12 inches)
        fig = plt.figure(figsize=(18, 12), dpi=200)
        gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1.0, 1.0], width_ratios=[1.0, 1.0], hspace=0.35, wspace=0.25)
        
        # Set Title
        fig.suptitle(rf"\textbf{{AstroPT Target Inspector | ID: {tid} | z = {z:.4f}}}", fontsize=20, y=0.96)
        
        # --- SUBPLOT 1: Image Stamp (Lupton Composite) ---
        ax_img = fig.add_subplot(gs[0, 0])
        vis = data["image_vis"]
        y = data["image_nisp_y"]
        j = data["image_nisp_j"]
        h = data["image_nisp_h"]
        
        has_image = (vis is not None and y is not None and j is not None and h is not None)
        
        if has_image:
            try:
                rgb_stamp = make_rgb_lupton(vis, y, j, h)
                ax_img.imshow(rgb_stamp, origin='lower')
                ax_img.set_title(r"\textbf{Euclid False-Color Stamp (VIS + NISP Y/J/H)}")
            except Exception as e:
                ax_img.text(0.5, 0.5, f"RGB Lupton failed:\n{e}", ha='center', va='center', color='red', fontsize=10)
        else:
            ax_img.fill([0, 1, 1, 0], [0, 0, 1, 1], color="#eaeaea", edgecolor="#d9534f", linewidth=2)
            ax_img.text(0.5, 0.5, r"\textbf{IMAGE DATA MISSING OR CORRUPT}", color="#d9534f", ha="center", va="center", fontsize=12)
            ax_img.set_xlim(0, 1)
            ax_img.set_ylim(0, 1)
        ax_img.axis('off')
        
        # --- SUBPLOT 2: Statistics & Diagnostics Panel (Textual) ---
        ax_text = fig.add_subplot(gs[0, 1])
        ax_text.axis('off')
        
        # Compute band statistics
        stats_rows = []
        for ch_name, arr in [("VIS", vis), ("Y", y), ("J", j), ("H", h)]:
            if arr is None:
                stats_rows.append(rf"\textbf{{{ch_name}}}: & MISSING")
            else:
                nan_mask = np.isnan(arr) | np.isinf(arr)
                nan_frac = np.mean(nan_mask)
                clean_arr = arr[~nan_mask]
                
                if len(clean_arr) == 0:
                    stats_rows.append(rf"\textbf{{{ch_name}}}: & NaN Corrupted")
                else:
                    std_val = np.std(clean_arr)
                    mean_val = np.mean(clean_arr)
                    max_val = np.max(clean_arr)
                    
                    flat_tag = " [FLAT]" if std_val < 1e-4 else ""
                    stats_rows.append(
                        rf"\textbf{{{ch_name}}}: & Mean: {mean_val:.2f} | Std: {std_val:.1e}{flat_tag} | Max: {max_val:.2f} | NaNs: {nan_frac:.1%}"
                    )
                    
        stats_block = "\\\\\n".join(stats_rows)
        
        panel_text = rf"""
        \begin{{tabular}}{{ll}}
        \textbf{{Target Profile}} & \\\\
        \hline
        Split/Partition: & {data['split']} / {data['partition']} \\\\
        Coordinates: & {ra_dec_str} \\\\
        Redshift: & $z = {z:.5f}$ \\\\
        & \\\\
        \textbf{{Quality Classification}} & \\\\
        \hline
        Audit Status: & \textcolor{{{q_color}}}{{\textbf{{{q_class}}}}} \\\\
        Survey Flags: & \texttt{{{flags_str if flags_str else '(No catalog data)'}}} \\\\
        & \\\\
        \textbf{{Image Pixel Statistics}} & \\\\
        \hline
        {stats_block}
        \end{{tabular}}
        """
        ax_text.text(
            0.0, 0.9, panel_text, 
            ha='left', va='top', transform=ax_text.transAxes, fontsize=11, family='serif'
        )
        
        # --- SUBPLOTS 3 & 4: Redshift-Corrected Spectra (Blue & Red Channels) ---
        spec_flux = data["spectrum_flux"]
        spec_wave = data["spectrum_wave"]
        has_spec = (spec_flux is not None and spec_wave is not None)
        
        if has_spec and len(spec_flux) > 0:
            min_wl, max_wl = spec_wave.min(), spec_wave.max()
            mid_wl = (min_wl + max_wl) / 2.0
            
            # Channel 1: Blue Channel (Left/Top Half)
            ax_blue = fig.add_subplot(gs[1, :])
            ax_blue.plot(spec_wave, spec_flux, 'k-', lw=1.2, alpha=0.8)
            ax_blue.set_xlim(min_wl, mid_wl)
            ax_blue.set_title(r"\textbf{DESI Ground-Truth Spectrum: Blue Channel}")
            ax_blue.set_ylabel(r"Flux [$10^{-17}\text{ erg s}^{-1}\text{ cm}^{-2}\text{ \AA}^{-1}$]")
            ax_blue.grid(True, linestyle=":", alpha=0.5)
            # Find local min/max inside xlim for proper vertical scaling
            mask_blue = (spec_wave >= min_wl) & (spec_wave <= mid_wl)
            if np.any(mask_blue):
                flux_blue = spec_flux[mask_blue]
                f_min, f_max = np.percentile(flux_blue, 1), np.percentile(flux_blue, 99)
                ax_blue.set_ylim(f_min - 0.1 * abs(f_min), f_max + 0.15 * abs(f_max))
            plot_spectral_lines(ax_blue, min_wl, mid_wl, z)
            
            # Channel 2: Red Channel (Right/Bottom Half)
            ax_red = fig.add_subplot(gs[2, :])
            ax_red.plot(spec_wave, spec_flux, 'k-', lw=1.2, alpha=0.8)
            ax_red.set_xlim(mid_wl, max_wl)
            ax_red.set_title(r"\textbf{DESI Ground-Truth Spectrum: Red Channel}")
            ax_red.set_xlabel(r"Observed Wavelength [\AA]")
            ax_red.set_ylabel(r"Flux [$10^{-17}\text{ erg s}^{-1}\text{ cm}^{-2}\text{ \AA}^{-1}$]")
            ax_red.grid(True, linestyle=":", alpha=0.5)
            # Find local min/max inside xlim for proper vertical scaling
            mask_red = (spec_wave >= mid_wl) & (spec_wave <= max_wl)
            if np.any(mask_red):
                flux_red = spec_flux[mask_red]
                f_min, f_max = np.percentile(flux_red, 1), np.percentile(flux_red, 99)
                ax_red.set_ylim(f_min - 0.1 * abs(f_min), f_max + 0.15 * abs(f_max))
            plot_spectral_lines(ax_red, mid_wl, max_wl, z)
        else:
            # Draw empty placeholder for spectra
            ax_spec_none = fig.add_subplot(gs[1:, :])
            ax_spec_none.fill([0, 1, 1, 0], [0, 0, 1, 1], color="#eaeaea", edgecolor="#d9534f", linewidth=2)
            ax_spec_none.text(0.5, 0.5, r"\textbf{DESI SPECTRA DATA MISSING OR CORRUPT}", color="#d9534f", ha="center", va="center", fontsize=15)
            ax_spec_none.set_xlim(0, 1)
            ax_spec_none.set_ylim(0, 1)
            ax_spec_none.axis('off')
            
        # Save Dashboard
        save_path = os.path.join(args.output_dir, f"target_diagnostic_{tid}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Dashboard saved successfully to: {save_path}")

    logger.info("=" * 80)
    logger.info("ASTROPT TARGET AUDIT COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
