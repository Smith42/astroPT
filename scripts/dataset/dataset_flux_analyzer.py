#!/usr/bin/env python3
"""
AstroPT Dataset Flux Analyzer
=============================
Author: Senior Astro-Data Scientist & PyTorch/HPC Infrastructure Engineer
Description: 
    This script performs a rigorous data quality audit of Euclid VIS and NISP 
    image fluxes in the processed AstroPT Apache Arrow dataset. It detects 
    completely black (zero-flux) or NaN-corrupted galaxy images, cross-references 
    them with the FITS catalog, identifies the underlying quality flags causing 
    the blanks, and generates a recommended filter string for training.
"""

import os
import sys
import json
import glob
import logging
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.table import Table
from datasets import load_from_disk, concatenate_datasets

# Setup premium logging to console and log files
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("astropt_flux_analyzer")

# Apply premium plot aesthetics
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 15,
    'figure.dpi': 200,
    'savefig.bbox': 'tight'
})

def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="AstroPT Processed Dataset Flux Integrity Analyzer")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated",
        help="Root directory containing processed Arrow datasets splits."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which split of the dataset to audit (train or test)."
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits",
        help="Absolute path to the FITS catalog containing all survey metadata."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/astropt_dataset_audit",
        help="Directory to save generated CSV reports, JSON summary, and diagnostic plots."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="Max samples to analyze. Set to -1 to audit the entire dataset split."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for high-speed HuggingFace Arrow batch loading."
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
    
    return rgb_scaled.transpose(1, 2, 0) # Return shape (H, W, 3) for plotting

def main():
    args = parse_args()
    
    # -------------------------------------------------------------------------
    # STAGE 0: Directory Initialization & Setup
    # -------------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    stamps_dir = os.path.join(args.output_dir, "stamps")
    os.makedirs(stamps_dir, exist_ok=True)
    
    # Configure file logging
    log_file = os.path.join(args.output_dir, "flux_analyzer.log")
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)
    
    logger.info("=" * 80)
    logger.info("STARTING ASTROPT DATASET FLUX AND QUALITY ANALYZER")
    logger.info("=" * 80)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Metadata path: {args.metadata_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # -------------------------------------------------------------------------
    # STAGE 1: Dataset Ingestion (Apache Arrow)
    # -------------------------------------------------------------------------
    logger.info("Stage 1: Ingesting processed Arrow dataset splits...")
    arrow_pattern = os.path.join(args.data_dir, f"{args.split}_*")
    arrow_folders = sorted(glob.glob(arrow_pattern))
    
    if not arrow_folders:
        logger.error(f"No processed Arrow folders found matching: {arrow_pattern}")
        sys.exit(1)
        
    logger.info(f"Found {len(arrow_folders)} dataset split directories.")
    
    datasets_list = []
    for path in arrow_folders:
        logger.info(f"Loading partition: {os.path.basename(path)}")
        try:
            datasets_list.append(load_from_disk(path))
        except Exception as e:
            logger.warning(f"Failed to load dataset folder '{path}' directly. Attempting raw arrow loader fallback: {e}")
            arrow_files = sorted(glob.glob(os.path.join(path, "*.arrow")))
            if arrow_files:
                from datasets import load_dataset
                datasets_list.append(load_dataset("arrow", data_files=arrow_files, split="train"))
                
    if not datasets_list:
        logger.error("No valid HuggingFace datasets loaded. Audit aborted.")
        sys.exit(1)
        
    ds = concatenate_datasets(datasets_list)
    total_len = len(ds)
    logger.info(f"HuggingFace dataset concatenated successfully. Total rows: {total_len:,}")
    
    # Setup sample indices
    # OPTIMIZATION: Random sampling on HuggingFace Datasets with .select() is extremely slow because 
    # non-contiguous indices trigger PyArrow to perform random disk seeks across 180 GB of Arrow shards.
    # By selecting a contiguous slice sequentially, we keep PyArrow in sequential disk-read mode.
    # Since Arrow shards are pre-randomized during creation, a contiguous slice is perfectly representative.
    if args.max_samples > 0 and args.max_samples < total_len:
        logger.info(f"Representative contiguous subset requested. Selecting the first {args.max_samples:,} sources sequentially to leverage C++ PyArrow sequential memory mapping...")
        sample_indices = list(range(args.max_samples))
    else:
        logger.info("Full dataset audit requested. Processing all records...")
        sample_indices = list(range(total_len))
        
    # Column checking
    required_cols = ["targetid", "redshift", "image_vis", "image_nisp_y", "image_nisp_j", "image_nisp_h"]
    missing_cols = [c for c in required_cols if c not in ds.column_names]
    if missing_cols:
        logger.error(f"Missing required columns in dataset schema: {missing_cols}")
        sys.exit(1)
        
    # -------------------------------------------------------------------------
    # STAGE 2: Image Flux Auditing (High-Speed Batch Parsing)
    # -------------------------------------------------------------------------
    logger.info("Stage 2: Starting high-speed image flux audit across VIS, Y, J, and H filters...")
    
    audited_data = []
    
    # Setup sub-slices for visual stamp extraction
    good_stamps = []
    dark_stamps = []
    
    # Process dataset in batches to avoid CPU memory overhead
    num_samples = len(sample_indices)
    batch_size = args.batch_size
    
    # Check if sample_indices is a contiguous range to allow direct dictionary slicing
    is_contiguous = (sample_indices == list(range(num_samples)))
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        
        # Load batch of rows as dict
        logger.info(f"Processing batch: sources {start_idx:,} to {end_idx:,} / {num_samples:,}...")
        
        if is_contiguous:
            # OPTIMIZATION: Direct dictionary slicing ds[start:end] is up to 1000x faster than .select()
            # because it completely bypasses pointer index resolution and reads memory-mapped columns sequentially!
            batch_data = ds[start_idx:end_idx]
        else:
            batch_indices = sample_indices[start_idx:end_idx]
            batch_ds = ds.select(batch_indices)
            batch_data = {
                "targetid": batch_ds["targetid"],
                "redshift": batch_ds["redshift"],
                "image_vis": batch_ds["image_vis"],
                "image_nisp_y": batch_ds["image_nisp_y"],
                "image_nisp_j": batch_ds["image_nisp_j"],
                "image_nisp_h": batch_ds["image_nisp_h"]
            }
        
        targetids = batch_data["targetid"]
        redshifts = batch_data["redshift"]
        image_vis = batch_data["image_vis"]
        image_y = batch_data["image_nisp_y"]
        image_j = batch_data["image_nisp_j"]
        image_h = batch_data["image_nisp_h"]
        
        for i in range(len(targetids)):
            tid = int(targetids[i])
            z = float(redshifts[i]) if redshifts[i] is not None else 0.0
            
            ch_stats = {}
            bands = {"VIS": image_vis[i], "Y": image_y[i], "J": image_j[i], "H": image_h[i]}
            
            for band_name, band_data in bands.items():
                if band_data is None:
                    ch_stats[f"{band_name}_missing"] = True
                    ch_stats[f"{band_name}_black"] = False
                    ch_stats[f"{band_name}_nan_fraction"] = 1.0
                    ch_stats[f"{band_name}_mean"] = 0.0
                    ch_stats[f"{band_name}_max"] = 0.0
                    ch_stats[f"{band_name}_std"] = 0.0
                else:
                    arr = np.array(band_data, dtype=np.float32)
                    ch_stats[f"{band_name}_missing"] = False
                    
                    # Identify NaNs or Infs
                    nan_mask = np.isnan(arr) | np.isinf(arr)
                    nan_fraction = float(np.mean(nan_mask))
                    ch_stats[f"{band_name}_nan_fraction"] = nan_fraction
                    
                    clean_arr = arr[~nan_mask]
                    if clean_arr.size == 0:
                        ch_stats[f"{band_name}_black"] = True
                        ch_stats[f"{band_name}_mean"] = 0.0
                        ch_stats[f"{band_name}_max"] = 0.0
                        ch_stats[f"{band_name}_std"] = 0.0
                    else:
                        mean_val = float(np.mean(clean_arr))
                        max_val = float(np.max(clean_arr))
                        std_val = float(np.std(clean_arr))
                        
                        ch_stats[f"{band_name}_mean"] = mean_val
                        ch_stats[f"{band_name}_max"] = max_val
                        ch_stats[f"{band_name}_std"] = std_val
                        
                        # An image is black if max is zero, extremely close to zero, or if variance is zero
                        ch_stats[f"{band_name}_black"] = (std_val == 0.0) or (max_val <= 1e-7)
            
            # Outlier Quality Classification
            vis_black = ch_stats["VIS_black"] or ch_stats["VIS_missing"]
            y_black = ch_stats["Y_black"] or ch_stats["Y_missing"]
            j_black = ch_stats["J_black"] or ch_stats["J_missing"]
            h_black = ch_stats["H_black"] or ch_stats["H_missing"]
            
            is_all_black = vis_black and y_black and j_black and h_black
            is_partially_black = (vis_black or y_black or j_black or h_black) and not is_all_black
            
            vis_nan = ch_stats["VIS_nan_fraction"] == 1.0
            y_nan = ch_stats["Y_nan_fraction"] == 1.0
            j_nan = ch_stats["J_nan_fraction"] == 1.0
            h_nan = ch_stats["H_nan_fraction"] == 1.0
            is_nan_corrupted = vis_nan or y_nan or j_nan or h_nan
            
            if is_all_black:
                quality_class = "All-Black"
            elif is_nan_corrupted:
                quality_class = "NaN-Corrupted"
            elif is_partially_black:
                quality_class = "Partially-Black"
            else:
                quality_class = "Good"
                
            entry = {
                "TargetID": tid,
                "redshift": z,
                "quality_class": quality_class,
                "VIS_missing": ch_stats["VIS_missing"],
                "VIS_black": ch_stats["VIS_black"],
                "VIS_mean": ch_stats["VIS_mean"],
                "VIS_max": ch_stats["VIS_max"],
                "VIS_std": ch_stats["VIS_std"],
                "Y_missing": ch_stats["Y_missing"],
                "Y_black": ch_stats["Y_black"],
                "Y_mean": ch_stats["Y_mean"],
                "Y_max": ch_stats["Y_max"],
                "Y_std": ch_stats["Y_std"],
                "J_missing": ch_stats["J_missing"],
                "J_black": ch_stats["J_black"],
                "J_mean": ch_stats["J_mean"],
                "J_max": ch_stats["J_max"],
                "J_std": ch_stats["J_std"],
                "H_missing": ch_stats["H_missing"],
                "H_black": ch_stats["H_black"],
                "H_mean": ch_stats["H_mean"],
                "H_max": ch_stats["H_max"],
                "H_std": ch_stats["H_std"],
            }
            audited_data.append(entry)
            
            # Pull stamps for diagnostic visual checker (save raw matrices for plotting later)
            if quality_class == "Good" and len(good_stamps) < 3 and bands["VIS"] is not None:
                good_stamps.append((tid, np.array(bands["VIS"]), np.array(bands["Y"]), np.array(bands["J"]), np.array(bands["H"])))
            elif quality_class == "All-Black" and len(dark_stamps) < 3 and bands["VIS"] is not None:
                dark_stamps.append((tid, np.array(bands["VIS"]), np.array(bands["Y"]), np.array(bands["J"]), np.array(bands["H"])))

    # Convert audit results to DataFrame
    df_audit = pd.DataFrame(audited_data)
    
    # -------------------------------------------------------------------------
    # STAGE 3: FITS Catalog Cross-Referencing & Flag Profiling
    # -------------------------------------------------------------------------
    logger.info("Stage 3: Cross-referencing anomalous black galaxies with FITS metadata catalog...")
    
    if not os.path.exists(args.metadata_path):
        logger.error(f"FITS catalog not found at: {args.metadata_path}. Unable to execute Stage 3.")
        sys.exit(1)
        
    try:
        logger.info("Reading FITS binary table metadata...")
        t = Table.read(args.metadata_path)
        logger.info(f"Loaded FITS catalog successfully. Rows: {len(t):,}")
        
        # Resolve ID column name in the catalog
        id_col = None
        for candidate in ['TARGETID', 'targetid', 'ID', 'id', 'OBJID', 'objid']:
            if candidate in t.colnames:
                id_col = candidate
                break
        if id_col is None:
            raise KeyError(f"Could not identify target ID column in catalog columns: {t.colnames}")
            
        # Convert table to Pandas DataFrame for high-speed indexing
        df_meta = t[[id_col, 'det_quality_flag', 'spurious_flag', 'flag_vis', 'flag_h', 'flag_j', 'flag_y']].to_pandas()
        del t # Free up memory
        
        # Clean columns and index
        df_meta.columns = [c.strip() for c in df_meta.columns]
        df_meta = df_meta.rename(columns={id_col: "TargetID"}).drop_duplicates(subset=["TargetID"]).set_index("TargetID")
        
        # Join audit results with FITS metadata flags
        df_joined = df_audit.join(df_meta, on="TargetID", how="left")
    except Exception as e:
        logger.error(f"FITS catalog loading or cross-referencing failed: {str(e)}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # STAGE 4: Quality Flag Conditional Probability Analysis
    # -------------------------------------------------------------------------
    logger.info("Stage 4: Conducting conditional probability analysis on quality flags...")
    
    # Global metrics
    total_audited = len(df_joined)
    good_count = len(df_joined[df_joined["quality_class"] == "Good"])
    black_count = len(df_joined[df_joined["quality_class"] == "All-Black"])
    part_black_count = len(df_joined[df_joined["quality_class"] == "Partially-Black"])
    nan_count = len(df_joined[df_joined["quality_class"] == "NaN-Corrupted"])
    
    logger.info("-" * 50)
    logger.info("DATASET QUALITY SUMMARY STATS:")
    logger.info(f"  Total Audited Samples:  {total_audited:,}")
    logger.info(f"  Operational (Good):     {good_count:,} ({good_count/total_audited:.2%})")
    logger.info(f"  All-Black Anomalies:    {black_count:,} ({black_count/total_audited:.2%})")
    logger.info(f"  Part-Black Anomalies:   {part_black_count:,} ({part_black_count/total_audited:.2%})")
    logger.info(f"  NaN-Corrupted Samples:  {nan_count:,} ({nan_count/total_audited:.2%})")
    logger.info("-" * 50)
    
    # Profiling quality flags
    flag_stats = {}
    
    # We analyze Euclid flag columns
    quality_flags = ['det_quality_flag', 'spurious_flag', 'flag_vis', 'flag_h', 'flag_j', 'flag_y']
    
    for flag in quality_flags:
        if flag not in df_joined.columns:
            logger.warning(f"Flag column '{flag}' was not found in the cross-referenced catalog.")
            continue
            
        # Drop NaN values for flag analysis
        df_valid_flag = df_joined.dropna(subset=[flag])
        if len(df_valid_flag) == 0:
            continue
            
        unique_vals = sorted(df_valid_flag[flag].unique())
        flag_stats[flag] = {}
        
        for val in unique_vals:
            subset = df_valid_flag[df_valid_flag[flag] == val]
            sub_total = len(subset)
            sub_black = len(subset[subset["quality_class"] == "All-Black"])
            
            # P(Black | Flag == val)
            cond_prob = sub_black / sub_total if sub_total > 0 else 0.0
            
            flag_stats[flag][str(val)] = {
                "total_records": sub_total,
                "black_records": sub_black,
                "conditional_black_probability": cond_prob
            }
            logger.info(f"Flag Analysis: P(All-Black | {flag} == {val}) = {cond_prob:.2%} ({sub_black}/{sub_total})")

    # Generate a recommended clean filtering string
    # We inspect which flags heavily predict "All-Black" or corrupted entries.
    # Typically, spurious_flag == 1 or det_quality_flag == 1 or flag_vis == 0 represent bad images.
    filters_list = []
    
    # 1. VIS Flag check
    if 'flag_vis' in df_joined.columns:
        # Check if flag_vis == 0 correlates with black VIS channel
        df_vis_0 = df_joined[df_joined['flag_vis'] == 0]
        if len(df_vis_0) > 0:
            vis_0_black = len(df_vis_0[df_vis_0['VIS_black'] == True])
            if vis_0_black / len(df_vis_0) > 0.90:
                filters_list.append("flag_vis > 0")
                
    # 2. Spurious Flag check
    if 'spurious_flag' in df_joined.columns:
        df_spurious_1 = df_joined[df_joined['spurious_flag'] == 1]
        if len(df_spurious_1) > 0:
            spurious_1_black = len(df_spurious_1[df_spurious_1['quality_class'] == "All-Black"])
            if spurious_1_black / len(df_spurious_1) > 0.80:
                filters_list.append("spurious_flag == 0")
                
    # 3. Detection Quality Flag check
    if 'det_quality_flag' in df_joined.columns:
        df_quality_1 = df_joined[df_joined['det_quality_flag'] == 1]
        if len(df_quality_1) > 0:
            quality_1_black = len(df_quality_1[df_quality_1['quality_class'] == "All-Black"])
            if quality_1_black / len(df_quality_1) > 0.80:
                filters_list.append("det_quality_flag == 0")

    # Fallback to standard Euclid clean-up if flags correlate
    if not filters_list:
        filters_list = ["flag_vis > 0", "spurious_flag == 0", "det_quality_flag == 0"]
        
    recommended_filter_str = " && ".join(filters_list)
    logger.info(f"RECOMMENDED AstroPT CONFIG DATASET FILTER: ['{recommended_filter_str}']")
    
    # -------------------------------------------------------------------------
    # STAGE 5: Save Quality Audit Reports (CSV and JSON)
    # -------------------------------------------------------------------------
    logger.info("Stage 5: Saving structured quality audit reports and registries...")
    
    # Save the registry of completely black/dark galaxies
    df_anomalies = df_joined[df_joined["quality_class"] != "Good"]
    anomalies_csv_path = os.path.join(args.output_dir, "dark_galaxies_registry.csv")
    df_anomalies.to_csv(anomalies_csv_path, index=False)
    logger.info(f"Saved registry of anomalous/dark galaxies to: {anomalies_csv_path}")
    
    # Save full processed dataset stats for debugging
    all_stats_path = os.path.join(args.output_dir, "dataset_flux_statistics.csv")
    df_joined.to_csv(all_stats_path, index=False)
    
    # Save summary stats JSON
    summary_data = {
        "dataset_split": args.split,
        "total_audited_samples": total_audited,
        "operational_good_samples": good_count,
        "operational_percentage": good_count / total_audited,
        "all_black_samples": black_count,
        "all_black_percentage": black_count / total_audited,
        "partially_black_samples": part_black_count,
        "partially_black_percentage": part_black_count / total_audited,
        "nan_corrupted_samples": nan_count,
        "nan_corrupted_percentage": nan_count / total_audited,
        "conditional_probabilities_quality_flags": flag_stats,
        "recommended_config_filter_string": [recommended_filter_str]
    }
    
    summary_json_path = os.path.join(args.output_dir, "flux_quality_summary.json")
    with open(summary_json_path, 'w') as f_json:
        json.dump(summary_data, f_json, indent=4)
    logger.info(f"Saved quality summary statistics report to: {summary_json_path}")
    
    # -------------------------------------------------------------------------
    # STAGE 6: Generate Diagnostic Charts & Visual Stamps
    # -------------------------------------------------------------------------
    logger.info("Stage 6: Rendering diagnostic charts and visual False RGB verification stamps...")
    
    # Plot 1: Flux Value Distributions (VIS vs NISP bands)
    plt.figure(figsize=(15, 10))
    channels = ["VIS", "Y", "J", "H"]
    colors = ["blue", "green", "orange", "red"]
    
    for idx, (ch, color) in enumerate(zip(channels, colors)):
        plt.subplot(2, 2, idx + 1)
        # Drop missing values and filter to positive values for logarithmic plotting
        valid_means = df_joined[df_joined[f"{ch}_missing"] == False][f"{ch}_mean"].values
        # Separate good vs all-black
        good_means = df_joined[(df_joined[f"{ch}_missing"] == False) & (df_joined["quality_class"] == "Good")][f"{ch}_mean"].values
        black_means = df_joined[(df_joined[f"{ch}_missing"] == False) & (df_joined["quality_class"] == "All-Black")][f"{ch}_mean"].values
        
        if good_means.size > 0:
            sns.histplot(good_means, kde=True, color=color, label="Operational (Good)", bins=50, alpha=0.6)
        if black_means.size > 0:
            sns.histplot(black_means, kde=False, color="black", label="All-Black Anomalies", bins=10, alpha=0.9, log_scale=False)
            
        plt.title(f"Euclid {ch} Mean Pixel Flux Distribution", fontsize=11, fontweight="bold")
        plt.xlabel("Mean Flux", fontsize=9)
        plt.ylabel("Density", fontsize=9)
        plt.grid(True, alpha=0.15)
        plt.legend(fontsize=8)
        
    plt.suptitle("AstroPT Processed Dataset Image Flux Distributions (Good vs. Dark Galaxies)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    flux_plot_path = os.path.join(args.output_dir, "flux_distribution_comparison.png")
    plt.savefig(flux_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Correlation of Quality Flags to Blank Images
    # We plot P(Black | flag == val) for different quality flags
    plt.figure(figsize=(12, 6))
    flag_plot_data = []
    for flag, vals in flag_stats.items():
        for val, stats in vals.items():
            flag_plot_data.append({
                "Quality Flag Indicator": f"{flag} == {val}",
                "Outlier Ratio (%)": stats["conditional_black_probability"] * 100.0,
                "Sample Count": stats["total_records"]
            })
            
    df_plot_flags = pd.DataFrame(flag_plot_data)
    if not df_plot_flags.empty:
        sns.barplot(
            data=df_plot_flags,
            x="Outlier Ratio (%)",
            y="Quality Flag Indicator",
            palette="viridis",
            edgecolor="black"
        )
        plt.title("Euclid Quality Metadata Flags vs. Blank Image Outlier Ratios\n(Identifying which survey markers correlate with zero-flux images)", fontsize=13, fontweight="bold", pad=15)
        plt.xlabel("Probability of Galaxy Image Being Blank P(All-Black | Flag) (%)", fontsize=10, fontweight="bold")
        plt.ylabel("Survey Metadata Flag State", fontsize=10, fontweight="bold")
        plt.grid(True, alpha=0.15)
        plt.xlim(0, 105)
        
        # Label bars with count of samples
        for index, row in df_plot_flags.iterrows():
            plt.text(
                row["Outlier Ratio (%)"] + 1,
                index,
                f"N={row['Sample Count']:,}",
                color="black",
                ha="left",
                va="center",
                fontsize=9
            )
            
        plt.tight_layout()
        flag_plot_path = os.path.join(args.output_dir, "quality_flags_correlation.png")
        plt.savefig(flag_plot_path, dpi=200, bbox_inches='tight')
        plt.close()

    # Plot 3: Visual Verification Stamps Grid (3x2 Comparison)
    # Renders 3 Good and 3 All-Black galaxies side-by-side using false-color RGB Lupton stretching
    if len(good_stamps) > 0 or len(dark_stamps) > 0:
        logger.info("Generating false-color RGB visual stamp grid...")
        fig = plt.figure(figsize=(18, 12), dpi=150)
        gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.2)
        
        # Plot Good Stamps (Row 0)
        for idx, (tid, vis, y, j, h) in enumerate(good_stamps):
            ax = fig.add_subplot(gs[0, idx])
            try:
                rgb_stamp = make_rgb_lupton(vis, y, j, h)
                ax.imshow(rgb_stamp, origin='lower')
                ax.set_title(f"Operational Galaxy\nTargetID: {tid}", color="darkgreen", fontsize=11, fontweight="bold")
            except Exception as e:
                ax.text(0.5, 0.5, f"RGB Render Failed: {e}", ha='center', va='center', fontsize=9)
            ax.axis('off')
            
        # Plot All-Black Stamps (Row 1)
        for idx, (tid, vis, y, j, h) in enumerate(dark_stamps):
            ax = fig.add_subplot(gs[1, idx])
            try:
                rgb_stamp = make_rgb_lupton(vis, y, j, h)
                ax.imshow(rgb_stamp, origin='lower')
                ax.set_title(f"Zero-Flux Outlier Stamp\nTargetID: {tid}", color="darkred", fontsize=11, fontweight="bold")
            except Exception as e:
                ax.text(0.5, 0.5, f"RGB Render Failed: {e}", ha='center', va='center', fontsize=9)
            ax.axis('off')
            
        plt.suptitle("AstroPT Visual Quality Stamps: Operational Galaxies vs. Zero-Flux Outlier Stamps\n(Euclid VIS+NISP False-Color RGB composite plots with asinh scaling)", fontsize=14, fontweight="bold", y=0.96)
        
        stamps_plot_path = os.path.join(args.output_dir, "dark_vs_good_stamps.png")
        plt.savefig(stamps_plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"Visual composite stamps grid successfully saved to: {stamps_plot_path}")
        
    logger.info("=" * 80)
    logger.info("ASTROPT DATASET FLUX INTEGRITY AUDIT COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
