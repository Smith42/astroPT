#!/usr/bin/env python3
"""
AstroPT Dataset Audit Pipeline
=============================
Author: Senior Astro-Data Scientist & PyTorch/HPC Infrastructure Engineer
Description: 
    This script performs a multi-stage forensic data analysis of the AstroPT FITS
    metadata catalog. It dynamically segregates instrumental control variables from
    true physical galaxy properties, computes their Spearman rank correlations,
    automatically detects highly coupled "Hot Variables" (selection biases), and
    generates diagnostics via targeted pairplots (corner plots) to flag selection biases.
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.table import Table

# Set up clean, professional logging to both stdout and a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("astropt_audit")

# Apply highly premium, professional plot aesthetics
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

def parse_arguments():
    """Parses command-line arguments for the dataset audit."""
    parser = argparse.ArgumentParser(
        description="Forensic Dataset Audit Pipeline for AstroPT Foundation Model Catalog."
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="Absolute path to the Euclid/DESI FITS metadata catalog."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Absolute path to the directory where output audit plots and reports will be saved."
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # -------------------------------------------------------------------------
    # STAGE 0: Directory Initialization
    # -------------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Add file logging to the output directory
    log_file = os.path.join(args.output_dir, "audit_pipeline_execution.log")
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)
    
    logger.info("=" * 80)
    logger.info("STARTING ASTROPT FORENSIC DATASET AUDIT")
    logger.info("=" * 80)
    logger.info(f"Metadata Path: {args.metadata_path}")
    logger.info(f"Output Directory: {args.output_dir}")
    
    # -------------------------------------------------------------------------
    # STAGE 1: Data Ingestion & Categorization
    # -------------------------------------------------------------------------
    logger.info("Stage 1: Ingesting FITS metadata catalog...")
    
    if not os.path.exists(args.metadata_path):
        logger.error(f"Catalog file not found at: {args.metadata_path}")
        sys.exit(1)
        
    try:
        # Ingest the FITS table using astropy.table.Table
        logger.info("Reading FITS binary table using astropy.table.Table...")
        fits_table = Table.read(args.metadata_path)
        logger.info(f"FITS catalog loaded successfully. Found {len(fits_table)} rows and {len(fits_table.colnames)} columns.")
        
        # Convert table to Pandas DataFrame for high-performance statistical operations
        logger.info("Converting Astropy Table to Pandas DataFrame...")
        df = fits_table.to_pandas()
        del fits_table # Free up memory
    except Exception as e:
        logger.error(f"Failed to ingest FITS catalog: {str(e)}")
        sys.exit(1)

    # Clean up column names (strip trailing/leading whitespace and decode if necessary)
    df.columns = [col.strip() for col in df.columns]

    # Stage 1: Dynamic Column Parsing & Categorization
    # Identify all numeric columns dynamically from the DataFrame to avoid hardcoding
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude typical ID, row-index, and coordinate columns which are non-informative for correlation analysis
    id_keywords = {'id', 'index', 'name', 'obj', 'coord', 'ra', 'dec', 'targetid', 'field', 'tile', 'row_id'}
    active_cols = []
    for col in numeric_cols:
        col_lower = col.lower()
        if any(kw in col_lower for kw in id_keywords):
            continue
        # Drop columns with zero variance (single unique value) or all NaN values
        if df[col].nunique() > 1 and not df[col].isna().all():
            active_cols.append(col)
            
    logger.info(f"Identified {len(active_cols)} active numeric columns out of {len(df.columns)} catalog columns.")
    
    # Dynamically categorize these columns into instrumental controls and physical properties
    # Instrumental features include SNR, error flags, quality indices, CHI2, observation weights, etc.
    # Physical target features include stellar properties, line fluxes, redshift, segmentation area, etc.
    instrumental_keywords = {'snr', 'flag', 'chi2', 'err', 'unc', 'ivar', 'mask', 'weight', 'quality', 'spurious', 'exptime', 'nobs'}
    
    instrumental_control_cols = []
    physical_target_cols = []
    
    for col in active_cols:
        col_lower = col.lower()
        is_inst = any(kw in col_lower for kw in instrumental_keywords)
        if is_inst:
            instrumental_control_cols.append(col)
        else:
            physical_target_cols.append(col)
            
    logger.info(f"Dynamically segregated: {len(instrumental_control_cols)} Instrumental Controls, {len(physical_target_cols)} Physical Targets.")
    
    # Combine active columns ensuring instrumental variables are grouped first, followed by physical variables
    combined_cols = instrumental_control_cols + physical_target_cols
    if not combined_cols:
        combined_cols = active_cols
        
    # Robust NaN filtration:
    # Catalogs combined from diverse surveys (e.g., Euclid MER + DESI) may have high NaN density in distinct columns.
    # Drop columns that have more than 15% NaN entries to avoid dropping the entire row matrix during .dropna()
    initial_rows = len(df)
    nan_threshold = 0.85 * initial_rows
    valid_cols = [col for col in combined_cols if df[col].notna().sum() >= nan_threshold]
    
    logger.info(f"NaN filtration: Retained {len(valid_cols)} / {len(combined_cols)} columns with >= 85% non-NaN values.")
    
    # Update our logical families with columns that passed the NaN density check
    instrumental_control_cols = [c for c in instrumental_control_cols if c in valid_cols]
    physical_target_cols = [c for c in physical_target_cols if c in valid_cols]
    combined_cols = instrumental_control_cols + physical_target_cols
    
    # Fallback in case the strict NaN density threshold leaves too few columns
    if not combined_cols or len(combined_cols) < 2:
        logger.warning("Strict NaN threshold resulted in too few columns. Relaxing threshold to retain all active columns with data...")
        combined_cols = [c for c in active_cols if df[c].notna().sum() > 0]
        
    df_clean = df[combined_cols].dropna()
    final_rows = len(df_clean)
    
    # If the clean matrix is still extremely small, perform a relaxed fallback where columns with at least 50% non-NaNs are kept
    if final_rows < 100:
        logger.warning(f"Extremely small sample size ({final_rows} rows) after strict dropna. Slicing columns with >= 50% non-NaNs...")
        relaxed_cols = [c for c in active_cols if df[c].notna().sum() >= 0.5 * initial_rows]
        if len(relaxed_cols) >= 2:
            df_clean = df[relaxed_cols].dropna()
            combined_cols = relaxed_cols
            instrumental_control_cols = [c for c in instrumental_control_cols if c in combined_cols]
            physical_target_cols = [c for c in physical_target_cols if c in combined_cols]
            final_rows = len(df_clean)
            logger.info(f"Relaxed columns: {len(combined_cols)}, Rows retained: {final_rows:,}")
            
    logger.info(f"Final synchronous analysis matrix: {final_rows:,} rows x {len(combined_cols)} columns.")
    
    if final_rows == 0:
        logger.error("Error: Synchronous data matrix is empty. Audit aborted.")
        sys.exit(1)
        
    # -------------------------------------------------------------------------
    # STAGE 2: Macro Analysis via Spearman Rank Correlation
    # -------------------------------------------------------------------------
    logger.info("Stage 2: Computing full Spearman Rank Correlation Matrix...")
    
    # Compute the full Spearman rank correlation matrix
    corr_matrix = df_clean[combined_cols].corr(method='spearman')
    
    # Generate and save the Spearman Correlation Matrix as a CSV file (highly requested)
    csv_path = os.path.join(args.output_dir, "spearman_correlation_matrix.csv")
    corr_matrix.to_csv(csv_path)
    logger.info(f"Spearman rank correlation matrix successfully exported to: {csv_path}")
    
    logger.info("Generating and saving Full Macro Correlation Heatmap (All Variables)...")
    
    # Dynamic layout setup based on matrix dimensionality
    n_vars = len(combined_cols)
    figsize_width = max(15, min(35, n_vars * 0.8))
    figsize_height = max(12, min(30, n_vars * 0.7))
    plt.figure(figsize=(figsize_width, figsize_height))
    
    # Determine tick label font size dynamically to prevent overlapping
    tick_fontsize = max(4, min(10, 250 // n_vars))
    
    # Only render cell annotations if the feature set is small enough (<= 35 columns) to avoid rendering bottlenecks/overlaps
    show_annotations = n_vars <= 35
    
    # Plot using a divergent, high-premium colormap (coolwarm)
    sns.heatmap(
        corr_matrix,
        annot=show_annotations,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        linewidths=0.5 if n_vars < 50 else 0.1,
        linecolor="#efefef",
        cbar_kws={"label": "Spearman Rank Correlation Coefficient ($r_s$)"},
        annot_kws={"size": max(5, min(8, 200 // n_vars)), "weight": "bold"}
    )
    
    plt.title(
        f"AstroPT Macro Dataset Audit: Full Feature Correlation Matrix\n"
        f"(Spearman Rank Correlation - {n_vars} variables with dynamic survey cross-talk)",
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    plt.xlabel("Variables (Instrumental Controls & Galactic Targets)", fontsize=12, fontweight='bold', labelpad=12)
    plt.ylabel("Variables (Instrumental Controls & Galactic Targets)", fontsize=12, fontweight='bold', labelpad=12)
    
    plt.xticks(rotation=45, ha='right', fontsize=tick_fontsize)
    plt.yticks(rotation=0, fontsize=tick_fontsize)
    
    heatmap_path = os.path.join(args.output_dir, "macro_correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Macro correlation heatmap successfully saved to: {heatmap_path}")
    
    # -------------------------------------------------------------------------
    # STAGE 3: Automated "Hot Variables" (Bias Alarm) Detector
    # -------------------------------------------------------------------------
    logger.info("Stage 3: Running algorithmic threshold watcher for selection bias detection...")
    
    BIAS_THRESHOLD = 0.65
    hot_variables = set()
    alarms_triggered = 0
    
    alert_log_path = os.path.join(args.output_dir, "bias_alarms.txt")
    
    with open(alert_log_path, 'w') as f_alert:
        f_alert.write("=" * 80 + "\n")
        f_alert.write("ASTROPT DATASET AUDIT: SELECTION BIAS ALARM REGISTER\n")
        f_alert.write(f"Threshold: |Spearman r| > {BIAS_THRESHOLD}\n")
        f_alert.write(f"Total Columns Evaluated: {n_vars}\n")
        f_alert.write("=" * 80 + "\n\n")
        
        # If we have distinct physical and instrumental columns, watch the cross-coupling
        if physical_target_cols and instrumental_control_cols:
            for phys in physical_target_cols:
                for inst in instrumental_control_cols:
                    if phys in corr_matrix.index and inst in corr_matrix.columns:
                        r_val = corr_matrix.loc[phys, inst]
                        if abs(r_val) > BIAS_THRESHOLD:
                            alarms_triggered += 1
                            alert_msg = f"[BIAS ALARM] Physical property '{phys}' is highly coupled with Instrumental feature '{inst}' (Spearman r = {r_val:.2f})"
                            print(f"\033[91m\033[1m{alert_msg}\033[0m")
                            logger.warning(alert_msg)
                            f_alert.write(f"{alert_msg}\n")
                            hot_variables.add(phys)
                            hot_variables.add(inst)
        else:
            # Fallback: compare all unique variable pairs in the dataset
            logger.info("Performing comprehensive all-to-all pairing analysis for alarms...")
            for i in range(len(combined_cols)):
                for j in range(i + 1, len(combined_cols)):
                    col1 = combined_cols[i]
                    col2 = combined_cols[j]
                    r_val = corr_matrix.loc[col1, col2]
                    if abs(r_val) > BIAS_THRESHOLD:
                        alarms_triggered += 1
                        alert_msg = f"[BIAS ALARM] Variable '{col1}' is highly coupled with Variable '{col2}' (Spearman r = {r_val:.2f})"
                        print(f"\033[91m\033[1m{alert_msg}\033[0m")
                        logger.warning(alert_msg)
                        f_alert.write(f"{alert_msg}\n")
                        hot_variables.add(col1)
                        hot_variables.add(col2)
                        
        f_alert.write(f"\nTotal alarms triggered: {alarms_triggered}\n")
        f_alert.write(f"Unique hot variables identified: {sorted(list(hot_variables))}\n")
        
    logger.info(f"Bias Detector complete. Alarms triggered: {alarms_triggered}. Hot variables found: {len(hot_variables)}.")
    
    # -------------------------------------------------------------------------
    # STAGE 4: Micro Analysis via Targeted Diagonal Pairplots (Corner Plots)
    # -------------------------------------------------------------------------
    if hot_variables:
        logger.info("Stage 4: Performing targeted micro analysis of isolated Hot Variables...")
        
        # If there are too many hot variables, limit the pairplot to the top 15 variables
        # involved in the strongest selection biases. This avoids rendering crashes/hangs and
        # ensures the corner plot remains visually legible and interpretable.
        MAX_PLOT_VARS = 15
        if len(hot_variables) > MAX_PLOT_VARS:
            logger.warning(f"Found {len(hot_variables)} hot variables. Limiting pairplot to the top {MAX_PLOT_VARS} involved in the strongest selection biases for readability.")
            
            # Gather all alarmed pairs with their correlation strength
            alarm_pairs = []
            if physical_target_cols and instrumental_control_cols:
                for phys in physical_target_cols:
                    for inst in instrumental_control_cols:
                        if phys in corr_matrix.index and inst in corr_matrix.columns:
                            r_val = corr_matrix.loc[phys, inst]
                            if abs(r_val) > BIAS_THRESHOLD:
                                alarm_pairs.append((abs(r_val), phys, inst))
            else:
                for i in range(len(combined_cols)):
                    for j in range(i + 1, len(combined_cols)):
                        col1 = combined_cols[i]
                        col2 = combined_cols[j]
                        r_val = corr_matrix.loc[col1, col2]
                        if abs(r_val) > BIAS_THRESHOLD:
                            alarm_pairs.append((abs(r_val), col1, col2))
            
            # Sort pairs by correlation strength descending
            alarm_pairs = sorted(alarm_pairs, key=lambda x: x[0], reverse=True)
            
            # Select top variables until we reach MAX_PLOT_VARS
            selected_vars = set()
            for strength, col1, col2 in alarm_pairs:
                selected_vars.add(col1)
                selected_vars.add(col2)
                if len(selected_vars) >= MAX_PLOT_VARS:
                    break
            hot_list = sorted(list(selected_vars))
        else:
            hot_list = sorted(list(hot_variables))
            
        logger.info(f"Targeted variables list for corner plot ({len(hot_list)} columns): {hot_list}")
        
        # Downsample the dataframe to a maximum of 10,000 random rows to prevent out-of-memory errors
        # or rendering hangs during high-dpi vector plots creation on Slurm nodes.
        MAX_SAMPLES = 10000
        if len(df_clean) > MAX_SAMPLES:
            logger.info(f"Downsampling dataset to {MAX_SAMPLES} random rows using seed 42 to avoid cluster memory starvation...")
            df_sub = df_clean[hot_list].sample(n=MAX_SAMPLES, random_state=42)
        else:
            logger.info(f"Dataset size ({len(df_clean)} rows) is below {MAX_SAMPLES}. Using full clean slice.")
            df_sub = df_clean[hot_list]
            
        logger.info("Generating high-resolution targeted bias pairplot (corner plot)...")
        
        # Setup modern plotting style with seaborn
        sns.set_theme(style="ticks", rc={
            'axes.grid': True, 
            'grid.color': '#f1f1f1',
            'grid.linestyle': '--'
        })
        
        try:
            # Generate a comprehensive corner plot (lower-triangle matrix)
            # - Diagonal: Kernel Density Estimation (KDE) to visualize shape & multi-modality
            # - Off-Diagonal: Scatter plots with low alpha (0.4) and small size (5) to map density boundary ellipses
            g = sns.pairplot(
                df_sub,
                kind='scatter',
                diag_kind='kde',
                corner=True,
                plot_kws={
                    'alpha': 0.4, 
                    's': 5, 
                    'color': '#1f77b4',
                    'edgecolor': 'none'
                },
                diag_kws={
                    'fill': True, 
                    'color': '#2ca02c', 
                    'alpha': 0.6
                }
            )
            
            # Decorate the corner matrix beautifully
            g.figure.suptitle("Targeted Hot-Variables Anomaly & Bias Corner Plot", y=1.02, fontsize=15, fontweight='bold')
            
            pairplot_path = os.path.join(args.output_dir, "targeted_bias_pairplot.png")
            g.savefig(pairplot_path, dpi=200, bbox_inches='tight')
            plt.close()
            logger.info(f"Targeted micro pairplot saved successfully to: {pairplot_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate seaborn pairplot due to plotting exception: {str(e)}")
            logger.info("Attempting fallback simpler pairplot with histograms on the diagonal...")
            try:
                plt.close('all')
                g = sns.pairplot(
                    df_sub,
                    kind='scatter',
                    diag_kind='hist',
                    corner=True,
                    plot_kws={'alpha': 0.4, 's': 5, 'color': '#1f77b4', 'edgecolor': 'none'},
                    diag_kws={'color': '#2ca02c', 'alpha': 0.6}
                )
                g.figure.suptitle("Targeted Hot-Variables Anomaly & Bias Corner Plot (Fallback)", y=1.02, fontsize=15, fontweight='bold')
                pairplot_path = os.path.join(args.output_dir, "targeted_bias_pairplot.png")
                g.savefig(pairplot_path, dpi=200, bbox_inches='tight')
                plt.close()
                logger.info(f"Fallback targeted micro pairplot saved successfully to: {pairplot_path}")
            except Exception as ex:
                logger.error(f"Fallback pairplot generation also failed: {str(ex)}")
    else:
        logger.info("Stage 4 Skipped: No Hot Variables exceeded the selection bias Spearman threshold (|r| > 0.65).")
        # Write an empty marker file to show it was clean
        with open(os.path.join(args.output_dir, "no_hot_variables_detected.txt"), 'w') as f_ok:
            f_ok.write("AstroPT Dataset Audit: Checked for selection biases and found no high-coupling variables exceeding r=0.65.\n")
            
    # -------------------------------------------------------------------------
    # STAGE 5: Multi-Regime Correlation Analysis (Positive & Negative Bins)
    # -------------------------------------------------------------------------
    logger.info("Stage 5: Generating multi-regime correlation tables and matrices...")
    
    # 1. Generate masked matrix CSV files for four specific correlation regimes
    regimes = {
        "above_0.5": (corr_matrix >= 0.5),
        "0_to_0.5": (corr_matrix >= 0) & (corr_matrix < 0.5),
        "minus_0.5_to_0": (corr_matrix > -0.5) & (corr_matrix < 0),
        "below_minus_0.5": (corr_matrix <= -0.5)
    }
    
    for name, mask in regimes.items():
        filtered_matrix = corr_matrix.where(mask)
        matrix_path = os.path.join(args.output_dir, f"spearman_matrix_{name}.csv")
        filtered_matrix.to_csv(matrix_path)
        logger.info(f"Saved filtered correlation grid matrix: {matrix_path}")
        
    # 2. Extract flat list of unique non-self pairs sorted by correlation strength
    variables = corr_matrix.columns.tolist()
    pairs_data = []
    
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            v1 = variables[i]
            v2 = variables[j]
            r_val = corr_matrix.loc[v1, v2]
            if not np.isnan(r_val):
                pairs_data.append({"Variable_1": v1, "Variable_2": v2, "Spearman_r": r_val})
                
    df_pairs = pd.DataFrame(pairs_data)
    if not df_pairs.empty:
        regime_filters = {
            "above_0.5": df_pairs[df_pairs["Spearman_r"] >= 0.5].sort_values("Spearman_r", ascending=False),
            "0_to_0.5": df_pairs[(df_pairs["Spearman_r"] >= 0) & (df_pairs["Spearman_r"] < 0.5)].sort_values("Spearman_r", ascending=False),
            "minus_0.5_to_0": df_pairs[(df_pairs["Spearman_r"] > -0.5) & (df_pairs["Spearman_r"] < 0)].sort_values("Spearman_r", ascending=True),
            "below_minus_0.5": df_pairs[df_pairs["Spearman_r"] <= -0.5].sort_values("Spearman_r", ascending=True)
        }
        
        for name, df_filtered in regime_filters.items():
            pairs_path = os.path.join(args.output_dir, f"correlation_pairs_{name}.csv")
            df_filtered.to_csv(pairs_path, index=False)
            logger.info(f"Successfully generated correlation report for regime '{name}': {len(df_filtered)} unique pairs saved.")

    # -------------------------------------------------------------------------
    # STAGE 6: Science vs. Instrumental Controls Cross-Correlation (Physics + Spectral Lines + Morphology)
    # -------------------------------------------------------------------------
    logger.info("Stage 6: Generating premium scientific vs. instrumental cross-talk correlation analysis...")
    
    specified_instrumental = [
        'SNR_SPEC_B', 'SNR_SPEC_R', 'SNR_SPEC_Z', 'SNR_R', 'SNR_G', 'SNR_Z', 
        'SNR_W1', 'SNR_W2', 'det_quality_flag', 'spurious_flag', 'flag_vis', 
        'flag_h', 'flag_j', 'flag_y', 'CHI2', 'HALPHA_BROAD_CHI2'
    ]
    
    specified_scientific = [
        # Physics & Fluxes
        'Z', 'LOGMSTAR', 'LOGSFR', 'AGNLUM', 'segmentation_area', 'flux_detection_total',
        # Spectral lines
        'HALPHA_FLUX', 'NII_6584_FLUX', 'OIII_5007_FLUX', 'HBETA_FLUX', 'OII_3726_FLUX',
        # Morphology (Sersic)
        'sersic_sersic_vis_radius', 'sersic_sersic_vis_index', 'sersic_sersic_vis_axis_ratio',
        # Spiral Arms & Shape
        'has_spiral_arms_yes', 'smoothness'
    ]
    
    # Map exact column names case-insensitively using columns available in clean DataFrame
    df_cols_lower = {col.lower(): col for col in df_clean.columns}
    
    active_instrumental = [df_cols_lower[col.lower()] for col in specified_instrumental if col.lower() in df_cols_lower]
    active_scientific = [df_cols_lower[col.lower()] for col in specified_scientific if col.lower() in df_cols_lower]
    
    if active_instrumental and active_scientific:
        logger.info(f"Rendering scientific cross-talk heatmap ({len(active_scientific)} targets vs {len(active_instrumental)} controls)...")
        
        # Extract the rectangular correlation slice
        cross_corr = corr_matrix.loc[active_scientific, active_instrumental]
        
        # Setup plot dimensions and aesthetics
        plt.figure(figsize=(16, 12))
        
        sns.heatmap(
            cross_corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            linewidths=0.5,
            linecolor="#efefef",
            cbar_kws={"label": "Spearman Rank Correlation Coefficient ($r_s$)"},
            annot_kws={"size": 8, "weight": "bold"}
        )
        
        plt.title(
            "AstroPT Target Properties vs. Instrumental Controls Cross-Correlation\n"
            "(Isolating selection biases across galaxy physics, spectral lines, and morphology)",
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        plt.xlabel("Instrumental Controls & Survey Noise Markers", fontsize=11, fontweight='bold', labelpad=12)
        plt.ylabel("Scientific Target Properties (Physics, Spectral Lines, Morphology)", fontsize=11, fontweight='bold', labelpad=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        cross_talk_path = os.path.join(args.output_dir, "scientific_vs_instrumental_bias_heatmap.png")
        plt.savefig(cross_talk_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"Scientific cross-talk correlation heatmap successfully saved to: {cross_talk_path}")
    else:
        logger.warning("Could not map active columns to construct the scientific cross-talk rectangular heatmap.")

    logger.info("=" * 80)
    logger.info("ASTROPT FORENSIC DATASET AUDIT COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
