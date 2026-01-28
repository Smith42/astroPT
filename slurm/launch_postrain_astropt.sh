#!/bin/bash

#--- ASTROPT POST-TRAINING LAUNCHER ---#
echo "-------------------------------------------------"
echo "Launching AstroPT Analysis"
echo "-------------------------------------------------"

# PATH AND DIR CONFIGURATION
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
SCRIPT_DIR=$(dirname "$0")
OUT_DIR=${1:-"logs/astropt_100M_250K_arrow_new"} 
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"
META_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.0.fits"

# Launching scripts
PLOT_MET_SCRIPT="$SCRIPT_DIR/plot_training_metrics.sh"
PLOT_IMG_SCRIPT="$SCRIPT_DIR/plot_images_spectra.sh"
EXT_EMBD_SCRIPT="$SCRIPT_DIR/extract_embeddings.sh"
COS_SIM_SCRIPT="$SCRIPT_DIR/cosine_similarity.sh"
UMAPS_SCRIPT="$SCRIPT_DIR/plot_umaps.sh"

# Printing Output Directory
OUT_DIR=${OUT_DIR%/}
echo "Target Output Directory: $OUT_DIR"

# LAUNCH ANALYSIS FUNCTION (STANDALONE)
launch_analysis_standalone() {
    echo "--> Launching Analysis Pipeline"

    # Plotting Metrics 
    local J_MET=$(sbatch --parsable "$PLOT_MET_SCRIPT" "$REPO_ROOT" "$OUT_DIR")
    echo "    [Metric]  Job sent. ID: $J_MET"

    # Plotting Images 
    local J_IMG=$(sbatch --parsable "$PLOT_IMG_SCRIPT" "$REPO_ROOT" "$OUT_DIR" "$DATA_DIR")
    echo "    [ImgSpec] Job sent. ID: $J_IMG"

    # Extract Embeddings 
    local J_EMB=$(sbatch --parsable "$EXT_EMBD_SCRIPT" "$REPO_ROOT" "$OUT_DIR" "$DATA_DIR")
    echo "    [Embeds]  Job sent. ID: $J_EMB"

    # Cosine Similarity 
    local J_COS=$(sbatch --parsable --dependency=afterok:$J_EMB "$COS_SIM_SCRIPT" "$REPO_ROOT" "$OUT_DIR")
    echo "    [CosSim]  Job sent. ID: $J_COS (Depends on Embeds: ok)"

    # UMAPS 
    local J_UMAP=$(sbatch --parsable --dependency=afterok:$J_EMB "$UMAPS_SCRIPT" "$REPO_ROOT" "$OUT_DIR" "$META_PATH")
    echo "    [UMAPS]   Job sent. ID: $J_UMAP (Depends on Embeds: ok)"
    
    echo "--> Analysis Batch Sent."
    echo ""
}

# EXECUTION
launch_analysis_standalone

# QUEUE STATUS
echo "-------------------------------------------------"
echo "Waiting for resources..."
sleep 15
echo "Pipeline Status Queue:"
squeue -u $USER -S i -o "%.10i %.6P %.25j %.8T %.10M %.35E %R"
echo "-------------------------------------------------"