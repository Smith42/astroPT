#!/bin/bash

#--- ASTROPT PIPELINE LAUNCHER (OPTIMIZED) ---#
echo "-------------------------------------------------"
echo "Launching Full AstroPT Pipeline (Modular)"
echo "-------------------------------------------------"


# PATH AND DIR CONFIGURATION
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
SCRIPT_DIR=$(dirname "$0")
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"
META_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.0.fits"

# LAUNCHING SCRIPTS
TRAIN_SCRIPT="$SCRIPT_DIR/train_astropt_multiGPU.sh"
PLOT_MET_SCRIPT="$SCRIPT_DIR/plot_training_metrics.sh"
PLOT_IMG_SCRIPT="$SCRIPT_DIR/plot_images_spectra.sh"
EXT_EMBD_SCRIPT="$SCRIPT_DIR/extract_embeddings.sh"
COS_SIM_SCRIPT="$SCRIPT_DIR/cosine_similarity.sh"
UMAPS_SCRIPT="$SCRIPT_DIR/plot_umaps.sh"
PROBING_SCRIPT="$SCRIPT_DIR/probing_downstream.sh"

# ARGUMENTS DEFAULT VALUES
TRAIN_NAME="New Train"              # Training name
TRAIN_DESC="New AstroPT Training"   # Training description
OUT_DIR=""                          # Output directory

while getopts ":n:d:o:h" opt; do
  case $opt in
    n) TRAIN_NAME="$OPTARG" ;;
    d) TRAIN_DESC="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    h) 
       echo "Usage: $0 [-n 'Name'] [-d 'Description'] [-o 'output_path/']"
       exit 0 
       ;;
    \?) 
       echo "Invalid option: -$OPTARG" >&2
       exit 1 
       ;;
  esac
done

# AUTOMATIC CONFIGURATION
SUFIX_NAME=$(echo "$TRAIN_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/ /g' | xargs | tr ' ' '_')
TRAIN_DATE="$(date +%Y%m%d)"
DEFAULT_PATH="logs/astropt_100M_250K_arrow_${TRAIN_DATE}_${SUFIX_NAME}"

if [ -z "$OUT_DIR" ]; then
    OUT_DIR="$DEFAULT_PATH"
fi

echo "Target Output Directory: $OUT_DIR"

# LAUNCH ANALYSIS FUNCTION
launch_analysis() {
    local PARENT_JOB_ID=$1  
    local STEP_NAME=$2      

    echo "--> Launching Analysis Battery for: $STEP_NAME (Parent: $PARENT_JOB_ID)"

    # Plotting Metrics
    local J_MET=$(sbatch --parsable --dependency=afterany:$PARENT_JOB_ID "$PLOT_MET_SCRIPT" \
                -r "$REPO_ROOT" \
                -o "$OUT_DIR"
            )
    echo "    [Metric]  Job sent. ID: $J_MET (Depends on Train: any)"

    # Plotting Images
    local J_IMG=$(sbatch --parsable --dependency=afterany:$PARENT_JOB_ID "$PLOT_IMG_SCRIPT" \
                -r "$REPO_ROOT" \
                -o "$OUT_DIR" \
                -a "$DATA_DIR"
            )
    echo "    [ImgSpec] Job sent. ID: $J_IMG (Depends on Train: any)"

    # Extract Embeddings 
    local J_EMB=$(sbatch --parsable --dependency=afterok:$PARENT_JOB_ID "$EXT_EMBD_SCRIPT" \
                -r "$REPO_ROOT" \
                -o "$OUT_DIR" \
                -a "$DATA_DIR"
            )
    echo "    [Embeds]  Job sent. ID: $J_EMB (Depends on Train: any)"

    # Cosine Similarity
    local J_COS=$(sbatch --parsable --dependency=afterok:$J_EMB "$COS_SIM_SCRIPT" \
                -r "$REPO_ROOT" \
                -o "$OUT_DIR"
            )
    echo "    [CosSim]  Job sent. ID: $J_COS (Depends on Embeds: ok)"

    # UMAPS
    local J_UMAP=$(sbatch --parsable --dependency=afterok:$J_EMB "$UMAPS_SCRIPT" \
                -r "$REPO_ROOT" \
                -o "$OUT_DIR" \
                -f "$META_PATH"
            )
    echo "    [UMAPS]   Job sent. ID: $J_UMAP (Depends on Embeds: ok)"

    # PROBING DOWNSTREAM TASKS
    local J_PROB=$(sbatch --parsable --dependency=afterok:$J_EMB "$PROBING_SCRIPT" \
                -r "$REPO_ROOT" \
                -o "$OUT_DIR" \
                -f "$META_PATH"
            )
    echo "    [PROBING] Job sent. ID: $J_PROB (Depends on Embeds: ok)"
    
    echo "--> Analysis Batch Sent."
    echo ""
}

# PART 1 - TRAIN FROM SCRATCH
echo "-------------------------------------------------"
echo "STEP 1: TRAINING FROM SCRATCH"
JOB_TRAIN_1=$(sbatch --parsable "$TRAIN_SCRIPT" \
    -r "$REPO_ROOT" \
    -o "$OUT_DIR" \
    -a "$DATA_DIR" \
    -n "$TRAIN_NAME" \
    -d "$TRAIN_DESC" \
    -m "scratch" \
    -k "all")
echo "Training Job (Scratch) launched. ID: $JOB_TRAIN_1"

# Calling the execution function
launch_analysis "$JOB_TRAIN_1" "Part 1 (SCRATCH)"


# PART 2 - TRAIN FROM RESUME
echo "-------------------------------------------------"
echo "STEP 2: TRAINING FROM RESUME"
JOB_TRAIN_2=$(sbatch --parsable --dependency=afterok:$JOB_TRAIN_1 "$TRAIN_SCRIPT" \
    -r "$REPO_ROOT" \
    -o "$OUT_DIR" \
    -a "$DATA_DIR" \
    -n "$TRAIN_NAME" \
    -d "$TRAIN_DESC" \
    -m "resume" \
    -k "all")
echo "Training Job (Resume) launched.  ID: $JOB_TRAIN_2 (Depends on Train 1: ok $JOB_TRAIN_1)"

# Calling the execution function
launch_analysis "$JOB_TRAIN_2" "Part 2 (Resume)"


# QUEUE STATUS
echo "-------------------------------------------------"
echo "Waiting for resources..."
sleep 15
echo "Pipeline Status Queue:"
squeue -u $USER -S i -o "%.10i %.6P %.25j %.8T %.10M %.35E %R"
echo "-------------------------------------------------"