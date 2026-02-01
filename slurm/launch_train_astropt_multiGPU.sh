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
WORKFLOW_SCRIPT="$SCRIPT_DIR/workflow_controller.sh"

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
DEFAULT_PATH="$REPO_ROOT/logs/astropt_100M_250K_arrow_${TRAIN_DATE}_${SUFIX_NAME}"

if [ -z "$OUT_DIR" ]; then
    OUT_DIR="$DEFAULT_PATH"
fi

# Absolute output path
OUT_DIR=$(readlink -f "$OUT_DIR")

echo "Target Output Directory: $OUT_DIR"

# LAUNCH ANALYSIS FUNCTION
launch_analysis() {
    local PARENT_JOB_ID=$1
    local JOB_SUFFIX=$2  
    local STEP_NAME=$3      

    echo " --> Launching Analysis Battery for: $STEP_NAME (Parent: $PARENT_JOB_ID)"

    # Plotting Metrics
    local J_MET=$(sbatch --parsable \
                --dependency=afterany:$PARENT_JOB_ID \
                --job-name="Plot_Metrics$JOB_SUFFIX" \
                "$PLOT_MET_SCRIPT" \
                -r "$REPO_ROOT" \
                -o "$OUT_DIR"
            )
    echo "    [Metric]  Job sent. ID: $J_MET (Depends on Train: any)"

    # Workflow controller
    J_WOR=$(sbatch --parsable \
                --dependency=afterany:$PARENT_JOB_ID \
                --job-name="Workflow_Controller$JOB_SUFFIX" \
                "$WORKFLOW_SCRIPT" \
                -r "$REPO_ROOT" \
                -o "$OUT_DIR" \
                -s "$JOB_SUFFIX"
            )
    echo "    [Work]    Job sent. ID: $J_WOR (Depends on Train: any)"

    # Plotting Images
    local J_IMG=$(sbatch --parsable \
                --dependency=afterok:$J_WOR \
                --job-name="Plot_Im_Sp$JOB_SUFFIX" \
                "$PLOT_IMG_SCRIPT" \
                -r "$REPO_ROOT" \
                -o "$OUT_DIR" \
                -a "$DATA_DIR"
            )
    echo "    [ImgSpec] Job sent. ID: $J_IMG (Depends on Train: any)"

    # Extract Embeddings 
    local J_EMB=$(sbatch --parsable \
                --dependency=afterok:$J_WOR \
                --job-name="Extract_Embed$JOB_SUFFIX" \
                "$EXT_EMBD_SCRIPT" \
                -r "$REPO_ROOT" \
                -o "$OUT_DIR" \
                -a "$DATA_DIR"
            )
    echo "    [Embeds]  Job sent. ID: $J_EMB (Depends on Train: any)"

    # Cosine Similarity
    local J_COS=$(sbatch --parsable \
                --dependency=afterok:$J_EMB \
                --job-name="Cos_Sim$JOB_SUFFIX" \
                "$COS_SIM_SCRIPT" \
                -r "$REPO_ROOT" \
                -o "$OUT_DIR"
            )
    echo "    [CosSim]  Job sent. ID: $J_COS (Depends on Embeds: ok)"

    # UMAPS
    local J_UMAP=$(sbatch --parsable \
                --dependency=afterok:$J_EMB \
                --job-name="Plot_Umaps$JOB_SUFFIX" \
                "$UMAPS_SCRIPT" \
                -r "$REPO_ROOT" \
                -o "$OUT_DIR" \
                -f "$META_PATH"
            )
    echo "    [UMAPS]   Job sent. ID: $J_UMAP (Depends on Embeds: ok)"

    # PROBING DOWNSTREAM TASKS
    local J_PROB=$(sbatch --parsable \
                --dependency=afterok:$J_EMB \
                --job-name="Probing_Tasks$JOB_SUFFIX" \
                "$PROBING_SCRIPT" \
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
JOB_SUFFIX="_T1"
JOB_TRAIN_1=$(sbatch --parsable \
    --job-name="Train_AstroPT_DDP$JOB_SUFFIX" \
    "$TRAIN_SCRIPT" \
    -r "$REPO_ROOT" \
    -o "$OUT_DIR" \
    -a "$DATA_DIR" \
    -n "$TRAIN_NAME" \
    -d "$TRAIN_DESC" \
    -m "scratch" \
    -k "all")
echo "Training Job (Scratch) launched. ID: $JOB_TRAIN_1"

# Calling the execution function
launch_analysis "$JOB_TRAIN_1" "$JOB_SUFFIX" "Part 1 (SCRATCH)"


# PART 2 - TRAIN FROM RESUME
echo "-------------------------------------------------"
echo "STEP 2: TRAINING FROM RESUME"
JOB_SUFFIX="_T2"
JOB_TRAIN_2=$(sbatch --parsable \
    --dependency=afterok:$J_WOR \
    --job-name="Train_AstroPT_DDP$JOB_SUFFIX" \
    "$TRAIN_SCRIPT" \
    -r "$REPO_ROOT" \
    -o "$OUT_DIR" \
    -a "$DATA_DIR" \
    -n "$TRAIN_NAME" \
    -d "$TRAIN_DESC" \
    -m "resume" \
    -k "all")
echo "Training Job (Resume) launched.  ID: $JOB_TRAIN_2 (Depends on Train 1: ok $JOB_TRAIN_1)"

# Calling the execution function
launch_analysis "$JOB_TRAIN_2" "$JOB_SUFFIX" "Part 2 (Resume)"


# QUEUE STATUS
echo "-------------------------------------------------"
echo "Waiting for resources..."
sleep 15
echo "Pipeline Status Queue:"
squeue -u $USER -S i -o "%.10i %.6P %.25j %.8T %.10M %.35E %R"
echo "-------------------------------------------------"