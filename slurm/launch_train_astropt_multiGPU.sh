#!/bin/bash

#--- ASTROPT PIPELINE LAUNCHER (OPTIMIZED) ---#
echo "-------------------------------------------------"
echo "Launching Full AstroPT Pipeline (Modular)"
echo "-------------------------------------------------"


# PATH AND DIR CONFIGURATION
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
SCRIPT_DIR=$(dirname "$0")
#DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_filter_corrupt"
#DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_large_P50"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated"
META_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits"

# LAUNCHING SCRIPTS
TRAIN_SCRIPT="$SCRIPT_DIR/train_astropt_multiGPU.sh"
PLOT_MET_SCRIPT="$SCRIPT_DIR/plot_training_metrics.sh"
ATTN_MAPS_SCRIPT="$SCRIPT_DIR/attention_maps.sh"
PLOT_IMG_SCRIPT="$SCRIPT_DIR/plot_images_spectra.sh"
CROSS_REC_SCRIPT="$SCRIPT_DIR/cross_reconstruction.sh"
EXT_EMBD_SCRIPT="$SCRIPT_DIR/extract_embeddings.sh"
COS_SIM_SCRIPT="$SCRIPT_DIR/cosine_similarity.sh"
UMAPS_SCRIPT="$SCRIPT_DIR/plot_umaps.sh"
PROBING_SCRIPT="$SCRIPT_DIR/probing_downstream.sh"
PROBING_DASH_SCRIPT="$SCRIPT_DIR/probing_downstream_dashboard.sh"
LATENT_SCRIPT="$SCRIPT_DIR/probing_latent_mapper.sh"
LATENT_DASH_SCRIPT="$SCRIPT_DIR/probing_latent_dashboard.sh"
WORKFLOW_SCRIPT="$SCRIPT_DIR/workflow_controller.sh"

# ARGUMENTS DEFAULT VALUES
TRAIN_NAME="New Train"              # Training name
TRAIN_DESC="New AstroPT Training"   # Training description
TRAIN_EXTRA_ARGS=""                 # Extra flags forwarded to train_multimodal_arrow.py
J_PROB_DASH=""
LAST_PROB_JOB_ID=""

while getopts ":n:d:t:l:x:h" opt; do
  case $opt in
    n) TRAIN_NAME="$OPTARG" ;;
    d) TRAIN_DESC="$OPTARG" ;;
    t) TRAIN_DIR="$OPTARG" ;;
    l) LONG_TRAINING="$OPTARG" ;;
    x) TRAIN_EXTRA_ARGS="$OPTARG" ;;
    h) 
       echo "Usage: $0 [-n 'Name'] [-d 'Description'] [-t 'output_path/'] [-l TRUE] [-x 'extra train flags']"
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

if [ -z "$TRAIN_DIR" ]; then
    TRAIN_DIR="$DEFAULT_PATH"
fi

# Absolute output paths
TRAIN_DIR=$(readlink -f "$TRAIN_DIR")
WEIGHTS_DIR="$TRAIN_DIR/weights"
LOGS_DIR="$TRAIN_DIR/logs"
PLOTS_DIR="$TRAIN_DIR/plots"
EMB_DIR="$TRAIN_DIR/embeddings"

echo "Target Training Directory: $TRAIN_DIR"

# LAUNCH ANALYSIS FUNCTION
launch_analysis() {
    local PARENT_JOB_ID=$1
    local JOB_SUFFIX=$2  
    local STEP_NAME=$3      
    local EMB_STAGE_TAG="${JOB_SUFFIX#_}"

    # Stage-aware output folders to avoid T1/T2/T3 overlap.
    local ATTN_SAVE_DIR="$PLOTS_DIR/attention_maps$JOB_SUFFIX"
    local IMG_SPEC_SAVE_DIR="$PLOTS_DIR/images_spectra_reconstruction$JOB_SUFFIX"
    local CROSS_REC_SAVE_DIR="$PLOTS_DIR/cross_reconstruction$JOB_SUFFIX"

    echo " --> Launching Analysis Battery for: $STEP_NAME (Parent: $PARENT_JOB_ID)"

    # Plotting Metrics
    local J_MET=$(sbatch --parsable \
                --dependency=afterany:$PARENT_JOB_ID \
                --job-name="Plot_Metrics$JOB_SUFFIX" \
                "$PLOT_MET_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -l "$LOGS_DIR" \
                -s "$PLOTS_DIR"
            )
    echo "    [Metric]  Job sent.       ID: $J_MET (Depends on Train: any)"

    # Workflow controller
    J_WOR=$(sbatch --parsable \
                --dependency=afterany:$PARENT_JOB_ID \
                --job-name="Workflow_Controller$JOB_SUFFIX" \
                "$WORKFLOW_SCRIPT" \
                -r "$REPO_ROOT" \
                -t "$TRAIN_DIR" \
                -x "$JOB_SUFFIX"
            )
    echo "    [Work]    Job sent.       ID: $J_WOR (Depends on Train: any)"
        if [[ -z "$J_WOR" ]]; then
            echo "    [ERROR] Workflow controller submission failed. Stopping analysis chain."
            return 1
        fi

    # Plotting Attention Maps
    local J_ATT=$(sbatch --parsable \
                --dependency=afterok:$J_WOR \
                --job-name="Plot_Attn_Maps$JOB_SUFFIX" \
                "$ATTN_MAPS_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -s "$ATTN_SAVE_DIR" \
                -a "$DATA_DIR"
            )
    echo "    [Att Maps] Job sent.      ID: $J_ATT (Depends on Train: any)"
    

    # Plotting Images and Spectra
    local J_IMG=$(sbatch --parsable \
                --dependency=afterok:$J_WOR \
                --job-name="Plot_Im_Sp$JOB_SUFFIX" \
                "$PLOT_IMG_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -s "$IMG_SPEC_SAVE_DIR" \
                -a "$DATA_DIR"
            )
    echo "    [ImgSpec] Job sent.       ID: $J_IMG (Depends on Train: any)"

    # Cross Reconstruction
    local J_CROSS=$(sbatch --parsable \
                --dependency=afterok:$J_WOR \
                --job-name="Cross_Rec$JOB_SUFFIX" \
                "$CROSS_REC_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -s "$CROSS_REC_SAVE_DIR" \
                -a "$DATA_DIR"
            )
    echo "    [CrossRec] Job sent.      ID: $J_CROSS (Depends on Train: any)"

    # Extract Embeddings 
    local J_EMB=$(sbatch --parsable \
                --dependency=afterany:$J_CROSS \
                --job-name="Extract_Embed$JOB_SUFFIX" \
                "$EXT_EMBD_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -s "$EMB_DIR" \
                -x "$EMB_STAGE_TAG" \
                -a "$DATA_DIR"
            )
    echo "    [Embeds]  Job sent.       ID: $J_EMB (Depends on Train: any)"
        if [[ -z "$J_EMB" ]]; then
            echo "    [ERROR] Embedding extraction submission failed. Stopping analysis chain."
            return 1
        fi

    # Cosine Similarity
    local J_COS=$(sbatch --parsable \
                --dependency=afterok:$J_EMB \
                --job-name="Cos_Sim$JOB_SUFFIX" \
                "$COS_SIM_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -e "$EMB_DIR"
            )
    echo "    [CosSim]  Job sent.       ID: $J_COS (Depends on Embeds: ok)"

    # UMAPS
    local J_UMAP=$(sbatch --parsable \
                --dependency=afterok:$J_EMB \
                --job-name="Plot_Umaps$JOB_SUFFIX" \
                "$UMAPS_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -e "$EMB_DIR" \
                -a "$DATA_DIR" \
                -f "$META_PATH"
            )
    echo "    [UMAPS]   Job sent.       ID: $J_UMAP (Depends on Embeds: ok)"

    # PROBING DOWNSTREAM TASKS
    local J_PROB=$(sbatch --parsable \
                --dependency=afterok:$J_EMB \
                --job-name="Probing_Tasks$JOB_SUFFIX" \
                "$PROBING_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -e "$EMB_DIR" \
                -f "$META_PATH"
            )
    echo "    [PROBING] Job sent.       ID: $J_PROB (Depends on Embeds: ok)"
        if [[ -z "$J_PROB" ]]; then
            echo "    [ERROR] Downstream probing submission failed. Stopping analysis chain."
            return 1
        fi
        LAST_PROB_JOB_ID="$J_PROB"

    # PROBING DOWNSTREAM TASKS DASHBOARD
    local J_PROB_DASH=$(sbatch --parsable \
                --dependency=afterok:$J_PROB \
                --job-name="Probing_Tasks_Dash$JOB_SUFFIX" \
                "$PROBING_DASH_SCRIPT" \
                -r "$REPO_ROOT" \
                -e "$EMB_DIR"
            )
    echo "    [PROB DASH] Job sent.     ID: $J_PROB_DASH (Depends on Probing: ok)"
        if [[ -z "$J_PROB_DASH" ]]; then
            echo "    [ERROR] Downstream dashboard submission failed. Stopping analysis chain."
            return 1
        fi

    # PROBING LATENT MAPPING TASKS
    local J_LATENT=$(sbatch --parsable \
                --dependency=afterok:$J_EMB \
                --job-name="Latent_Tasks$JOB_SUFFIX" \
                "$LATENT_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -e "$EMB_DIR" \
                -f "$META_PATH"
            )
    echo "    [LATENT] Job sent.        ID: $J_LATENT (Depends on Embeds: ok)"
        if [[ -z "$J_LATENT" ]]; then
            echo "    [ERROR] Latent probing submission failed. Stopping analysis chain."
            return 1
        fi

    # PROBING LATENT MAPPING DASHBOARD
    J_LATENT_DASH=$(sbatch --parsable \
                --dependency=afterok:$J_LATENT \
                --job-name="Latent_Tasks_Dash$JOB_SUFFIX" \
                "$LATENT_DASH_SCRIPT" \
                -r "$REPO_ROOT" \
                -e "$EMB_DIR"
            )
    echo "    [LATENT DASH] Job sent.   ID: $J_LATENT_DASH (Depends on Latent: ok)"
        if [[ -z "$J_LATENT_DASH" ]]; then
            echo "    [ERROR] Latent dashboard submission failed. Stopping analysis chain."
            return 1
        fi
    
    echo " --> Analysis Batch Sent."
}

# PART 1 - TRAIN FROM SCRATCH
echo "-------------------------------------------------"
echo "STEP 1: TRAINING FROM SCRATCH"
JOB_SUFFIX="_T1"
JOB_TRAIN_1=$(sbatch --parsable \
    --job-name="Train_AstroPT_DDP$JOB_SUFFIX" \
    "$TRAIN_SCRIPT" \
    -r "$REPO_ROOT" \
    -t "$TRAIN_DIR" \
    -a "$DATA_DIR" \
    -n "$TRAIN_NAME" \
    -d "$TRAIN_DESC" \
    -m "scratch" \
    -k "all" \
    -x "$TRAIN_EXTRA_ARGS")
echo "Training Job (Scratch) launched. ID: $JOB_TRAIN_1"
if [[ -z "$JOB_TRAIN_1" ]]; then
    echo "[ERROR] Step 1 training submission failed. Aborting pipeline."
    exit 1
fi

# Calling the execution function
launch_analysis "$JOB_TRAIN_1" "$JOB_SUFFIX" "Part 1 (SCRATCH)" || {
    echo "[ERROR] Analysis submission failed for STEP 1. Aborting pipeline."
    exit 1
}


# PART 2 - TRAIN FROM RESUME
echo "-------------------------------------------------"
echo "STEP 2: TRAINING FROM RESUME"
JOB_SUFFIX="_T2"
JOB_TRAIN_2=$(sbatch --parsable \
    --dependency=afterany:$J_LATENT_DASH \
    --job-name="Train_AstroPT_DDP$JOB_SUFFIX" \
    "$TRAIN_SCRIPT" \
    -r "$REPO_ROOT" \
    -t "$TRAIN_DIR" \
    -a "$DATA_DIR" \
    -n "$TRAIN_NAME" \
    -d "$TRAIN_DESC" \
    -m "resume" \
    -k "all" \
    -x "$TRAIN_EXTRA_ARGS")
echo "Training Job (Resume) launched.  ID: $JOB_TRAIN_2 (Depends on Latent Dash T1: $J_LATENT_DASH)"
if [[ -z "$JOB_TRAIN_2" ]]; then
    echo "[ERROR] Step 2 training submission failed. Aborting pipeline."
    exit 1
fi

# Calling the execution function
launch_analysis "$JOB_TRAIN_2" "$JOB_SUFFIX" "Part 2 (Resume)" || {
    echo "[ERROR] Analysis submission failed for STEP 2. Aborting pipeline."
    exit 1
}


# Implementing a third training round for longer trainings
# LONG_TRAINING must be explicitly passed via -l TRUE
if [ -z "${LONG_TRAINING:-}" ]; then
    LONG_TRAINING="FALSE"
fi

if [[ "$LONG_TRAINING" == "TRUE" ]]; then
    # PART 3 - TRAIN FROM RESUME
    echo "-------------------------------------------------"
    echo "STEP 3: TRAINING FROM RESUME"
    JOB_SUFFIX="_T3"
    JOB_TRAIN_3=$(sbatch --parsable \
        --dependency=afterany:$LAST_PROB_JOB_ID \
        --job-name="Train_AstroPT_DDP$JOB_SUFFIX" \
        "$TRAIN_SCRIPT" \
        -r "$REPO_ROOT" \
        -t "$TRAIN_DIR" \
        -a "$DATA_DIR" \
        -n "$TRAIN_NAME" \
        -d "$TRAIN_DESC" \
        -m "resume" \
        -k "all" \
        -x "$TRAIN_EXTRA_ARGS")
    echo "Training Job (Resume) launched.  ID: $JOB_TRAIN_3 (Depends on Train 2: ok $JOB_TRAIN_2)"
        if [[ -z "$JOB_TRAIN_3" ]]; then
            echo "[ERROR] Step 3 training submission failed. Aborting pipeline."
            exit 1
        fi

    # Calling the execution function
        launch_analysis "$JOB_TRAIN_3" "$JOB_SUFFIX" "Part 3 (Resume)" || {
            echo "[ERROR] Analysis submission failed for STEP 3. Aborting pipeline."
            exit 1
        }

fi


# QUEUE STATUS
echo "-------------------------------------------------"
echo "Waiting for resources..."
sleep 15
echo "Pipeline Status Queue:"
squeue -u $USER -S i -o "%.10i %.6P %.25j %.8T %.10M %.35E %R"
echo "-------------------------------------------------"