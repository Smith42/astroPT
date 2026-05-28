#!/bin/bash

# --- MODULAR ANALYSIS PIPELINE (SLURM ONLY) --- #
# WARNING: This script is designed to be executed in a SLURM environment.
# It submits multiple background jobs via 'sbatch' and will NOT work
# on a local machine or a simple interactive session without Slurm.

# Technical check: ensure sbatch is available
if ! command -v sbatch &> /dev/null; then
    echo "-----------------------------------------------------------------"
    echo "[ERROR] 'sbatch' command not found."
    echo "This script is a SLURM orchestrator and must be run on a cluster."
    echo "-----------------------------------------------------------------"
    exit 1
fi

# Robust repository root detection based on script location
# (Script is in slurm/trainers/, so REPO_ROOT is two levels up)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# If SLURM has moved us to /var/spool, correct REPO_ROOT to the launch working directory
if [[ "$REPO_ROOT" == "/var/spool"* ]]; then
    if [ -d "$PWD/astroPT" ]; then
        REPO_ROOT="$PWD/astroPT"
    else
        REPO_ROOT="$PWD"
    fi
fi

# --- DEFAULTS ---
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated_tokenized"
META_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits"
PARENT_JOB_ID="any"
JOB_SUFFIX=""
STEP_NAME="Post-Training Analysis"

# --- LAUNCHING SCRIPTS ---

# 📊 Analysis & Plotting
PLOT_MET_SCRIPT="$REPO_ROOT/slurm/analysis/dash_training_metrics.sh"
ATTN_MAPS_SCRIPT="$REPO_ROOT/slurm/analysis/dash_attention_maps.sh"
PLOT_IMG_SCRIPT="$REPO_ROOT/slurm/analysis/dash_internal_reconstruction_samples.sh"
CROSS_REC_SCRIPT="$REPO_ROOT/slurm/analysis/dash_zero_shot_reconstruction.sh"
COS_SIM_SCRIPT="$REPO_ROOT/slurm/analysis/dash_embeddings_alignment.sh"
UMAPS_SCRIPT="$REPO_ROOT/slurm/analysis/dash_latent_projection.sh"

# 🧠 Inference & Embeddings
EXT_EMBD_SCRIPT="$REPO_ROOT/slurm/inference/embeddings_extraction.sh"

# 🎯 Probing & Dashboards
PROBING_SCRIPT="$REPO_ROOT/slurm/probing/probing_downstream_benchmark.sh"
PROBING_DASH_SCRIPT="$REPO_ROOT/slurm/analysis/dash_probing_downstream.sh"
PREDICT_DASH_SCRIPT="$REPO_ROOT/slurm/analysis/dash_probing_predictions.sh"
LATENT_SCRIPT="$REPO_ROOT/slurm/probing/probing_latent_retrieval.sh"
LATENT_DASH_SCRIPT="$REPO_ROOT/slurm/analysis/dash_probing_latent.sh"

# 🛠️ Utilities
WORKFLOW_SCRIPT="$REPO_ROOT/slurm/utils/workflow_controller.sh"

while getopts ":t:p:s:n:a:f:h" opt; do
  case $opt in
    t) TRAIN_DIR="$OPTARG" ;;
    p) PARENT_JOB_ID="$OPTARG" ;;
    s) JOB_SUFFIX="$OPTARG" ;;
    n) STEP_NAME="$OPTARG" ;;
    a) DATA_DIR="$OPTARG" ;;
    f) META_PATH="$OPTARG" ;;
    h) 
       echo "Usage: $0 -t 'path/to/training_dir' [-p 'parent_job_id'] [-s '_suffix'] [-n 'Step Name'] [-a 'data_dir'] [-f 'meta_path']"
       exit 0 
       ;;
    \?) 
       echo "Invalid option: -$OPTARG" >&2
       exit 1 
       ;;
  esac
done

if [ -z "$TRAIN_DIR" ]; then
    echo "[ERROR] You must provide a training directory using -t."
    exit 1
fi

# Absolute output paths
TRAIN_DIR=$(readlink -f "$TRAIN_DIR")
WEIGHTS_DIR="$TRAIN_DIR/weights"
LOGS_DIR="$TRAIN_DIR/logs"
PLOTS_DIR="$TRAIN_DIR/plots"
EMB_DIR="$TRAIN_DIR/embeddings"

EM_STAGE_TAG="${JOB_SUFFIX#_}"
ATTN_SAVE_DIR="$PLOTS_DIR/attention_maps$JOB_SUFFIX"
IMG_SPEC_SAVE_DIR="$PLOTS_DIR/images_spectra_reconstruction$JOB_SUFFIX"
CROSS_REC_SAVE_DIR="$PLOTS_DIR/cross_reconstruction$JOB_SUFFIX"

# Change to REPO_ROOT so that any sbatch calls inherit the correct SLURM_SUBMIT_DIR
cd "$REPO_ROOT" || { echo "[ERROR] Failed to cd to $REPO_ROOT"; exit 1; }

echo "-------------------------------------------------"
echo " --> Launching Analysis Battery for: $STEP_NAME"
echo "     Target Dir : $TRAIN_DIR"
echo "     Parent Job : $PARENT_JOB_ID"
echo "-------------------------------------------------"

# Plotting Metrics
J_MET=$(sbatch --parsable \
            --dependency=afterany:$PARENT_JOB_ID \
            --job-name="Plot_Metrics$JOB_SUFFIX" \
            "$PLOT_MET_SCRIPT" \
            -r "$REPO_ROOT" \
            -w "$WEIGHTS_DIR" \
            -l "$LOGS_DIR" \
            -s "$PLOTS_DIR")
echo "    [Metric]  Job sent.       ID: $J_MET"

# Workflow controller
J_WOR=$(sbatch --parsable \
            --dependency=afterany:$PARENT_JOB_ID \
            --job-name="Workflow_Controller$JOB_SUFFIX" \
            "$WORKFLOW_SCRIPT" \
            -r "$REPO_ROOT" \
            -t "$TRAIN_DIR" \
            -x "$JOB_SUFFIX")
echo "    [Work]    Job sent.       ID: $J_WOR"
if [[ -z "$J_WOR" ]]; then
    echo "    [ERROR] Workflow controller submission failed. Stopping analysis chain."
    exit 1
fi

# Plotting Attention Maps
J_ATT=$(sbatch --parsable \
            --dependency=afterok:$J_WOR \
            --job-name="Plot_Attn_Maps$JOB_SUFFIX" \
            "$ATTN_MAPS_SCRIPT" \
            -r "$REPO_ROOT" \
            -w "$WEIGHTS_DIR" \
            -s "$ATTN_SAVE_DIR" \
            -a "$DATA_DIR")
echo "    [Att Maps] Job sent.      ID: $J_ATT"

# Plotting Images and Spectra
J_IMG=$(sbatch --parsable \
            --dependency=afterok:$J_WOR \
            --job-name="Plot_Im_Sp$JOB_SUFFIX" \
            "$PLOT_IMG_SCRIPT" \
            -r "$REPO_ROOT" \
            -w "$WEIGHTS_DIR" \
            -s "$IMG_SPEC_SAVE_DIR" \
            -a "$DATA_DIR")
echo "    [ImgSpec] Job sent.       ID: $J_IMG"

# Cross Reconstruction
J_CROSS=$(sbatch --parsable \
            --dependency=afterok:$J_WOR \
            --job-name="Cross_Rec$JOB_SUFFIX" \
            "$CROSS_REC_SCRIPT" \
            -r "$REPO_ROOT" \
            -w "$WEIGHTS_DIR" \
            -s "$CROSS_REC_SAVE_DIR" \
            -a "$DATA_DIR")
echo "    [CrossRec] Job sent.      ID: $J_CROSS"

# Extract Embeddings 
J_EMB=$(sbatch --parsable \
            --dependency=afterany:$J_CROSS \
            --job-name="Extract_Embed$JOB_SUFFIX" \
            "$EXT_EMBD_SCRIPT" \
            -r "$REPO_ROOT" \
            -w "$WEIGHTS_DIR" \
            -s "$EMB_DIR" \
            -x "$EM_STAGE_TAG" \
            -a "$DATA_DIR")
echo "    [Embeds]  Job sent.       ID: $J_EMB"
if [[ -z "$J_EMB" ]]; then
    echo "    [ERROR] Embedding extraction submission failed. Stopping analysis chain."
    exit 1
fi

# Cosine Similarity
J_COS=$(sbatch --parsable \
            --dependency=afterok:$J_EMB \
            --job-name="Cos_Sim$JOB_SUFFIX" \
            "$COS_SIM_SCRIPT" \
            -r "$REPO_ROOT" \
            -w "$WEIGHTS_DIR" \
            -e "$EMB_DIR")
echo "    [CosSim]  Job sent.       ID: $J_COS"

# UMAPS
J_UMAP=$(sbatch --parsable \
            --dependency=afterok:$J_EMB \
            --job-name="Plot_Umaps$JOB_SUFFIX" \
            "$UMAPS_SCRIPT" \
            -r "$REPO_ROOT" \
            -w "$WEIGHTS_DIR" \
            -e "$EMB_DIR" \
            -a "$DATA_DIR" \
            -f "$META_PATH")
echo "    [UMAPS]   Job sent.       ID: $J_UMAP"

# PROBING DOWNSTREAM TASKS
J_PROB=$(sbatch --parsable \
            --dependency=afterok:$J_EMB \
            --job-name="Probing_Tasks$JOB_SUFFIX" \
            "$PROBING_SCRIPT" \
            -r "$REPO_ROOT" \
            -w "$WEIGHTS_DIR" \
            -e "$EMB_DIR" \
            -f "$META_PATH")
echo "    [PROBING] Job sent.       ID: $J_PROB"
if [[ -z "$J_PROB" ]]; then
    echo "    [ERROR] Downstream probing submission failed. Stopping analysis chain."
    exit 1
fi

# PROBING DOWNSTREAM TASKS DASHBOARD
J_PROB_DASH=$(sbatch --parsable \
            --dependency=afterok:$J_PROB \
            --job-name="Probing_Tasks_Dash$JOB_SUFFIX" \
            "$PROBING_DASH_SCRIPT" \
            -r "$REPO_ROOT" \
            -e "$EMB_DIR")
echo "    [PROB DASH] Job sent.     ID: $J_PROB_DASH"

# PROBING PREDICTIONS DASHBOARD
J_PRED_DASH=$(sbatch --parsable \
            --dependency=afterok:$J_PROB \
            --job-name="Predict_Dash$JOB_SUFFIX" \
            "$PREDICT_DASH_SCRIPT" \
            -r "$REPO_ROOT" \
            -e "$EMB_DIR")
echo "    [PRED DASH] Job sent.     ID: $J_PRED_DASH"

# PROBING LATENT MAPPING TASKS
J_LATENT=$(sbatch --parsable \
            --dependency=afterok:$J_EMB \
            --job-name="Latent_Tasks$JOB_SUFFIX" \
            "$LATENT_SCRIPT" \
            -r "$REPO_ROOT" \
            -w "$WEIGHTS_DIR" \
            -e "$EMB_DIR" \
            -f "$META_PATH")
echo "    [LATENT] Job sent.        ID: $J_LATENT"

# PROBING LATENT MAPPING DASHBOARD
J_LATENT_DASH=$(sbatch --parsable \
            --dependency=afterok:$J_LATENT \
            --job-name="Latent_Tasks_Dash$JOB_SUFFIX" \
            "$LATENT_DASH_SCRIPT" \
            -r "$REPO_ROOT" \
            -e "$EMB_DIR")
echo "    [LATENT DASH] Job sent.   ID: $J_LATENT_DASH"

echo " --> Analysis Batch Sent Successfully."
