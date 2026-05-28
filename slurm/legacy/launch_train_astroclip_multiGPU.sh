#!/bin/bash

#--- ASTROCLIP PIPELINE LAUNCHER ---#
echo "-------------------------------------------------"
echo "Launching AstroCLIP Pipeline (Train + Post-Analysis)"
echo "-------------------------------------------------"

# PATH AND DIR CONFIGURATION
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
SCRIPT_DIR=$(dirname "$0")
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated"
META_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits"

# LAUNCHING SCRIPTS
TRAIN_SCRIPT="$SCRIPT_DIR/train_astroclip_multiGPU.sh"
PLOT_MET_SCRIPT="$SCRIPT_DIR/plot_training_metrics.sh"
EXT_EMBD_SCRIPT="$SCRIPT_DIR/extract_embeddings_astroclip.sh"
COS_SIM_SCRIPT="$SCRIPT_DIR/cosine_similarity.sh"
UMAPS_SCRIPT="$SCRIPT_DIR/plot_umaps.sh"
PROBING_SCRIPT="$SCRIPT_DIR/probing_downstream.sh"
PROBING_DASH_SCRIPT="$SCRIPT_DIR/probing_downstream_dashboard.sh"
LATENT_SCRIPT="$SCRIPT_DIR/probing_latent_mapper.sh"
LATENT_DASH_SCRIPT="$SCRIPT_DIR/probing_latent_dashboard.sh"

# ARGUMENTS DEFAULT VALUES
TRAIN_NAME="New Train"
TRAIN_DESC="New AstroCLIP Training"
TRAIN_EXTRA_ARGS=""
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
DEFAULT_PATH="$REPO_ROOT/logs/astroclip_${TRAIN_DATE}_${SUFIX_NAME}"

if [ -z "${TRAIN_DIR:-}" ]; then
    TRAIN_DIR="$DEFAULT_PATH"

# Absolute output paths
TRAIN_DIR=$(readlink -f "$TRAIN_DIR")
WEIGHTS_DIR="$TRAIN_DIR/weights"
LOGS_DIR="$TRAIN_DIR/logs"
PLOTS_DIR="$TRAIN_DIR/plots"
EMB_DIR="$TRAIN_DIR/embeddings"

echo "Target Training Directory: $TRAIN_DIR"

ASTROCLIP_ROOT_PATH="${ASTROCLIP_ROOT:-$REPO_ROOT/../AstroCLIP}"
ASTRODINO_CKPT="$ASTROCLIP_ROOT_PATH/pretrained/astrodino.ckpt"
SPECFORMER_CKPT="$ASTROCLIP_ROOT_PATH/pretrained/specformer.ckpt"

if [[ ! -f "$ASTRODINO_CKPT" || ! -f "$SPECFORMER_CKPT" ]]; then
    echo "[ERROR] Missing AstroCLIP pretrained checkpoints:"
    [[ ! -f "$ASTRODINO_CKPT" ]] && echo "  - $ASTRODINO_CKPT"
    [[ ! -f "$SPECFORMER_CKPT" ]] && echo "  - $SPECFORMER_CKPT"
    echo "[ERROR] Aborting submission to avoid wasting queue time."
    echo "        Download from:"
    echo "        https://huggingface.co/polymathic-ai/astrodino"
    echo "        https://huggingface.co/polymathic-ai/specformer"
    exit 1

# Submit-time preflight so we do not enqueue long chains with broken imports.
if [[ ! -f "$REPO_ROOT/.venv/bin/python" ]]; then
    echo "[ERROR] Missing virtualenv python at $REPO_ROOT/.venv/bin/python"
    exit 1

echo "Running dependency preflight (dinov2 + astroclip imports)..."
if ! REPO_ROOT="$REPO_ROOT" "$REPO_ROOT/.venv/bin/python" - <<'PY'
import importlib
import os
from pathlib import Path
import sys

repo_root = Path(os.environ["REPO_ROOT"])
astroclip_dir = repo_root.parent / "AstroCLIP"
astroclip_module_file = astroclip_dir / "astroclip" / "models" / "astroclip.py"

if not astroclip_module_file.exists():
    print("[ERROR] Dependency preflight failed:")
    print(f"  - Missing AstroCLIP module file: {astroclip_module_file}")
    sys.exit(2)

sys.path.insert(0, str(astroclip_dir))

mods = ["dinov2", "dotenv"]
errors = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        errors.append((m, str(e)))

if errors:
    print("[ERROR] Dependency preflight failed:")
    for m, e in errors:
        print(f"  - {m}: {e}")
    sys.exit(2)

print("[OK] Dependency preflight passed.")
PY
then
    echo "[ERROR] Aborting: missing dependencies in astroPT/.venv."
    exit 1

# LAUNCH ANALYSIS FUNCTION
launch_analysis() {
    local PARENT_JOB_ID=$1
    local JOB_SUFFIX=$2
    local STEP_NAME=$3
    local EMB_STAGE_TAG="${JOB_SUFFIX#_}"

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
    echo "    [Metric]      Job sent. ID: $J_MET (Depends on Train: any)"

    # Extract Embeddings (AstroCLIP-aware)
    local J_EMB=$(sbatch --parsable \
                --dependency=afterany:$PARENT_JOB_ID \
                --job-name="Extract_Embed$JOB_SUFFIX" \
                "$EXT_EMBD_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -s "$EMB_DIR" \
                -x "$EMB_STAGE_TAG" \
                -a "$DATA_DIR"
            )
    echo "    [Embeds]      Job sent. ID: $J_EMB (Depends on Train: any)"
    if [[ -z "$J_EMB" ]]; then
        echo "    [ERROR] Embedding extraction submission failed. Stopping analysis chain."
        return 1

    # Cosine Similarity
    local J_COS=$(sbatch --parsable \
                --dependency=afterok:$J_EMB \
                --job-name="Cos_Sim$JOB_SUFFIX" \
                "$COS_SIM_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -e "$EMB_DIR"
            )
    echo "    [CosSim]      Job sent. ID: $J_COS (Depends on Embeds: ok)"

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
    echo "    [UMAPS]       Job sent. ID: $J_UMAP (Depends on Embeds: ok)"

    # Probing Downstream Tasks
    local J_PROB=$(sbatch --parsable \
                --dependency=afterok:$J_EMB \
                --job-name="Probing_Tasks$JOB_SUFFIX" \
                "$PROBING_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -e "$EMB_DIR" \
                -f "$META_PATH"
            )
    echo "    [PROBING]     Job sent. ID: $J_PROB (Depends on Embeds: ok)"
    if [[ -z "$J_PROB" ]]; then
        echo "    [ERROR] Downstream probing submission failed. Stopping analysis chain."
        return 1
    LAST_PROB_JOB_ID="$J_PROB"

    # Probing Dashboard
    J_PROB_DASH=$(sbatch --parsable \
                --dependency=afterok:$J_PROB \
                --job-name="Probing_Tasks_Dash$JOB_SUFFIX" \
                "$PROBING_DASH_SCRIPT" \
                -r "$REPO_ROOT" \
                -e "$EMB_DIR"
            )
    echo "    [PROB DASH]   Job sent. ID: $J_PROB_DASH (Depends on Probing: ok)"
    if [[ -z "$J_PROB_DASH" ]]; then
        echo "    [ERROR] Downstream dashboard submission failed. Stopping analysis chain."
        return 1

    # Latent Mapper
    local J_LATENT=$(sbatch --parsable \
                --dependency=afterok:$J_EMB \
                --job-name="Latent_Tasks$JOB_SUFFIX" \
                "$LATENT_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -e "$EMB_DIR" \
                -f "$META_PATH"
            )
    echo "    [LATENT]      Job sent. ID: $J_LATENT (Depends on Embeds: ok)"
    if [[ -z "$J_LATENT" ]]; then
        echo "    [ERROR] Latent probing submission failed. Stopping analysis chain."
        return 1

    # Latent Dashboard
    local J_LATENT_DASH=$(sbatch --parsable \
                --dependency=afterok:$J_LATENT \
                --job-name="Latent_Tasks_Dash$JOB_SUFFIX" \
                "$LATENT_DASH_SCRIPT" \
                -r "$REPO_ROOT" \
                -e "$EMB_DIR"
            )
    echo "    [LATENT DASH] Job sent. ID: $J_LATENT_DASH (Depends on Latent: ok)"
    if [[ -z "$J_LATENT_DASH" ]]; then
        echo "    [ERROR] Latent dashboard submission failed. Stopping analysis chain."
        return 1

    echo " --> Analysis Batch Sent."
}

# STEP 1 - TRAIN FROM SCRATCH
echo "-------------------------------------------------"
echo "STEP 1: TRAINING FROM SCRATCH"
JOB_SUFFIX="_T1"
JOB_TRAIN_1=$(sbatch --parsable \
    --job-name="Train_AstroCLIP_DDP$JOB_SUFFIX" \
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

launch_analysis "$JOB_TRAIN_1" "$JOB_SUFFIX" "Part 1 (SCRATCH)" || {
    echo "[ERROR] Analysis submission failed for STEP 1. Aborting pipeline."
    exit 1
}

# STEP 2 - TRAIN FROM RESUME
echo "-------------------------------------------------"
echo "STEP 2: TRAINING FROM RESUME"
JOB_SUFFIX="_T2"
JOB_TRAIN_2=$(sbatch --parsable \
    --dependency=afterany:$J_PROB_DASH \
    --job-name="Train_AstroCLIP_DDP$JOB_SUFFIX" \
    "$TRAIN_SCRIPT" \
    -r "$REPO_ROOT" \
    -t "$TRAIN_DIR" \
    -a "$DATA_DIR" \
    -n "$TRAIN_NAME" \
    -d "$TRAIN_DESC" \
    -m "resume" \
    -k "all" \
    -x "$TRAIN_EXTRA_ARGS")
echo "Training Job (Resume) launched. ID: $JOB_TRAIN_2"
if [[ -z "$JOB_TRAIN_2" ]]; then
    echo "[ERROR] Step 2 training submission failed. Aborting pipeline."
    exit 1

launch_analysis "$JOB_TRAIN_2" "$JOB_SUFFIX" "Part 2 (Resume)" || {
    echo "[ERROR] Analysis submission failed for STEP 2. Aborting pipeline."
    exit 1
}

# Optional step 3 for longer trainings
if [ -z "${LONG_TRAINING:-}" ]; then
    LONG_TRAINING="FALSE"

if [[ "$LONG_TRAINING" == "TRUE" ]]; then
    echo "-------------------------------------------------"
    echo "STEP 3: TRAINING FROM RESUME"
    JOB_SUFFIX="_T3"
    JOB_TRAIN_3=$(sbatch --parsable \
        --dependency=afterany:$LAST_PROB_JOB_ID \
        --job-name="Train_AstroCLIP_DDP$JOB_SUFFIX" \
        "$TRAIN_SCRIPT" \
        -r "$REPO_ROOT" \
        -t "$TRAIN_DIR" \
        -a "$DATA_DIR" \
        -n "$TRAIN_NAME" \
        -d "$TRAIN_DESC" \
        -m "resume" \
        -k "all" \
        -x "$TRAIN_EXTRA_ARGS")
    echo "Training Job (Resume) launched. ID: $JOB_TRAIN_3"
    if [[ -z "$JOB_TRAIN_3" ]]; then
        echo "[ERROR] Step 3 training submission failed. Aborting pipeline."
        exit 1

    launch_analysis "$JOB_TRAIN_3" "$JOB_SUFFIX" "Part 3 (Resume)" || {
        echo "[ERROR] Analysis submission failed for STEP 3. Aborting pipeline."
        exit 1
    }

echo "-------------------------------------------------"
echo "Pipeline Status Queue:"
squeue -u "$USER" -S i -o "%.10i %.6P %.25j %.8T %.10M %.35E %R"
echo "-------------------------------------------------"

# Note:
# Reconstruction/attention-specific AstroPT jobs are intentionally excluded.
# This launcher copies the analysis chain from embedding extraction onward.
