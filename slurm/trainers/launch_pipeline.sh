#!/bin/bash

#--- ASTROPT PIPELINE LAUNCHER ---#
echo "-------------------------------------------------"
echo "Launching Full AstroPT Pipeline (Modular)"
echo "-------------------------------------------------"

# Robust repository root detection based on script location
# (Script is in slurm/trainers/, so REPO_ROOT is two levels up)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# If SLURM has moved us to /var/spool, correct REPO_ROOT to the launch working directory
if [[ "$REPO_ROOT" == "/var/spool"* ]]; then
    REPO_ROOT="$PWD"
fi

# LAUNCHING SCRIPTS
TRAIN_SCRIPT="$SCRIPT_DIR/train.sh"
ANALYSIS_PIPELINE="$REPO_ROOT/slurm/analysis/pipeline.sh"

# --- USER CONFIGURATION ---
# Change these paths to match your cluster environment
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated_tokenized"
META_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits"
# --------------------------

# ARGUMENTS DEFAULT VALUES
CONFIG_FILE="$REPO_ROOT/config/default_config.yaml"
CLI_TRAIN_NAME=""
CLI_TRAIN_DESC=""
TRAIN_DIR=""
TRAIN_EXTRA_ARGS=""

while getopts ":c:n:d:t:x:h" opt; do
  case $opt in
    c) CONFIG_FILE="$OPTARG" ;;
    n) CLI_TRAIN_NAME="$OPTARG" ;;
    d) CLI_TRAIN_DESC="$OPTARG" ;;
    t) TRAIN_DIR="$OPTARG" ;;
    x) TRAIN_EXTRA_ARGS="$OPTARG" ;;
    h) 
       echo "Usage: $0 [-c 'config.yaml'] [-n 'Name'] [-d 'Description'] [-t 'output_path/'] [-x 'extra flags']"
       exit 0 
       ;;
    \?) 
       echo "Invalid option: -$OPTARG" >&2
       exit 1 
       ;;
  esac
done

CONFIG_FILE=$(readlink -f "$CONFIG_FILE")

# Helper to read YAML values using python
read_yaml() {
    local key=$1
    "$REPO_ROOT/.venv/bin/python3" -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE')).get('$key', ''))" 2>/dev/null
}

# 1. Get values from YAML
YAML_TRAIN_NAME=$(read_yaml "train_name")
YAML_TRAIN_DESC=$(read_yaml "train_description")

# 2. Priority: CLI > YAML > DEFAULT
TRAIN_NAME="${CLI_TRAIN_NAME:-${YAML_TRAIN_NAME:-astropt_run}}"
TRAIN_DESC="${CLI_TRAIN_DESC:-${YAML_TRAIN_DESC:-AstroPT training run}}"

# AUTOMATIC DIRECTORY CONFIGURATION
SUFIX_NAME=$(echo "$TRAIN_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/ /g' | xargs | tr ' ' '_')
TRAIN_DATE="$(date +%Y%m%d)"

if [ -z "$TRAIN_DIR" ]; then
    TRAIN_DIR="$REPO_ROOT/logs/astropt_${TRAIN_DATE}_${SUFIX_NAME}"
fi

TRAIN_DIR=$(readlink -f "$TRAIN_DIR")

echo "Config File: $CONFIG_FILE"
echo "Target Training Directory: $TRAIN_DIR"
echo "-------------------------------------------------"

# Change to REPO_ROOT so that any sbatch calls inherit the correct SLURM_SUBMIT_DIR
cd "$REPO_ROOT" || { echo "[ERROR] Failed to cd to $REPO_ROOT"; exit 1; }

# 1. LAUNCH TRAINING
# We pass --train_dir via extra args so Python overrides the config and uses our known path.
COMBINED_EXTRA_ARGS="--train_dir \"$TRAIN_DIR\" --train_name \"$TRAIN_NAME\" --train_description \"$TRAIN_DESC\" --data_dir \"$DATA_DIR\" $TRAIN_EXTRA_ARGS"

echo "STEP 1: SUBMITTING TRAINING JOB"
JOB_TRAIN=$(sbatch --parsable \
    "$TRAIN_SCRIPT" \
    -c "$CONFIG_FILE" \
    -x "$COMBINED_EXTRA_ARGS")

if [[ -z "$JOB_TRAIN" ]]; then
    echo "[ERROR] Training submission failed. Aborting pipeline."
    exit 1
fi
echo "Training Job launched successfully. ID: $JOB_TRAIN"

# 2. CHAIN ANALYSIS PIPELINE
echo "-------------------------------------------------"
echo "STEP 2: CHAINING ANALYSIS PIPELINE"
$ANALYSIS_PIPELINE -t "$TRAIN_DIR" -p "$JOB_TRAIN" -a "$DATA_DIR" -f "$META_PATH"

# QUEUE STATUS
echo "-------------------------------------------------"
echo "Waiting for resources..."
sleep 5
echo "Pipeline Status Queue:"
squeue -u $USER -S i -o "%.10i %.6P %.25j %.8T %.10M %.35E %R"
echo "-------------------------------------------------"
