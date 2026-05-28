#!/bin/bash

# --- ASTROPT FAST TEST SCRIPT ---
# Use this script on an interactive GPU node to quickly test 
# if the model builds, the dataloader works, and it trains for a few iterations.
# It DOES NOT use torchrun (runs on a single GPU).

# Robust repository root detection based on script location
# (Script is in slurm/trainers/, so REPO_ROOT is two levels up)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# If SLURM has moved us to /var/spool, correct REPO_ROOT to the launch working directory
if [[ "$REPO_ROOT" == "/var/spool"* ]]; then
    REPO_ROOT="$PWD"
fi

cd "$REPO_ROOT" || exit 1

# Activate environment
source "$REPO_ROOT/.venv/bin/activate"

# Environment Configuration
# We use paths relative to REPO_ROOT to make the script portable
export UV_CACHE_DIR="$REPO_ROOT/../cache/uv_cache"
export HF_HOME="$REPO_ROOT/../cache/huggingface"
export OMP_NUM_THREADS=4 

# Default values
CONFIG_FILE="$REPO_ROOT/config/default_config.yaml"
T_NAME="Debug_Run"

while getopts "c:n:" opt; do
  case $opt in
    c) CONFIG_FILE="$OPTARG" ;;
    n) T_NAME="$OPTARG" ;;
    *) echo "Usage: $0 [-c config.yaml] [-n name]" ;;
  esac
done

CONFIG_FILE=$(readlink -f "$CONFIG_FILE")

echo "========================================="
echo "Running Fast Test: $T_NAME"
echo "Config: $CONFIG_FILE"
echo "========================================="

# We override the training length and evaluation intervals to fail fast
python scripts/train.py "$CONFIG_FILE" \
    --train_name "$T_NAME" \
    --eval_interval 20 \
    --log_interval 10 \
    --checkpoint_interval 20 \
    --max_run_hours "00:20:00" \
    --compile "False"

echo "========================================="
echo "Test Finished."
echo "========================================="
