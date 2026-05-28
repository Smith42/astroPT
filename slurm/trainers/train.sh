#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Train_AstroPT_DDP
#SBATCH --partition=gpu
#SBATCH --nodes=1                # Nodes number
#SBATCH --ntasks=1               # Total task number (1 task invoking torchrun)
#SBATCH --cpus-per-task=64       # CPUs for task (16 per GPU * 4 GPUs)
#SBATCH --gpus-per-task=4        # GPUs for task - DDP
#SBATCH --mem=256G               # Requested RAM
#SBATCH --time=12:00:00          # Requested time in cluster

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_train_DDP_%j.out
#SBATCH --error=logs/astropt_train_DDP_%j.err

MAX_RUN_HOURS=11:55:00           # Time limit to training (in case of early stopping)

# Creating logs directory
mkdir -p logs

#--- Environment configuration ---#
NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "------------------------------------------------------"
echo "Training AstroPT ($SLURM_JOB_ID) on node $SLURM_NODELIST - $NOW"
echo "------------------------------------------------------"

# Robust repository root detection based on script location
# (Script is in slurm/trainers/, so REPO_ROOT is two levels up)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# If SLURM has moved us to /var/spool, correct REPO_ROOT to the launch working directory
if [[ "$REPO_ROOT" == "/var/spool"* ]]; then
    REPO_ROOT="$PWD"
fi

#--- DEFAULT VALUES ---#
CONFIG_FILE="$REPO_ROOT/config/default_config.yaml"
EXTRA_ARGS=""

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":c:x:" opt; do
  case $opt in
    c) CONFIG_FILE="$OPTARG" ;;
    x) EXTRA_ARGS="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

CONFIG_FILE=$(readlink -f "$CONFIG_FILE")

# Changing directory to run astropt
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# Activating AstroPT enviroment
source "$REPO_ROOT/.venv/bin/activate"

# Exports cache
export UV_CACHE_DIR="$REPO_ROOT/../cache/uv_cache"
export HF_HOME="$REPO_ROOT/../cache/huggingface"

# DDP optimization
# Matches cpus-per-task (64) / gpus (4) = 16
export OMP_NUM_THREADS=16 

# Running the training
echo "Training AstroPT in DDP GPU:"
echo "  CONFIG FILE:   $CONFIG_FILE"
if [[ -n "$EXTRA_ARGS" ]]; then
  echo "  EXTRA ARGS:    $EXTRA_ARGS"
fi

# Running Python Script with torchrun
CMD=(torchrun --standalone --nproc_per_node=4 "$REPO_ROOT/scripts/train.py" "$CONFIG_FILE" --max_run_hours "$MAX_RUN_HOURS")

if [[ -n "$EXTRA_ARGS" ]]; then
  eval "EXTRA_ARGS_ARRAY=($EXTRA_ARGS)"
  CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

"${CMD[@]}"

echo "------------------------------------------------------"
echo "Training Finished"
echo "------------------------------------------------------"
