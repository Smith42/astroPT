#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Train_AstroPT_DDP
#SBATCH --partition=gpu
#SBATCH --nodes=1                # Nodes number
#SBATCH --ntasks=1               # Total task number (1 task invoking torchrun)
#SBATCH --cpus-per-task=64       # CPUs for task (16 per GPU * 4 GPUs)
#SBATCH --gpus-per-task=4        # GPUs for task - DDP
#SBATCH --mem=256G               # Requested RAM
#SBATCH --time=12:00:00          # Requested time

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_train_DDP_%j.out   # Output logs file
#SBATCH --error=logs/astropt_train_DDP_%j.err    # Error logs file

# Creating logs directory
mkdir -p logs

#--- Environment configuration ---#
NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "------------------------------------------------------"
echo "Training AstroPT ($SLURM_JOB_ID) on node $SLURM_NODELIST - $NOW"
echo "------------------------------------------------------"

#--- DEFAULT VALUES ---#
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
PYTHON_SCRIPT="scripts/$PYTHON_SCRIPT"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_filter_corrupt"
TRAIN_DIR="logs/astropt_100M_arrow"
TRAIN_NAME="New Train"
TRAIN_DESC="New AstroPT Training"
TRAIN_MODE="resume"
CHECK_MODE="both"
EXTRA_ARGS=""

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:t:a:n:d:m:k:x:" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    t) TRAIN_DIR="$OPTARG" ;;
    a) DATA_DIR="$OPTARG" ;;
    n) TRAIN_NAME="$OPTARG" ;;
    d) TRAIN_DESC="$OPTARG" ;;
    m) TRAIN_MODE="$OPTARG" ;;
    k) CHECK_MODE="$OPTARG" ;;
    x) EXTRA_ARGS="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Absolute output path
TRAIN_DIR=$(readlink -f "$TRAIN_DIR")

# Changing directory to run astropt
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# Activating AstroPT enviroment
source .venv/bin/activate

# Exports cache
export UV_CACHE_DIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/uv_cache"
export HF_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache/huggingface"

# DDP optimization
# Matches cpus-per-task (64) / gpus (4) = 16
export OMP_NUM_THREADS=16 

# Running the training
echo "Training AstroPT in DDP GPU:"
echo "  DATA DIR:      $DATA_DIR"
echo "  TRAIN DIR:     $TRAIN_DIR"
echo "  TRAIN NAME:    $TRAIN_NAME"
echo "  TRAIN MODE:    $TRAIN_MODE"
if [[ -n "$EXTRA_ARGS" ]]; then
  echo "  EXTRA ARGS:    $EXTRA_ARGS"
fi

# Running Python Script with torch
# Note: --max_run_hours is set slightly lower than SBATCH time to allow clean autosave
CMD=(torchrun --standalone --nproc_per_node=4 scripts/train_multimodal_arrow.py \
    --data_dir "$DATA_DIR" \
    --train_dir "$TRAIN_DIR" \
    --train_name "$TRAIN_NAME" \
    --train_description "$TRAIN_DESC" \
    --init_from "$TRAIN_MODE" \
    --checkpoint_save_type "$CHECK_MODE" \
    --max_run_hours "11:55:00")

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=($EXTRA_ARGS)
  CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

"${CMD[@]}"

echo "------------------------------------------------------"
echo "Training Finished"
echo "------------------------------------------------------"