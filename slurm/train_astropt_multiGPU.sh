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
echo "------------------------------------------------------"
echo "Training AstroPT ($SLURM_JOB_ID) on node $SLURM_NODELIST"
echo "------------------------------------------------------"

#--- DEFAULT VALUES ---#
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
PYTHON_SCRIPT="scripts/$PYTHON_SCRIPT"
OUT_DIR="logs/astropt_100M_arrow"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"
TRAIN_NAME="New Train"
TRAIN_DESC="New AstroPT Training"
TRAIN_MODE="resume"
CHECK_MODE="both"

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:o:a:n:d:m:k:" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    a) DATA_DIR="$OPTARG" ;;
    n) TRAIN_NAME="$OPTARG" ;;
    d) TRAIN_DESC="$OPTARG" ;;
    m) TRAIN_MODE="$OPTARG" ;;
    k) CHECK_MODE="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Absolute output path
OUT_DIR=$(readlink -f "$OUT_DIR")

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
echo "  OUT DIR:       $OUT_DIR"
echo "  DATA DIR:      $DATA_DIR"
echo "  TRAIN NAME:    $TRAIN_NAME"
echo "  TRAIN MODE:    $TRAIN_MODE"

# Running Python Script with torch
# Note: --max_run_hours is set slightly lower than SBATCH time to allow clean autosave
torchrun --standalone --nproc_per_node=4 scripts/train_multimodal_arrow.py \
    --out_dir "$OUT_DIR" \
    --data_dir "$DATA_DIR" \
    --train_name "$TRAIN_NAME" \
    --train_description "$TRAIN_DES" \
    --init_from "$TRAIN_MODE" \
    --checkpoint_save_type "$CHECK_MODE" \
    --loss_type "l1" \
    --max_run_hours "11:55:00"

echo "------------------------------------------------------"
echo "Training Finished"
echo "------------------------------------------------------"