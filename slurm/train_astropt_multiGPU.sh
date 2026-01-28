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

#--- Environment configuration ---#
echo "------------------------------------------------------"
echo "Starting the job $SLURM_JOB_ID on node $SLURM_NODELIST"
echo "------------------------------------------------------"

# Creating logs directory
mkdir -p logs

# Changing directory to run astropt
REPO_ROOT=${1:-"/home/valonso/iac18_mhuertas_shared/valonso/astroPT"}
shift
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || exit 1

# Activating AstroPT enviroment
source .venv/bin/activate

# Exports cache
export UV_CACHE_DIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/uv_cache"
export HF_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache/huggingface"

# DDP optimization
# Matches cpus-per-task (64) / gpus (4) = 16
export OMP_NUM_THREADS=16 

# Arguments
# Output Dir (Required):
OUT_DIR=${1:-"logs/astropt_100M_arrow"}

# Dataset Directory (Arrow)
DATA_DIR=${2:-"/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"}

# Training Mode (Optional): Default resume
TRAIN_MODE=${3:-resume}

# Saving checkpoint mode (Optional): Default both
CHECK_MODE=${4:-both}

# Running the training
echo "Training AstroPT in DDP GPU:"
echo "  OUT DIR:       $OUT_DIR"
echo "  DATA DIR:      $DATA_DIR"
echo "  TRAIN MODE:    $TRAIN_MODE"
echo "  CHECKPOINT:    $CHECK_MODE"

# Running Python Script with torch
# Note: --max_run_hours is set slightly lower than SBATCH time to allow clean autosave
torchrun --standalone --nproc_per_node=4 scripts/train_multimodal_arrow.py \
    --out_dir "$OUT_DIR" \
    --data_dir "$DATA_DIR" \
    --init_from "$TRAIN_MODE" \
    --checkpoint_save_type "$CHECK_MODE" \
    --loss_type "l1" \
    --max_run_hours "11:55:00"

echo "------------------------------------------------------"
echo "Job finished"
echo "------------------------------------------------------"