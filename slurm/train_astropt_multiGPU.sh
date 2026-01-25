#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Train_AstroPT_DDP
#SBATCH --partition=gpu
#SBATCH --nodes=1                # Nodes number
#SBATCH --ntasks=1               # Total task number
#SBATCH --cpus-per-task=64       # CPUs for task
#SBATCH --gpus-per-task=4        # GPUs for task - DDP
#SBATCH --mem=256G               # Requested RAM
#SBATCH --time=01:00:00          # Requested time

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_train_DDP_%j.out   # Output logs file
#SBATCH --error=logs/astropt_train_DDP_%j.err    # Error logs file

#--- Enviroment configuration ---#
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
source  .venv/bin/activate

# Exports cache
export UV_CACHE_DIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/uv_cache"
export HF_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache/huggingface"

# DDP optimization
export OMP_NUM_THREADS=16

# Arguments
# Output Dir (Required):
OUT_DIR=${1:-"logs/astropt_100M_arrow"}

# Dataset Directory (Arrow)
DATA_DIR=${2:-"/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"}

# Training Mode (Optional): Default resume
MODE=${3:-resume} 

# Running the training
echo "Training AstroPT in DDP GPU:"
echo "   OUT DIR:   $OUT_DIR"
echo "   DATA DIR:  $DATA_DIR"
echo "   MODE:      $MODE"

# Running Python Script with torch
torchrun --standalone --nproc_per_node=4 scripts/train_multimodal_arrow.py \
    --out_dir "$OUT_DIR" \
    --data_dir "$DATA_DIR" \
    --init_from $MODE \
    --eval_interval 100 \
    --log_interval 50 \
    --max_run_hours "00:55:00" \
    --checkpoint_interval 200

echo "------------------------------------------------------"
echo "Job finished"
echo "------------------------------------------------------"