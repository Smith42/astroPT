#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Extract_Embed
#SBATCH --partition=gpu
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=16       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=64G                
#SBATCH --time=02:00:00          

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_extract_embed_%j.out
#SBATCH --error=logs/astropt_extract_embed_%j.err

#--- Enviroment configuration ---#
echo "-----------------------------------------------"
echo "Starting Embedding Extraction Job $SLURM_JOB_ID"
echo "-----------------------------------------------"

# Changing directory to run astropt
REPO_ROOT=${1:-"/home/valonso/iac18_mhuertas_shared/valonso/astroPT"}
shift
echo "Changing directory to: $REPO_ROOT"
cd "$PROJ_ROOT" || exit 1

# Activating AstroPT enviroment
source .venv/bin/activate

# Exports cache
export UV_CACHE_DIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/uv_cache"
export HF_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache/huggingface"

# Arguments
# Input/Output Dir (Required):
TRAIN_OUT_DIR=${1:-"logs/astropt_100M_arrow"}

# Dataset Directory (Arrow)
DATA_DIR=${2:-"/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"}

# Checkpoint Name (Optional): Default ckpt_best.pt
CKPT_NAME=${3:-"ckpt_best.pt"}

echo "Extracting from:"
echo "   DIR:  $TRAIN_OUT_DIR"
echo "   CHECKPOINT: $CKPT_NAME"

# Run Python Script
python scripts/extract_embeddings.py \
    --out_dir "$TRAIN_OUT_DIR" \
    --data_dir "$DATA_DIR" \
    --ckpt_name "$CKPT_NAME" \
    --batch_size 64 \
    --num_workers 8

echo "-----------------------------------------------"
echo "Extraction Finished"
echo "-----------------------------------------------"