#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Extract_Embed
#SBATCH --partition=gpu
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=16       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=64G                
#SBATCH --time=01:00:00          

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
cd "$REPO_ROOT" || exit 1
source .venv/bin/activate

# Exports cache
export UV_CACHE_DIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/uv_cache"
export HF_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache/huggingface"

# Arguments
# Input/Output Dir (Required):
OUT_DIR=${1:-"logs/astropt_100M_250K_arrow_20260128"}

# Dataset Directory (Arrow)
DATA_DIR=${2:-"/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"}

echo "Embedding Extraction:"
echo "   DIR:        $OUT_DIR"
echo "   DATA DIR:   $DATA_DIR"

# Run Python Script
python scripts/extract_embeddings.py \
    --out_dir "$OUT_DIR" \
    --data_dir "$DATA_DIR"

echo "-----------------------------------------------"
echo "Embedding Extraction Finished"
echo "-----------------------------------------------"