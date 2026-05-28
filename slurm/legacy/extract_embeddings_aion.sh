#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=AION_Extract
#SBATCH --partition=gpu
#SBATCH --account=iac18
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=16       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=64G                
#SBATCH --time=04:00:00          

#--- LOGS FILES ---#
#SBATCH --output=logs/aion_extract_%j.out
#SBATCH --error=logs/aion_extract_%j.err

set -euo pipefail

# Robust repository root detection based on script location
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# If SLURM has moved us to /var/spool, correct REPO_ROOT to the launch working directory
if [[ "$REPO_ROOT" == "/var/spool"* ]]; then
    if [ -d "$PWD/astroPT" ]; then
        REPO_ROOT="$PWD/astroPT"
    else
        REPO_ROOT="$PWD"
    fi
fi

# Activate Environment
source "$REPO_ROOT/.venv/bin/activate"

# Configure Hugging Face Cache and Offline Mode
export UV_CACHE_DIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/uv_cache"
export HF_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_OFFLINE=1

# Change directory
cd "$REPO_ROOT"

# Run AION embedding extraction script
python3 scripts/legacy/extract_embeddings_aion.py \
    --data_dir "/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated_tokenized" \
    --save_dir "/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/AION_freeze/embeddings/aion_embeddings" \
    --resnet_weights_path "/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/Euclid2AION_resnet_adapter_weights/adapters_final.pt" \
    --batch_size 64 \
    --device "cuda" \
    --num_workers 16

echo "-----------------------------------------------"
echo "AION Embedding Extraction Finished"
echo "-----------------------------------------------"
