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

#--- DEFAULT VALUES ---#
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
PYTHON_SCRIPT="scripts/extract_embeddings.py"
OUT_DIR=""
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:o:a:" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    a) DATA_DIR="$OPTARG" ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

#--- ENVIRONMENT SETUP ---#
echo "-----------------------------------------------"
echo "Starting Embedding Extraction Job $SLURM_JOB_ID"
echo "-----------------------------------------------"

# 1. Change directory
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# 2. Activate Environment
source .venv/bin/activate

# 3. Exports cache
export UV_CACHE_DIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/uv_cache"
export HF_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache/huggingface"

#--- EXECUTION ---#
echo "Embedding Extraction Configuration:"
echo "   DIR:        $OUT_DIR"
echo "   DATA DIR:   $DATA_DIR"


# Run Python Script
python "$PYTHON_SCRIPT" \
    --out_dir "$OUT_DIR" \
    --data_dir "$DATA_DIR"

echo "-----------------------------------------------"
echo "Embedding Extraction Finished"
echo "-----------------------------------------------"