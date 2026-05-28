#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=DiffSpec
#SBATCH --partition=gpu
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=4      
#SBATCH --gpus-per-task=1        
#SBATCH --mem=128G                
#SBATCH --time=5:00:00         
#SBATCH --output=logs/diffusion_%A.out
#SBATCH --error=logs/diffusion_%A.err  

set -euo pipefail

# Robust repository root detection based on script location
# (Script is in slurm/trainers/, so REPO_ROOT is two levels up)
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

PYTHON_SCRIPT="$REPO_ROOT/scripts/experimental/train_diffusion.py"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated"
EMBEDDINGS_PATH="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/astropt_20260516_hybrid_cliploss/embeddings/best_img-mean_spec-rank_final_iso_j-mean/embeddings_all.npz"
SAVE_DIR="$REPO_ROOT/logs/astropt_20260516_hybrid_cliploss/spectrum_diffusion"
CONTEXT_DIM=512
MAX_RUN_HOURS="4:55:00"

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":e:a:s:c:" opt; do
  case $opt in
    e) EMBEDDINGS_PATH="$OPTARG" ;;
    a) DATA_DIR="$OPTARG" ;;
    s) SAVE_DIR="$OPTARG" ;;
    c) CONTEXT_DIM="$OPTARG" ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

shift $((OPTIND - 1))

# Absolute paths
EMBEDDINGS_PATH=$(readlink -f "$EMBEDDINGS_PATH" || echo "$EMBEDDINGS_PATH")
DATA_DIR=$(readlink -f "$DATA_DIR" || echo "$DATA_DIR")
SAVE_DIR=$(readlink -f "$SAVE_DIR" || echo "$SAVE_DIR")

NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "-----------------------------------------------"
echo "Spectrum Diffusion Model ${SLURM_JOB_ID:-LOCAL} - $NOW"
echo "Embeddings: $EMBEDDINGS_PATH"
echo "Data:       $DATA_DIR"
echo "Save:       $SAVE_DIR"
echo "Context:    $CONTEXT_DIM"
echo "-----------------------------------------------"

cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# Activate virtual environment
if [ -d ".venv" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "Running Diffusion Training..."
python3 "$PYTHON_SCRIPT" \
    --embeddings_path "$EMBEDDINGS_PATH" \
    --data_dir "$DATA_DIR" \
    --train_dir "$SAVE_DIR" \
    --context_dim "$CONTEXT_DIM" \
    --max_run_hours "$MAX_RUN_HOURS" \
    --init_from "resume"

echo "-----------------------------------------------"
echo "Diffusion Training Finished"
echo "-----------------------------------------------"
