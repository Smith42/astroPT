#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Extract_Embed
#SBATCH --partition=gpu
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=32       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=128G                
#SBATCH --time=01:00:00          

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_extract_embed_%j.out
#SBATCH --error=logs/astropt_extract_embed_%j.err

set -euo pipefail

#--- DEFAULT VALUES ---#
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
PYTHON_SCRIPT="scripts/extract_embeddings.py"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_filter_corrupt"
PCA_DIM=0
EXP_TAG=""

print_usage() {
  echo "Usage: $0 -w WEIGHTS_DIR -s SAVE_DIR [options]"
  echo "Required:"
  echo "  -w <path>    Weights directory"
  echo "  -s <path>    Save directory"
  echo "Optional:"
  echo "  -r <path>    Repository root"
  echo "  -a <path>    Arrow data directory"
  echo "  -i <name>    Pool method for images (mean|max|mixed|lp|rank)"
  echo "  -m <name>    Pool method for spectra (mean|max|mixed|lp|rank)"
  echo "  -p <int>     PCA dimensions (0 disables PCA)"
  echo "  -x <tag>     Extra experiment tag appended to embedding folder name"
  echo "  -h           Help"
  echo
  echo "Experimental sweeps: use slurm/extract_embeddings_experimental.sh"
}

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:w:s:a:p:i:m:x:h" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    w) WEIGHTS_DIR="$OPTARG" ;;
    s) SAVE_DIR="$OPTARG" ;;
    a) DATA_DIR="$OPTARG" ;;
    p) PCA_DIM="$OPTARG" ;;
    i) POOL_MET_IMG="$OPTARG" ;;
    m) POOL_MET_SPEC="$OPTARG" ;;
    x) EXP_TAG="$OPTARG" ;;
    h) print_usage; exit 0 ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

if [[ -z "$WEIGHTS_DIR" || -z "$SAVE_DIR" ]]; then
  echo "[ERROR] Both -w and -s are required."
  print_usage
  exit 1
fi

# Absolute output path
WEIGHTS_DIR=$(readlink -f "$WEIGHTS_DIR")
SAVE_DIR=$(readlink -f "$SAVE_DIR")
DATA_DIR=$(readlink -f "$DATA_DIR")

#--- ENVIRONMENT SETUP ---#
NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "-----------------------------------------------"
echo "Starting Embedding Extraction Job $SLURM_JOB_ID - $NOW"
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
echo "    DATA DIR:       $DATA_DIR" 
echo "    WEIGHTS DIR:    $WEIGHTS_DIR"
echo "    SAVE DIR:       $SAVE_DIR"
if [[ -n "$EXP_TAG" ]]; then
  echo "    EXP TAG:        $EXP_TAG"
fi

# Run stable post-analysis extraction once per training run.
CMD=(python "$PYTHON_SCRIPT" \
    --weights_dir "$WEIGHTS_DIR" \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR")

if [[ -n "$EXP_TAG" ]]; then
  CMD+=(--exp_tag "$EXP_TAG")
fi

"${CMD[@]}"

echo "-----------------------------------------------"
echo "Embedding Extraction Finished"
echo "-----------------------------------------------"
