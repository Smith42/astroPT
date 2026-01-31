#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Probing_Tasks
#SBATCH --partition=gpu
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=16       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=64G                
#SBATCH --time=04:00:00         

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_probing_%j.out
#SBATCH --error=logs/astropt_probing_%j.err

#--- DEFAULT VALUES ---#
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
PYTHON_SCRIPT="scripts/probing_downstream.py"
OUT_DIR=""
META_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.0.fits"

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:o:f:" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    f) META_PATH="$OPTARG" ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Absolute output path
OUT_DIR=$(readlink -f "$OUT_DIR")

#--- ENVIRONMENT SETUP ---#
echo "-----------------------------------------------"
echo "Starting Probing/MLP Tasks Job $SLURM_JOB_ID"
echo "-----------------------------------------------"

# 1. Change directory
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "❌ Error: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# 2. Activate Environment
source .venv/bin/activate

# 3. Activating LaTeX (Required for confusion matrix plots)
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

#--- EMBEDDING DETECTION LOGIC ---#
EMB_DIR=$(ls -td "$OUT_DIR"/embeddings_* 2>/dev/null | head -n 1)

if [ -z "$EMB_DIR" ]; then
    echo "[ERROR]: No 'embeddings_*' directory found in $OUT_DIR"
    echo "[WARNING]: Run extract_embeddings.sh first"
    exit 1
fi

#--- EXECUTION ---#
echo "Probing Configuration:"
echo "   OUT DIR:       $OUT_DIR"
echo "   METADATA:      $META_PATH"
echo "   EMBEDDINGS:    $EMB_DIR"

# Run Python Script
python "$PYTHON_SCRIPT" \
    --out_dir "$OUT_DIR" \
    --metadata_path "$META_PATH" \
    --embeddings_dir "$EMB_DIR" \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-3

echo "-----------------------------------------------"
echo "Probing Tasks Finished"
echo "-----------------------------------------------"