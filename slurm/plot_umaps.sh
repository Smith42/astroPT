#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Plot_Umaps
#SBATCH --partition=gpu
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=4       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=64G                
#SBATCH --time=01:00:00         

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_umaps_%j.out
#SBATCH --error=logs/astropt_umaps_%j.err

#--- DEFAULT VALUES ---#
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
PYTHON_SCRIPT="scripts/plot_umaps.py"
OUT_DIR=""
META_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.0.fits"

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:o:c:" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    f) META_PATH="$OPTARG" ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

#--- ENVIRONMENT SETUP ---#
echo "-----------------------------------------------"
echo "Starting Plotting UMAPS Job $SLURM_JOB_ID"
echo "-----------------------------------------------"

# 1. Change directory
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# 2. Activate Environment
source .venv/bin/activate

# 3. Activating LaTeX
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

#--- EMBEDDING DETECTION LOGIC ---#
EMB_DIR=$(ls -td "$OUT_DIR"/embeddings_* 2>/dev/null | head -n 1)

if [ -z "$EMB_DIR" ]; then
    echo "[ERROR]: No 'embeddings_*' directory found in $OUT_DIR"
    echo "[WARNING]: Run extract_embeddings.sh first"
    exit 1
fi

#--- EXECUTION ---#
echo "Plotting UMAPS Configuration:"
echo "   OUT DIR:       $OUT_DIR"
echo "   METADATA:      $META_PATH"
echo "   EMBEDDINGS:    $EMB_DIR (Auto-detected)"

# Run Python Script
python "$PYTHON_SCRIPT" \
    --out_dir "$OUT_DIR" \
    --metadata_path "$META_PATH" \
    --embeddings_dir "$EMB_DIR"

echo "-----------------------------------------------"
echo "Plotting UMAPS Finished"
echo "-----------------------------------------------"