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
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"
META_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits"

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:e:o:f:" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    e) EMB_DIR="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    f) META_PATH="$OPTARG" ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Absolute output path
OUT_DIR=$(readlink -f "$OUT_DIR")

#--- EMBEDDING DETECTION LOGIC ---#
if [ -z "$EMB_DIR" ]; then
    echo "[INFO] EMB_DIR not set. Searching for latest embeddings in $OUT_DIR..."
    EMB_DIR=$(ls -td "$OUT_DIR"/embeddings_* 2>/dev/null | head -n 1)
fi

if [ -z "$EMB_DIR" ]; then
    echo "[ERROR]: No 'embeddings_*' directory found in $OUT_DIR"
    echo "[WARNING]: Run extract_embeddings.sh first"
    #exit 1
fi

EMB_DIR=$(readlink -f "$EMB_DIR")

#--- ENVIRONMENT SETUP ---#
echo "-----------------------------------------------"
echo "Starting Plotting UMAPS Job $SLURM_JOB_ID"
echo "-----------------------------------------------"

# 1. Change directory
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# 2. Activate Environment
source .venv/bin/activate

# 3. Activating LaTeX (Required for plots)
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

#--- EXECUTION ---#
echo "Plotting UMAPS Configuration:"
echo "   OUT DIR:       $EMB_DIR"
echo "   METADATA:      $META_PATH"
echo "   EMBEDDINGS:    $EMB_DIR"

# Run Python Script
python "$PYTHON_SCRIPT" \
    --out_dir "$EMB_DIR" \
    --data_dir "$DATA_DIR" \
    --metadata_path "$META_PATH" \
    --emb_dir "$EMB_DIR" \
    --plot_spectral \
    --plot_visual \
    --plot_standard
    

echo "-----------------------------------------------"
echo "Plotting UMAPS Finished"
echo "-----------------------------------------------"