#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Latent_Proj
#SBATCH --partition=gpu
#SBATCH --account=iac18
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=16       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=64G                
#SBATCH --time=03:00:00         

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_latent_%j.out
#SBATCH --error=logs/astropt_latent_%j.err

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

PYTHON_SCRIPT="$REPO_ROOT/scripts/analysis/dashboards/dash_latent_projection.py"

# Default metadata and data paths (can be overridden with flags)
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"
META_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits"

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:w:s:e:a:f:" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    w) WEIGHTS_DIR="$OPTARG" ;;
    s) SAVE_DIR="$OPTARG" ;;
    e) EMB_DIR="$OPTARG" ;;
    a) DATA_DIR="$OPTARG" ;;
    f) META_PATH="$OPTARG" ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

if [ -z "$WEIGHTS_DIR" ] || [ -z "$EMB_DIR" ]; then
  echo "[ERROR]: WEIGHTS_DIR and EMB_DIR are required"
  echo "Usage: $0 -w <weights_dir> -e <embedding_dir_or_root> [-s save_dir] [-a data_dir] [-f metadata_path]"
  exit 1
fi

# Absolute paths
WEIGHTS_DIR=$(readlink -f "$WEIGHTS_DIR")
DATA_DIR=$(readlink -f "$DATA_DIR")
META_PATH=$(readlink -f "$META_PATH")
EMB_DIR=$(readlink -f "$EMB_DIR")

#--- ENVIRONMENT SETUP ---#
NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "-----------------------------------------------"
echo "Starting Latent Projections Job $SLURM_JOB_ID - $NOW"
echo "-----------------------------------------------"

# 1. Change directory
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# 2. Activate Environment
source "$REPO_ROOT/.venv/bin/activate"

# 3. Activating LaTeX (Required for plots)
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
export MPLCONFIGDIR="$REPO_ROOT/../cache/matplotlib"
export XDG_CACHE_HOME="$REPO_ROOT/../cache"

#--- EMBEDDING DETECTION LOGIC ---#
# Check if the provided directory contains the files directly
if [ -f "$EMB_DIR/EuclidImage.npy" ] || [ -f "$EMB_DIR/images.npy" ] || [ -f "$EMB_DIR/embeddings_all.npz" ]; then
    DETECTED_EMB="$EMB_DIR"
else
    # Otherwise, search for the most recent subdirectory
    SUBDIR=$(ls -td "${EMB_DIR}"/*/ 2>/dev/null | head -n 1)
    if [ -n "$SUBDIR" ]; then
        DETECTED_EMB="${SUBDIR%/}"
    else
        DETECTED_EMB=""
    fi
fi

if [ -n "$DETECTED_EMB" ]; then
    DETECTED_EMB=$(readlink -f "$DETECTED_EMB")
fi

if [ -z "$DETECTED_EMB" ]; then
    echo "[ERROR]: No valid embedding files (.npy/.npz) found in $EMB_DIR or its subdirectories."
    echo "[WARNING]: Run extract_multimodal_embeddings.sh first"
    exit 1
fi

if [ -n "$SAVE_DIR" ]; then
    SAVE_DIR=$(readlink -f "$SAVE_DIR")
    SAVE_ARG="--save_dir $SAVE_DIR"
fi

#--- EXECUTION ---#
echo "Latent Projections Configuration:"
echo "    METADATA:       $META_PATH"
echo "    DATA DIR:       $DATA_DIR" 
echo "    WEIGHTS DIR:    $WEIGHTS_DIR"
echo "    EMB DIR:        $DETECTED_EMB"
if [ -n "$SAVE_DIR" ]; then
    echo "    SAVE DIR:       $SAVE_DIR (User-Specified)"
fi

# Run Python Script
python "$PYTHON_SCRIPT" \
    --metadata_path "$META_PATH" \
    --weights_dir "$WEIGHTS_DIR" \
    --emb_dir "$DETECTED_EMB" \
    $SAVE_ARG \
    --data_dir "$DATA_DIR" \
    --plot_spectral \
    --plot_visual \
    --plot_standard

echo "-----------------------------------------------"
echo "Latent Projections Finished"
echo "-----------------------------------------------"
