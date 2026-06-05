#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=AstroPT_Anoms
#SBATCH --partition=gpu
#SBATCH --account=iac18
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=16       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=64G                
#SBATCH --time=03:00:00         

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_anom_hunter_%j.out
#SBATCH --error=logs/astropt_anom_hunter_%j.err

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

PYTHON_SCRIPT="$REPO_ROOT/scripts/analysis/anomalies/anomalies_analysis.py"

# Default metadata and data paths (can be overridden with flags)
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"
META_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1_FILTERED.fits"
N_ANOMALIES=10
BASE_MODALITY=""

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:w:s:e:f:a:n:m:" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    w) WEIGHTS_DIR="$OPTARG" ;;
    s) SAVE_DIR="$OPTARG" ;;
    e) EMB_DIR="$OPTARG" ;;
    f) META_PATH="$OPTARG" ;;
    a) DATA_DIR="$OPTARG" ;;
    n) N_ANOMALIES="$OPTARG" ;;
    m) BASE_MODALITY="$OPTARG" ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

if [ -z "$WEIGHTS_DIR" ] || [ -z "$EMB_DIR" ]; then
  echo "[ERROR]: WEIGHTS_DIR and EMB_DIR are required"
  echo "Usage: $0 -w <weights_dir> -e <embeddings_dir_or_root> [-s save_dir] [-a data_dir] [-f metadata_path] [-n n_anomalies] [-m base_modality]"
  exit 1
fi

# Absolute paths
WEIGHTS_DIR=$(readlink -f "$WEIGHTS_DIR")
DATA_DIR=$(readlink -f "$DATA_DIR")
META_PATH=$(readlink -f "$META_PATH")
EMB_DIR=$(readlink -f "$EMB_DIR")

# Checkpoint Resolution
if [ -f "$WEIGHTS_DIR" ]; then
    CKPT_PATH="$WEIGHTS_DIR"
elif [ -d "$WEIGHTS_DIR" ]; then
    if [ -f "$WEIGHTS_DIR/ckpt_best.pt" ]; then
        CKPT_PATH="$WEIGHTS_DIR/ckpt_best.pt"
    else
        # Find the first .pt file in the directory
        FIRST_PT=$(find "$WEIGHTS_DIR" -maxdepth 1 -name "*.pt" | head -n 1)
        if [ -n "$FIRST_PT" ]; then
            CKPT_PATH="$FIRST_PT"
        else
            echo "[ERROR]: No checkpoint (.pt) found in $WEIGHTS_DIR"
            exit 1
        fi
    fi
else
    echo "[ERROR]: WEIGHTS_DIR ($WEIGHTS_DIR) does not exist."
    exit 1
fi

#--- ENVIRONMENT SETUP ---#
NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "--------------------------------------------------------"
echo "Starting AstroPT Multimodal Anomaly Hunter Job $SLURM_JOB_ID - $NOW"
echo "--------------------------------------------------------"

# 1. Change directory
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# 2. Activate Environment
source "$REPO_ROOT/.venv/bin/activate"

# 3. Activating LaTeX (Required for high-quality astrophysical plots & reports)
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
export MPLCONFIGDIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/matplotlib"
export XDG_CACHE_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache"

#--- EMBEDDING DETECTION LOGIC ---#
# Check if the provided directory contains the files directly
if [ -f "$EMB_DIR/EuclidImage.npy" ] || [ -f "$EMB_DIR/EuclidImage_phase1.npy" ] || [ -f "$EMB_DIR/ids.npy" ]; then
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
    echo "[ERROR]: No valid embedding files (.npy) found in $EMB_DIR or its subdirectories."
    exit 1
fi

if [ -n "$SAVE_DIR" ]; then
    SAVE_DIR=$(readlink -f "$SAVE_DIR")
    OUTPUT_ARG="--output_dir $SAVE_DIR"
fi

#--- EXECUTION ---#
echo "Multimodal Anomaly Hunter Configuration:"
echo "    PYTHON SCRIPT:  $PYTHON_SCRIPT"
echo "    CHECKPOINT:     $CKPT_PATH"
echo "    EMB DIR:        $DETECTED_EMB"
echo "    METADATA:       $META_PATH"
echo "    DATA DIR:       $DATA_DIR"
echo "    N ANOMALIES:    $N_ANOMALIES"
echo "    BASE MODALITY:  $BASE_MODALITY"
if [ -n "$SAVE_DIR" ]; then
    echo "    SAVE DIR:       $SAVE_DIR (User-Specified)"
fi

# Run Python Script
python3 "$PYTHON_SCRIPT" \
    --embeddings_dir "$DETECTED_EMB" \
    --ckpt_path "$CKPT_PATH" \
    --data_dir "$DATA_DIR" \
    --metadata_path "$META_PATH" \
    --n_anomalies "$N_ANOMALIES" \
    $OUTPUT_ARG \
    --plot_projection

echo "--------------------------------------------------------"
echo "AstroPT Multimodal Anomaly Hunter Finished"
echo "--------------------------------------------------------"
