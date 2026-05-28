#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Cos_Sim
#SBATCH --partition=gpu
#SBATCH --account=iac18
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G        
#SBATCH --time=00:20:00

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_cos_%j.out
#SBATCH --error=logs/astropt_cos_%j.err

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

PYTHON_SCRIPT="$REPO_ROOT/scripts/analysis/dashboards/dash_embeddings_alignment.py"

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:w:s:e:" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    w) WEIGHTS_DIR="$OPTARG" ;;
    s) SAVE_DIR="$OPTARG" ;;
    e) EMB_DIR="$OPTARG" ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

if [ -z "$WEIGHTS_DIR" ] || [ -z "$EMB_DIR" ]; then
  echo "[ERROR]: WEIGHTS_DIR and EMB_DIR are required"
  echo "Usage: $0 -w <weights_dir> -e <embedding_dir_or_root> [-s save_dir]"
  exit 1
fi

# Absolute paths
WEIGHTS_DIR=$(readlink -f "$WEIGHTS_DIR")
EMB_DIR=$(readlink -f "$EMB_DIR")

#--- ENVIRONMENT SETUP ---#
NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "-----------------------------------------------"
echo "Starting Cosine Similarity Job $SLURM_JOB_ID - $NOW" 
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
# Check if the provided directory contains the files directly (EuclidImage or legacy images)
if [ -f "$EMB_DIR/EuclidImage.npy" ] || [ -f "$EMB_DIR/images.npy" ] || [ -f "$EMB_DIR/embeddings_all.npz" ]; then
    DETECTED_EMB="$EMB_DIR"
else
    # Otherwise, search for the most recent subdirectory (standard extract_embeddings structure)
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
echo "Cosine Similarity Configuration:"
echo "    WEIGHTS DIR:  $WEIGHTS_DIR"
echo "    EMB DIR:      $DETECTED_EMB"
if [ -n "$SAVE_DIR" ]; then
    echo "    SAVE DIR:     $SAVE_DIR (User-specified)"
    echo "    SAVE DIR:     (Auto-inferring from Python script)"
fi

# Running Python Script
python "$PYTHON_SCRIPT" \
    --weights_dir "$WEIGHTS_DIR" \
    --emb_dir "$DETECTED_EMB" \
    $SAVE_ARG

echo "-----------------------------------------------"
echo "Cosine Similarity Finished"
echo "-----------------------------------------------"
