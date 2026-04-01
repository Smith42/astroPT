#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Probing_Tasks
#SBATCH --partition=gpu
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=16       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=64G                
#SBATCH --time=01:00:00         

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_probing_%j.out
#SBATCH --error=logs/astropt_probing_%j.err

#--- DEFAULT VALUES ---#
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
PYTHON_SCRIPT="scripts/probing_downstream.py"
META_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits"

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:w:s:e:f:u:n:" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    w) WEIGHTS_DIR="$OPTARG" ;;
    s) SAVE_DIR="$OPTARG" ;;
    e) EMB_DIR="$OPTARG" ;;
    f) META_PATH="$OPTARG" ;;
    u) SUBSET_ID_PATH="$OPTARG" ;;
    n) SAVE_CSV_NAME="$OPTARG" ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Absolute output path
WEIGHTS_DIR=$(readlink -f "$WEIGHTS_DIR")
META_PATH=$(readlink -f "$META_PATH")
EMB_DIR=$(readlink -f "$EMB_DIR")

#--- ENVIRONMENT SETUP ---#
NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "-----------------------------------------------"
echo "Starting Probing/MLP Tasks Job $SLURM_JOB_ID - $NOW"
echo "-----------------------------------------------"

# 1. Change directory
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# 2. Activate Environment
source .venv/bin/activate

# 3. Activating LaTeX (Required for confusion matrix plots)
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

#--- EMBEDDING DETECTION LOGIC ---#
DETECTED_EMB=$(ls -td "${EMB_DIR}"/*/ 2>/dev/null | head -n 1)
DETECTED_EMB="${DETECTED_EMB%/}"
DETECTED_EMB=$(readlink -f "$DETECTED_EMB")

if [ -z "$DETECTED_EMB" ]; then
    echo "[ERROR]: No sub-directory found in $EMB_DIR"
    echo "[WARNING]: Run extract_embeddings.sh first"
    exit 1
fi

if [ -z "$SAVE_DIR" ]; then
    SAVE_DIR="$DETECTED_EMB"
fi
SAVE_DIR=$(readlink -f "$SAVE_DIR")

# Extra Arguments
EXTRA_ARGS=()
if [ -n "$SUBSET_ID_PATH" ]; then
    SUBSET_ID_PATH=$(readlink -f "$SUBSET_ID_PATH")
    EXTRA_ARGS+=("--filter_ids_path" "$SUBSET_ID_PATH")
fi

if [ -n "$SAVE_CSV_NAME" ]; then
    EXTRA_ARGS+=("--save_name" "$SAVE_CSV_NAME")
fi

#--- EXECUTION ---#
echo "Probing Configuration:"
echo "    METADATA:       $META_PATH"
echo "    WEIGHTS DIR:    $WEIGHTS_DIR"
echo "    EMB DIR:        $DETECTED_EMB"
echo "    SAVE DIR:       $SAVE_DIR (Auto-Detected)"

# Run Python Script
python "$PYTHON_SCRIPT" \
    --metadata_path "$META_PATH" \
    --weights_dir "$WEIGHTS_DIR" \
    --emb_dir "$DETECTED_EMB" \
    --save_dir "$SAVE_DIR" \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-3
    "${EXTRA_ARGS[@]}"

echo "-----------------------------------------------"
echo "Probing Tasks Finished"
echo "-----------------------------------------------"