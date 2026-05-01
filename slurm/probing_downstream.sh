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
EPOCHS=50
BATCH_SIZE=128
LEARNING_RATE=1e-3
SEEDS="61,21,278"
MLP_VAL_SPLIT=0.1
MLP_PATIENCE=8
MLP_MIN_DELTA=1e-4
MLP_WEIGHT_DECAY=1e-3
MLP_EASY_EPOCH_FACTOR=0.6
EASY_TARGETS="SPECTYPE,data_set_release,has_spiral_arms_yes"

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

if [ -z "$WEIGHTS_DIR" ] || [ -z "$EMB_DIR" ]; then
        echo "[ERROR]: WEIGHTS_DIR and EMB_DIR are required"
        echo "Usage: $0 -w <weights_dir> -e <embeddings_root> [-s save_dir] [other flags]"
        exit 1
fi

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
export MPLCONFIGDIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/matplotlib"
export XDG_CACHE_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache"

#--- EMBEDDING DETECTION LOGIC ---#
if [ -f "$EMB_DIR/images.npy" ] || [ -f "$EMB_DIR/spectra.npy" ] || [ -f "$EMB_DIR/embeddings_all.npz" ]; then
    DETECTED_EMB="$EMB_DIR"
else
    DETECTED_EMB=$(ls -td "${EMB_DIR}"/*/ 2>/dev/null | head -n 1)
    DETECTED_EMB="${DETECTED_EMB%/}"
    DETECTED_EMB=$(readlink -f "$DETECTED_EMB")
fi

if [ -z "$DETECTED_EMB" ]; then
    echo "[ERROR]: No sub-directory found in $EMB_DIR"
    echo "[WARNING]: Run extract_embeddings.sh first"
    exit 1
fi

if [ -n "$SAVE_DIR" ]; then
    SAVE_DIR=$(readlink -f "$SAVE_DIR")
    SAVE_ARG="--save_dir $SAVE_DIR"
else
    SAVE_ARG=""
fi

# Extra Arguments
EXTRA_ARGS=()
if [ -n "$SUBSET_ID_PATH" ]; then
    SUBSET_ID_PATH=$(readlink -f "$SUBSET_ID_PATH")
    EXTRA_ARGS+=("--filter_ids_path" "$SUBSET_ID_PATH")
fi

if [ -n "$SAVE_CSV_NAME" ]; then
    EXTRA_ARGS+=("--save_name" "$SAVE_CSV_NAME")
fi

# Parse comma-separated values into repeated CLI args
IFS=',' read -r -a SEEDS_ARRAY <<< "$SEEDS"
IFS=',' read -r -a EASY_TARGETS_ARRAY <<< "$EASY_TARGETS"

EXTRA_ARGS+=("--seeds" "${SEEDS_ARRAY[@]}")
EXTRA_ARGS+=("--mlp_val_split" "$MLP_VAL_SPLIT")
EXTRA_ARGS+=("--mlp_patience" "$MLP_PATIENCE")
EXTRA_ARGS+=("--mlp_min_delta" "$MLP_MIN_DELTA")
EXTRA_ARGS+=("--mlp_weight_decay" "$MLP_WEIGHT_DECAY")
EXTRA_ARGS+=("--mlp_easy_epoch_factor" "$MLP_EASY_EPOCH_FACTOR")
EXTRA_ARGS+=("--easy_targets" "${EASY_TARGETS_ARRAY[@]}")

#--- EXECUTION ---#
echo "Probing Configuration:"
echo "    METADATA:       $META_PATH"
echo "    WEIGHTS DIR:    $WEIGHTS_DIR"
echo "    EMB DIR:        $DETECTED_EMB"
if [ -n "$SAVE_DIR" ]; then
    echo "    SAVE DIR:       $SAVE_DIR (User-Specified)"
else
    echo "    SAVE DIR:       (Auto-inferring from Python script)"
fi
echo "    SEEDS:          ${SEEDS_ARRAY[*]}"
echo "    EPOCHS:         $EPOCHS"
echo "    BATCH SIZE:     $BATCH_SIZE"
echo "    LR:             $LEARNING_RATE"

# Run Python Script
python "$PYTHON_SCRIPT" \
    --metadata_path "$META_PATH" \
    --weights_dir "$WEIGHTS_DIR" \
    --emb_dir "$DETECTED_EMB" \
    $SAVE_ARG \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    "${EXTRA_ARGS[@]}"

echo "-----------------------------------------------"
echo "Probing Tasks Finished"
echo "-----------------------------------------------"