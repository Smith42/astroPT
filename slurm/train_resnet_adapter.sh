#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Train_Resnet
#SBATCH --partition=gpu
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=64       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=128G                
#SBATCH --time=12:00:00          

#--- LOGS FILES ---#
#SBATCH --output=logs/resnet_adapter_%j.out
#SBATCH --error=logs/resnet_adapter_%j.err

set -euo pipefail

#--- DEFAULT VALUES ---#
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated"
SAVE_DIR="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/resnet_adapter_weights"
RETRAIN_SCRIPT="${REPO_ROOT}/scripts/train_resnet_adapter.py"

BATCH_SIZE=32
EPOCHS=20
LR=1e-4
TARGET_SIZE=112
TRANSFORM_METHOD="resize"

print_usage() {
  echo "Usage: $0 [options]"
  echo "Optional:"
  echo "  -r <path>    AstroPT Repository root"
  echo "  -a <path>    Arrow data directory (cache dir)"
  echo "  -s <path>    Save directory for adapter weights"
  echo "  -b <int>     Batch size"
  echo "  -e <int>     Epochs"
  echo "  -l <float>   Learning rate"
  echo "  -t <int>     Target size for AION ImageCodec (default: 112)"
  echo "  -m <string>  Transform method: 'crop' or 'resize' (default: crop)"
  echo "  -h           Help"
  echo
}

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:s:a:b:e:l:t:m:h" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    s) SAVE_DIR="$OPTARG" ;;
    a) DATA_DIR="$OPTARG" ;;
    b) BATCH_SIZE="$OPTARG" ;;
    e) EPOCHS="$OPTARG" ;;
    l) LR="$OPTARG" ;;
    t) TARGET_SIZE="$OPTARG" ;;
    m) TRANSFORM_METHOD="$OPTARG" ;;
    h) print_usage; exit 0 ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Absolute output path
SAVE_DIR=$(readlink -f "$SAVE_DIR")
DATA_DIR=$(readlink -f "$DATA_DIR")

#--- ENVIRONMENT SETUP ---#
NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "-----------------------------------------------"
echo "Starting Resnet Adapter Training Job $SLURM_JOB_ID - $NOW"
echo "-----------------------------------------------"

# 1. Change directory
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# 2. Activate Environment
source .venv/bin/activate

# 3. Exports cache
export UV_CACHE_DIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/uv_cache"
export HF_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache/huggingface"

mkdir -p "$SAVE_DIR"

#--- EXECUTION ---#
echo "U-Net Adapter Training Configuration:"
echo "    DATA DIR:       $DATA_DIR" 
echo "    SAVE DIR:       $SAVE_DIR"
echo "    BATCH SIZE:     $BATCH_SIZE"
echo "    EPOCHS:         $EPOCHS"
echo "    LR:             $LR"
echo "    TARGET SIZE:    $TARGET_SIZE"
echo "    METHOD:         $TRANSFORM_METHOD"

CMD=(python "$RETRAIN_SCRIPT" \
    --data-dir "$DATA_DIR" \
    --output "$SAVE_DIR" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --target-size "$TARGET_SIZE" \
    --transform-method "$TRANSFORM_METHOD" \
    --num-workers 4)

"${CMD[@]}"

echo "-----------------------------------------------"
echo "Resnet Adapter Training Finished"
echo "-----------------------------------------------"
