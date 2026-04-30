#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Train_AstroCLIP_DDP
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=18:00:00

#--- LOGS FILES ---#
#SBATCH --output=logs/astroclip_train_DDP_%j.out
#SBATCH --error=logs/astroclip_train_DDP_%j.err

set -euo pipefail

mkdir -p logs

NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "------------------------------------------------------"
echo "Training AstroCLIP ($SLURM_JOB_ID) on node $SLURM_NODELIST - $NOW"
echo "------------------------------------------------------"

#--- DEFAULT VALUES ---#
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
PYTHON_SCRIPT="scripts/train_astroclip_arrow.py"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated"
TRAIN_DIR="logs/astroclip_arrow"
TRAIN_NAME="New Train"
TRAIN_DESC="New AstroCLIP Training"
TRAIN_MODE="resume"
CHECK_MODE="both"
EXTRA_ARGS=""

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:t:a:n:d:m:k:x:" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    t) TRAIN_DIR="$OPTARG" ;;
    a) DATA_DIR="$OPTARG" ;;
    n) TRAIN_NAME="$OPTARG" ;;
    d) TRAIN_DESC="$OPTARG" ;;
    m) TRAIN_MODE="$OPTARG" ;;
    k) CHECK_MODE="$OPTARG" ;;
    x) EXTRA_ARGS="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

TRAIN_DIR=$(readlink -f "$TRAIN_DIR")

echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

source .venv/bin/activate

export UV_CACHE_DIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/uv_cache"
export HF_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache/huggingface"
export OMP_NUM_THREADS=16

ASTROCLIP_ROOT_PATH="${ASTROCLIP_ROOT:-$REPO_ROOT/../AstroCLIP}"
ASTRODINO_CKPT="$ASTROCLIP_ROOT_PATH/pretrained/astrodino.ckpt"
SPECFORMER_CKPT="$ASTROCLIP_ROOT_PATH/pretrained/specformer.ckpt"

if [[ ! -f "$ASTRODINO_CKPT" || ! -f "$SPECFORMER_CKPT" ]]; then
  echo "[FATAL] Missing AstroCLIP pretrained checkpoints:"
  [[ ! -f "$ASTRODINO_CKPT" ]] && echo "  - $ASTRODINO_CKPT"
  [[ ! -f "$SPECFORMER_CKPT" ]] && echo "  - $SPECFORMER_CKPT"
  echo "[FATAL] Aborting before GPU initialization to save compute time."
  exit 2
fi

# Fast dependency check to avoid burning GPU time on immediate import crashes.
REPO_ROOT="$REPO_ROOT" python - <<'PY'
import importlib
import os
from pathlib import Path
import sys

repo_root = Path(os.environ["REPO_ROOT"]).resolve()
astroclip_dir = repo_root.parent / "AstroCLIP"
astroclip_module_file = astroclip_dir / "astroclip" / "models" / "astroclip.py"

if not astroclip_module_file.exists():
  print("[FATAL] Missing AstroCLIP module file:")
  print(f"  - {astroclip_module_file}")
  sys.exit(2)

sys.path.insert(0, str(astroclip_dir))

required = [
  "dinov2",
  "dotenv",
]

missing = []
for name in required:
  try:
    importlib.import_module(name)
  except Exception as exc:
    missing.append((name, str(exc)))

if missing:
  print("[FATAL] Missing or broken Python dependencies for AstroCLIP training:")
  for mod, err in missing:
    print(f"  - {mod}: {err}")
  print("[FATAL] Install dependencies in astroPT/.venv before re-submitting.")
  sys.exit(2)

print("[OK] Python dependency preflight passed (dinov2 + astroclip).")
PY

echo "Training AstroCLIP in DDP GPU:"
echo "  DATA DIR:      $DATA_DIR"
echo "  TRAIN DIR:     $TRAIN_DIR"
echo "  TRAIN NAME:    $TRAIN_NAME"
echo "  TRAIN MODE:    $TRAIN_MODE"
if [[ -n "$EXTRA_ARGS" ]]; then
  echo "  EXTRA ARGS:    $EXTRA_ARGS"
fi

# Use torchrun to spawn 4 DDP workers, matching train_astropt_multiGPU.sh.
# Plain `python script.py` lets Lightning try to spawn subprocesses internally,
# which deadlocks in SLURM because Lightning detects the SLURM environment and
# expects 4 separate tasks (ntasks=4) rather than spawning them itself.
CMD=(torchrun --standalone --nproc_per_node=4 "$PYTHON_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --train_dir "$TRAIN_DIR" \
    --train_name "$TRAIN_NAME" \
    --train_description "$TRAIN_DESC" \
    --init_from "$TRAIN_MODE" \
    --checkpoint_save_type "$CHECK_MODE" \
    --ddp_num_gpus 4)

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=($EXTRA_ARGS)
  CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

"${CMD[@]}"

echo "------------------------------------------------------"
echo "Training Finished"
echo "------------------------------------------------------"
