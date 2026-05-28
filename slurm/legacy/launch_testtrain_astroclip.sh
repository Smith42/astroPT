#!/bin/bash

set -euo pipefail

# Environment Configuration
export REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
export DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated"
export UV_CACHE_DIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/uv_cache"
export HF_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache/huggingface"
export OMP_NUM_THREADS=4

# Test defaults
T_NAME="AstroCLIP Dependency Test"
T_DESC="Quick AstroCLIP run to validate dependencies and imports"
MAX_STEPS=2
BATCH_SIZE=4
NUM_WORKERS=2
TRAIN_MODE="scratch"
RUN_TRAIN=0

while getopts "n:d:a:s:b:w:m:th" opt; do
  case $opt in
    n) T_NAME="$OPTARG" ;;
    d) T_DESC="$OPTARG" ;;
    a) DATA_DIR="$OPTARG" ;;
    s) MAX_STEPS="$OPTARG" ;;
    b) BATCH_SIZE="$OPTARG" ;;
    w) NUM_WORKERS="$OPTARG" ;;
    m) TRAIN_MODE="$OPTARG" ;;
        t) RUN_TRAIN=1 ;;
    h)
            echo "Usage: $0 [-n name] [-d description] [-a data_dir] [-s max_steps] [-b batch_size] [-w num_workers] [-m scratch|resume] [-t]"
            echo "  -t : Run tiny training after dependency preflight (requires pretrained ckpts)."
      exit 0
      ;;
    *)
            echo "Usage: $0 [-n name] [-d description] [-a data_dir] [-s max_steps] [-b batch_size] [-w num_workers] [-m scratch|resume] [-t]"
      exit 1
      ;;
  esac
done

cd "$REPO_ROOT" || { echo "[ERROR] Cannot enter REPO_ROOT: $REPO_ROOT"; exit 1; }
source "/.venv/bin/activate"

echo "Running AstroCLIP dependency test: $T_NAME"
echo "  DATA_DIR:     $DATA_DIR"
echo "  MAX_STEPS:    $MAX_STEPS"
echo "  BATCH_SIZE:   $BATCH_SIZE"
echo "  NUM_WORKERS:  $NUM_WORKERS"
echo "  RUN_TRAIN:    $RUN_TRAIN"

echo "[Preflight] Checking imports and local AstroCLIP path..."
python - <<'PY'
import importlib
import os
import sys
from pathlib import Path

repo_root = Path(os.environ["REPO_ROOT"]).resolve()
astroclip_dir = repo_root.parent / "AstroCLIP"

if not astroclip_dir.exists():
    raise SystemExit(f"[FATAL] AstroCLIP repo not found at: {astroclip_dir}")

sys.path.insert(0, str(astroclip_dir))

mods = [
    "dinov2",
    "dotenv",
    "astroclip.models.astroclip",
]

errors = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        errors.append((m, str(e)))

if errors:
    print("[FATAL] Dependency preflight failed:")
    for m, e in errors:
        print(f"  - {m}: {e}")
    raise SystemExit(2)

print("[OK] Dependency preflight passed.")
PY

ASTROCLIP_ROOT_PATH="${ASTROCLIP_ROOT:-$REPO_ROOT/../AstroCLIP}"
ASTRODINO_CKPT="$ASTROCLIP_ROOT_PATH/pretrained/astrodino.ckpt"
SPECFORMER_CKPT="$ASTROCLIP_ROOT_PATH/pretrained/specformer.ckpt"

if [[ ! -f "$ASTRODINO_CKPT" || ! -f "$SPECFORMER_CKPT" ]]; then
    echo "[WARN] Pretrained checkpoints not found under: $ASTROCLIP_ROOT_PATH/pretrained"
    echo "       Missing files required for training run:"
    [[ ! -f "$ASTRODINO_CKPT" ]] && echo "       - $ASTRODINO_CKPT"
    [[ ! -f "$SPECFORMER_CKPT" ]] && echo "       - $SPECFORMER_CKPT"
    echo "       This does NOT block dependency-only preflight."
    echo "       To run training, download from README links:"
    echo "       https://huggingface.co/polymathic-ai/astrodino"
    echo "       https://huggingface.co/polymathic-ai/specformer"

if [[ "$RUN_TRAIN" -ne 1 ]]; then
    echo "[OK] Dependency preflight finished (no training requested)."
    exit 0

if [[ ! -f "$ASTRODINO_CKPT" || ! -f "$SPECFORMER_CKPT" ]]; then
    echo "[FATAL] Tiny training requested (-t) but pretrained checkpoints are missing."
    exit 2

RUN_TAG=$(date +%Y%m%d_%H%M%S)
TRAIN_DIR="$REPO_ROOT/logs/astroclip_depcheck_${RUN_TAG}"

echo "[Run] Starting tiny AstroCLIP train in: $TRAIN_DIR"
python scripts/train_astroclip_arrow.py \
    --data_dir "$DATA_DIR" \
    --train_dir "$TRAIN_DIR" \
    --train_name "$T_NAME" \
    --train_description "$T_DESC" \
    --init_from "$TRAIN_MODE" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --max_epochs 1 \
    --max_steps "$MAX_STEPS" \
    --val_check_interval 1 \
    --limit_val_batches 1 \
    --log_interval 1 \
    --checkpoint_save_type "last" \
    --checkpoint_interval 1 \
    --early_stopping_patience 2 \
    --ddp_num_gpus 1 \
    --precision "32-true"

echo "AstroCLIP test end."
