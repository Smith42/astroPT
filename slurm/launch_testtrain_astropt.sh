#!/bin/bash

# Evironment Configuration
export REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
export DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"
export UV_CACHE_DIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/uv_cache"
export HF_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache/huggingface"
export OMP_NUM_THREADS=4 

# Repo directory
T_NAME="Default Test Name"
T_DESC="Default Test Description"

while getopts "n:d:" opt; do
  case $opt in
    n) T_NAME="$OPTARG" ;;
    d) T_DESC="$OPTARG" ;;
    *) echo "Usage: $0 [-n name] [-d description]" ;;
  esac
done

cd "$REPO_ROOT" || exit 1
source .venv/bin/activate

echo "Running Test: $T_NAME"

python3 scripts/train_multimodal_arrow.py \
    --data_dir "$DATA_DIR" \
    --train_name "$T_NAME" \
    --train_description "$T_DESC" \
    --init_from "scratch" \
    --loss_type "mae" \
    --eval_interval 50 \
    --log_interval 10 \
    --checkpoint_save_type "all" \
    --checkpoint_interval 50 \
    --max_run_hours "00:20:00" \
    --spectra_inverse \
    --no-compile 

echo "Test end."