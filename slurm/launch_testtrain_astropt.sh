#!/bin/bash

# Evironment Configuration
export REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
export DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"
export UV_CACHE_DIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/uv_cache"
export HF_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache/huggingface"
export OMP_NUM_THREADS=4  # Limitamos hilos para no molestar si hay gente en el nodo

# Repo directory
cd "$REPO_ROOT" || { echo "[ERROR] Repo Directory not found $REPO_ROOT"; exit 1; }


source .venv/bin/activate
echo "Testing AstroPT Training"

# Running the test
python3 scripts/train_multimodal_arrow.py \
    --train_name "Flash Test Manual +" \
    --train_description "Prueba rapida desde terminal sin Slurm" \
    --data_dir "$DATA_DIR" \
    --init_from "scratch" \
    --loss_type "l1" \
    --checkpoint_save_type "min" \
    --max_run_hours "00:10:00" \
    --no-compile 

echo "Test end."