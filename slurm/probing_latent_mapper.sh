#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Latent_Mapper
#SBATCH --partition=gpu
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=16       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=64G                
#SBATCH --time=01:00:00         

#--- LOGS FILES ---#
#SBATCH --output=logs/latent_mapper_%j.out
#SBATCH --error=logs/latent_mapper_%j.err

#--- DEFAULT VALUES ---#
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
PYTHON_SCRIPT="scripts/probing_latent_mapper.py"
META_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits"

# Modality Defaults
SOURCE="images"
TARGET="joint"
K_NEIGHBORS=10

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:w:s:e:f:x:y:k:" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    w) WEIGHTS_DIR="$OPTARG" ;;
    s) SAVE_DIR="$OPTARG" ;;
    e) EMB_DIR="$OPTARG" ;;
    f) META_PATH="$OPTARG" ;;
    x) SOURCE="$OPTARG" ;;
    y) TARGET="$OPTARG" ;;
    k) K_NEIGHBORS="$OPTARG" ;;
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
echo "Starting Latent Mapper Job $SLURM_JOB_ID - $NOW"
echo "Mapping Mode: $SOURCE -> $TARGET (k=$K_NEIGHBORS)"
echo "-----------------------------------------------"

cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }
source .venv/bin/activate

#--- EMBEDDING DETECTION LOGIC ---#
DETECTED_EMB=$(ls -td "${EMB_DIR}"/*/ 2>/dev/null | head -n 1)
DETECTED_EMB="${DETECTED_EMB%/}"
DETECTED_EMB=$(readlink -f "$DETECTED_EMB")

if [ -z "$DETECTED_EMB" ]; then
    echo "[ERROR]: No sub-directory found in $EMB_DIR"
    exit 1
fi

if [ -z "$SAVE_DIR" ]; then
    SAVE_DIR="$DETECTED_EMB"
fi
SAVE_DIR=$(readlink -f "$SAVE_DIR")

#--- EXECUTION ---#
python "$PYTHON_SCRIPT" \
    --metadata_path "$META_PATH" \
    --weights_dir "$WEIGHTS_DIR" \
    --emb_dir "$DETECTED_EMB" \
    --save_dir "$SAVE_DIR" \
    --source "$SOURCE" \
    --target "$TARGET" \
    --all_targets \
    --k_neighbors "$K_NEIGHBORS" \
    --overwrite

echo "-----------------------------------------------"
echo "Latent Mapper Job Finished"
echo "-----------------------------------------------"