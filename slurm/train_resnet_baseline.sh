#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=ResNetBase
#SBATCH --partition=gpu
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=4      
#SBATCH --gpus-per-task=1        
#SBATCH --mem=64G                
#SBATCH --time=10:00:00         
#SBATCH --output=logs/resnet_base_%A_%a.out
#SBATCH --error=logs/resnet_base_%A_%a.err
#SBATCH --array=0-3  # Adjust based on number of targets to run concurrently

set -euo pipefail

#--- DEFAULT VALUES ---#
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
PYTHON_SCRIPT="scripts/train_resnet_supervised.py"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated"
METADATA_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits"
SAVE_DIR="$REPO_ROOT/logs/resnet18_images_supervised"

# Targets List
TARGETS=(
    "Z" 
    "LOGMSTAR"
    "LOGSFR"
    "GR" 
    #'flux_detection_total'
    #'HALPHA_EW'
    #'HALPHA_FLUX'
    #'NII_6584_FLUX'
    #'OIII_5007_FLUX'
    #'HBETA_FLUX'
    #'sersic_sersic_vis_radius'
    #'sersic_sersic_vis_index'
    #'sersic_sersic_vis_axis_ratio'
    #'has_spiral_arms_yes'
    #'smoothness'
    #'gini'
    #'SPECTYPE'
    #'data_set_release'
)

# Get current target from array task id (defauls to 0 if run locally)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
CURRENT_TARGET="${TARGETS[$TASK_ID]}"

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":m:a:s:" opt; do
  case $opt in
    m) METADATA_PATH="$OPTARG" ;;
    a) DATA_DIR="$OPTARG" ;;
    s) SAVE_DIR="$OPTARG" ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

shift $((OPTIND - 1))

# Absolute paths
METADATA_PATH=$(readlink -f "$METADATA_PATH" || echo "$METADATA_PATH")
DATA_DIR=$(readlink -f "$DATA_DIR" || echo "$DATA_DIR")
SAVE_DIR=$(readlink -f "$SAVE_DIR" || echo "$SAVE_DIR")

NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "-----------------------------------------------"
echo "ResNet18 Supervised Baseline ${SLURM_JOB_ID:-LOCAL} - $NOW"
echo "Target: $CURRENT_TARGET"
echo "-----------------------------------------------"

cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }
source .venv/bin/activate

echo "Running Python Script..."
python "$PYTHON_SCRIPT" \
    --metadata_path "$METADATA_PATH" \
    --data_dir "$DATA_DIR" \
    --target "$CURRENT_TARGET" \
    --train_dir "$SAVE_DIR" \
    --max_run_hours "09:55:00"

echo "-----------------------------------------------"
echo "ResNet18 Baseline Finished"
echo "-----------------------------------------------"
