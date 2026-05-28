#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=SpectraBase
#SBATCH --partition=gpu
#SBATCH --account=iac18
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=4      
#SBATCH --gpus-per-task=1        
#SBATCH --mem=64G                
#SBATCH --time=10:00:00         
#SBATCH --output=logs/spectra_base_%A_%a.out
#SBATCH --error=logs/spectra_base_%A_%a.err  
#SBATCH --array=0-3

set -euo pipefail

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

PYTHON_SCRIPT="$REPO_ROOT/scripts/baselines/train_supervised_baseline_spectra.py"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated"
METADATA_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits"
SAVE_DIR="$REPO_ROOT/logs/spectra_supervised_baseline"

# Targets List
TARGETS=(
    #"Z" 
    #"LOGMSTAR"
    #"LOGSFR"
    #"GR" 
    'flux_detection_total'
    'HALPHA_EW'
    'HALPHA_FLUX'
    'NII_6584_FLUX'
    #'OIII_5007_FLUX'
    #'HBETA_FLUX'
    #'sersic_sersic_vis_radius'
    #'sersic_sersic_vis_index'
    #'sersic_sersic_vis_axis_ratio'
    #'has_spiral_arms_yes'
    #'smoothness'
    #'smooth_or_featured_smooth'
    #'gini'
    #'SPECTYPE'
    #'data_set_release'
)

# Get current target from array task id
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
echo "Spectra Supervised Baseline ${SLURM_JOB_ID:-LOCAL} - $NOW"
echo "Target: $CURRENT_TARGET"
echo "-----------------------------------------------"

cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# Activate virtual environment
if [ -d ".venv" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "Running Python Script..."
python3 "$PYTHON_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --metadata_path "$METADATA_PATH" \
    --target "$CURRENT_TARGET" \
    --train_dir "$SAVE_DIR" \
    --train_name "spectra_baseline_${CURRENT_TARGET}"

echo "-----------------------------------------------"
echo "Spectra Baseline Finished"
echo "-----------------------------------------------"
