#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Mapper_Dash
#SBATCH --partition=gpu
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=2       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=32G                
#SBATCH --time=00:10:00         

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_mapper_dash_%j.out
#SBATCH --error=logs/astropt_mapper_dash_%j.err

#--- DEFAULT VALUES ---#
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
PYTHON_SCRIPT="scripts/probing_latent_dashboard.py"
LOGS_BASE="${REPO_ROOT}/logs"

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:e:s:n:" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    s) SAVE_DIR="$OPTARG" ;;
    e) EMB_DIR="$OPTARG" ;;
    n) SAVE_NAME="$OPTARG" ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

#--- ENVIRONMENT SETUP ---#
NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "--------------------------------------------------"
echo "Starting Latent Mapper Dashboard Job $SLURM_JOB_ID - $NOW"
echo "--------------------------------------------------"

cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }
source .venv/bin/activate
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

# SAVE_DIR is now optional; python script will auto-save near the chosen CSV
SAVE_ARG=""
if [ -n "${SAVE_DIR:-}" ]; then
    SAVE_DIR=$(readlink -f "$SAVE_DIR")
    SAVE_ARG="--save_dir $SAVE_DIR"
fi

SAVE_NAME_ARGS=()
if [[ -n "${SAVE_NAME:-}" ]]; then
  SAVE_NAME_ARGS=(--save_name "$SAVE_NAME")
fi

#--- UNIMODAL GUARD ---#
CURRENT_RUN_DIR=$(echo "$EMB_DIR" | sed 's|/embeddings.*||')
CONFIG_FILE="$CURRENT_RUN_DIR/weights/config.json"
if [ -f "$CONFIG_FILE" ]; then
    IMG_TRAIN=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('images_train', c.get('img_train', True)))")
    SPEC_TRAIN=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('spectra_train', c.get('spec_train', True)))")

    if [[ "$IMG_TRAIN" == "False" ]] || [[ "$SPEC_TRAIN" == "False" ]]; then
        echo "[INFO] Unimodal architecture detected. Latent Mapper Dashboard requires multimodal runs. Exiting cleanly."
        exit 0
    fi
fi

#--- RUN DISCOVERY ---#

# Baseline runs for comparison
INPUT_DIRS=(
    "${LOGS_BASE}/astropt_100M_250K_arrow_20260406_tokmix5_mask0p5_forceimg"
)

# Auto-add current run if not already there
if [[ ! " ${INPUT_DIRS[*]} " =~ " ${CURRENT_RUN_DIR} " ]]; then
    INPUT_DIRS+=("$CURRENT_RUN_DIR")
fi

NAMES=()
for d in "${INPUT_DIRS[@]}"; do
    NAMES+=("$(basename "$d")")
done

#--- EXECUTION ---#
echo "Mapper Dashboard Configuration:"
echo "    EMB ROOT:       $EMB_DIR"

# RUN PYTHON SCRIPT

python "$PYTHON_SCRIPT" \
  $SAVE_ARG \
  --input_dirs "${INPUT_DIRS[@]}" \
  --names "${NAMES[@]}" \
  --targets \
      Z LOGMSTAR LOGSFR GR \
      flux_detection_total HALPHA_EW HALPHA_FLUX NII_6584_FLUX OIII_5007_FLUX HBETA_FLUX \
      sersic_sersic_vis_radius sersic_sersic_vis_index sersic_sersic_vis_axis_ratio has_spiral_arms_yes smoothness gini \
      SPECTYPE data_set_release \
  "${SAVE_NAME_ARGS[@]}"

echo "-----------------------------------------------"
echo "Latent Mapper Dashboard Finished"
echo "-----------------------------------------------"