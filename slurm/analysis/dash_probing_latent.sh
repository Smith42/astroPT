#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Mapper_Dash
#SBATCH --partition=gpu
#SBATCH --account=iac18
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=2       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=32G                
#SBATCH --time=00:10:00         

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_mapper_dash_%j.out
#SBATCH --error=logs/astropt_mapper_dash_%j.err

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

PYTHON_SCRIPT="$REPO_ROOT/scripts/analysis/dashboards/dash_probing_latent.py"
ALL_TARGETS_MODE=0

# If invoked as: sbatch script.sh -- -e <emb_root> ...
if [[ "${1:-}" == "--" ]]; then
  shift
fi

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:e:s:n:Ah" opt; do
  case $opt in
    s) SAVE_DIR="$OPTARG" ;;
    e) EMB_DIR="$OPTARG" ;;
    n) SAVE_NAME="$OPTARG" ;;
    A) ALL_TARGETS_MODE=1 ;;
    h)
      echo "Usage: $0 -e EMB_DIR [-s SAVE_DIR] [-n SAVE_NAME] [-A] [-r REPO_ROOT]"
      echo "  -A: Plot all latent targets and compare all runs found in EMB_DIR/*/latent_mapper/mapper_*.csv"
      exit 0
      ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

shift $((OPTIND - 1))

if [[ -z "${EMB_DIR:-}" ]]; then
    echo "[ERROR]: EMB_DIR is required (-e <embeddings_root>)"
    exit 1
fi

EMB_DIR=$(readlink -f "$EMB_DIR")

#--- ENVIRONMENT SETUP ---#
NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "--------------------------------------------------"
echo "Starting Latent Mapper Dashboard Job ${SLURM_JOB_ID:-local} - $NOW"
echo "--------------------------------------------------"

# 1. Change directory
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# 2. Activate Environment
source "$REPO_ROOT/.venv/bin/activate"

# 3. Activating LaTeX
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
export MPLCONFIGDIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/matplotlib"
export XDG_CACHE_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache"

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
if [[ "$ALL_TARGETS_MODE" -eq 0 ]] && [ -f "$CONFIG_FILE" ]; then
    IMG_TRAIN=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('images_train', c.get('img_train', True)))")
    SPEC_TRAIN=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('spectra_train', c.get('spec_train', True)))")

    if [[ "$IMG_TRAIN" == "False" ]] || [[ "$SPEC_TRAIN" == "False" ]]; then
        echo "[INFO] Unimodal architecture detected. Latent Mapper Dashboard requires multimodal runs. Exiting cleanly."
        exit 0
    fi
fi

#--- EXECUTION ---#
echo "Mapper Dashboard Configuration:"
echo "    EMB ROOT:       $EMB_DIR"
echo "    ALL TARGETS:    $ALL_TARGETS_MODE"

if [[ "$ALL_TARGETS_MODE" -eq 1 ]]; then
  RUN_CSVS=()
  RUN_NAMES=()

  while IFS= read -r csv_path; do
    RUN_CSVS+=("$csv_path")
    run_dir=$(basename "$(dirname "$(dirname "$csv_path")")")
    RUN_NAMES+=("$run_dir")
  done < <(
    {
      find "$EMB_DIR" -mindepth 3 -maxdepth 3 -type f -path "*/latent_mapper/mapper_*.csv"
      find "$EMB_DIR" -mindepth 4 -maxdepth 4 -type f -path "*/downstream_tasks/latent_mapper/mapper_*.csv"
    } | sort -u
  )

  if [[ ${#RUN_CSVS[@]} -eq 0 ]]; then
    echo "[ERROR]: No mapper_*.csv found under $EMB_DIR/*/latent_mapper/"
    exit 1
  fi

  echo "    N RUNS FOUND:    ${#RUN_CSVS[@]}"

  python "$PYTHON_SCRIPT" \
    $SAVE_ARG \
    --input_dirs "${RUN_CSVS[@]}" \
    --names "${RUN_NAMES[@]}" \
    --all_targets \
    "${SAVE_NAME_ARGS[@]}"
  # Define the base log directory
  LOGS_BASE="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs"

  # Define the past baseline runs you want to compare
  RUNS_TO_COMPARE=(
    "${LOGS_BASE}/astropt_100M_250K_arrow_20260408_baseline_images/"
    "${LOGS_BASE}/astropt_100M_250K_arrow_20260409_baseline_spectra"
    "${LOGS_BASE}/astropt_100M_250K_arrow_20260407_tokmix5_mask0p25"
    "${LOGS_BASE}/astropt_100M_250K_arrow_20260411_tokmixran_crossrecloss_longblocks"
    "${LOGS_BASE}/astropt_100M_250K_arrow_20260412_tokmixran_crossrecloss_longblocks_dsinter"
    "${LOGS_BASE}/astropt_100M_250K_arrow_20260413_tokmix16_crossrecloss_longblocks_dsinter"
    "${LOGS_BASE}/astropt_100M_250K_arrow_20260414_tokmix16_crossrecloss_dsinter_smallmod"
  )

  # Auto-add the CURRENT run being evaluated by the automated SLURM pipeline
  # We extract the run root by removing the /embeddings... suffix from EMB_DIR
  CURRENT_RUN_DIR=$(echo "$EMB_DIR" | sed 's|/embeddings.*||')

  # Ensure it is not duplicated if it was already in the baseline list
  if [[ ! " ${RUNS_TO_COMPARE[*]} " =~ " ${CURRENT_RUN_DIR} " ]]; then
      RUNS_TO_COMPARE+=("$CURRENT_RUN_DIR")
  fi

  RUN_NAMES=()
  for run_dir in "${RUNS_TO_COMPARE[@]}"; do
    RUN_NAMES+=("$(basename "$run_dir")")
  done

  python "$PYTHON_SCRIPT" \
    $SAVE_ARG \
    --input_dirs "${RUNS_TO_COMPARE[@]}" \
    --names "${RUN_NAMES[@]}" \
    --targets \
        Z LOGMSTAR LOGSFR GR \
        flux_detection_total HALPHA_EW HALPHA_FLUX NII_6584_FLUX OIII_5007_FLUX HBETA_FLUX NII_6584_FLUX \
        sersic_sersic_vis_radius sersic_sersic_vis_index sersic_sersic_vis_axis_ratio has_spiral_arms_yes smoothness gini \
        SPECTYPE data_set_release \
      "${SAVE_NAME_ARGS[@]}"
fi

echo "-----------------------------------------------"
echo "Latent Mapper Dashboard Finished"
echo "-----------------------------------------------"