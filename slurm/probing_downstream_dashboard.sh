#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Probing_Tasks
#SBATCH --partition=gpu
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=2       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=32G                
#SBATCH --time=00:10:00         

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_probing_dash_%j.out
#SBATCH --error=logs/astropt_probing_dash_%j.err

set -euo pipefail

#--- DEFAULT VALUES ---#
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
PYTHON_SCRIPT="scripts/probing_downstream_dashboard.py"
ALL_TARGETS_MODE=0

# If invoked as: sbatch script.sh -- -e <emb_root> ...
if [[ "${1:-}" == "--" ]]; then
  shift
fi

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:e:s:n:Ah" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    s) SAVE_DIR="$OPTARG" ;;
    e) EMB_DIR="$OPTARG" ;;
    n) SAVE_NAME="$OPTARG" ;;
    A) ALL_TARGETS_MODE=1 ;;
    h)
      echo "Usage: $0 -e EMB_DIR [-s SAVE_DIR] [-n SAVE_NAME] [-A] [-r REPO_ROOT]"
      echo "  -A: Plot all downstream targets and compare all runs found in EMB_DIR/*/downstream_tasks/downstream_results.csv"
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
echo "Starting Probing Tasks Dashboard Job ${SLURM_JOB_ID:-local} - $NOW"
echo "--------------------------------------------------"

# 1. Change directory
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# 2. Activate Environment
source .venv/bin/activate

# 3. Activating LaTeX (Required for confusion matrix plots)
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

#--- EXECUTION ---#
echo "Probing Dashboard Configuration:"
echo "    EMB ROOT:       $EMB_DIR"
echo "    ALL TARGETS:    $ALL_TARGETS_MODE"

if [[ "$ALL_TARGETS_MODE" -eq 1 ]]; then
  RUN_CSVS=()
  RUN_NAMES=()

  while IFS= read -r csv_path; do
    RUN_CSVS+=("$csv_path")
    run_dir=$(basename "$(dirname "$(dirname "$csv_path")")")
    RUN_NAMES+=("$run_dir")
  done < <(find "$EMB_DIR" -mindepth 3 -maxdepth 3 -type f -path "*/downstream_tasks/downstream_results.csv" | sort)

  if [[ ${#RUN_CSVS[@]} -eq 0 ]]; then
    echo "[ERROR]: No downstream_results.csv found under $EMB_DIR/*/downstream_tasks/"
    exit 1
  fi

  echo "    N RUNS FOUND:    ${#RUN_CSVS[@]}"

  python "$PYTHON_SCRIPT" \
    $SAVE_ARG \
    --csv_path "${RUN_CSVS[@]}" \
    --names "${RUN_NAMES[@]}" \
    --all_targets \
    "${SAVE_NAME_ARGS[@]}"
else
  # Define the base log directory
  LOGS_BASE="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs"

  # Define the past baseline runs you want to compare
  # The python script will automatically find their downstream_results.csv and their names!
  RUNS_TO_COMPARE=(
    "${LOGS_BASE}/astropt_100M_250K_arrow_20260408_baseline_images/"
    "${LOGS_BASE}/astropt_100M_250K_arrow_20260409_baseline_spectra"
    "${LOGS_BASE}/astropt_100M_250K_arrow_20260407_tokmix5_mask0p25"
    "${LOGS_BASE}/astropt_100M_250K_arrow_20260411_tokmixran_crossrecloss_longblocks"
    "${LOGS_BASE}/astropt_100M_250K_arrow_20260412_tokmixran_crossrecloss_longblocks_dsinter"
    "${LOGS_BASE}/astropt_100M_250K_arrow_20260413_tokmix16_crossrecloss_longblocks_dsinter"
    "${LOGS_BASE}/astropt_100M_250K_arrow_20260414_tokmix16_crossrecloss_dsinter_smallmod"
    "${LOGS_BASE}/astropt_100M_250K_arrow_20260416_tokmix16_crossrecloss_dsinter_smallmod_shufftrue"
  )

  # Auto-add the CURRENT run being evaluated by the automated SLURM pipeline
  # We extract the run root by removing the /embeddings... suffix from EMB_DIR
  CURRENT_RUN_DIR=$(echo "$EMB_DIR" | sed 's|/embeddings.*||')
  
  # Ensure it is not duplicated if it was already in the baseline list
  if [[ ! " ${RUNS_TO_COMPARE[*]} " =~ " ${CURRENT_RUN_DIR} " ]]; then
      RUNS_TO_COMPARE+=("$CURRENT_RUN_DIR")
  fi

  # Comparative mode using auto-discovery
  python "$PYTHON_SCRIPT" \
    $SAVE_ARG \
    --run_dirs "${RUNS_TO_COMPARE[@]}" \
    --targets \
        Z LOGMSTAR LOGSFR GR \
        flux_detection_total HALPHA_EW HALPHA_FLUX NII_6584_FLUX OIII_5007_FLUX HBETA_FLUX NII_6584_FLUX \
        sersic_sersic_vis_radius sersic_sersic_vis_index sersic_sersic_vis_axis_ratio has_spiral_arms_yes smoothness gini \
        SPECTYPE data_set_release \
      "${SAVE_NAME_ARGS[@]}"
fi



echo "-----------------------------------------------"
echo "Probing Tasks Dashboard Finished"
echo "-----------------------------------------------"