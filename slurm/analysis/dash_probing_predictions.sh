#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Preds_Dash
#SBATCH --partition=gpu
#SBATCH --account=iac18
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=2       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=32G                
#SBATCH --time=00:30:00         

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_preds_dash_%j.out
#SBATCH --error=logs/astropt_preds_dash_%j.err

set -euo pipefail

# Robust repository root detection based on script location
# (Script is in slurm/trainers/, so REPO_ROOT is two levels up)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# If SLURM has moved us to /var/spool, correct REPO_ROOT to the launch working directory
if [[ "$REPO_ROOT" == "/var/spool"* ]]; then
    REPO_ROOT="$PWD"
fi

PYTHON_SCRIPT="$REPO_ROOT/scripts/analysis/dashboards/dash_probing_predictions.py"

# If invoked as: sbatch script.sh -- -e <emb_root> ...
if [[ "${1:-}" == "--" ]]; then
  shift
fi

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:w:e:s:h" opt; do
  case $opt in
    w) WEIGHTS_DIR="$OPTARG" ;;
    s) SAVE_DIR="$OPTARG" ;;
    e) EMB_DIR="$OPTARG" ;;
    h)
      echo "Usage: $0 -e EMB_DIR [-w WEIGHTS_DIR] [-s SAVE_DIR] [-r REPO_ROOT]"
      exit 0
      ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

shift $((OPTIND - 1))

if [[ -z "${EMB_DIR:-}" ]]; then
    echo "[ERROR]: EMB_DIR is required (-e <embeddings_dir>)"
    exit 1
fi

EMB_DIR=$(readlink -f "$EMB_DIR")
PRED_DIR="$EMB_DIR/downstream_tasks/predictions"

if [[ ! -d "$PRED_DIR" ]]; then
    echo "[ERROR]: Predictions directory not found at $PRED_DIR"
    exit 1
fi

#--- ENVIRONMENT SETUP ---#
NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "--------------------------------------------------"
echo "Starting Predictions Dashboard Job ${SLURM_JOB_ID:-local} - $NOW"
echo "--------------------------------------------------"

# 1. Change directory
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# 2. Activate Environment
source "$REPO_ROOT/.venv/bin/activate"

# 3. Activating LaTeX (Required for plots)
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
export MPLCONFIGDIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/matplotlib"
export XDG_CACHE_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache"

SAVE_ARG=""
if [ -n "${SAVE_DIR:-}" ]; then
    SAVE_DIR=$(readlink -f "$SAVE_DIR")
    SAVE_ARG="--save_dir $SAVE_DIR"
fi

#--- EXECUTION ---#
echo "Running predictions dashboard for:"
echo "    PRED_DIR:     $PRED_DIR"

python "$PYTHON_SCRIPT" \
    --pred_dir "$PRED_DIR" \
    $SAVE_ARG

echo "-----------------------------------------------"
echo "Predictions Dashboard Finished"
echo "-----------------------------------------------"
