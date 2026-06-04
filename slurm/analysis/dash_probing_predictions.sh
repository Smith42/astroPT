#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Preds_Dash
#SBATCH --partition=gpu
#SBATCH --account=iac18
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=4       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=32G                
#SBATCH --time=04:00:00         

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_preds_dash_%j.out
#SBATCH --error=logs/astropt_preds_dash_%j.err

set -euo pipefail

# Robust repository root detection based on script location
# (Script is in slurm/analysis/, so REPO_ROOT is two levels up)
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

PYTHON_SCRIPT="$REPO_ROOT/scripts/analysis/dashboards/dash_probing_predictions.py"
INTERACTIVE_SCRIPT="$REPO_ROOT/scripts/analysis/dashboards/dash_probing_predictions_interactive.py"

PORT=8502
EMB_DIR=""
SAVE_DIR=""
WEIGHTS_DIR=""
LOGS_DIR="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs"
FITS_PATH=""
INTERACTIVE_MODE=0

# If invoked as: sbatch script.sh -- -e <emb_root> ...
if [[ "${1:-}" == "--" ]]; then
  shift
fi

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:e:s:p:f:l:w:ih" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    e) EMB_DIR="$OPTARG" ;;
    s) SAVE_DIR="$OPTARG" ;;
    p) PORT="$OPTARG" ;;
    f) FITS_PATH="$OPTARG" ;;
    l) LOGS_DIR="$OPTARG" ;;
    w) WEIGHTS_DIR="$OPTARG" ;;
    i) INTERACTIVE_MODE=1 ;;
    h)
      echo "Usage: $0 [-e EMB_DIR] [-w WEIGHTS_DIR] [-s SAVE_DIR] [-r REPO_ROOT] [-i] [-p PORT] [-f FITS_PATH] [-l LOGS_DIR]"
      echo "  -e: Specify embeddings directory (required for static mode, optional for interactive mode)"
      echo "  -w: Specify weights directory containing config.json"
      echo "  -s: Specify static plot save directory (only used in static mode)"
      echo "  -r: Specify repository root directory (default: auto-detected)"
      echo "  -i: Launch interactive Streamlit Dashboard instead of static Matplotlib PNG generation"
      echo "  -p: Specify Streamlit Server port (default: 8501, only used in interactive mode)"
      echo "  -f: Specify custom metadata FITS catalog path (only used in interactive mode)"
      echo "  -l: Specify custom logs root directory to scan (only used in interactive mode)"
      exit 0
      ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

shift $((OPTIND - 1))

# Validation: EMB_DIR is required ONLY in static mode
if [[ "$INTERACTIVE_MODE" -eq 0 ]]; then
    if [[ -z "${EMB_DIR:-}" ]]; then
        echo "[ERROR]: EMB_DIR is required in static mode (-e <embeddings_dir>)"
        exit 1
    fi
fi

#--- ENVIRONMENT SETUP ---#
NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "--------------------------------------------------"
echo "Starting Predictions Dashboard Job ${SLURM_JOB_ID:-local} - $NOW"
echo "Interactive Mode: $INTERACTIVE_MODE"
echo "--------------------------------------------------"

echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# Activate Environment
source "$REPO_ROOT/.venv/bin/activate"

# Configure cache directories for stable rendering on the cluster
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
export MPLCONFIGDIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/matplotlib"
export XDG_CACHE_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache"

#--- EXECUTION ---#
if [[ "$INTERACTIVE_MODE" -eq 1 ]]; then
    echo "Launching Interactive Dashboard..."
    echo "    Script: $INTERACTIVE_SCRIPT"
    echo "    Port:   $PORT"
    echo "    Host:   0.0.0.0"
    
    PYTHON_ARGS=()
    if [[ -n "${EMB_DIR:-}" ]]; then
        RESOLVED_EMB=$(readlink -f "$EMB_DIR")
        # Try to resolve downstream_tasks/predictions subfolder if present
        if [[ -d "$RESOLVED_EMB/downstream_tasks/predictions" ]]; then
            PYTHON_ARGS+=("--pred_dir" "$RESOLVED_EMB/downstream_tasks/predictions")
        else
            PYTHON_ARGS+=("--pred_dir" "$RESOLVED_EMB")
        fi
    fi
    if [[ -n "${LOGS_DIR:-}" ]]; then
        PYTHON_ARGS+=("--logs_dir" "$(readlink -f "$LOGS_DIR")")
    fi
    if [[ -n "${FITS_PATH:-}" ]]; then
        PYTHON_ARGS+=("--metadata_path" "$(readlink -f "$FITS_PATH")")
    fi
    if [[ -n "${WEIGHTS_DIR:-}" ]]; then
        PYTHON_ARGS+=("--weights_dir" "$(readlink -f "$WEIGHTS_DIR")")
    fi

    streamlit run "$INTERACTIVE_SCRIPT" --server.port "$PORT" --server.address 0.0.0.0 -- "${PYTHON_ARGS[@]}"
else
    EMB_DIR=$(readlink -f "$EMB_DIR")
    PRED_DIR="$EMB_DIR/downstream_tasks/predictions"

    if [[ ! -d "$PRED_DIR" ]]; then
        echo "[ERROR]: Predictions directory not found at $PRED_DIR"
        exit 1
    fi

    SAVE_ARG=""
    if [ -n "${SAVE_DIR:-}" ]; then
        SAVE_DIR=$(readlink -f "$SAVE_DIR")
        SAVE_ARG="--save_dir $SAVE_DIR"
    fi

    echo "Running static predictions dashboard for:"
    echo "    PRED_DIR:     $PRED_DIR"

    python3 "$PYTHON_SCRIPT" \
        --pred_dir "$PRED_DIR" \
        $SAVE_ARG
fi

echo "-----------------------------------------------"
echo "Predictions Dashboard Finished"
echo "-----------------------------------------------"
