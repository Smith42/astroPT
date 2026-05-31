#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Saliency_Maps
#SBATCH --partition=gpu
#SBATCH --account=iac18
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=2       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=32G                
#SBATCH --time=00:30:00         

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_saliency_%j.out
#SBATCH --error=logs/astropt_saliency_%j.err

set -euo pipefail

# Robust repository root detection based on script location
# (Script is in slurm/probing/, so REPO_ROOT is two levels up)
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

PYTHON_SCRIPT="$REPO_ROOT/scripts/probing/task_saliency_maps.py"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated_tokenized"
NUM_PLOT=5
SAVE_DIR=""
TARGET_IDS=""

# If invoked with leading -- (standard in some SLURM invocations)
if [[ "${1:-}" == "--" ]]; then
  shift
fi

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:w:p:d:t:m:s:n:i:h" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    w) WEIGHTS_DIR="$OPTARG" ;;
    p) PROBE_FILE="$OPTARG" ;;
    d) DATA_DIR="$OPTARG" ;;
    t) TARGET="$OPTARG" ;;
    m) MODALITY="$OPTARG" ;;
    s) SAVE_DIR="$OPTARG" ;;
    n) NUM_PLOT="$OPTARG" ;;
    i) TARGET_IDS="$OPTARG" ;;
    h)
      echo "Usage: $0 -w WEIGHTS_DIR -p PROBE_FILE -t TARGET -m MODALITY [options]"
      echo "  -w: Directory with AstroPT checkpoint (Required)"
      echo "  -p: Path to probing_MLP.pt or probing_LP.pt (Required)"
      echo "  -t: Target task column, e.g., sersic_index_VIS (Required)"
      echo "  -m: Expert Modality used in probing, e.g., EuclidImage_phase1 (Required)"
      echo "  -d: Arrow data root directory (Default: /home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated_tokenized)"
      echo "  -s: Directory to save plots (Default: weights_dir/saliency_maps)"
      echo "  -n: Number of random galaxies to plot if target IDs are not specified (Default: 5)"
      echo "  -i: Space-separated specific target IDs to interpret, e.g., \"39627061836389042 39627853679036156\""
      echo "  -r: Set custom REPO_ROOT"
      exit 0
      ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

shift $((OPTIND - 1))

if [[ -z "${WEIGHTS_DIR:-}" || -z "${PROBE_FILE:-}" || -z "${TARGET:-}" || -z "${MODALITY:-}" ]]; then
  echo "[ERROR] WEIGHTS_DIR (-w), PROBE_FILE (-p), TARGET (-t), and MODALITY (-m) are all required." >&2
  echo "Run $0 -h for usage instructions." >&2
  exit 1
fi

WEIGHTS_DIR=$(readlink -f "$WEIGHTS_DIR")
PROBE_FILE=$(readlink -f "$PROBE_FILE")
DATA_DIR=$(readlink -f "$DATA_DIR")
if [[ -n "${SAVE_DIR:-}" ]]; then
  SAVE_DIR=$(readlink -f "$SAVE_DIR")
fi

#--- ENVIRONMENT SETUP ---#
NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "--------------------------------------------------"
echo "Starting Task-Specific Saliency Maps Job ${SLURM_JOB_ID:-local} - $NOW"
echo "--------------------------------------------------"

# 1. Change directory
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# 2. Activate Environment
source "$REPO_ROOT/.venv/bin/activate"

# 3. Activating LaTeX (Required for some figure renderings)
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
export MPLCONFIGDIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/matplotlib"
export XDG_CACHE_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache"

# Build optional arguments for python execution
EXTRA_ARGS=()
if [[ -n "${SAVE_DIR:-}" ]]; then
  EXTRA_ARGS+=("--save_dir" "$SAVE_DIR")
fi

if [[ -n "${TARGET_IDS:-}" ]]; then
  # Split space-separated target ids into individual elements
  EXTRA_ARGS+=("--target_ids" ${TARGET_IDS})
else
  EXTRA_ARGS+=("--num_plot" "$NUM_PLOT")
fi

#--- EXECUTION ---#
echo "Saliency Maps Configuration:"
echo "    WEIGHTS DIR:  $WEIGHTS_DIR"
echo "    PROBE FILE:   $PROBE_FILE"
echo "    DATA DIR:     $DATA_DIR"
echo "    TARGET:       $TARGET"
echo "    MODALITY:     $MODALITY"
if [[ -n "${SAVE_DIR:-}" ]]; then
  echo "    SAVE DIR:     $SAVE_DIR"
fi
if [[ -n "${TARGET_IDS:-}" ]]; then
  echo "    TARGET IDS:   $TARGET_IDS"
else
  echo "    NUM PLOT:     $NUM_PLOT"
fi
echo "--------------------------------------------------"

# Run Python Script
python3 "$PYTHON_SCRIPT" \
    --weights_dir "$WEIGHTS_DIR" \
    --probe_file "$PROBE_FILE" \
    --data_dir "$DATA_DIR" \
    --target "$TARGET" \
    --modality "$MODALITY" \
    "${EXTRA_ARGS[@]}"

echo "--------------------------------------------------"
echo "Saliency Maps Generation Finished"
echo "--------------------------------------------------"
