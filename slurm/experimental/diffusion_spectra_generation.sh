#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=GenSpecIm
#SBATCH --partition=gpu
#SBATCH --account=iac18
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=4       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=64G                
#SBATCH --time=01:00:00         

#--- LOGS FILES ---#
#SBATCH --output=logs/generate_spec_%j.out
#SBATCH --error=logs/generate_spec_%j.err

set -euo pipefail

# Robust repository root detection based on script location
# (Script is in slurm/experimental/, so REPO_ROOT is two levels up)
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

PYTHON_SCRIPT="$REPO_ROOT/scripts/experimental/diffusion_spectra_generation.py"

# Default fallbacks
CHECKPOINT_PATH="$REPO_ROOT/logs/astropt_20260601_hybrid_optimized/diffusion/weights/ckpt_best.pt"
EMBEDDINGS_PATH="$REPO_ROOT/logs/astropt_20260601_hybrid_optimized/embeddings/best_img-mean_spec-rank_final_iso_j-mean/embeddings_all.npz"
EMBEDDINGS_KEY="EuclidImage_phase2"
GALAXY_ID=""
INDEX=""
REDSHIFT=""
REDSHIFT_PROBE_PATH="$REPO_ROOT/logs/astropt_20260601_hybrid_optimized/embeddings/best_img-mean_spec-rank_final_iso_j-mean/downstream_tasks/probing_MLP.pt"
OUTPUT_PATH="$REPO_ROOT/logs/astropt_20260601_hybrid_optimized/diffusion/samples/"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated"
METADATA_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1_FILTERED.fits"
PLOT_ORIGINAL=0
LAYOUT="dashboard"
ENSEMBLE=1

# If invoked as: sbatch script.sh -- -c <checkpoint> ...
if [[ "${1:-}" == "--" ]]; then
  shift
fi

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:c:e:k:g:i:z:p:o:d:xl:sm:" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    c) CHECKPOINT_PATH="$OPTARG" ;;
    e) EMBEDDINGS_PATH="$OPTARG" ;;
    k) EMBEDDINGS_KEY="$OPTARG" ;;
    g) GALAXY_ID="$OPTARG" ;;
    i) INDEX="$OPTARG" ;;
    z) REDSHIFT="$OPTARG" ;;
    p) REDSHIFT_PROBE_PATH="$OPTARG" ;;
    o) OUTPUT_PATH="$OPTARG" ;;
    d) DATA_DIR="$OPTARG" ;;
    m) METADATA_PATH="$OPTARG" ;;
    x) PLOT_ORIGINAL=1 ;;
    l) LAYOUT="$OPTARG" ;;
    s) ENSEMBLE=1 ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

shift $((OPTIND - 1))

# Validation of required parameters
if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
  echo "[ERROR] CHECKPOINT_PATH is required (-c <checkpoint_path>)" >&2
  exit 1
fi

if [[ -z "${EMBEDDINGS_PATH:-}" ]]; then
  echo "[ERROR] EMBEDDINGS_PATH is required (-e <embeddings_path>)" >&2
  exit 1
fi

# Resolve absolute paths
CHECKPOINT_PATH=$(readlink -f "$CHECKPOINT_PATH")
EMBEDDINGS_PATH=$(readlink -f "$EMBEDDINGS_PATH")

# Auto-resolve REDSHIFT_PROBE_PATH if not explicitly provided
if [[ -z "${REDSHIFT_PROBE_PATH:-}" ]]; then
  EMB_DIR=$(dirname "$EMBEDDINGS_PATH")
  if [[ -f "$EMB_DIR/downstream_tasks/probing_MLP.pt" ]]; then
    REDSHIFT_PROBE_PATH="$EMB_DIR/downstream_tasks/probing_MLP.pt"
  elif [[ -f "$EMB_DIR/downstream_tasks_V2/probing_MLP.pt" ]]; then
    REDSHIFT_PROBE_PATH="$EMB_DIR/downstream_tasks_V2/probing_MLP.pt"
  else
    FALLBACK_PROBE="$REPO_ROOT/logs/astropt_20260516_hybrid_cliploss/embeddings/best_img-mean_spec-rank_final_iso_j-mean/downstream_tasks/probing_MLP.pt"
    if [[ -f "$FALLBACK_PROBE" ]]; then
      REDSHIFT_PROBE_PATH="$FALLBACK_PROBE"
    fi
  fi
fi

# If REDSHIFT_PROBE_PATH is a directory, automatically try to append probing_MLP.pt or probing_LP.pt
if [[ -d "${REDSHIFT_PROBE_PATH:-}" ]]; then
  if [[ -f "$REDSHIFT_PROBE_PATH/probing_MLP.pt" ]]; then
    REDSHIFT_PROBE_PATH="$REDSHIFT_PROBE_PATH/probing_MLP.pt"
  elif [[ -f "$REDSHIFT_PROBE_PATH/probing_LP.pt" ]]; then
    REDSHIFT_PROBE_PATH="$REDSHIFT_PROBE_PATH/probing_LP.pt"
  fi
fi

if [[ -n "${REDSHIFT_PROBE_PATH:-}" ]]; then
  REDSHIFT_PROBE_PATH=$(readlink -f "$REDSHIFT_PROBE_PATH")
else
  echo "[WARNING] REDSHIFT_PROBE_PATH is empty and could not be auto-discovered!" >&2
fi

if [[ -n "${DATA_DIR:-}" ]]; then
  DATA_DIR=$(readlink -f "$DATA_DIR")
fi

if [[ -n "${METADATA_PATH:-}" ]]; then
  METADATA_PATH=$(readlink -f "$METADATA_PATH")
fi

#--- ENVIRONMENT SETUP ---#
NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "-----------------------------------------------"
echo "Generating Spectrum from Image ${SLURM_JOB_ID:-local} - $NOW"
echo "-----------------------------------------------"

# 1. Change directory
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# 2. Activate Environment
source "$REPO_ROOT/.venv/bin/activate"

# 3. Activating LaTeX and matplotlib paths
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
export MPLCONFIGDIR="$REPO_ROOT/../cache/matplotlib"
export XDG_CACHE_HOME="$REPO_ROOT/../cache"

#--- EXECUTION ---#
echo "Generating Spectrum:"
echo "    CHECKPOINT:   $CHECKPOINT_PATH"
echo "    EMBEDDINGS:   $EMBEDDINGS_PATH" 

# Building dynamically optional flags
OPTIONAL_FLAGS=""
if [[ -n "${GALAXY_ID:-}" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --galaxy_id $GALAXY_ID"
fi
if [[ -n "${INDEX:-}" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --index $INDEX"
fi
if [[ -n "${REDSHIFT:-}" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --redshift $REDSHIFT"
fi

if [[ -n "${EMBEDDINGS_KEY:-}" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --embeddings_key $EMBEDDINGS_KEY"
fi

if [[ -n "${METADATA_PATH:-}" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --metadata_path $METADATA_PATH"
fi

if [[ -n "${OUTPUT_PATH:-}" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --output_path $OUTPUT_PATH"
fi
if [[ -n "${DATA_DIR:-}" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --data_dir $DATA_DIR"
fi
if [[ "$PLOT_ORIGINAL" -eq 1 ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --plot_original"
fi
if [[ -n "${LAYOUT:-}" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --layout $LAYOUT"
fi
if [[ "$ENSEMBLE" -eq 1 ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --ensemble"
fi

python3 "$PYTHON_SCRIPT" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --embeddings_path "$EMBEDDINGS_PATH" \
    --probing_weights_path "$REDSHIFT_PROBE_PATH" \
    $OPTIONAL_FLAGS

echo "-----------------------------------------------"
echo "Spectrum Generation Finished"
echo "-----------------------------------------------"
