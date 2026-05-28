#!/bin/bash

# --- SLURM options ---
#SBATCH --job-name=ArrowInterpPipe
#SBATCH --partition=gpu
#SBATCH --account=iac18
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=logs/astropt_arrow_interp_pipe_%j.out
#SBATCH --error=logs/astropt_arrow_interp_pipe_%j.err

set -euo pipefail

# --- Defaults ---
# Automatically detect the repository root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
INTERP_SCRIPT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/scripts/arrow_dataset_interpolation.py"
VIEWER_SCRIPT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/scripts/arrow_dataset_interpolation_viewer.py"

DATA_DIR_ORIGINAL="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_filter_corrupt"
DATA_DIR_INTERP="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated"
METADATA_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits"
VIEWER_SAVE_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated_stamps"

SPLIT="all"
NUM_WORKERS=8
RE_MULTIPLIER=5.0
COL_RADIUS="sersic_sersic_vis_radius"
COL_Q_RATIO="sersic_sersic_vis_axis_ratio"

# Optional wavelength zoom for viewer
WL_MIN=""
WL_MAX=""

# If invoked as: sbatch script.sh -- -m ...
if [[ "${1:-}" == "--" ]]; then
  shift

usage() {
  cat <<EOF
Usage:
  sbatch slurm/arrow_dataset_interpolation_pipeline.sh -- [options]
  bash slurm/arrow_dataset_interpolation_pipeline.sh [options]

Required:
  -m <metadata_path>            FITS metadata catalog path

Optional:
  -r <repo_root>                Repo root (default: ${REPO_ROOT})
  -i <data_dir_original>        Original Arrow root
  -o <data_dir_interpolated>    Output Arrow root for interpolated dataset
  -v <viewer_save_dir>          Output folder for viewer plots
  -p <split>                    Split for viewer: train|test|all (default: ${SPLIT})
  -n <num_workers>              Workers for interpolation map (default: ${NUM_WORKERS})
  -k <re_multiplier>            Multiplier for Re crop window (default: ${RE_MULTIPLIER})
  -c <col_radius>               Radius column (default: ${COL_RADIUS})
  -q <col_q_ratio>              Axis-ratio column (default: ${COL_Q_RATIO})
  -x <wl_min>                   Optional viewer wavelength min
  -y <wl_max>                   Optional viewer wavelength max
  -h                            Show this help
EOF
}

while getopts ":r:i:o:m:v:p:n:k:c:q:x:y:h" opt; do
  case $opt in
    i) DATA_DIR_ORIGINAL="$OPTARG" ;;
    o) DATA_DIR_INTERP="$OPTARG" ;;
    m) METADATA_PATH="$OPTARG" ;;
    v) VIEWER_SAVE_DIR="$OPTARG" ;;
    p) SPLIT="$OPTARG" ;;
    n) NUM_WORKERS="$OPTARG" ;;
    k) RE_MULTIPLIER="$OPTARG" ;;
    c) COL_RADIUS="$OPTARG" ;;
    q) COL_Q_RATIO="$OPTARG" ;;
    x) WL_MIN="$OPTARG" ;;
    y) WL_MAX="$OPTARG" ;;
    h)
      usage
      exit 0
      ;;
    \?)
      echo "[ERROR] Invalid option: -$OPTARG" >&2
      usage
      exit 1
      ;;
    :)
      echo "[ERROR] Option -$OPTARG requires an argument." >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$METADATA_PATH" ]]; then
  echo "[ERROR] Missing required -m <metadata_path>" >&2
  usage
  exit 1

DATA_DIR_ORIGINAL=$(readlink -f "$DATA_DIR_ORIGINAL")
DATA_DIR_INTERP=$(readlink -f "$DATA_DIR_INTERP")
METADATA_PATH=$(readlink -f "$METADATA_PATH")

export HF_DATASETS_CACHE="/home/valonso/iac18_mhuertas_shared/valonso/cache/huggingface/datasets"

if [[ -z "$VIEWER_SAVE_DIR" ]]; then
  VIEWER_SAVE_DIR="${REPO_ROOT}/logs/interpolation_viewer_$(date +%Y%m%d_%H%M%S)"
VIEWER_SAVE_DIR=$(readlink -m "$VIEWER_SAVE_DIR")

NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")
echo "-----------------------------------------------"
echo "Arrow Interpolation Pipeline - ${NOW}"
echo "-----------------------------------------------"

echo "[1/6] Changing directory to repo: ${REPO_ROOT}"
cd "$REPO_ROOT"

echo "[2/6] Activating Python environment"
source "$REPO_ROOT/.venv/bin/activate""

echo "[3/6] Running interpolation"
echo "      Original dataset:   ${DATA_DIR_ORIGINAL}"
echo "      Interpolated save:  ${DATA_DIR_INTERP}"
echo "      Metadata:           ${METADATA_PATH}"
echo "      Workers:            ${NUM_WORKERS}"
echo "      Re Multiplier:      ${RE_MULTIPLIER}"

python "$INTERP_SCRIPT" \
  --data_dir "$DATA_DIR_ORIGINAL" \
  --save_dir "$DATA_DIR_INTERP" \
  --metadata_path "$METADATA_PATH" \
  --col_radius "$COL_RADIUS" \
  --col_q_ratio "$COL_Q_RATIO" \
  --num_workers "$NUM_WORKERS" \
  --re_multiplier "$RE_MULTIPLIER"

echo "[4/6] Interpolation done"

echo "[5/6] Running interpolation viewer"
echo "      Viewer save dir:    ${VIEWER_SAVE_DIR}"
echo "      Viewer split:       ${SPLIT}"

VIEWER_CMD=(
  python "$VIEWER_SCRIPT"
  --data_dir_original "$DATA_DIR_ORIGINAL"
  --data_dir_interpolated "$DATA_DIR_INTERP"
  --save_dir "$VIEWER_SAVE_DIR"
  --split "$SPLIT"
)

if [[ -n "$WL_MIN" && -n "$WL_MAX" ]]; then
  VIEWER_CMD+=(--wl_range "$WL_MIN" "$WL_MAX")

"${VIEWER_CMD[@]}"

echo "[6/6] Pipeline finished successfully"
echo "      Plots saved in: ${VIEWER_SAVE_DIR}"
echo "-----------------------------------------------"
