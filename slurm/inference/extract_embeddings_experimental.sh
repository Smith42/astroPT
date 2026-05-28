#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Extract_Embed
#SBATCH --partition=gpu
#SBATCH --account=iac18
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=32       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=128G                
#SBATCH --time=01:00:00          

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_extract_embed_%j.out
#SBATCH --error=logs/astropt_extract_embed_%j.err

# Robust repository root detection based on script location
# (Script is in slurm/trainers/, so REPO_ROOT is two levels up)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# If SLURM has moved us to /var/spool, correct REPO_ROOT to the launch working directory
if [[ "$REPO_ROOT" == "/var/spool"* ]]; then
    REPO_ROOT="$PWD"
fi

PYTHON_SCRIPT="$REPO_ROOT/scripts/extract_embeddings.py"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"
POOL_MET_IMG="mean"
POOL_MET_SPEC="rank"
PCA_DIM=0
ORDER_MODE="bidirectional"
JOINT_MODE="mean"
JOINT_ALPHA="0.5"
DRAW_FROM_CENTRE=0
EXP_TAG=""
BATTERY_PRESET=""

print_usage() {
  echo "Usage: $0 -w WEIGHTS_DIR -s SAVE_DIR [options]"
  echo "Required:"
  echo "  -w <path>    Weights directory"
  echo "  -s <path>    Save directory"
  echo "Optional:"
  echo "  -r <path>    Repository root"
  echo "  -a <path>    Arrow data directory"
  echo "  -i <name>    Pool method for images (mean|max|mixed|lp|rank)"
  echo "  -m <name>    Pool method for spectra (mean|max|mixed|lp|rank)"
  echo "  -p <int>     PCA dimensions (0 disables PCA)"
  echo "  -o <mode>    Order mode (bidirectional|images_first|spectra_first)"
  echo "  -j <mode>    Joint mode (mean|l2mean|weighted)"
  echo "  -l <float>   Joint alpha for weighted mode [0,1]"
  echo "  -e <tag>     Experiment tag"
  echo "  -c           Extract from middle layer"
  echo "  -b <preset>  Run battery preset (quick|orders|fusion|different)"
  echo "  -h           Help"
}

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:w:s:a:p:i:m:o:j:l:e:b:ch" opt; do
  case $opt in
    w) WEIGHTS_DIR="$OPTARG" ;;
    s) SAVE_DIR="$OPTARG" ;;
    a) DATA_DIR="$OPTARG" ;;
    p) PCA_DIM="$OPTARG" ;;
    i) POOL_MET_IMG="$OPTARG" ;;
    m) POOL_MET_SPEC="$OPTARG" ;;
    o) ORDER_MODE="$OPTARG" ;;
    j) JOINT_MODE="$OPTARG" ;;
    l) JOINT_ALPHA="$OPTARG" ;;
    e) EXP_TAG="$OPTARG" ;;
    b) BATTERY_PRESET="$OPTARG" ;;
    c) DRAW_FROM_CENTRE=1 ;;
    h) print_usage; exit 0 ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

if [[ -z "$WEIGHTS_DIR" || -z "$SAVE_DIR" ]]; then
  echo "[ERROR] Both -w and -s are required."
  print_usage
  exit 1

# Absolute output path
WEIGHTS_DIR=$(readlink -f "$WEIGHTS_DIR")
SAVE_DIR=$(readlink -f "$SAVE_DIR")
DATA_DIR=$(readlink -f "$DATA_DIR")

#--- ENVIRONMENT SETUP ---#
NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "-----------------------------------------------"
echo "Starting Embedding Extraction Job $SLURM_JOB_ID - $NOW"
echo "-----------------------------------------------"

# 1. Change directory
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# 2. Activate Environment
source "$REPO_ROOT/.venv/bin/activate""

# 3. Exports cache
export UV_CACHE_DIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/uv_cache"
export HF_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache/huggingface"

#--- EXECUTION ---#
echo "Embedding Extraction Configuration:"
echo "    DATA DIR:       $DATA_DIR" 
echo "    WEIGHTS DIR:    $WEIGHTS_DIR"
echo "    SAVE DIR:       $SAVE_DIR"
echo "    POOL MET IMG:   $POOL_MET_IMG"
echo "    POOL MET SPEC:  $POOL_MET_SPEC"
echo "    PCA DIM:        $PCA_DIM"
echo "    ORDER MODE:     $ORDER_MODE"
echo "    JOINT MODE:     $JOINT_MODE"
echo "    JOINT ALPHA:    $JOINT_ALPHA"
echo "    MID LAYER:      $DRAW_FROM_CENTRE"
echo "    EXP TAG:        ${EXP_TAG:-none}"
echo "    BATTERY:        ${BATTERY_PRESET:-none}"

run_extraction() {
  local run_tag="$1"
  local run_order="$2"
  local run_joint="$3"
  local run_alpha="$4"
  local run_draw_center="$5"

  echo "-----------------------------------------------"
  echo "Running experiment: $run_tag"
  echo "    order_mode:    $run_order"
  echo "    joint_mode:    $run_joint"
  echo "    joint_alpha:   $run_alpha"
  echo "    draw_centre:   $run_draw_center"

  CMD=(python "$PYTHON_SCRIPT"
      --weights_dir "$WEIGHTS_DIR"
      --data_dir "$DATA_DIR"
      --save_dir "$SAVE_DIR"
      --pool_method_img "$POOL_MET_IMG"
      --pool_method_spec "$POOL_MET_SPEC"
      --pca_dim "$PCA_DIM"
      --order_mode "$run_order"
      --joint_mode "$run_joint"
      --joint_alpha "$run_alpha"
      --exp_tag "$run_tag")

  if [[ "$run_draw_center" -eq 1 ]]; then
    CMD+=(--draw_from_centre)

  "${CMD[@]}"
}

if [[ -n "$BATTERY_PRESET" ]]; then
  case "$BATTERY_PRESET" in
    quick)
      run_extraction "baseline_final" "bidirectional" "mean" "0.5" 0 || exit 1
      run_extraction "joint_l2mean" "bidirectional" "l2mean" "0.5" 0 || exit 1
      run_extraction "joint_weighted_03" "bidirectional" "weighted" "0.3" 0 || exit 1
      run_extraction "joint_weighted_07" "bidirectional" "weighted" "0.7" 0 || exit 1
      run_extraction "middle_layer" "bidirectional" "mean" "0.5" 1 || exit 1
      ;;
    orders)
      run_extraction "ord_bidirectional" "bidirectional" "mean" "0.5" 0 || exit 1
      run_extraction "ord_images_first" "images_first" "mean" "0.5" 0 || exit 1
      run_extraction "ord_spectra_first" "spectra_first" "mean" "0.5" 0 || exit 1
      ;;
    fusion)
      run_extraction "fusion_mean" "bidirectional" "mean" "0.5" 0 || exit 1
      run_extraction "fusion_l2mean" "bidirectional" "l2mean" "0.5" 0 || exit 1
      run_extraction "fusion_weighted_03" "bidirectional" "weighted" "0.3" 0 || exit 1
      run_extraction "fusion_weighted_05" "bidirectional" "weighted" "0.5" 0 || exit 1
      run_extraction "fusion_weighted_07" "bidirectional" "weighted" "0.7" 0 || exit 1
      ;;
    different)
      # Unique union of quick + orders + fusion (no repeated parameter combinations).
      run_extraction "diff_baseline_final" "bidirectional" "mean" "0.5" 0 || exit 1
      run_extraction "diff_joint_l2mean" "bidirectional" "l2mean" "0.5" 0 || exit 1
      run_extraction "diff_joint_weighted_03" "bidirectional" "weighted" "0.3" 0 || exit 1
      run_extraction "diff_joint_weighted_05" "bidirectional" "weighted" "0.5" 0 || exit 1
      run_extraction "diff_joint_weighted_07" "bidirectional" "weighted" "0.7" 0 || exit 1
      run_extraction "diff_middle_layer" "bidirectional" "mean" "0.5" 1 || exit 1
      run_extraction "diff_ord_images_first" "images_first" "mean" "0.5" 0 || exit 1
      run_extraction "diff_ord_spectra_first" "spectra_first" "mean" "0.5" 0 || exit 1
      ;;
    *)
      echo "[ERROR] Invalid battery preset: $BATTERY_PRESET"
      echo "        Allowed presets: quick, orders, fusion, different"
      exit 1
      ;;
  esac
  SINGLE_TAG="${EXP_TAG:-manual}"
  run_extraction "$SINGLE_TAG" "$ORDER_MODE" "$JOINT_MODE" "$JOINT_ALPHA" "$DRAW_FROM_CENTRE" || exit 1

echo "-----------------------------------------------"
echo "Embedding Extraction Finished"
echo "-----------------------------------------------"
