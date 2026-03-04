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
#SBATCH --output=logs/astropt_probing_dash%j.out
#SBATCH --error=logs/astropt_probing_dash%j.err

#--- DEFAULT VALUES ---#
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
PYTHON_SCRIPT="scripts/probing_downstream_dashboard.py"
OUT_DIR=""

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:e:o:s:" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    e) EMB_DIR="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    s) SAVE_NAME="$OPTARG" ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Absolute output path
OUT_DIR=$(readlink -f "$OUT_DIR")

#--- EMBEDDING DETECTION LOGIC ---#
if [ -z "$EMB_DIR" ]; then
    echo "[INFO] EMB_DIR not set. Searching for latest embeddings in $OUT_DIR..."
    EMB_DIR=$(ls -td "$OUT_DIR"/embeddings_* 2>/dev/null | head -n 1)
fi

if [ -z "$EMB_DIR" ]; then
    echo "[ERROR]: No 'embeddings_*' directory found in $OUT_DIR"
    echo "[WARNING]: Run extract_embeddings.sh first"
    exit 1
fi

EMB_DIR=$(readlink -f "$EMB_DIR")

#--- ENVIRONMENT SETUP ---#
echo "--------------------------------------------------"
echo "Starting Probing Tasks Dashboard Job $SLURM_JOB_ID"
echo "--------------------------------------------------"

# 1. Change directory
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# 2. Activate Environment
source .venv/bin/activate

# 3. Activating LaTeX (Required for confusion matrix plots)
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

#--- EXECUTION ---#
echo "Probing Dashboard Configuration:"
echo "   OUT DIR:       $OUT_DIR"
echo "   EMBEDDINGS:    $EMB_DIR"

# Run Python Script
python "$PYTHON_SCRIPT" \
  --out_dir "$OUT_DIR" \
  --files \
      "/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/astropt_100M_250K_arrow_20260130_asinh_mae_rot_lessmaxiters/embeddings_astropt_100M_250K_arrow_20260130_asinh_mae_rot_lessmaxiters_best_mean/probing_results_all.csv" \
      "/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/astropt_100M_250K_arrow_20260130_asinh_mae_rot_lessmaxiters/embeddings_astropt_100M_250K_arrow_20260130_asinh_mae_rot_lessmaxiters_best_rank/probing_results_all.csv" \
      "/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/astropt_100M_250K_arrow_20260130_asinh_mae_rot_lessmaxiters/embeddings_astropt_100M_250K_arrow_20260130_asinh_mae_rot_lessmaxiters_best_meanrank/probing_results_all.csv" \
      "/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/astropt_100M_250K_arrow_20260202_asinh_huber_rot_lr_moreiters/embeddings_astropt_100M_250K_arrow_20260202_asinh_huber_rot_lr_moreiters_best_meanrank/probing_results_all.csv" \
      "$EMB_DIR/probing_results_all.csv" \
  --names \
      "MAE Mean Pooling (0.40)" \
      "MAE Rank Pooling (0.76)" \
      "MAE Mean-Rank Pooling (0.26)" \
      "Huber Mean-Rank Pooling (0.37)" \
      "Last Training Pooling" \
  --targets \
      Z LOGMSTAR LOGSFR GR \
      flux_detection_total HALPHA_EW HALPHA_FLUX NII_6584_FLUX OIII_5007_FLUX HBETA_FLUX NII_6584_FLUX \
      sersic_sersic_vis_radius sersic_sersic_vis_index sersic_sersic_vis_axis_ratio has_spiral_arms_yes smoothness gini \
      SPECTYPE data_set_release \
  ${SAVE_NAME:+--save_name "$SAVE_NAME"}

echo "-----------------------------------------------"
echo "Probing Tasks Dashboard Finished"
echo "-----------------------------------------------"


