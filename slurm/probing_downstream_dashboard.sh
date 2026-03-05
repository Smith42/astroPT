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

#--- DEFAULT VALUES ---#
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
PYTHON_SCRIPT="scripts/probing_downstream_dashboard.py"

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:e:o:n:" opt; do
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
echo "Starting Probing Tasks Dashboard Job $SLURM_JOB_ID - $NOW"
echo "--------------------------------------------------"

# 1. Change directory
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# 2. Activate Environment
source .venv/bin/activate

# 3. Activating LaTeX (Required for confusion matrix plots)
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

#--- EMBEDDING DETECTION LOGIC ---#
DETECTED_EMB=$(ls -td "${EMB_DIR}"/*/ 2>/dev/null | head -n 1)
DETECTED_EMB="${DETECTED_EMB%/}"
DETECTED_EMB=$(readlink -f "$DETECTED_EMB")

if [ -z "$DETECTED_EMB" ]; then
    echo "[ERROR]: No sub-directory found in $EMB_DIR"
    echo "[WARNING]: Run extract_embeddings.sh first"
    exit 1
fi

if [ -z "$SAVE_DIR" ]; then
    SAVE_DIR="$DETECTED_EMB"
fi
SAVE_DIR=$(readlink -f "$SAVE_DIR")

#--- EXECUTION ---#
echo "Probing Dashboard Configuration:"
echo "    SAVE DIR:       $SAVE_DIR (Auto-Detected)"

# Run Python Script
python "$PYTHON_SCRIPT" \
  --save_dir "$SAVE_DIR" \
  --csv_path \
      "/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/astropt_100M_250K_arrow_20260130_asinh_mae_rot_lessmaxiters/embeddings_astropt_100M_250K_arrow_20260130_asinh_mae_rot_lessmaxiters_best_mean/downstream_results.csv" \
      "/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/astropt_100M_250K_arrow_20260130_asinh_mae_rot_lessmaxiters/embeddings_astropt_100M_250K_arrow_20260130_asinh_mae_rot_lessmaxiters_best_rank/downstream_results.csv" \
      "/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/astropt_100M_250K_arrow_20260130_asinh_mae_rot_lessmaxiters/embeddings_astropt_100M_250K_arrow_20260130_asinh_mae_rot_lessmaxiters_best_meanrank/downstream_results.csv" \
      "/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/astropt_100M_250K_arrow_20260202_asinh_huber_rot_lr_moreiters/embeddings_astropt_100M_250K_arrow_20260202_asinh_huber_rot_lr_moreiters_best_meanrank/downstream_results.csv" \
      "$SAVE_DIR/downstream_tasks/downstream_results.csv" \
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


