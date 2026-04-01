#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Mapper_Dash
#SBATCH --partition=gpu
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=2       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=32G                
#SBATCH --time=00:10:00         

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_mapper_dash_%j.out
#SBATCH --error=logs/astropt_mapper_dash_%j.err

#--- DEFAULT VALUES ---#
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
PYTHON_SCRIPT="scripts/probing_latent_dashboard.py"

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:e:s:n:" opt; do
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
echo "Starting Latent Mapper Dashboard Job $SLURM_JOB_ID - $NOW"
echo "--------------------------------------------------"

cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }
source .venv/bin/activate
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

#--- EMBEDDING DETECTION LOGIC ---#
DETECTED_EMB=$(ls -td "${EMB_DIR}"/*/ 2>/dev/null | head -n 1)
DETECTED_EMB="${DETECTED_EMB%/}"
DETECTED_EMB=$(readlink -f "$DETECTED_EMB")

if [ -z "$DETECTED_EMB" ]; then
    echo "[ERROR]: No sub-directory found in $EMB_DIR"
    exit 1
fi

if [ -z "$SAVE_DIR" ]; then
    SAVE_DIR="$DETECTED_EMB"
fi
SAVE_DIR=$(readlink -f "$SAVE_DIR")

#--- EXECUTION ---#
echo "Mapper Dashboard Configuration:"
echo "    SAVE DIR:       $SAVE_DIR (Auto-Detected)"

# RUN PYTHON SCRIPT
# Nota: Puedes pasar múltiples CSVs aquí para comparar distintas versiones del modelo,
# o simplemente pasar varios mapeos (images->joint, images->spectra) del MISMO modelo
# para verlos todos en la misma gráfica.

python "$PYTHON_SCRIPT" \
  --save_dir "$SAVE_DIR" \
  --input_dirs \
      "/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/astropt_100M_250K_arrow_20260307_asinh_rot_mae/embeddings/best_meanrank/latent_mapper/" \
      "/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/astropt_100M_250K_arrow_20260312_asinh_rot_mae_imgpatchsize8/embeddings/best_meanrank/latent_mapper/" \
      "/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/astropt_100M_250K_arrow_20260312_asinh_rot_mae_imgpatchsize8_copy/embeddings/best_meanrank/latent_mapper/" \
      "$SAVE_DIR/downstream_results.csv" \
  --names \
      "MAE Mean-Rank Pooling (IMG PS 16 / SPEC PS 10) (0.24)" \
      "MAE Mean-Rank IMG Patch Size 8 (0.14)" \
      "MAE Mean-Rank IMG Patch Size 8 (0.14) EMBEDDING CHANGES" \
      "MAE Mean-Rank IMG Patch Size 8 SHUFFLE TRUE" \
  --targets \
      Z LOGMSTAR LOGSFR GR \
      flux_detection_total HALPHA_EW HALPHA_FLUX NII_6584_FLUX OIII_5007_FLUX HBETA_FLUX \
      sersic_sersic_vis_radius sersic_sersic_vis_index sersic_sersic_vis_axis_ratio has_spiral_arms_yes smoothness gini \
      SPECTYPE data_set_release \
  

echo "-----------------------------------------------"
echo "Latent Mapper Dashboard Finished"
echo "-----------------------------------------------"


# ${SAVE_NAME:+--save_name "$SAVE_NAME"}