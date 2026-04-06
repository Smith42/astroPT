#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Plot_Im_Sp
#SBATCH --partition=gpu
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=1       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=64G                
#SBATCH --time=01:00:00         

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_img_spec_%j.out
#SBATCH --error=logs/astropt_img_spec_%j.err

set -euo pipefail

#--- DEFAULT VALUES ---#
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
PYTHON_SCRIPT="scripts/plot_images_spectra.py"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"

# If invoked as: sbatch script.sh -- -w <weights> ...
if [[ "${1:-}" == "--" ]]; then
  shift
fi

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:w:s:a:" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    w) WEIGHTS_DIR="$OPTARG" ;;
    s) SAVE_DIR="$OPTARG" ;;
    a) DATA_DIR="$OPTARG" ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

shift $((OPTIND - 1))

if [[ -z "${WEIGHTS_DIR:-}" ]]; then
  echo "[ERROR] WEIGHTS_DIR is required (-w <weights_dir>)" >&2
  exit 1
fi

if [[ -z "${SAVE_DIR:-}" ]]; then
  echo "[ERROR] SAVE_DIR is required (-s <save_dir>)" >&2
  exit 1
fi

# Absolute output path
WEIGHTS_DIR=$(readlink -f "$WEIGHTS_DIR")
SAVE_DIR=$(readlink -f "$SAVE_DIR")
DATA_DIR=$(readlink -f "$DATA_DIR")

#--- ENVIRONMENT SETUP ---#
NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "-----------------------------------------------"
echo "Plotting Images and Spectra $SLURM_JOB_ID - $NOW"
echo "-----------------------------------------------"

# 1. Change directory
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# 2. Activate Environment
source .venv/bin/activate

# 3. Activating LaTeX
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

#--- EXECUTION ---#
echo "Plotting Images and Spectra:"
echo "    WEIGHTS DIR:  $WEIGHTS_DIR"
echo "    DATA DIR:     $DATA_DIR" 
echo "    SAVE DIR:     $SAVE_DIR"

# Run Python Script
python "$PYTHON_SCRIPT" \
    --weights_dir "$WEIGHTS_DIR" \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --num_plot 25 \
    --target_ids \
        39627061836389042 \
        39627853679036156 \
        39633445487378968 \
        39627346218590254 \
        39633442895301960 \
        39633491763136752 \
        39633523014894811 \
        39633312192397870 \
        39633476848190817 \
        39633526185788566 \
        39633516366922547 \
        39633118423944455 \
        39633448033322139 \
        39633530795330888 \
        39089837394909544 \
        39633478949537029 \
        39633312192397870 \
        39633414239814702 \
        39633493688322559 \
        39627859714640945

echo "-----------------------------------------------"
echo "Plotting Images and Spectra Finished"
echo "-----------------------------------------------"