#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Plot_Metrics
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_plot_%j.out
#SBATCH --error=logs/astropt_plot_%j.err

#--- DEFAULT VALUES ---#
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
PYTHON_SCRIPT="scripts/plot_training_metrics.py"
OUT_DIR=""

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:o:" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Absolute output path
OUT_DIR=$(readlink -f "$OUT_DIR")

#--- ENVIRONMENT SETUP ---#
echo "-----------------------------------------------"
echo "Starting Plotting Job $SLURM_JOB_ID"
echo "-----------------------------------------------"

# 1. Change directory
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# 2. Activate Environment
source .venv/bin/activate

# 3. Activating LaTeX (Required for plots)
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

#--- EXECUTION ---#
echo "Plotting Metrics Configuration:"
echo "   OUT DIR:   $OUT_DIR"

# Running Python Script
python "$PYTHON_SCRIPT" \
    --out_dir "$OUT_DIR" \
    --csv_name "metrics.csv" \
    --save_name "training_metrics.png"

echo "-----------------------------------------------"
echo "Plotting Finished"
echo "-----------------------------------------------"