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

echo "-----------------------------------------------"
echo "Starting Plotting Job $SLURM_JOB_ID"
echo "-----------------------------------------------"


# Changing directory to run astropt
REPO_ROOT=${1:-"/home/valonso/iac18_mhuertas_shared/valonso/astroPT"}
shift
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || exit 1
source .venv/bin/activate

# Activating LaTeX
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

# Arguments
# Output Dir (Required)
OUT_DIR=${1:-"logs/astropt_100M_250K_arrow_20260126"}

echo "Plotting metrics from:"
echo "   OUT DIR:   $OUT_DIR"

# Run Python Script
python scripts/plot_training_metrics.py \
    --out_dir "$OUT_DIR" \
    --csv_name "metrics.csv" \
    --save_name "training_metrics.png"

echo "-----------------------------------------------"
echo "Plotting Finished"
echo "-----------------------------------------------"