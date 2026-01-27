#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Plot_Im_Sp
#SBATCH --partition=gpu
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=1       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=64G                
#SBATCH --time=00:20:00         

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

# Dataset Directory (Arrow)
DATA_DIR=${2:-"/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"}

echo "Plotting Images and Spectra:"
echo "   OUT DIR:    $OUT_DIR"
echo "   DATA DIR:   $DATA_DIR"

# Run Python Script
python scripts/plot_images_spectra.py \
    --out_dir "$OUT_DIR" \
    --data_dir "$DATA_DIR" \
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
    --num_plot 15

echo "-----------------------------------------------"
echo "Plotting Images and Spectra Finished"
echo "-----------------------------------------------"