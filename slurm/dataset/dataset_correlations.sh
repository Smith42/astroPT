#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Dataset_Corr
#SBATCH --partition=gpu
#SBATCH --nodes=1                # Nodes number
#SBATCH --ntasks=1               # Total task number (1 task invoking torchrun)
#SBATCH --cpus-per-task=64       # CPUs for task (16 per GPU * 4 GPUs)
#SBATCH --gpus-per-task=1        # GPUs for task - DDP
#SBATCH --mem=128G               # Requested RAM
#SBATCH --time=2:00:00          # Requested time in cluster

#--- LOGS FILES ---#
#SBATCH --output=logs/dataset_corr_%j.out
#SBATCH --error=logs/dataset_corr_%j.err

# --- ROBUST REPOSITORY ROOT DETECTION ---
# Robust repository root detection based on script location
# (Script is in slurm/trainers/, so REPO_ROOT is two levels up)
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

# Move to the repository root directory
cd "$REPO_ROOT" || { echo "[ERROR] Failed to navigate to repository root: $REPO_ROOT"; exit 1; }

# Create logs directory if it does not already exist to prevent Slurm/Output failures
mkdir -p "$REPO_ROOT/logs"

# --- ENVIRONMENT ACTIVATION ---
source "$REPO_ROOT/.venv/bin/activate"

# Verify the currently running Python version and environment path
echo "Current Python path: $(which python)"
echo "Current Python version: $(python --version)"

# --- CONFIGURATION VARIABLES ---
# Define strict, absolute environment variables for paths as specified in the instructions
export METADATA_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits"
export OUTPUT_DIR="/mnt/data_proj/iac18_mhuertas_shared/valonso/astroPT/logs/astropt_dataset_audit"

# Verify file existence before running to generate meaningful, proactive logs
echo "--------------------------------------------------------"
echo "  AstroPT Forensic Dataset Audit Orchestration Launch"
echo "--------------------------------------------------------"
echo "  Job ID:         $SLURM_JOB_ID"
echo "  Metadata Path:  $METADATA_PATH"
echo "  Output Dir:     $OUTPUT_DIR"
echo "  Script Root:    $REPO_ROOT"
echo "--------------------------------------------------------"

if [ ! -f "$METADATA_PATH" ]; then
    echo "[CRITICAL ERROR] Target FITS catalog not found at: $METADATA_PATH"
    exit 1
fi

# Ensure output directory exists before launching the Python script
mkdir -p "$OUTPUT_DIR"

# --- EXECUTION ---
# Execute the Python data-audit script, forwarding all environment variables as arguments.
# Using 'python' from the activated environment ensures that Astropy/Pandas are run under astropt miniconda.
python "$REPO_ROOT/scripts/dataset/dataset_correlations.py" \
    --metadata_path "$METADATA_PATH" \
    --output_dir "$OUTPUT_DIR"

# Capture exit status from python execution to propagate error codes correctly
EXIT_STATUS=$?
echo "--------------------------------------------------------"
if [ $EXIT_STATUS -eq 0 ]; then
    echo "[SUCCESS] AstroPT Forensic Dataset Audit completed successfully."
else
    echo "[FAILURE] AstroPT Dataset Audit failed with exit code $EXIT_STATUS."
fi
echo "--------------------------------------------------------"

exit $EXIT_STATUS
