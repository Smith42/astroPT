#!/bin/bash

# SLURM BATCH DIRECTIVES
#SBATCH --job-name=dataset_flux_audit
#SBATCH --partition=gpu
#SBATCH --account=iac18
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=32       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=64G                
#SBATCH --time=04:00:00         

#--- LOGS FILES ---#
#SBATCH --output=logs/dataset_flux_audit_%j.out
#SBATCH --error=logs/dataset_flux_audit_%j.err

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
# Activate the project's custom python environment
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    echo "Activating project virtual environment..."
    source "$REPO_ROOT/.venv/bin/activate"
else
    echo "[WARNING] Virtual environment activator not found at: $REPO_ROOT/.venv/bin/activate"
fi

PYTHON_SCRIPT="$REPO_ROOT/scripts/dataset/dataset_flux_analyzer.py"

# Default values
MAX_SAMPLES=-1
SPLIT="train"
OUTPUT_DIR="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/dataset_images_audit"
METADATA_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated"

# If invoked as: sbatch script.sh -- -e <emb_root> ...
if [[ "${1:-}" == "--" ]]; then
  shift
fi

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":n:s:o:m:h" opt; do
  case $opt in
    n) MAX_SAMPLES="$OPTARG" ;;
    s) SPLIT="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    m) METADATA_PATH="$OPTARG" ;;
    h)
      echo "Usage: $0 [-n MAX_SAMPLES] [-s SPLIT] [-o OUTPUT_DIR] [-m METADATA_PATH]"
      echo "  -n: Max samples to audit (default: -1, i.e., all)"
      echo "  -s: Dataset split to audit (default: 'train')"
      echo "  -o: Directory to save the audit outputs"
      echo "  -m: Path to the catalog metadata FITS file"
      exit 0
      ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

shift $((OPTIND - 1))

# Verify the currently running Python version and environment path
echo "Current Python path: $(which python)"
echo "Current Python version: $(python --version)"

# Verify file existence before running to generate meaningful, proactive logs
echo "--------------------------------------------------------"
echo "  AstroPT Processed Dataset Flux Integrity Audit"
echo "--------------------------------------------------------"
echo "  Job ID:         ${SLURM_JOB_ID:-local}"
echo "  Data Directory: $DATA_DIR"
echo "  Metadata Path:  $METADATA_PATH"
echo "  Output Dir:     $OUTPUT_DIR"
echo "  Script Root:    $REPO_ROOT"
echo "  Max Samples:    $MAX_SAMPLES"
echo "  Split:          $SPLIT"
echo "--------------------------------------------------------"

if [ ! -d "$DATA_DIR" ]; then
    echo "[CRITICAL ERROR] Target processed data folder not found at: $DATA_DIR"
    exit 1
fi

if [ ! -f "$METADATA_PATH" ]; then
    echo "[CRITICAL ERROR] Target FITS catalog not found at: $METADATA_PATH"
    exit 1
fi

# Ensure output directory exists before launching the Python script
mkdir -p "$OUTPUT_DIR"

# --- EXECUTION ---
python3 "$PYTHON_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --split "$SPLIT" \
    --metadata_path "$METADATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples "$MAX_SAMPLES"

# Capture exit status from python execution to propagate error codes correctly
EXIT_STATUS=$?
echo "--------------------------------------------------------"
if [ $EXIT_STATUS -eq 0 ]; then
    echo "[SUCCESS] AstroPT Processed Dataset Flux Audit completed successfully."
else
    echo "[FAILURE] AstroPT Dataset Flux Audit failed with exit code $EXIT_STATUS."
fi
echo "--------------------------------------------------------"

exit $EXIT_STATUS
