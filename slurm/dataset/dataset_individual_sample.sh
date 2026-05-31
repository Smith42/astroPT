#!/bin/bash

# --- SLURM BATCH DIRECTIVES ---
#SBATCH --job-name=astropt_inspect_target
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/target_inspection_%j.log
#SBATCH --error=logs/target_inspection_%j.log

# --- ROBUST REPOSITORY ROOT DETECTION ---
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "$SCRIPT_DIR" == *"/slurm/dataset"* ]]; then
    REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
else
    REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
fi

cd "$REPO_ROOT" || { echo "[ERROR] Failed to navigate to repository root: $REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"

# --- ENVIRONMENT ACTIVATION ---
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    echo "Activating project virtual environment..."
    source "$REPO_ROOT/.venv/bin/activate"
else
    echo "[WARNING] Virtual environment activator not found at: $REPO_ROOT/.venv/bin/activate"
fi

# --- CONFIGURATION VARIABLES & ARGUMENT PARSING ---
TARGET_IDS=""
OUTPUT_DIR="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/dataset_individual_samples"
METADATA_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated"
DARK_REGISTRY="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/dataset_images_flux_analysis/dark_galaxies_registry.csv"

# If arguments are passed to sbatch (e.g., sbatch script.sh -t "123 456"), we parse them:
while getopts "t:o:m:d:h" opt; do
  case $opt in
    t) TARGET_IDS="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    m) METADATA_PATH="$OPTARG" ;;
    d) DATA_DIR="$OPTARG" ;;
    h)
      echo "Usage: $0 -t \"TARGET_ID1 TARGET_ID2 ...\" [-o OUTPUT_DIR] [-m METADATA_PATH] [-d DATA_DIR]"
      echo "  -t: Space-separated list of TargetIDs in quotes (e.g. -t \"39633411307996619 2305843017606509207\")"
      echo "  -o: Output directory to save diagnostic dashboard images"
      echo "  -m: Path to the FITS catalog containing survey metadata"
      echo "  -d: Path to Arrow dataset directory splits"
      exit 0
      ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

if [ -z "$TARGET_IDS" ]; then
    echo "[CRITICAL ERROR] No TargetIDs provided! You must specify -t followed by a space-separated list of TargetIDs in quotes."
    echo "Usage example:"
    echo "  sbatch slurm/dataset/inspect_target.sh -t \"39633411307996619 2305843017606509207\""
    exit 1
fi

echo "--------------------------------------------------------"
echo "  AstroPT Target Audit & Inspector Slurm Job"
echo "--------------------------------------------------------"
echo "  Job ID:            ${SLURM_JOB_ID:-local}"
echo "  Target IDs:        $TARGET_IDS"
echo "  Output Directory:  $OUTPUT_DIR"
echo "  Metadata Path:     $METADATA_PATH"
echo "  Arrow Data Dir:    $DATA_DIR"
echo "  Dark Registry:     $DARK_REGISTRY"
echo "--------------------------------------------------------"

# Verify file existence before running
if [ ! -d "$DATA_DIR" ]; then
    echo "[CRITICAL ERROR] Target processed data folder not found at: $DATA_DIR"
    exit 1
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# --- EXECUTION ---
# Run the inspection python script
python3 "$REPO_ROOT/scripts/dataset/dataset_individual_sample.py" \
    --target_ids $TARGET_IDS \
    --data_dir "$DATA_DIR" \
    --metadata_path "$METADATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --dark_registry "$DARK_REGISTRY"

EXIT_STATUS=$?
echo "--------------------------------------------------------"
if [ $EXIT_STATUS -eq 0 ]; then
    echo "[SUCCESS] AstroPT Target Audit & Inspection completed successfully."
    echo "  Dashboards saved in: $OUTPUT_DIR"
else
    echo "[FAILURE] AstroPT Target Audit & Inspection failed with exit code $EXIT_STATUS."
fi
echo "--------------------------------------------------------"

exit $EXIT_STATUS
