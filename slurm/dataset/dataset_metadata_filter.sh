#!/bin/bash

# --- SLURM BATCH DIRECTIVES ---
#SBATCH --job-name=astropt_catalog_filter
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/catalog_filter_%j.log
#SBATCH --error=logs/catalog_filter_%j.log

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

echo "Current Python path: $(which python)"
echo "Current Python version: $(python --version)"

# --- CONFIGURATION VARIABLES ---
# Input: Original FITS catalog
export INPUT_CATALOG="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits"

# Output: Filtered clean FITS catalog
export OUTPUT_CATALOG="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1_FILTERED.fits"

# Dark galaxies registry from the flux analyzer audit
export DARK_REGISTRY="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/dataset_images_flux_analysis/dark_galaxies_registry.csv"

# Output directory for logs and summary report
export OUTPUT_DIR="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/astropt_dataset_audit"

# --- PRE-FLIGHT CHECKS ---
echo "--------------------------------------------------------"
echo "  AstroPT Catalog Filter & Clean Dataset Generator"
echo "--------------------------------------------------------"
echo "  Job ID:            $SLURM_JOB_ID"
echo "  Input Catalog:     $INPUT_CATALOG"
echo "  Output Catalog:    $OUTPUT_CATALOG"
echo "  Dark Registry:     $DARK_REGISTRY"
echo "  Output Directory:  $OUTPUT_DIR"
echo "  Script Root:       $REPO_ROOT"
echo "--------------------------------------------------------"

if [ ! -f "$INPUT_CATALOG" ]; then
    echo "[CRITICAL ERROR] Input FITS catalog not found at: $INPUT_CATALOG"
    exit 1
fi

if [ ! -f "$DARK_REGISTRY" ]; then
    echo "[WARNING] Dark galaxies registry not found at: $DARK_REGISTRY"
    echo "          Image quality filtering will be skipped. Run the full flux audit first."
    DARK_REGISTRY_FLAG=""
else
    DARK_REGISTRY_FLAG="--dark_registry $DARK_REGISTRY"
fi

mkdir -p "$OUTPUT_DIR"

# --- EXECUTION ---
# Apply both the SNR metadata filter AND the image quality blacklist filter.
# The SNR filter keeps only sources with adequate spectral signal-to-noise ratio:
#   - Low-z (Z < 0.15): requires SNR_SPEC_R > 3.0
#   - High-z (Z >= 0.15): requires SNR_SPEC_Z > 3.0
# The dark registry removes All-Black, Partially-Black, and NaN-Corrupted images.

python "$REPO_ROOT/scripts/dataset/dataset_metadata_filter.py" \
    --input_catalog "$INPUT_CATALOG" \
    --output_catalog "$OUTPUT_CATALOG" \
    --filters '((Z < 0.15) && (SNR_SPEC_R > 3.0)) || ((Z >= 0.15) && (SNR_SPEC_Z > 3.0))' \
    $DARK_REGISTRY_FLAG \
    --remove_classes All-Black Partially-Black NaN-Corrupted \
    --output_dir "$OUTPUT_DIR"

# --- EXIT STATUS ---
EXIT_STATUS=$?
echo "--------------------------------------------------------"
if [ $EXIT_STATUS -eq 0 ]; then
    echo "[SUCCESS] Catalog filtering completed successfully."
    echo "  Output FITS: $OUTPUT_CATALOG"
    echo ""
    echo "  NEXT STEP: Update your AstroPT training YAML config:"
    echo "    metadata_path: \"$OUTPUT_CATALOG\""
    echo "    applied_filters: []"
else
    echo "[FAILURE] Catalog filtering failed with exit code $EXIT_STATUS."
fi
echo "--------------------------------------------------------"

exit $EXIT_STATUS
