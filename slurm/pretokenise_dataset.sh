#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Pretokenise_AION
#SBATCH --partition=gpu
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=16       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=64G                
#SBATCH --time=24:00:00          

#--- LOGS FILES ---#
#SBATCH --output=logs/pretokenise_%j.out
#SBATCH --error=logs/pretokenise_%j.err

set -euo pipefail

#--- DEFAULT VALUES ---#
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated"
OUTPUT_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated_tokenized"
UNET_WEIGHTS="$REPO_ROOT/logs/unet_adapter_weights/adapters_final.pt"
BATCH_SIZE=128
DEVICE="cuda"

print_usage() {
  echo "Usage: $0 [options]"
  echo "Optional:"
  echo "  -r <path>    AstroPT Repository root (default: $REPO_ROOT)"
  echo "  -a <path>    Input Arrow data directory (default: $DATA_DIR)"
  echo "  -o <path>    Output directory for tokenized data (default: $OUTPUT_DIR)"
  echo "  -w <path>    Path to U-Net adapter weights (default: $UNET_WEIGHTS)"
  echo "  -b <int>     Batch size (default: $BATCH_SIZE)"
  echo "  -h           Help"
  echo
}

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:a:o:w:b:h" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    a) DATA_DIR="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    w) UNET_WEIGHTS="$OPTARG" ;;
    b) BATCH_SIZE="$OPTARG" ;;
    h) print_usage; exit 0 ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Changing directory to run astropt
echo "------------------------------------------------------"
echo "Pre-tokenizing Dataset ($SLURM_JOB_ID) - $(date)"
echo "------------------------------------------------------"
echo "REPO_ROOT:  $REPO_ROOT"
echo "DATA_DIR:   $DATA_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "------------------------------------------------------"

cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# Creating logs directory if it doesn't exist
mkdir -p logs

# Launching Python script
python3 scripts/pretokenise_dataset_arrow.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --unet_weights "$UNET_WEIGHTS" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE"

echo "------------------------------------------------------"
echo "Pre-tokenization Finished - $(date)"
echo "------------------------------------------------------"
