#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Pretokenise_AION
#SBATCH --partition=gpu
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=32       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=64G                
#SBATCH --time=10:00:00          

#--- LOGS FILES ---#
#SBATCH --output=logs/pretokenise_%j.out
#SBATCH --error=logs/pretokenise_%j.err

set -euo pipefail

#--- DEFAULT VALUES ---#
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated_tokenized/test_0/"
OUTPUT_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated_tokenized/test_0/"
RESNET_WEIGHTS="$REPO_ROOT/logs/resnet_adapter_weights/adapters_final.pt"
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
    w) RESNET_WEIGHTS="$OPTARG" ;;
    b) BATCH_SIZE="$OPTARG" ;;
    h) print_usage; exit 0 ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Changing directory to run astropt
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

source .venv/bin/activate

export UV_CACHE_DIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/uv_cache"
export HF_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache/huggingface"

# Creating logs directory if it doesn't exist
mkdir -p logs

#--- EXECUTION ---#
echo "Pre-tokenization Configuration:"
echo "    DATA DIR:       $DATA_DIR"
echo "    OUTPUT DIR:     $OUTPUT_DIR"
echo "    RESNET WEIGHTS: $RESNET_WEIGHTS"
echo "    BATCH SIZE:     $BATCH_SIZE"
echo "    DEVICE:         $DEVICE"

CMD=(python scripts/pretokenise_dataset_arrow.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --resnet_weights "$RESNET_WEIGHTS" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE")

"${CMD[@]}"

echo "-----------------------------------------------"
echo "Pre-tokenization Finished"
echo "-----------------------------------------------"
