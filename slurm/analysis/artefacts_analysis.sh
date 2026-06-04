#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=AstroPT_Audit
#SBATCH --partition=gpu
#SBATCH --account=iac18
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=16       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=64G                
#SBATCH --time=02:00:00         

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_audit_%j.out
#SBATCH --error=logs/astropt_audit_%j.err

echo "--------------------------------------------------------"
echo "AstroPT Dataset Observational Quality Audit (SLURM)"
echo "--------------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "--------------------------------------------------------"

# Robust repository root detection based on script location
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

cd "$REPO_ROOT" || { echo "[ERROR] Failed to cd to $REPO_ROOT"; exit 1; }

# Environment Setup
export OMP_NUM_THREADS=16
export PYTHONWARNINGS="ignore"
export PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH"

# Enable Virtual Environment
VENV_PATH="$REPO_ROOT/.venv"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Using python environment: $(which python)"
else
    echo "[WARNING] Virtual environment not found at $VENV_PATH. Using system python."
fi

# Default Paths
CKPT_PATH="logs/astropt_20260601_hybrid_optimized/weights/ckpt_best.pt"
WEIGHTS_DIR=""
EMB_DIR=""
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"
META_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1_FILTERED.fits"
N_SAMPLES=5000  # Audit 5000 random/sequential samples by default to get super-fast results
SPLIT="both"    # Audit both splits by default

# Parse arguments to override defaults if desired
while getopts ":w:e:d:m:n:p:" opt; do
  case $opt in
    w) WEIGHTS_DIR="$OPTARG" ;;
    e) EMB_DIR="$OPTARG" ;;
    d) DATA_DIR="$OPTARG" ;;
    m) META_PATH="$OPTARG" ;;
    n) N_SAMPLES="$OPTARG" ;;
    p) SPLIT="$OPTARG" ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done

# Resolve Checkpoint Path from Weights (file or directory)
if [ -n "$WEIGHTS_DIR" ]; then
    WEIGHTS_DIR=$(readlink -f "$WEIGHTS_DIR")
    if [ -f "$WEIGHTS_DIR" ]; then
        CKPT_PATH="$WEIGHTS_DIR"
    elif [ -d "$WEIGHTS_DIR" ]; then
        if [ -f "$WEIGHTS_DIR/ckpt_best.pt" ]; then
            CKPT_PATH="$WEIGHTS_DIR/ckpt_best.pt"
        else
            FIRST_PT=$(find "$WEIGHTS_DIR" -maxdepth 1 -name "*.pt" | head -n 1)
            if [ -n "$FIRST_PT" ]; then
                CKPT_PATH="$FIRST_PT"
            else
                echo "[ERROR]: No checkpoint (.pt) found in weights directory $WEIGHTS_DIR"
                exit 1
            fi
        fi
    else
        echo "[ERROR]: WEIGHTS_DIR ($WEIGHTS_DIR) does not exist."
        exit 1
    fi
else
    CKPT_PATH=$(readlink -f "$CKPT_PATH")
fi

DATA_DIR=$(readlink -f "$DATA_DIR")
META_PATH=$(readlink -f "$META_PATH")

OUTPUT_ARG=""
if [ -n "$EMB_DIR" ]; then
    EMB_DIR=$(readlink -f "$EMB_DIR")
    # Resolve artifacts save directory
    OUTPUT_ARG="--output_dir $EMB_DIR/anomalies"
fi

echo "Checkpoint Path:  $CKPT_PATH"
echo "Dataset Dir:      $DATA_DIR"
echo "Catalog Path:     $META_PATH"
echo "Samples count:    $N_SAMPLES"
echo "Split Selected:   $SPLIT"
if [ -n "$OUTPUT_ARG" ]; then
    echo "Output Dir:       $EMB_DIR/anomalies"
fi
echo "--------------------------------------------------------"

# Run Python Auditor
python3 "$REPO_ROOT/scripts/analysis/anomalies/artefacts_analysis.py" \
    --ckpt_path "$CKPT_PATH" \
    --data_dir "$DATA_DIR" \
    --metadata_path "$META_PATH" \
    --n_samples "$N_SAMPLES" \
    --split "$SPLIT" \
    $OUTPUT_ARG

echo "--------------------------------------------------------"
echo "AstroPT Dataset Quality Audit Finished Successfully"
echo "--------------------------------------------------------"
