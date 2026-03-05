#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Cos_Sim
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G        
#SBATCH --time=00:20:00

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_cos_%j.out
#SBATCH --error=logs/astropt_cos_%j.err

#--- DEFAULT VALUES ---#
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
PYTHON_SCRIPT="scripts/cosine_similarity.py"

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:w:s:e:" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    w) WEIGHTS_DIR="$OPTARG" ;;
    s) SAVE_DIR="$OPTARG" ;;
    e) EMB_DIR="$OPTARG" ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Absolute paths
WEIGHTS_DIR=$(readlink -f "$WEIGHTS_DIR")
EMB_DIR=$(readlink -f "$EMB_DIR")

#--- ENVIRONMENT SETUP ---#
NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "-----------------------------------------------"
echo "Starting Cosine Similarity Job $SLURM_JOB_ID - $NOW" 
echo "-----------------------------------------------"



# 1. Change directory
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# 2. Activate Environment
source .venv/bin/activate

# 3. Activating LaTeX (Required for plots)
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

#--- EMBEDDING DETECTION LOGIC ---#
DETECTED_EMB=$(ls -td "${EMB_DIR}"/*/ 2>/dev/null | head -n 1)
DETECTED_EMB="${DETECTED_EMB%/}"
DETECTED_EMB=$(readlink -f "$DETECTED_EMB")

if [ -z "$DETECTED_EMB" ]; then
    echo "[ERROR]: No sub-directory found in $EMB_DIR"
    echo "[WARNING]: Run extract_embeddings.sh first"
    exit 1
fi

if [ -z "$SAVE_DIR" ]; then
    SAVE_DIR="$DETECTED_EMB"
fi
SAVE_DIR=$(readlink -f "$SAVE_DIR")

#--- EXECUTION ---#
echo "Cosine Similarity Configuration:"
echo "    WEIGHTS DIR:  $WEIGHTS_DIR"
echo "    EMB DIR:      $DETECTED_EMB"
echo "    SAVE DIR:     $SAVE_DIR (Auto-detected)"


# Running Python Script
python "$PYTHON_SCRIPT" \
    --weights_dir "$WEIGHTS_DIR" \
    --emb_dir "$DETECTED_EMB" \
    --save_dir "$SAVE_DIR"

echo "-----------------------------------------------"
echo "Cosine Similarity Finished"
echo "-----------------------------------------------"
