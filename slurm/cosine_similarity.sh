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

echo "-----------------------------------------------"
echo "Starting Cosine Similarity Job $SLURM_JOB_ID"
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
OUT_DIR=${1:-"logs/astropt_100M_250K_arrow_20260128"}

# Embeddings Directory
EMB_DIR=$(ls -td "$OUT_DIR"/embeddings_* 2>/dev/null | head -n 1)

if [ -z "$EMB_DIR" ]; then
    echo "[ERROR]: No 'embeddings_*' directory found in $OUT_DIR"
    echo "[WARNING]: Run astropt/scripts/extract_embeddings.py first"
    exit 1
fi

echo "Cosine Similarity:"
echo "   OUT DIR:   $OUT_DIR"
echo "   EMB DIR:   $EMB_DIR"

# Run Python Script
python scripts/cosine_similarity.py \
    --out_dir "$OUT_DIR" \
    --emb_dir "$EMB_DIR"

echo "-----------------------------------------------"
echo "Cosine Similarity Finished"
echo "-----------------------------------------------"