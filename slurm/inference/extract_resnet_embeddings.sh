#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Resnet_Emb
#SBATCH --partition=gpu
#SBATCH --account=iac18
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=16       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=64G                
#SBATCH --time=04:00:00         

#--- LOGS FILES ---#
#SBATCH --output=logs/resnet_emb_%j.out
#SBATCH --error=logs/resnet_emb_%j.err

cd "/home/valonso/iac18_mhuertas_shared/valonso/astroPT" || exit 1
source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="${PYTHONPATH}:/home/valonso/iac18_mhuertas_shared/valonso/astroPT"

# Automatically detect the repository root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
PYTHON_SCRIPT="$REPO_ROOT/scripts/extract_resnet_embeddings.py"
BASE_DIR="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/resnet18_images_supervised"

# Iterate over all target folders inside resnet18_images_supervised
for target_dir in "$BASE_DIR"/*/; do
    if [ -d "$target_dir" ] && [ -d "${target_dir}weights" ]; then
        echo "========================================="
        echo "Extracting embeddings for ResNet Model: $(basename "$target_dir")"
        python "$PYTHON_SCRIPT" --model_dir "$target_dir" --output_name "test_set_embeddings"
done

echo "Done extracting all ResNet embeddings."
