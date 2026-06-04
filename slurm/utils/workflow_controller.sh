#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Workflow_Controller
#SBATCH --partition=gpu
#SBATCH --account=iac18
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:05:00

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_workflow_%j.out
#SBATCH --error=logs/astropt_workflow_%j.err

#--- DEFAULT VALUES ---#
# Automatically detect the repository root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
#--- ARGUMENT PARSING (FLAGS) ---#
FORCE_ANALYSIS=false
while getopts ":r:t:x:f" opt; do
  case $opt in
    t) TRAIN_DIR="$OPTARG" ;;
    x) SUFFIX="$OPTARG" ;;
    f) FORCE_ANALYSIS=true ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Absolute output path
TRAIN_DIR=$(readlink -f "$TRAIN_DIR")

#--- ENVIRONMENT SETUP ---#
NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "-----------------------------------------------"
echo "Workflow Controller Job $SLURM_JOB_ID - $NOW"
echo "-----------------------------------------------"

# Checking if .improved exists
if [ "$FORCE_ANALYSIS" = false ] && [ ! -f "$TRAIN_DIR/.improved" ]; then
    echo "[CRITICAL]: No detected improvements in $TRAIN_DIR"
    echo "[ACTION]: Cancelling dependent jobs for suffix: $SUFFIX"

    JOBS_TO_KILL=(
        "Plot_Attn_Maps${SUFFIX}"
        "Plot_Im_Sp${SUFFIX}"
        "Cross_Rec${SUFFIX}"
        "Extract_Embed${SUFFIX}"
        "Cos_Sim${SUFFIX}"
        "Plot_Umaps${SUFFIX}"
        "Probing_Tasks${SUFFIX}"
        "Probing_Tasks_Dash${SUFFIX}"
        "Latent_Tasks${SUFFIX}"
        "Latent_Tasks_Dash${SUFFIX}"
        "Predict_Dash${SUFFIX}"
    )

    for JOB in "${JOBS_TO_KILL[@]}"; do
        echo " --> Cancelling: $JOB"
        scancel --name="$JOB" --user=$USER
    done

    echo "[DONE]: Kill sequence completed."
    exit 1
fi

echo "[OK]: Improvement confirmed. Running post analysis"
exit 0
