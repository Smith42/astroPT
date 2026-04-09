#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Workflow_Controller
#SBATCH --partition=gpu
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
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:t:x:" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    t) TRAIN_DIR="$OPTARG" ;;
    x) SUFFIX="$OPTARG" ;;
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
if [ ! -f "$TRAIN_DIR/.improved" ]; then
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
    )

    for JOB in "${JOBS_TO_KILL[@]}"; do
        echo " --> Cancelling: $JOB"
        scancel --name="$JOB" --user=$USER
    done

    echo "[DONE]: Kill sequence completed."

    exit 1

else
    echo "[OK]: Improvement confirmed. Running post analysis"
    exit 0
fi