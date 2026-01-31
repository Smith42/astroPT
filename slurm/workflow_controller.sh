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
OUT_DIR=""

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:o:s:" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    s) SUFFIX="$OPTARG" ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Absolute output path
OUT_DIR=$(readlink -f "$OUT_DIR")

# Checking if .improved exists
if [ ! -f "$OUT_DIR/.improved" ]; then
    echo "[CRITICAL]: No detected improvements in $OUT_DIR"
    echo "[ACTION]: Cancelling dependent jobs for suffix: $SUFFIX"

    JOBS_TO_KILL=(
        "Plot_Im_Sp${SUFFIX}"
        "Extract_Embed${SUFFIX}"
        "Cos_Sim${SUFFIX}"
        "Plot_Umaps${SUFFIX}"
        "Probing_Tasks${SUFFIX}"
    )

    for JOB in "${JOBS_TO_KILL[@]}"; do
        echo " -> Cancelling: $JOB"
        scancel --name="$JOB" --user=$USER
    done

    echo "[DONE]: Kill sequence completed."

    exit 1

else
    echo "[OK]: Improvement confirmed. Running post analysis"
    exit 0
fi