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
while getopts ":r:o:" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Absolute output path
OUT_DIR=$(readlink -f "$OUT_DIR")

# Checking if .improved exists
if [ ! -f "$OUT_DIR/.improved" ]; then
    echo "[CRITICAL]: No detected improvements in $OUT_DIR"
    echo "[ACTION]: Cancelling jobs depending on ckpt_best.pt file for Training Part 2"
    
    scancel --name Plot_Im_Sp,Extract_Embed,Extract_Embed,Cos_Sim,Plot_Umaps,Probing_Tasks --user=$USER
else
    echo "[OK]: Improvement confirmed. Running post analysis"
fi