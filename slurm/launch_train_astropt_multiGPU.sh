#!/bin/bash


#--- ASTROPT PIPELINE LAUNCHER ---#
echo "-------------------------------------------------"
echo "Launching Full AstroPT Pipeline"
echo "-------------------------------------------------"

# Repository directory
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"

# Launching directory
SCRIPT_DIR=$(dirname "$0")

# Training output directory
OUT_DIR="logs/astropt_100M_250K_arrow_T1_20260125_test"
echo "Target Output Directory: $OUT_DIR"

# Dataset Directory (Arrow)
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"


#--- PART 1: Model Training ---#

# First part of the training from scratch
JOB1=$(sbatch --parsable "$SCRIPT_DIR/train_astropt_multiGPU.sh" "$REPO_ROOT" "$OUT_DIR" "$DATA_DIR" scratch)
echo "[Step 1] Job sent (Train Scratch).  ID: $JOB1"

# Second part of the training from resume
JOB2=$(sbatch --parsable --dependency=afterany:$JOB1 "$SCRIPT_DIR/train_astropt_multiGPU.sh" "$REPO_ROOT" "$OUT_DIR" "$DATA_DIR" resume)
echo "[Step 2] Job sent (Train Resume).   ID: $JOB2 (Depends on $JOB1)"


#--- PART 2: Post Training Analysis ---#

# Plotting the metrics
JOB3=$(sbatch --parsable --dependency=afterany:$JOB2 "$SCRIPT_DIR/plot_training_metrics.sh" "$REPO_ROOT" "$OUT_DIR")
echo "[Step 3] Job sent (Plot Dashboard). ID: $JOB3 (Depends on $JOB2)"


#--- JOB INFORMATION ---#
echo "-------------------------------------------------"
echo "Pipeline Status Queue:"
squeue -u $USER
echo "-------------------------------------------------"