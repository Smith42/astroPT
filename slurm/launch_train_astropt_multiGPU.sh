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
OUT_DIR="logs/astropt_100M_250K_arrow_20260129"
echo "Target Output Directory: $OUT_DIR"

# Dataset Directory (Arrow)
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"

# Launching scripts
TRAIN_SCRIPT="$SCRIPT_DIR/train_astropt_multiGPU.sh"
PLOT_MET_SCRIPT="$SCRIPT_DIR/plot_training_metrics.sh"
PLOT_IMG_SCRIPT="$SCRIPT_DIR/plot_images_spectra.sh"


#--- PART 1: First Training Session ---#

# First part of the training from SCRATCH
JOB1=$(sbatch --parsable "$TRAIN_SCRIPT" "$REPO_ROOT" "$OUT_DIR" "$DATA_DIR" scratch all)
echo "[Step 1] Job sent (Train Scratch).            ID: $JOB1"

# Plotting the metrics
JOB2=$(sbatch --parsable --dependency=afterany:$JOB1 "$PLOT_MET_SCRIPT" "$REPO_ROOT" "$OUT_DIR")
echo "[Step 2] Job sent (Plot Dashboard).           ID: $JOB2 (Depends on $JOB1)"

# Plotting the images and spectra
JOB3=$(sbatch --parsable --dependency=afterany:$JOB1 "$PLOT_IMG_SCRIPT" "$REPO_ROOT" "$OUT_DIR" "$DATA_DIR")
echo "[Step 3] Job sent (Plot Images & Spectra).    ID: $JOB3 (Depends on $JOB1)"


#--- PART 2: Second Training Session ---#

# Second part of the training from RESUME
JOB4=$(sbatch --parsable --dependency=afterok:$JOB1 "$TRAIN_SCRIPT" "$REPO_ROOT" "$OUT_DIR" "$DATA_DIR" resume all)
echo "[Step 4] Job sent (Train Resume).             ID: $JOB4 (Depends on $JOB1)"

# Plotting the metrics
JOB5=$(sbatch --parsable --dependency=afterany:$JOB4 "$PLOT_MET_SCRIPT" "$REPO_ROOT" "$OUT_DIR")
echo "[Step 5] Job sent (Plot Dashboard).           ID: $JOB5 (Depends on $JOB4)"

# Plotting the images and spectra
JOB6=$(sbatch --parsable --dependency=afterany:$JOB4 "$PLOT_IMG_SCRIPT" "$REPO_ROOT" "$OUT_DIR" "$DATA_DIR")
echo "[Step 6] Job sent (Plot Images & Spectra).    ID: $JOB6 (Depends on $JOB4)"




# Waiting for resources
sleep 10

#--- JOB INFORMATION ---#
echo "-------------------------------------------------"
echo "Pipeline Status Queue:"
squeue -u $USER
echo "-------------------------------------------------"