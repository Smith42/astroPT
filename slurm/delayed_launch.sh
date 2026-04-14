#!/bin/bash

# Check if the correct number of arguments is provided
TARGET_DATE="2026-04-14 22:35:00"
COMMAND_TO_RUN="./astroPT/slurm/launch_train_astropt_multiGPU.sh -n 'TokMix16 + CrossRecLoss + DsInter + SmallMod' -d 'Training with a token mixing with fixed 16 block size. Training with a cross-rconstruction loss added to the usual train loss. This configuration expects to improve the cross reconstruction and the probing tasks forcing the modal to learn the cross modality physics. Some hyperparameters have been updated to balance both modalities. Interpolate Zoomin and Zoomout dataset has been used. This dataset contains galaxies with the same size inside the 224x224 images. Decreasing the size of the model to test how this change in the learning.'"

# Convert dates to seconds since epoch
TARGET_TIME=$(date -d "$TARGET_DATE" +%s)
CURRENT_TIME=$(date +%s)

# Calculate remaining seconds
SLEEP_TIME=$((TARGET_TIME - CURRENT_TIME))

if [ "$SLEEP_TIME" -le 0 ]; then
    echo "[!] The target date has already passed. Executing immediately..."
    eval "$COMMAND_TO_RUN"
else
    # Display formatted wait time (optional)
    HOURS=$((SLEEP_TIME / 3600))
    MINS=$(((SLEEP_TIME % 3600) / 60))
    
    echo "-----------------------------------------------------------------"
    echo "⏳ Waiting $HOURS hours and $MINS minutes (until $TARGET_DATE)..."
    echo "Scheduled command: $COMMAND_TO_RUN"
    echo "-----------------------------------------------------------------"
    
    # Pause execution
    sleep "$SLEEP_TIME"
    
    echo "▶️ [$(date)] Executing command..."
    eval "$COMMAND_TO_RUN"
fi

