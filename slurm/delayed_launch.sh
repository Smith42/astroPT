#!/bin/bash

set -euo pipefail

# USER CONFIGURATION
# Supported values: "astropt" or "astroclip"
MODEL="astropt"

# Format accepted by GNU date: YYYY-MM-DD HH:MM:SS
TARGET_DATE="2026-05-05 00:00:30"

TRAIN_NAME="60M + TokMix128 + CLS + DisTok"
TRAIN_DESC="Training a small 60M parameters AstroPt model in a discrete dataset (30M from the model + 30M from discrete vocab). This dataset was tokenized using AION tokenizers, FSQ for images and LFQ for spectra. Images used a U-Net translator from euclid to HSC images (original from AION). This mehtod expects to break the multimodality gap or at least, improve th perpormance of images in spectral probing tasks. This training also implements CLS and modality embeddings to improve the model's understanding of the multimodality. Longer blocks of 128 tokens per modality in token mixing to avaoid constant changes in the sequence."

# Optional extra flags forwarded to the selected launcher.
TRAIN_EXTRA_ARGS=""


# INTERNAL RESOLUTION
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

case "${MODEL,,}" in
    astropt)
        LAUNCH_SCRIPT="$SCRIPT_DIR/launch_train_astropt_multiGPU.sh"
        ;;
    astroclip)
        LAUNCH_SCRIPT="$SCRIPT_DIR/launch_train_astroclip_multiGPU.sh"
        ;;
    *)
        echo "[ERROR] MODEL must be 'astropt' or 'astroclip'. Got: $MODEL"
        exit 1
        ;;
esac

if [[ ! -x "$LAUNCH_SCRIPT" ]]; then
    echo "[ERROR] Launch script not found or not executable: $LAUNCH_SCRIPT"
    exit 1
fi

COMMAND_TO_RUN=("$LAUNCH_SCRIPT" -n "$TRAIN_NAME" -d "$TRAIN_DESC")
if [[ -n "$TRAIN_EXTRA_ARGS" ]]; then
    COMMAND_TO_RUN+=( -x "$TRAIN_EXTRA_ARGS" )
fi

# Convert dates to seconds since epoch
TARGET_TIME=$(date -d "$TARGET_DATE" +%s)
CURRENT_TIME=$(date +%s)

# Calculate remaining seconds
SLEEP_TIME=$((TARGET_TIME - CURRENT_TIME))

if [ "$SLEEP_TIME" -le 0 ]; then
    echo "[!] The target date has already passed. Executing immediately..."
    "${COMMAND_TO_RUN[@]}"
else
    # Display formatted wait time (optional)
    HOURS=$((SLEEP_TIME / 3600))
    MINS=$(((SLEEP_TIME % 3600) / 60))
    
    echo "-----------------------------------------------------------------"
    echo "Waiting $HOURS hours and $MINS minutes (until $TARGET_DATE)..."
    echo "Model: ${MODEL,,}"
    echo "Scheduled script: $LAUNCH_SCRIPT"
    echo "-----------------------------------------------------------------"
    
    # Pause execution
    sleep "$SLEEP_TIME"
    
    echo "[$(date)] Executing scheduled launcher..."
    "${COMMAND_TO_RUN[@]}"
fi

