#!/bin/bash

# --- ASTROPT SCHEDULED TRAIN --- #
# Use this script to program a training run at a specific date/time.
# It will wait until the target time and then invoke the master launcher.

set -euo pipefail

# Robust repository root detection based on script location
# (Script is in slurm/trainers/, so REPO_ROOT is two levels up)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# If SLURM has moved us to /var/spool, correct REPO_ROOT to the launch working directory
if [[ "$REPO_ROOT" == "/var/spool"* ]]; then
    REPO_ROOT="$PWD"
fi

LAUNCH_SCRIPT="$REPO_ROOT/slurm/trainers/launch_pipeline.sh"

# --- USER CONFIGURATION ---
# Target date format: YYYY-MM-DD HH:MM:SS (GNU date format)
TARGET_DATE="2026-05-15 00:00:00"

# Training configuration
CONFIG_FILE="$REPO_ROOT/configs/default_config.yaml"
TRAIN_NAME="scheduled_experiment"
TRAIN_DESC="Add your detailed description here"
EXTRA_ARGS=""
# --------------------------

if [[ ! -x "$LAUNCH_SCRIPT" ]]; then
    echo "[ERROR] Master launcher not found or not executable: $LAUNCH_SCRIPT"
    exit 1
fi

# Convert dates to seconds since epoch
TARGET_TIME=$(date -d "$TARGET_DATE" +%s)
CURRENT_TIME=$(date +%s)

# Calculate remaining seconds
SLEEP_TIME=$((TARGET_TIME - CURRENT_TIME))

if [ "$SLEEP_TIME" -le 0 ]; then
    echo "[!] The target date has already passed. Executing immediately..."
    $LAUNCH_SCRIPT -c "$CONFIG_FILE" -n "$TRAIN_NAME" -d "$TRAIN_DESC" -x "$EXTRA_ARGS"
    exit 0
fi

HOURS=$((SLEEP_TIME / 3600))
MINS=$(((SLEEP_TIME % 3600) / 60))

echo "-----------------------------------------------------------------"
echo "SCHEDULED TRAINING"
echo "Waiting $HOURS hours and $MINS minutes (until $TARGET_DATE)..."
echo "Config: $CONFIG_FILE"
echo "Name  : $TRAIN_NAME"
echo "Desc  : $TRAIN_DESC"
echo "-----------------------------------------------------------------"

# Pause execution
sleep "$SLEEP_TIME"

echo "[$(date)] Time reached. Launching pipeline..."
$LAUNCH_SCRIPT -c "$CONFIG_FILE" -n "$TRAIN_NAME" -d "$TRAIN_DESC" -x "$EXTRA_ARGS"
