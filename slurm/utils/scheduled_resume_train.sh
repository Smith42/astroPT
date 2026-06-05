#!/bin/bash

# --- ASTROPT SCHEDULED RESUME (Job 187209) --- #
# Scheduled to run 4 hours from 2026-05-16 20:25

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

# --- CONFIGURATION --- #
TARGET_DATE="2026-06-04 10:45:00"
CONFIG_FILE="$REPO_ROOT/config/20260531_Hybrid_Optimized.yaml"

# IMPORTANT: Point to the same directory and set init_from to resume
TRAIN_DIR="/mnt/data_proj/iac18_mhuertas_shared/valonso/astroPT/logs/astropt_20260601_hybrid_optimized"

# RESUMING ARGUMENTS
EXTRA_ARGS="--init_from resume"


TARGET_TIME=$(date -d "$TARGET_DATE" +%s)
CURRENT_TIME=$(date +%s)
SLEEP_TIME=$((TARGET_TIME - CURRENT_TIME))

if [ "$SLEEP_TIME" -le 0 ]; then
    echo "[!] Target date reached. Launching..."
else
    HOURS=$((SLEEP_TIME / 3600))
    MINS=$(((SLEEP_TIME % 3600) / 60))
    echo "Waiting $HOURS hours and $MINS minutes until $TARGET_DATE..."
    sleep "$SLEEP_TIME"
fi

echo "Launching resume pipeline..."
$LAUNCH_SCRIPT -c "$CONFIG_FILE" -t "$TRAIN_DIR" -x "$EXTRA_ARGS"
