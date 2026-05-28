#!/bin/bash

set -u

REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
SLEEP_SECS=60

while getopts ":r:w:e:a:f:d:s:" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    w) WEIGHTS_DIR="$OPTARG" ;;
    e) EMB_ROOT="$OPTARG" ;;
    a) DATA_DIR="$OPTARG" ;;
    f) META_PATH="$OPTARG" ;;
    d) DEP_JOB="$OPTARG" ;;
    s) SLEEP_SECS="$OPTARG" ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

if [[ -z "${WEIGHTS_DIR:-}" || -z "${EMB_ROOT:-}" || -z "${DATA_DIR:-}" || -z "${META_PATH:-}" || -z "${DEP_JOB:-}" ]]; then
  echo "Usage: $0 -w WEIGHTS_DIR -e EMB_ROOT -a DATA_DIR -f META_PATH -d DEP_JOB [-r REPO_ROOT] [-s SLEEP_SECS]"
  exit 1
fi

WEIGHTS_DIR=$(readlink -f "$WEIGHTS_DIR")
EMB_ROOT=$(readlink -f "$EMB_ROOT")
DATA_DIR=$(readlink -f "$DATA_DIR")
META_PATH=$(readlink -f "$META_PATH")

DISPATCH_SCRIPT="$REPO_ROOT/slurm/posttraining_embeddings_dispatch.sh"

if [[ ! -f "$DISPATCH_SCRIPT" ]]; then
  echo "[ERROR] Missing dispatcher script: $DISPATCH_SCRIPT"
  exit 1
fi

echo "[$(date '+%F %T')] Retry loop started. Waiting for scheduler slot..."

while true; do
  out=$(sbatch --parsable --dependency=afterok:"$DEP_JOB" "$DISPATCH_SCRIPT" -w "$WEIGHTS_DIR" -e "$EMB_ROOT" -a "$DATA_DIR" -f "$META_PATH" 2>&1)
  if [[ "$out" =~ ^[0-9]+$ ]]; then
    echo "[$(date '+%F %T')] SUBMITTED dispatcher job $out"
    exit 0
  fi
  echo "[$(date '+%F %T')] RETRY ($SLEEP_SECS s): $out"
  sleep "$SLEEP_SECS"
done
