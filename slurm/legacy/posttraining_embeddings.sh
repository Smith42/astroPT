#!/bin/bash

set -euo pipefail

# Post-training launcher for embedding sweeps + automatic downstream analysis.

REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"
META_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits"
PRESETS="quick,orders,fusion"

print_usage() {
  echo "Usage: $0 -w WEIGHTS_DIR [-e EMB_ROOT] [options]"
  echo "Required:"
  echo "  -w <path>    Weights directory"
  echo "Optional:"
  echo "  -e <path>    Embeddings root directory (default: <weights_parent>/embeddings)"
  echo "  -r <path>    Repository root"
  echo "  -a <path>    Arrow data directory"
  echo "  -f <path>    Metadata FITS path"
  echo "  -p <list>    Comma-separated presets (default: quick,orders,fusion)"
  echo "  -h           Help"
}

while getopts ":r:w:e:a:f:p:h" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    w) WEIGHTS_DIR="$OPTARG" ;;
    e) EMB_ROOT="$OPTARG" ;;
    a) DATA_DIR="$OPTARG" ;;
    f) META_PATH="$OPTARG" ;;
    p) PRESETS="$OPTARG" ;;
    h) print_usage; exit 0 ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

if [[ -z "${WEIGHTS_DIR:-}" ]]; then
  echo "[ERROR] WEIGHTS_DIR is required"
  print_usage
  exit 1

WEIGHTS_DIR=$(readlink -f "$WEIGHTS_DIR")
DATA_DIR=$(readlink -f "$DATA_DIR")
META_PATH=$(readlink -f "$META_PATH")

if [[ -z "${EMB_ROOT:-}" ]]; then
  TRAIN_DIR=$(dirname "$WEIGHTS_DIR")
  EMB_ROOT="$TRAIN_DIR/embeddings"
EMB_ROOT=$(readlink -m "$EMB_ROOT")
mkdir -p "$EMB_ROOT"

EXTRACT_EXP_SCRIPT="$REPO_ROOT/slurm/extract_embeddings_experimental.sh"
DISPATCH_SCRIPT="$REPO_ROOT/slurm/posttraining_embeddings_dispatch.sh"

if [[ ! -f "$EXTRACT_EXP_SCRIPT" ]]; then
  echo "[ERROR] Missing script: $EXTRACT_EXP_SCRIPT"
  exit 1

if [[ ! -f "$DISPATCH_SCRIPT" ]]; then
  echo "[ERROR] Missing script: $DISPATCH_SCRIPT"
  exit 1

IFS=',' read -r -a PRESET_ARRAY <<< "$PRESETS"
if [[ ${#PRESET_ARRAY[@]} -eq 0 ]]; then
  echo "[ERROR] Empty preset list"
  exit 1

NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")
echo "-------------------------------------------------"
echo "Posttraining Embeddings Pipeline Launcher - $NOW"
echo "-------------------------------------------------"
echo "REPO ROOT:      $REPO_ROOT"
echo "WEIGHTS DIR:    $WEIGHTS_DIR"
echo "EMB ROOT:       $EMB_ROOT"
echo "DATA DIR:       $DATA_DIR"
echo "META PATH:      $META_PATH"
echo "PRESETS:        ${PRESET_ARRAY[*]}"
echo "-------------------------------------------------"

prev_job=""
for preset in "${PRESET_ARRAY[@]}"; do
  preset_trimmed=$(echo "$preset" | xargs)
  if [[ -z "$preset_trimmed" ]]; then
    continue

  dep_args=()
  if [[ -n "$prev_job" ]]; then
    dep_args+=(--dependency="afterok:$prev_job")

  jid=$(sbatch --parsable \
    "${dep_args[@]}" \
    --job-name="ExtractExp_${preset_trimmed}" \
    "$EXTRACT_EXP_SCRIPT" \
    -r "$REPO_ROOT" \
    -w "$WEIGHTS_DIR" \
    -s "$EMB_ROOT" \
    -a "$DATA_DIR" \
    -b "$preset_trimmed")

  echo "[ENQUEUED] Extraction preset '$preset_trimmed' -> JobID: $jid"
  prev_job="$jid"
done

if [[ -z "$prev_job" ]]; then
  echo "[ERROR] No extraction jobs were submitted"
  exit 1

dispatch_jid=$(sbatch --parsable \
  --dependency="afterok:$prev_job" \
  --job-name="PostEmb_Dispatch" \
  "$DISPATCH_SCRIPT" \
  -r "$REPO_ROOT" \
  -w "$WEIGHTS_DIR" \
  -e "$EMB_ROOT" \
  -a "$DATA_DIR" \
  -f "$META_PATH")

echo "[ENQUEUED] Dispatcher -> JobID: $dispatch_jid (after extraction chain)"
echo "-------------------------------------------------"
echo "Pipeline sent successfully"
echo "Track: squeue -j $dispatch_jid"
echo "-------------------------------------------------"
