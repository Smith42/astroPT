#!/bin/bash

#--- ASTROPT DIAGNOSTIC ABLATION LAUNCHER ---#
# Submits short/medium ablation batteries using only optional CLI flags.

REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
SCRIPT_DIR=$(dirname "$0")
TRAIN_SCRIPT="$SCRIPT_DIR/train_astropt_multiGPU.sh"
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"

PRESET="mini"
RUN_NAME="AstroPT Diagnostic"
RUN_DESC="Diagnostic ablation battery"
RUN_DATE="$(date +%Y%m%d)"
BASE_DIR="$REPO_ROOT/logs/astropt_diagnostic_ablation_${RUN_DATE}"

print_usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -r <path>    Repository root"
  echo "  -a <path>    Arrow data directory"
  echo "  -o <path>    Base output directory for all runs"
  echo "  -n <name>    Base run name"
  echo "  -d <desc>    Base run description"
  echo "  -p <preset>  Battery preset: mini | full (default: mini)"
  echo "  -h           Show help"
}

while getopts ":r:a:o:n:d:p:h" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    a) DATA_DIR="$OPTARG" ;;
    o) BASE_DIR="$OPTARG" ;;
    n) RUN_NAME="$OPTARG" ;;
    d) RUN_DESC="$OPTARG" ;;
    p) PRESET="$OPTARG" ;;
    h) print_usage; exit 0 ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; print_usage; exit 1 ;;
  esac
done

if [[ "$PRESET" != "mini" && "$PRESET" != "full" ]]; then
  echo "[ERROR] Invalid preset: $PRESET"
  echo "        Allowed: mini, full"
  exit 1

BASE_DIR=$(readlink -f "$BASE_DIR")
mkdir -p "$BASE_DIR"

submit_run() {
  local run_id="$1"
  local run_label="$2"
  local extra_args="$3"
  local dependency="$4"

  local run_dir="$BASE_DIR/$run_id"
  local run_name="$RUN_NAME [$run_label]"
  local run_desc="$RUN_DESC | $run_label"

  local cmd=(sbatch --parsable)
  if [[ -n "$dependency" ]]; then
    cmd+=(--dependency="afterany:$dependency")

  cmd+=(
    --job-name="TrainDiag_${run_id}"
    "$TRAIN_SCRIPT"
    -r "$REPO_ROOT"
    -t "$run_dir"
    -a "$DATA_DIR"
    -n "$run_name"
    -d "$run_desc"
    -m "scratch"
    -k "both"
    -x "$extra_args"
  )

  local job_id
  job_id=$("${cmd[@]}")

  if [[ -z "$job_id" ]]; then
    echo "[ERROR] Failed to submit run: $run_id" >&2
    return 1

  echo "[OK] $run_id -> Job $job_id" >&2
  echo "     extra args: $extra_args" >&2
  echo "     output dir: $run_dir" >&2
  echo "$job_id"
}

echo "-------------------------------------------------"
echo "Launching AstroPT diagnostic ablation battery"
echo "Preset:      $PRESET"
echo "Repo root:   $REPO_ROOT"
echo "Data dir:    $DATA_DIR"
echo "Output root: $BASE_DIR"
echo "-------------------------------------------------"

# Common diagnostics flags for all runs in this battery
COMMON_DIAG="--diagnostics_enabled --diagnostics_interval 100"

# Run A: Baseline diagnostics only
JOB_A=$(submit_run \
  "A_baseline" \
  "Baseline" \
  "$COMMON_DIAG" \
  "") || exit 1

# Run B: Modality dropout stress test
JOB_B=$(submit_run \
  "B_dropout_random_p015" \
  "Dropout Random 0.15" \
  "$COMMON_DIAG --modality_dropout_prob 0.15 --modality_dropout_mode random" \
  "$JOB_A") || exit 1

# Run D: Reweight + branch LR multipliers
JOB_D=$(submit_run \
  "D_reweight_lrbranch" \
  "Reweight + LR Branch" \
  "$COMMON_DIAG --images_loss_weight 0.8 --spectra_loss_weight 1.2 --lr_mult_images 0.85 --lr_mult_spectra 1.15" \
  "$JOB_B") || exit 1

if [[ "$PRESET" == "full" ]]; then
  # Run E: Stronger dropout focused on images
  JOB_E=$(submit_run \
    "E_dropout_images_p030" \
    "Dropout Images 0.30" \
    "$COMMON_DIAG --modality_dropout_prob 0.30 --modality_dropout_mode images" \
    "$JOB_D") || exit 1

  # Run F: Stronger dropout focused on spectra + backbone LR boost
  JOB_F=$(submit_run \
    "F_dropout_spectra_p030" \
    "Dropout Spectra 0.30" \
    "$COMMON_DIAG --modality_dropout_prob 0.30 --modality_dropout_mode spectra" \
    "$JOB_E") || exit 1

echo "-------------------------------------------------"
echo "Submission complete. Queue snapshot:"
squeue -u "$USER" -S i -o "%.10i %.6P %.25j %.8T %.10M %.35E %R"
echo "-------------------------------------------------"
