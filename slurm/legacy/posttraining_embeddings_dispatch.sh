#!/bin/bash

set -euo pipefail

#--- SLURM option configuration ---#
#SBATCH --job-name=PostEmb_Dispatch
#SBATCH --partition=batch
#SBATCH --account=iac18
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_postemb_dispatch_%j.out
#SBATCH --error=logs/astropt_postemb_dispatch_%j.err

#--- DEFAULT VALUES ---#
# Automatically detect the repository root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"
META_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits"

COS_SIM_SCRIPT=""
UMAPS_SCRIPT=""
PROBING_SCRIPT=""
REPORT_SCRIPT=""

while getopts ":r:w:e:a:f:" opt; do
  case $opt in
    w) WEIGHTS_DIR="$OPTARG" ;;
    e) EMB_ROOT="$OPTARG" ;;
    a) DATA_DIR="$OPTARG" ;;
    f) META_PATH="$OPTARG" ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

if [[ -z "${WEIGHTS_DIR:-}" || -z "${EMB_ROOT:-}" ]]; then
  echo "[ERROR] WEIGHTS_DIR and EMB_ROOT are required"
  echo "Usage: $0 -w <weights_dir> -e <embeddings_root> [-r repo_root] [-a data_dir] [-f metadata_path]"
  exit 1

WEIGHTS_DIR=$(readlink -f "$WEIGHTS_DIR")
EMB_ROOT=$(readlink -f "$EMB_ROOT")
DATA_DIR=$(readlink -f "$DATA_DIR")
META_PATH=$(readlink -f "$META_PATH")

COS_SIM_SCRIPT="$REPO_ROOT/slurm/cosine_similarity.sh"
UMAPS_SCRIPT="$REPO_ROOT/slurm/plot_umaps.sh"
PROBING_SCRIPT="$REPO_ROOT/slurm/probing_downstream.sh"
REPORT_SCRIPT="$REPO_ROOT/slurm/embeddings_experiments_report.sh"

if [[ ! -d "$EMB_ROOT" ]]; then
  echo "[ERROR] Embeddings root not found: $EMB_ROOT"
  exit 1

NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")
JOB_ID="${SLURM_JOB_ID:-local}"
echo "-----------------------------------------------"
echo "Posttraining Embeddings Dispatcher $JOB_ID - $NOW"
echo "-----------------------------------------------"
echo "REPO ROOT:      $REPO_ROOT"
echo "WEIGHTS DIR:    $WEIGHTS_DIR"
echo "EMB ROOT:       $EMB_ROOT"
echo "DATA DIR:       $DATA_DIR"
echo "META PATH:      $META_PATH"

submit_with_retry() {
  local label="$1"
  shift

  while true; do
    local out=""
    if out=$(sbatch --parsable "$@" 2>&1); then
      echo "$out"
      return 0
    echo "[WARN] submit failed for $label: $out"
    echo "[WARN] retrying in 60s..."
    sleep 60
  done
}

mapfile -t CANDIDATE_DIRS < <(find "$EMB_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
VALID_DIRS=()

for d in "${CANDIDATE_DIRS[@]}"; do
  if [[ -f "$d/ids.npy" ]] && [[ -f "$d/images.npy" || -f "$d/spectra.npy" || -f "$d/joint.npy" || -f "$d/embeddings_all.npz" ]]; then
    VALID_DIRS+=("$d")
done

if [[ ${#VALID_DIRS[@]} -eq 0 ]]; then
  echo "[ERROR] No valid embedding experiment directories found in $EMB_ROOT"
  exit 1

echo "Found ${#VALID_DIRS[@]} embedding experiment directories"

ALL_ANALYSIS_JOBS=()

for emb_dir in "${VALID_DIRS[@]}"; do
  tag=$(basename "$emb_dir" | tr -cd '[:alnum:]_-' | cut -c1-48)

  jid_cos=$(submit_with_retry "Cos_${tag}" \
    --job-name="Cos_${tag}" \
    "$COS_SIM_SCRIPT" \
    -r "$REPO_ROOT" \
    -w "$WEIGHTS_DIR" \
    -e "$emb_dir" \
    -s "$emb_dir")

  jid_umap=$(submit_with_retry "UMAP_${tag}" \
    --job-name="UMAP_${tag}" \
    "$UMAPS_SCRIPT" \
    -r "$REPO_ROOT" \
    -w "$WEIGHTS_DIR" \
    -e "$emb_dir" \
    -s "$emb_dir" \
    -a "$DATA_DIR" \
    -f "$META_PATH")

  jid_probe=$(submit_with_retry "Probe_${tag}" \
    --job-name="Probe_${tag}" \
    "$PROBING_SCRIPT" \
    -r "$REPO_ROOT" \
    -w "$WEIGHTS_DIR" \
    -e "$emb_dir" \
    -s "$emb_dir" \
    -f "$META_PATH")

  echo "[ENQUEUED] $(basename "$emb_dir")"
  echo "    Cosine:  $jid_cos"
  echo "    UMAPS:   $jid_umap"
  echo "    Probing: $jid_probe"

  ALL_ANALYSIS_JOBS+=("$jid_cos" "$jid_umap" "$jid_probe")
done

if [[ ${#ALL_ANALYSIS_JOBS[@]} -gt 0 && -f "$REPORT_SCRIPT" ]]; then
  dep_chain=$(IFS=:; echo "${ALL_ANALYSIS_JOBS[*]}")
  jid_report=$(submit_with_retry "Emb_Report" \
    --dependency="afterok:$dep_chain" \
    --job-name="Emb_Report" \
    "$REPORT_SCRIPT" \
    -r "$REPO_ROOT" \
    -e "$EMB_ROOT")
  echo "[ENQUEUED] Global report job: $jid_report"

echo "-----------------------------------------------"
echo "Dispatch completed successfully"
echo "-----------------------------------------------"
