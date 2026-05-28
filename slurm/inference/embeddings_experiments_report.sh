#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Emb_Report
#SBATCH --partition=batch
#SBATCH --account=iac18
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_emb_report_%j.out
#SBATCH --error=logs/astropt_emb_report_%j.err

set -euo pipefail

# Robust repository root detection based on script location
# (Script is in slurm/trainers/, so REPO_ROOT is two levels up)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# If SLURM has moved us to /var/spool, correct REPO_ROOT to the launch working directory
if [[ "$REPO_ROOT" == "/var/spool"* ]]; then
    REPO_ROOT="$PWD"
fi

PYTHON_SCRIPT="$REPO_ROOT/scripts/embeddings_experiments_report.py"
PROBE="MLP"
TOP_K=12

# If invoked as: sbatch script.sh -- -e <path>
# consume the leading "--" before getopts.
if [[ "${1:-}" == "--" ]]; then
  shift

while getopts ":r:e:s:b:p:k:h" opt; do
  case $opt in
    e) EMB_ROOT="$OPTARG" ;;
    s) SAVE_DIR="$OPTARG" ;;
    b) BASELINE="$OPTARG" ;;
    p) PROBE="$OPTARG" ;;
    k) TOP_K="$OPTARG" ;;
    h)
      echo "Usage: $0 -e EMB_ROOT [-s SAVE_DIR] [-b BASELINE] [-p PROBE] [-k TOP_K] [-r REPO_ROOT]"
      echo "   or: $0 EMB_ROOT [-s SAVE_DIR] [-b BASELINE] [-p PROBE] [-k TOP_K] [-r REPO_ROOT]"
      exit 0
      ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

shift $((OPTIND - 1))

# Optional positional fallback to avoid conflicts with sbatch short options.
# Example: sbatch .../embeddings_experiments_report.sh /path/to/embeddings
if [[ -z "${EMB_ROOT:-}" && -n "${1:-}" ]]; then
  EMB_ROOT="$1"

if [[ -z "${EMB_ROOT:-}" ]]; then
  echo "[ERROR] EMB_ROOT is required"
  echo "Usage: $0 -e EMB_ROOT [-s SAVE_DIR] [-b BASELINE] [-p PROBE] [-k TOP_K] [-r REPO_ROOT]"
  echo "   or: $0 EMB_ROOT [-s SAVE_DIR] [-b BASELINE] [-p PROBE] [-k TOP_K] [-r REPO_ROOT]"
  exit 1

EMB_ROOT=$(readlink -f "$EMB_ROOT")
if [[ -z "${SAVE_DIR:-}" ]]; then
  SAVE_DIR="$EMB_ROOT/embeddings_report"
SAVE_DIR=$(readlink -m "$SAVE_DIR")

NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")
echo "-----------------------------------------------"
echo "Embeddings Experiments Report Job ${SLURM_JOB_ID:-local} - $NOW"
echo "-----------------------------------------------"
echo "REPO ROOT:  $REPO_ROOT"
echo "EMB ROOT:   $EMB_ROOT"
echo "SAVE DIR:   $SAVE_DIR"
echo "PROBE:      $PROBE"
echo "TOP K:      $TOP_K"

cd "$REPO_ROOT" || { echo "[ERROR] Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }
source "$REPO_ROOT/.venv/bin/activate""

CMD=(python "$PYTHON_SCRIPT"
  --emb_root "$EMB_ROOT"
  --save_dir "$SAVE_DIR"
  --probe "$PROBE"
  --top_k "$TOP_K")

if [[ -n "${BASELINE:-}" ]]; then
  CMD+=(--baseline "$BASELINE")

"${CMD[@]}"

echo "-----------------------------------------------"
echo "Embeddings report finished"
echo "-----------------------------------------------"
