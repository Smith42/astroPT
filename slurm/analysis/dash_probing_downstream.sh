#!/bin/bash

#--- SLURM option configuration ---#
#SBATCH --job-name=Probing_Tasks
#SBATCH --partition=gpu
#SBATCH --account=iac18
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=2       
#SBATCH --gpus-per-task=1        
#SBATCH --mem=32G                
#SBATCH --time=00:10:00         

#--- LOGS FILES ---#
#SBATCH --output=logs/astropt_probing_dash_%j.out
#SBATCH --error=logs/astropt_probing_dash_%j.err

set -euo pipefail

# Robust repository root detection based on script location
# (Script is in slurm/trainers/, so REPO_ROOT is two levels up)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# If SLURM has moved us to /var/spool, correct REPO_ROOT to the launch working directory
if [[ "$REPO_ROOT" == "/var/spool"* ]]; then
    if [ -d "$PWD/astroPT" ]; then
        REPO_ROOT="$PWD/astroPT"
    else
        REPO_ROOT="$PWD"
    fi
fi

PYTHON_SCRIPT="$REPO_ROOT/scripts/analysis/dashboards/dash_probing_downstream.py"
INTERACTIVE_SCRIPT="$REPO_ROOT/scripts/analysis/dashboards/dash_probing_downstream_interactive.py"
ALL_TARGETS_MODE=0
INTERACTIVE_MODE=0
STREAMLIT_PORT=8501

# If invoked as: sbatch script.sh -- -e <emb_root> ...
if [[ "${1:-}" == "--" ]]; then
  shift
fi

#--- ARGUMENT PARSING (FLAGS) ---#
while getopts ":r:e:s:n:Ap:ih" opt; do
  case $opt in
    r) REPO_ROOT="$OPTARG" ;;
    s) SAVE_DIR="$OPTARG" ;;
    e) EMB_DIR="$OPTARG" ;;
    n) SAVE_NAME="$OPTARG" ;;
    A) ALL_TARGETS_MODE=1 ;;
    p) STREAMLIT_PORT="$OPTARG" ;;
    i) INTERACTIVE_MODE=1 ;;
    h)
      echo "Usage: $0 -e EMB_DIR [-s SAVE_DIR] [-n SAVE_NAME] [-A] [-r REPO_ROOT] [-i] [-p PORT]"
      echo "  -A: Plot all downstream targets and compare all runs found in EMB_DIR/*/downstream_tasks/downstream_results.csv"
      echo "  -i: Launch interactive Streamlit Dashboard instead of static Matplotlib PNG generation"
      echo "  -p: Specify Streamlit Server port (default: 8501, only used if -i is enabled)"
      exit 0
      ;;
    \?) echo "[ERROR] Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

shift $((OPTIND - 1))

if [[ -z "${EMB_DIR:-}" ]]; then
    echo "[ERROR]: EMB_DIR is required (-e <embeddings_root>)"
    exit 1
fi

EMB_DIR=$(readlink -f "$EMB_DIR")

#--- ENVIRONMENT SETUP ---#
NOW=$(date "+[%Y-%m-%d - %H:%M:%S]")

echo "--------------------------------------------------"
echo "Starting Probing Tasks Dashboard Job ${SLURM_JOB_ID:-local} - $NOW"
echo "--------------------------------------------------"

# 1. Change directory
echo "Changing directory to: $REPO_ROOT"
cd "$REPO_ROOT" || { echo "[ERROR]: Cannot find REPO_ROOT: $REPO_ROOT"; exit 1; }

# 2. Activate Environment
source "$REPO_ROOT/.venv/bin/activate"

# 3. Activating LaTeX (Required for confusion matrix plots)
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
export MPLCONFIGDIR="/home/valonso/iac18_mhuertas_shared/valonso/cache/matplotlib"
export XDG_CACHE_HOME="/home/valonso/iac18_mhuertas_shared/valonso/cache"

# SAVE_DIR is now optional; defaults to EMB_DIR/downstream_tasks if not provided
if [ -n "${SAVE_DIR:-}" ]; then
    SAVE_DIR=$(readlink -f "$SAVE_DIR")
    SAVE_ARG="--save_dir $SAVE_DIR"
else
    # Default to current embedding folder's downstream_tasks
    SAVE_ARG="--save_dir $EMB_DIR"
fi

SAVE_NAME_ARGS=()
if [[ -n "${SAVE_NAME:-}" ]]; then
  SAVE_NAME_ARGS=(--save_name "$SAVE_NAME")
fi

#--- EXECUTION ---#
echo "Probing Dashboard Configuration:"
echo "    EMB ROOT:       $EMB_DIR"
echo "    ALL TARGETS:    $ALL_TARGETS_MODE"
echo "    INTERACTIVE:    $INTERACTIVE_MODE"

# --- COMPARATIVE CONFIGURATION ---
declare -A BASELINE_MAP
LOGS_BASE="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs"

BASELINE_MAP["${LOGS_BASE}/astropt_100M_250K_arrow_20260408_baseline_images"]="AstroPT Baseline (Images)"
BASELINE_MAP["${LOGS_BASE}/supervised_baseline_images_filter/supervised_baseline_images_results.csv"]="Images Supervised Filter"
BASELINE_MAP["${LOGS_BASE}/supervised_baseline_images/supervised_baseline_images_results.csv"]="Images Supervised"
BASELINE_MAP["${LOGS_BASE}/astropt_100M_250K_arrow_20260409_baseline_spectra"]="AstroPT Baseline (Spectra)"
BASELINE_MAP["${LOGS_BASE}/supervised_baseline_spectra_filter/supervised_baseline_spectra_results.csv"]="Spectra Supervised Filter"
BASELINE_MAP["${LOGS_BASE}/supervised_baseline_spectra/supervised_baseline_spectra_results.csv"]="Spectra Supervised"
BASELINE_MAP["${LOGS_BASE}/astroclip_20260422_matchingastropt/embeddings/astroclip-step0031000-valloss0.6454"]="AstroCLIP"
BASELINE_MAP["${LOGS_BASE}/AION_freeze/embeddings/aion_embeddings"]="AION"
BASELINE_MAP["${LOGS_BASE}/astropt_100M_250K_arrow_20260411_tokmixran_crossrecloss_longblocks"]="AstroPT TokMixRan"
BASELINE_MAP["${LOGS_BASE}/astropt_100M_250K_arrow_20260414_tokmix16_crossrecloss_dsinter_smallmod"]="AstroPT TokMix16 30M"
BASELINE_MAP["${LOGS_BASE}/astropt_100M_250K_arrow_20260513_VISOnly"]="VIS Only"
BASELINE_MAP["${LOGS_BASE}/astropt_20260516_hybrid_cliploss/embeddings/best_img-mean_spec-rank_final_iso_j-mean/downstream_tasks_V1/downstream_results.csv"]="Hybrid (RAW)"
BASELINE_MAP["${LOGS_BASE}/astropt_20260516_hybrid_cliploss/embeddings/best_img-mean_spec-rank_final_iso_j-mean/downstream_tasks/downstream_results.csv"]="Hybrid (Filter Prob)"
BASELINE_MAP["${LOGS_BASE}/astropt_20260519_hybrid_filter/embeddings/best_img-mean_spec-rank_final_iso_j-mean_T2/downstream_tasks/downstream_results.csv"]="Hybrid + Filter"

# Collect all paths and names for the comparative dashboard
CSV_PATHS=()
CSV_NAMES=()

# 1. Add current run (priority)
CURRENT_CSV=$(find "$EMB_DIR" -maxdepth 2 -type f -name "downstream_results.csv" | head -n 1)
if [[ -n "$CURRENT_CSV" ]]; then
    CURRENT_NAME=$(basename "$(echo "$EMB_DIR" | sed 's|/embeddings.*||')")
    CSV_PATHS+=("$CURRENT_CSV")
    CSV_NAMES+=("Current: $CURRENT_NAME")
fi

# 2. Add baselines from dictionary
for run_path in "${!BASELINE_MAP[@]}"; do
    # Skip if it's the same as current
    if [[ "$run_path" == *"$EMB_DIR"* ]]; then continue; fi
    
    # Find the best CSV for this baseline
    # Try downstream_results.csv directly, or search in embeddings
    found_csv=""
    if [[ -f "$run_path" ]]; then
        found_csv="$run_path"
    elif [[ -f "${run_path}/downstream_results.csv" ]]; then
        found_csv="${run_path}/downstream_results.csv"
    elif [[ -f "${run_path}/downstream_tasks/downstream_results.csv" ]]; then
        found_csv="${run_path}/downstream_tasks/downstream_results.csv"
    elif [[ -d "$run_path" ]]; then
        found_csv=$(find "$run_path" -maxdepth 4 -type f -name "downstream_results.csv" | head -n 1 || true)
    fi
    
    if [[ -n "$found_csv" ]]; then
        CSV_PATHS+=("$found_csv")
        CSV_NAMES+=("${BASELINE_MAP[$run_path]}")
    fi
done

if [[ "$INTERACTIVE_MODE" -eq 1 ]]; then
    echo "    Launching Interactive Streamlit Dashboard on port $STREAMLIT_PORT..."
    if [[ "$ALL_TARGETS_MODE" -eq 1 ]]; then
        echo "    Mode A (Intra-run): Plotting multiple internal embeddings under $EMB_DIR..."
        streamlit run "$INTERACTIVE_SCRIPT" --server.port "$STREAMLIT_PORT" --server.address 0.0.0.0 -- \
            --run_dirs "$EMB_DIR"
    else
        echo "    Mode B (Comparative): Plotting current run against baselines..."
        streamlit run "$INTERACTIVE_SCRIPT" --server.port "$STREAMLIT_PORT" --server.address 0.0.0.0 -- \
            --csv_path "${CSV_PATHS[@]}" \
            --names "${CSV_NAMES[@]}"
    fi
else
    # --- STATIC MODE ---
    if [[ "$ALL_TARGETS_MODE" -eq 1 ]]; then
        echo "    Mode A (Intra-run): Analyzing multiple internal embeddings under $EMB_DIR..."
        python3 "$PYTHON_SCRIPT" \
            $SAVE_ARG \
            --run_dirs "$EMB_DIR" \
            --all_targets \
            "${SAVE_NAME_ARGS[@]}"
    fi

    # Always run the comparative dashboard against baselines in static mode
    echo "    Mode B (Comparative): Generating Global Comparative Dashboard against ${#CSV_PATHS[@]} runs..."
    python3 "$PYTHON_SCRIPT" \
        $SAVE_ARG \
        --csv_path "${CSV_PATHS[@]}" \
        --names "${CSV_NAMES[@]}" \
        --all_targets \
        "${SAVE_NAME_ARGS[@]}"
fi

echo "-----------------------------------------------"
echo "Probing Tasks Dashboard Finished"
echo "-----------------------------------------------"
