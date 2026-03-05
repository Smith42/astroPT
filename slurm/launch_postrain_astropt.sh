#!/bin/bash

#--- ASTROPT PIPELINE LAUNCHER (OPTIMIZED) ---#
echo "-------------------------------------------------"
echo "Launching Full AstroPT Pipeline (Modular)"
echo "-------------------------------------------------"


# PATH AND DIR CONFIGURATION
REPO_ROOT="/home/valonso/iac18_mhuertas_shared/valonso/astroPT"
SCRIPT_DIR=$(dirname "$0")
DATA_DIR="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"
META_PATH="/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits"

# LAUNCHING SCRIPTS
TRAIN_SCRIPT="$SCRIPT_DIR/train_astropt_multiGPU.sh"
PLOT_MET_SCRIPT="$SCRIPT_DIR/plot_training_metrics.sh"
PLOT_IMG_SCRIPT="$SCRIPT_DIR/plot_images_spectra.sh"
EXT_EMBD_SCRIPT="$SCRIPT_DIR/extract_embeddings.sh"
COS_SIM_SCRIPT="$SCRIPT_DIR/cosine_similarity.sh"
UMAPS_SCRIPT="$SCRIPT_DIR/plot_umaps.sh"
PROBING_SCRIPT="$SCRIPT_DIR/probing_downstream.sh"
PROBING_DASH_SCRIPT="$SCRIPT_DIR/probing_downstream_dashboard.sh"
WORKFLOW_SCRIPT="$SCRIPT_DIR/workflow_controller.sh"

# ARGUMENTS DEFAULT VALUES
TRAIN_NAME="New Train"              # Training name
TRAIN_DESC="New AstroPT Training"   # Training description

while getopts ":n:d:t:h" opt; do
  case $opt in
    n) TRAIN_NAME="$OPTARG" ;;
    d) TRAIN_DESC="$OPTARG" ;;
    t) TRAIN_DIR="$OPTARG" ;;
    h) 
       echo "Usage: $0 [-n 'Name'] [-d 'Description'] [-o 'output_path/']"
       exit 0 
       ;;
    \?) 
       echo "Invalid option: -$OPTARG" >&2
       exit 1 
       ;;
  esac
done

# AUTOMATIC CONFIGURATION
SUFIX_NAME=$(echo "$TRAIN_NAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | sed 's/[^a-z0-9_]//g')
TRAIN_DATE="$(date +%Y%m%d)"
DEFAULT_PATH="$REPO_ROOT/logs/astropt_100M_250K_arrow_${TRAIN_DATE}_${SUFIX_NAME}"

if [ -z "$TRAIN_DIR" ]; then
    TRAIN_DIR="$DEFAULT_PATH"
fi

# Absolute output path
TRAIN_DIR=$(readlink -f "$TRAIN_DIR")
WEIGHTS_DIR="$TRAIN_DIR/weights"
LOGS_DIR="$TRAIN_DIR/logs"
PLOTS_DIR="$TRAIN_DIR/plots"
EMB_DIR="$TRAIN_DIR/embeddings"


echo "Analysis Directories:"
echo "      TRAIN DIR:      $TRAIN_DIR"      
echo "      WEIGHTS DIR:    $WEIGHTS_DIR"
echo "      LOGS DIR:       $LOGS_DIR"
echo "      PLOTS DIR:      $PLOTS_DIR"
echo "      EMB DIR:        $EMB_DIR"
echo "-------------------------------------------------"

# LAUNCH ANALYSIS FUNCTION
launch_analysis() {   

    echo " --> Launching Analysis Pipeline"

    # Plotting Metrics
    local J_MET=$(sbatch --parsable \
                --job-name="Plot_Metrics" \
                "$PLOT_MET_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -l "$LOGS_DIR" \
                -s "$PLOTS_DIR"
            )
    echo "    [Metric]  Job sent.       ID: $J_MET (Depends on Train: any)"

    # Plotting Images
    local J_IMG=$(sbatch --parsable \
                --job-name="Plot_Im_Sp" \
                "$PLOT_IMG_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -s "$PLOTS_DIR" \
                -a "$DATA_DIR"
            )
    echo "    [ImgSpec] Job sent.       ID: $J_IMG (Depends on Train: any)"

    # Extract Embeddings 
    local J_EMB=$(sbatch --parsable \
                --job-name="Extract_Embed" \
                "$EXT_EMBD_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -s "$EMB_DIR" \
                -a "$DATA_DIR"
            )
    echo "    [Embeds]  Job sent.       ID: $J_EMB (Depends on Train: any)"

    # Cosine Similarity
    local J_COS=$(sbatch --parsable \
                --dependency=afterok:$J_EMB \
                --job-name="Cos_Sim" \
                "$COS_SIM_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -e "$EMB_DIR"
            )
    echo "    [CosSim]  Job sent.       ID: $J_COS (Depends on Embeds: ok)"

    # UMAPS
    local J_UMAP=$(sbatch --parsable \
                --dependency=afterok:$J_EMB \
                --job-name="Plot_Umaps" \
                "$UMAPS_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -e "$EMB_DIR" \
                -a "$DATA_DIR" \
                -f "$META_PATH"
            )
    echo "    [UMAPS]   Job sent.       ID: $J_UMAP (Depends on Embeds: ok)"

    # PROBING DOWNSTREAM TASKS
    local J_PROB=$(sbatch --parsable \
                --dependency=afterok:$J_EMB \
                --job-name="Probing_Tasks" \
                "$PROBING_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -e "$EMB_DIR" \
                -f "$META_PATH"
            )
    echo "    [PROBING] Job sent.       ID: $J_PROB (Depends on Embeds: ok)"

    # PROBING DOWNSTREAM TASKS DASHBOARD
    local J_PROB_DASH=$(sbatch --parsable \
                --dependency=afterok:$J_PROB \
                --job-name="Probing_Tasks_Dash" \
                "$PROBING_DASH_SCRIPT" \
                -r "$REPO_ROOT" \
                -e "$EMB_DIR"
            )
    echo "    [PROB DASH] Job sent.     ID: $J_PROB_DASH (Depends on Probing: ok)"
    
    echo " --> Analysis Batch Sent."
}

# EXECUTION
launch_analysis

# QUEUE STATUS
echo "-------------------------------------------------"
echo "Waiting for resources..."
sleep 15
echo "Pipeline Status Queue:"
squeue -u $USER -S i -o "%.10i %.6P %.25j %.8T %.10M %.35E %R"
echo "-------------------------------------------------"