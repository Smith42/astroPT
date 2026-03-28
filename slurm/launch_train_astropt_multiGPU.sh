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
CROSS_REC_SCRIPT="$SCRIPT_DIR/cross_reconstruction.sh"
EXT_EMBD_SCRIPT="$SCRIPT_DIR/extract_embeddings.sh"
COS_SIM_SCRIPT="$SCRIPT_DIR/cosine_similarity.sh"
UMAPS_SCRIPT="$SCRIPT_DIR/plot_umaps.sh"
PROBING_SCRIPT="$SCRIPT_DIR/probing_downstream.sh"
PROBING_DASH_SCRIPT="$SCRIPT_DIR/probing_downstream_dashboard.sh"
LATENT_SCRIPT="$SCRIPT_DIR/probing_latent_mapper.sh"
LATENT_DASH_SCRIPT="$SCRIPT_DIR/probing_latent_dashboard.sh"
WORKFLOW_SCRIPT="$SCRIPT_DIR/workflow_controller.sh"

# ARGUMENTS DEFAULT VALUES
TRAIN_NAME="New Train"              # Training name
TRAIN_DESC="New AstroPT Training"   # Training description

while getopts ":n:d:t:l:h" opt; do
  case $opt in
    n) TRAIN_NAME="$OPTARG" ;;
    d) TRAIN_DESC="$OPTARG" ;;
    t) TRAIN_DIR="$OPTARG" ;;
    l) LONG_TRAINING="$OPTARG" ;;
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
SUFIX_NAME=$(echo "$TRAIN_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/ /g' | xargs | tr ' ' '_')
TRAIN_DATE="$(date +%Y%m%d)"
DEFAULT_PATH="$REPO_ROOT/logs/astropt_100M_250K_arrow_${TRAIN_DATE}_${SUFIX_NAME}"

if [ -z "$TRAIN_DIR" ]; then
    TRAIN_DIR="$DEFAULT_PATH"
fi

# Absolute output paths
TRAIN_DIR=$(readlink -f "$TRAIN_DIR")
WEIGHTS_DIR="$TRAIN_DIR/weights"
LOGS_DIR="$TRAIN_DIR/logs"
PLOTS_DIR="$TRAIN_DIR/plots"
EMB_DIR="$TRAIN_DIR/embeddings"

echo "Target Training Directory: $TRAIN_DIR"

# LAUNCH ANALYSIS FUNCTION
launch_analysis() {
    local PARENT_JOB_ID=$1
    local JOB_SUFFIX=$2  
    local STEP_NAME=$3      

    echo " --> Launching Analysis Battery for: $STEP_NAME (Parent: $PARENT_JOB_ID)"

    # Plotting Metrics
    local J_MET=$(sbatch --parsable \
                --dependency=afterany:$PARENT_JOB_ID \
                --job-name="Plot_Metrics$JOB_SUFFIX" \
                "$PLOT_MET_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -l "$LOGS_DIR" \
                -s "$PLOTS_DIR"
            )
    echo "    [Metric]  Job sent.       ID: $J_MET (Depends on Train: any)"

    # Workflow controller
    J_WOR=$(sbatch --parsable \
                --dependency=afterany:$PARENT_JOB_ID \
                --job-name="Workflow_Controller$JOB_SUFFIX" \
                "$WORKFLOW_SCRIPT" \
                -r "$REPO_ROOT" \
                -t "$TRAIN_DIR" \
                -x "$JOB_SUFFIX"
            )
    echo "    [Work]    Job sent.       ID: $J_WOR (Depends on Train: any)"

    # Plotting Images and Spectra
    local J_IMG=$(sbatch --parsable \
                --dependency=afterok:$J_WOR \
                --job-name="Plot_Im_Sp$JOB_SUFFIX" \
                "$PLOT_IMG_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -s "$PLOTS_DIR" \
                -a "$DATA_DIR"
            )
    echo "    [ImgSpec] Job sent.       ID: $J_IMG (Depends on Train: any)"

    # Cross Reconstruction
    local J_CROSS=$(sbatch --parsable \
                --dependency=afterok:$J_WOR \
                --job-name="Cross_Rec$JOB_SUFFIX" \
                "$CROSS_REC_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -s "$PLOTS_DIR" \
                -a "$DATA_DIR"
            )
    echo "    [CrossRec] Job sent.      ID: $J_CROSS (Depends on Train: any)"

    # Extract Embeddings 
    local J_EMB=$(sbatch --parsable \
                --dependency=afterok:$J_WOR \
                --job-name="Extract_Embed$JOB_SUFFIX" \
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
                --job-name="Cos_Sim$JOB_SUFFIX" \
                "$COS_SIM_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -e "$EMB_DIR"
            )
    echo "    [CosSim]  Job sent.       ID: $J_COS (Depends on Embeds: ok)"

    # UMAPS
    local J_UMAP=$(sbatch --parsable \
                --dependency=afterok:$J_EMB \
                --job-name="Plot_Umaps$JOB_SUFFIX" \
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
                --job-name="Probing_Tasks$JOB_SUFFIX" \
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
                --job-name="Probing_Tasks_Dash$JOB_SUFFIX" \
                "$PROBING_DASH_SCRIPT" \
                -r "$REPO_ROOT" \
                -e "$EMB_DIR"
            )
    echo "    [PROB DASH] Job sent.     ID: $J_PROB_DASH (Depends on Probing: ok)"

    # PROBING LATENT MAPPING TASKS
    local J_LATENT=$(sbatch --parsable \
                --dependency=afterok:$J_EMB \
                --job-name="Latent_Tasks$JOB_SUFFIX" \
                "$LATENT_SCRIPT" \
                -r "$REPO_ROOT" \
                -w "$WEIGHTS_DIR" \
                -e "$EMB_DIR" \
                -f "$META_PATH"
            )
    echo "    [LATENT] Job sent.        ID: $J_LATENT (Depends on Embeds: ok)"

    # PROBING LATENT MAPPING DASHBOARD
    local J_LATENT_DASH=$(sbatch --parsable \
                --dependency=afterok:$J_LATENT \
                --job-name="Latent_Tasks_Dash$JOB_SUFFIX" \
                "$LATENT_DASH_SCRIPT" \
                -r "$REPO_ROOT" \
                -e "$EMB_DIR"
            )
    echo "    [LATENT DASH] Job sent.   ID: $J_LATENT_DASH (Depends on Latent: ok)"
    
    echo " --> Analysis Batch Sent."
}

# PART 1 - TRAIN FROM SCRATCH
echo "-------------------------------------------------"
echo "STEP 1: TRAINING FROM SCRATCH"
JOB_SUFFIX="_T1"
JOB_TRAIN_1=$(sbatch --parsable \
    --job-name="Train_AstroPT_DDP$JOB_SUFFIX" \
    "$TRAIN_SCRIPT" \
    -r "$REPO_ROOT" \
    -t "$TRAIN_DIR" \
    -a "$DATA_DIR" \
    -n "$TRAIN_NAME" \
    -d "$TRAIN_DESC" \
    -m "scratch" \
    -k "all")
echo "Training Job (Scratch) launched. ID: $JOB_TRAIN_1"

# Calling the execution function
launch_analysis "$JOB_TRAIN_1" "$JOB_SUFFIX" "Part 1 (SCRATCH)"


# PART 2 - TRAIN FROM RESUME
echo "-------------------------------------------------"
echo "STEP 2: TRAINING FROM RESUME"
JOB_SUFFIX="_T2"
JOB_TRAIN_2=$(sbatch --parsable \
    --dependency=afterok:$J_WOR \
    --job-name="Train_AstroPT_DDP$JOB_SUFFIX" \
    "$TRAIN_SCRIPT" \
    -r "$REPO_ROOT" \
    -t "$TRAIN_DIR" \
    -a "$DATA_DIR" \
    -n "$TRAIN_NAME" \
    -d "$TRAIN_DESC" \
    -m "resume" \
    -k "all")
echo "Training Job (Resume) launched.  ID: $JOB_TRAIN_2 (Depends on Train 1: ok $JOB_TRAIN_1)"

# Calling the execution function
launch_analysis "$JOB_TRAIN_2" "$JOB_SUFFIX" "Part 2 (Resume)"


# Implementing a third training round for longer trainings
if [ -z "$TRAIN_DIR" ]; then
    LONG_TRAINING="FALSE"
fi

if [[ "$LONG_TRAINING" == "TRUE" ]]; then
    # PART 3 - TRAIN FROM RESUME
    echo "-------------------------------------------------"
    echo "STEP 3: TRAINING FROM RESUME"
    JOB_SUFFIX="_T3"
    JOB_TRAIN_3=$(sbatch --parsable \
        --dependency=afterok:$J_WOR \
        --job-name="Train_AstroPT_DDP$JOB_SUFFIX" \
        "$TRAIN_SCRIPT" \
        -r "$REPO_ROOT" \
        -t "$TRAIN_DIR" \
        -a "$DATA_DIR" \
        -n "$TRAIN_NAME" \
        -d "$TRAIN_DESC" \
        -m "resume" \
        -k "all")
    echo "Training Job (Resume) launched.  ID: $JOB_TRAIN_3 (Depends on Train 2: ok $JOB_TRAIN_2)"

    # Calling the execution function
    launch_analysis "$JOB_TRAIN_3" "$JOB_SUFFIX" "Part 3 (Resume)"

fi


# QUEUE STATUS
echo "-------------------------------------------------"
echo "Waiting for resources..."
sleep 15
echo "Pipeline Status Queue:"
squeue -u $USER -S i -o "%.10i %.6P %.25j %.8T %.10M %.35E %R"
echo "-------------------------------------------------"