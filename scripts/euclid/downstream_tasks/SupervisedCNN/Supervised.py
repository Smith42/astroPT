import numpy as np
import tensorflow as tf
import torch
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import seaborn as sns
from data import GalaxyImageDataset
from data import GalaxyImageDataset_VIS_NISP
from eval import ModelEvaluator, evaluate_model
from models import train_model_on_subset, train_model_on_subset_vis_nisp
from utils import write_results_to_file
from utils import save_and_plot_redshifts
# Disable GPU (for debugging or running on CPU)
#tf.config.set_visible_devices([], 'GPU')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)
          
# Input paraemters
modality = 'VIS_NISP' # set 'VIS' or 'VIS_NISP'

model_name = 'SupervisedCNN'

task_type = "regression" # choose the `regression` for predictin photo-z or stellar masses or `classifciation` for galaxy type classification
parameter = "logM" # set 'logM' or 'PhotoZ'

# File paths
if modality=='VIS':
    train_txt = "../astropt090M_euclid/train.txt"
    test_txt = "../astropt090M_euclid/test.txt"
elif modality=='VIS_NISP':
    train_all_txt = "../astropt090M_euclid_4chan/train_all.txt"
    test_all_txt = "../astropt090M_euclid_4chan/test_all.txt"
else:
    print("No input data")

if parameter == 'logM':
    label_file = "../../../Q1_data/DESI_logM.fits"
elif parameter == 'PhotoZ':
    label_file = "../../../Q1_data/DESISpecZ.fits"
elif parameter == 'Smooth':   
    label_file = "../../../Q1_data/EuclidMorphPhysPropSpecZ.fits"
else:
    print("Catalog not loaded. Specify the parameter")


# Load dataset and monitor progress
if modality =='VIS':
    train_image_paths, train_labels = GalaxyImageDataset.get_image_paths_and_labels(train_txt, label_file, parameter, task_type)
    test_image_paths, test_labels = GalaxyImageDataset.get_image_paths_and_labels(test_txt, label_file, parameter, task_type)
    # Combine train and test for easier sampling
    image_paths = train_image_paths + test_image_paths
    labels = np.concatenate([train_labels, test_labels])
elif modality =='VIS_NISP':
    # Load dataset
    train_vis, train_y, train_j, train_h, train_labels = GalaxyImageDataset_VIS_NISP.get_image_paths_and_labels(train_all_txt, label_file, parameter, task_type)
    test_vis, test_y, test_j, test_h, test_labels = GalaxyImageDataset_VIS_NISP.get_image_paths_and_labels(test_all_txt, label_file, parameter, task_type)
    # Combine train and test sets
    vis = train_vis + test_vis
    nir_y = train_y + test_y
    nir_j = train_j + test_j
    nir_h = train_h + test_h
    labels = np.concatenate([train_labels, test_labels])
else:
    print("Data not loaded. Specify the modality")


    
# Train and evaluate on different percentages 
n_mc_runs = 10 # set the MC runs
#percentages = [1, 5, 10, 20, 50, 80, 100]  
percentages = [10]  # set the % of the labes used for training
results = {}

for percentage in percentages:
    print(f"\nTraining on {percentage}% of the dataset...")
    models = []
    test_dataset_list = []
    all_final_metrics = []
    all_metrics_mean = []
    all_metrics_std = []

    for i in range(n_mc_runs):
        print(f"Training run {i+1}/{n_mc_runs} for {percentage}% of the data...")
        if modality=='VIS':
            model, test_dataset = train_model_on_subset(image_paths, labels, n_mc_runs, parameter, modality, percentage)
        elif modality=='VIS_NISP':
            model, test_dataset = train_model_on_subset_vis_nisp(vis, nir_y, nir_j, nir_h, labels, n_mc_runs, parameter, modality, percentage)
        else:
            print("no modality is given. Not able to create the model")
        
        # Evaluate the model
        final_metrics, metrics_mean, metrics_std, predictions = evaluate_model(
            model, test_dataset, task_type=task_type, parameter=parameter, n_mc_runs=1
        )
        
        true_labels = np.concatenate([batch[1] for batch in test_dataset])
        if i == 0:
            # Save and plot the performance of the predictions - this is done on the firtst MC run
            save_and_plot_redshifts(
            true_redshifts=true_labels,
            predicted_redshifts=predictions,  
            percentage=percentage,
            modality=modality,
            parameter=parameter,
            model_name="SupervisedCNN"
            )
            
        # Store metrics for this run
        all_final_metrics.append(final_metrics)
        all_metrics_mean.append(metrics_mean)
        all_metrics_std.append(metrics_std)

    # Compute mean and std of metrics across all MC runs
    final_metrics_aggregated = {
        k: np.mean([m[k] for m in all_final_metrics]) for k in all_final_metrics[0]
    }
    metrics_mean_aggregated = {
        k: np.mean([m[k] for m in all_metrics_mean]) for k in all_metrics_mean[0]
    }
    metrics_std_aggregated = {
        k: np.std([m[k] for m in all_metrics_mean]) for k in all_metrics_mean[0]
    }
    
    # Write results to file
    write_results_to_file(
        model_name="SupervisedCNN",
        percentage=percentage,
        modality=modality,
        parameter=parameter,
        final_metrics=final_metrics_aggregated,
        metrics_mean=metrics_mean_aggregated,
        metrics_std=metrics_std_aggregated
    )
    
    # Store the trained model and validation data
    models.append(model)
    test_dataset_list.append(test_dataset)

    # Store the models and validation data for future evaluation
    results[percentage] = {
        "models": models,
        "test_dataset": test_dataset_list
    }

