import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import seaborn as sns

    
def plot_loss(train_losses, val_losses, epochs):
    """
    Plots the training and validation loss over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_losses, label="Training Loss", color='b')
    plt.plot(range(epochs), val_losses, label="Validation Loss", color='r')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def format_metrics(metrics):
    """Format metrics dictionary to display values with 4 decimal places."""
    formatted_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (float, np.float32, np.float64)):
            formatted_metrics[key] = f"{value:.4f}"
        else:
            formatted_metrics[key] = str(value)
    return formatted_metrics
    


def write_results_to_file(model_name, percentage, modality, parameter, final_metrics, metrics_mean, metrics_std):
    """
    Write evaluation results to a file.

    Args:
        percentage: Percentage of data used for training.
        modality: Modality (e.g., 'VIS').
        parameter: Parameter (e.g., 'PhotoZ').
        final_metrics: Final metrics computed on aggregated predictions.
        metrics_mean: Mean metrics across all runs.
        metrics_std: Standard deviation of metrics across all runs.
    """
    # Create the directory if it doesn't exist
    os.makedirs(modality, exist_ok=True)

    # Define the file name
    file_name = f"{parameter}_{modality}_{percentage}_{model_name}.txt"
    file_path = os.path.join(modality, file_name)

    # Write to the file
    with open(file_path, 'w') as txt_file:
        txt_file.write("Final Evaluation Statistics\n")
        txt_file.write("===========================\n")
        txt_file.write(f"Percentage: {percentage}\n")
        # Write mean and std metrics
        for key in metrics_mean:
            txt_file.write(f"{key}: {metrics_mean[key]:.4f} ± {metrics_std[key]:.4f}\n")


def save_and_plot_redshifts(true_redshifts, predicted_redshifts, percentage, modality, parameter, model_name):
    """
    Save the true and predicted redshifts to a file and plot them using a hexbin plot.

    Args:
        true_redshifts: True redshifts (numpy array).
        predicted_redshifts: Predicted redshifts (numpy array).
        sigma_68: Sigma 68 value for annotation.
        outlier_fraction: Outlier fraction for annotation.
        percentage: Percentage of data used for training.
        modality: Modality (e.g., 'VIS').
        parameter: Parameter (e.g., 'PhotoZ').
    """
    # Create the directory if it doesn't exist
    os.makedirs(modality, exist_ok=True)

    true_redshifts = np.array(true_redshifts)
    predicted_redshifts = np.array(predicted_redshifts)
    predicted_redshifts = np.squeeze(predicted_redshifts)
    
    # Flatten the predicted_redshifts array if it's 2D
    if predicted_redshifts.ndim > 1:
        predicted_redshifts = np.squeeze(predicted_redshifts)

    # Verify shapes
    #print("True Redshifts Shape:", true_redshifts.shape)
    #print("Predicted Redshifts Shape:", predicted_redshifts.shape)
    
                
    # Save true and predicted redshifts to a file
    output_file = f"{modality}/{parameter}_{modality}_{percentage}_{model_name}_predictions.txt"
    with open(output_file, 'w') as f:
        f.write("True_Redshift\tPredicted_Redshift\n")
        for true_z, pred_z in zip(true_redshifts, predicted_redshifts):
            f.write(f"{true_z:.6f}\t{pred_z:.6f}\n")

    # Plot the true vs predicted redshifts (Hexbin Plot)
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    fig = plt.figure(figsize=(22, 10)) 
    plt.subplots_adjust(left=0.18, bottom=None, right=None, top=None, wspace=0.02, hspace=0)
    
    # Create hexbin plot
    #hexbin = plt.hexbin(
     #   true_redshifts, predicted_redshifts, 
     #   gridsize=10,  # Number of hexagons in the x-direction
     #   cmap="RdYlBu_r",  # Colormap
     #   mincnt=1,  # Minimum number of points to plot a hexagon
     #   bins='log'  # Use logarithmic scale for color mapping
    #)
    

    ax = sns.kdeplot(x=true_redshifts, y=predicted_redshifts, cmap="RdYlBu_r", fill=True, 
                     levels=15, thresh=0.05)  # `fill=False` avoids the filled area
    plt.gca().set_facecolor('white')

    # Add the y=x line
    max_val = max(true_redshifts.max(), predicted_redshifts.max())
    min_val = min(true_redshifts.min(), predicted_redshifts.min())
    if parameter =="PhotoZ":
            plt.plot([-0.5,1.1], [-0.5,1.1], '--', linewidth=2, color='black')
    else:
            plt.plot([0.5*min_val, 1.5*max_val], [0.5*min_val, 1.5*max_val], '--', linewidth=2, color='black')


    #plt.text(
    #0.15, 0.75, r'$\rm \eta_{out} = %.2f \pm %.2f\%%$' % (outlier_fraction, outlier_fraction_err),
    #horizontalalignment='left', verticalalignment='bottom',
    #fontsize=35, transform=plt.gca().transAxes
    #)
    if parameter =="PhotoZ":
        plt.xlabel(r'$z_{\mathrm{DESI}}$', fontsize=50)
        plt.ylabel(r'$\mathrm{photo-}z_{{predicted}}$', fontsize=50)
    elif parameter =="logM":
        plt.xlabel(r'$\mathrm{log(M_{star}/M_{\odot})_{DESI}}$', fontsize=50)
        plt.ylabel(r'$\mathrm{log(M_{star}/M_{\odot})_{predicted}}$', fontsize=50)
    else:
        plt.xlabel(r'$x$', fontsize=35)
        plt.ylabel(r"$y$", fontsize=35)

    # Adjust ticks and formatting
    plt.minorticks_on()
    plt.tick_params(axis='x', which='major', labelsize=45)
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=45)
    plt.tick_params(axis='both', which='major', length=15)  # Major ticks length
    plt.tick_params(axis='both', which='minor', length=10)   # Minor ticks length  
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    if parameter =="PhotoZ":
        plt.xlim(-0.18,1.1)
        plt.ylim(-0.18,1.1)   
    elif parameter =="logM":
        plt.xlim(6.8, 13)
        plt.ylim(6.8, 13)
    else:
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)        
    # Remove grid and legend, set white background
    plt.gca().set_facecolor('white')
    plt.grid(False)
    #plt.legend([], [], frameon=False)


    # Save the plot
    plt.tight_layout()
    plot_file = f"{modality}/{parameter}_{modality}_{percentage}_{model_name}.png"
    plt.savefig(plot_file)
    #plt.show()
    plt.close()
