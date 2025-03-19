import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import LogLocator, FuncFormatter

# Define function to load statistics from file
def load_statistics(file_path):
    """
    Load statistics from a file and return a dictionary of metrics.
    Handles both single-point and multi-point data.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    stats = {}
    for line in lines:
        if ":" in line:
            key, value = line.split(":")
            value = value.strip()
            if "±" in value:
                mean, error = value.split("\u00b1")
                if "%" in mean:
                    mean = float(mean.strip().replace("%", "")) 
                    error = float(error.strip().replace("%", "")) 
                else:
                    mean = float(mean.strip())
                    error = float(error.strip())
                stats[key.strip()] = (mean, error)
            else:
                if "%" in value:
                    stats[key.strip()] = float(value.replace("%", "").strip()) 
                else:
                    stats[key.strip()] = float(value.strip())
    return stats

# File paths for each method
percentages = [1, 5, 10, 20, 50, 80, 100]
methods = {
    "Supervised VIS": ["SupervisedCNN/VIS/Smooth_VIS_{}_SupervisedCNN.txt".format(p) for p in percentages],
    "MLP Model VIS": ["VIS/smooth_VIS_{}_MLP.txt".format(p) for p in percentages],
    "Linear Model VIS": ["VIS/smooth_VIS_{}_LR.txt".format(p) for p in percentages],
    "Supervised VIS+NISP": ["SupervisedCNN/VIS_NISP/Smooth_VIS_NISP_{}_SupervisedCNN.txt".format(p) for p in percentages],
    "MLP Model VIS+NISP": ["VIS_NISP/smooth_VIS_NISP_{}_MLP.txt".format(p) for p in percentages],
    "Linear Model VIS+NISP": ["VIS_NISP/smooth_VIS_NISP_{}_LR.txt".format(p) for p in percentages],       
    "MLP Model VIS+NISP+SED": ["VIS_NISP_SED/smooth_VIS_NISP_SED_{}_MLP.txt".format(p) for p in percentages],
    "Linear Model VIS+NISP+SED": ["VIS_NISP_SED/smooth_VIS_NISP_SED_{}_LR.txt".format(p) for p in percentages],
}


# Initialize data storage
data = {method: {"Percentage": [], "accuracy": [], "accuracy Error": [], "precision": [], "precision Error": [], "recall": [], "recall Error": []} for method in methods.keys()}

for method, files in methods.items():
    print(f"Processing method: {method}")
    for p, file in zip(percentages, files):
        if not os.path.exists(file):
            print(f"File not found: {file}")
            continue
        stats = load_statistics(file)
        print(f"Loaded stats for file: {file} -> {stats}")
        
        # Populate data dictionary
        data[method]["Percentage"].append(p)
        if "accuracy" in stats:
            data[method]["accuracy"].append(stats["accuracy"][0])
            data[method]["accuracy Error"].append(stats["accuracy"][1])
        else:
            print(f"Missing 'accuracy' in stats for {file}")
        
        if "precision" in stats:
            data[method]["precision"].append(stats["precision"][0])
            data[method]["precision Error"].append(stats["precision"][1])
        else:
            print(f"Missing 'NMAD' in stats for {file}")
        
        if "recall" in stats:
            data[method]["recall"].append(stats["recall"][0])
            data[method]["recall Error"].append(stats["recall"][1])
        else:
            print(f"Missing 'recall' in stats for {file}")




# Plotting setup
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, axes = plt.subplots(3, 3, figsize=(22, 18), sharex=True)
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.0, wspace=0.0)  # Set wspace=0 for no space between columns

# Define colors and linestyles
colors = {
    "Supervised VIS": "blue",
    "Linear Model VIS": "green",
    "MLP Model VIS": "red",
    "Supervised VIS+NISP": "blue",
    "Linear Model VIS+NISP": "green",
    "MLP Model VIS+NISP": "red",
    "MLP Model VIS+NISP+SED": "red",
    "Linear Model VIS+NISP+SED": "green",
}

linestyles = {
    "MLP Model VIS+NISP+SED": "-",
    "Linear Model VIS+NISP+SED": "-",
}

# Define labels
methods = ["Supervised CNN", "Linear Model", "MLP Model"]
vis_methods = ["Supervised VIS", "Linear Model VIS", "MLP Model VIS"]
vis_nisp_methods = ["Supervised VIS+NISP", "Linear Model VIS+NISP", "MLP Model VIS+NISP"]
sed_methods = ["MLP Model VIS+NISP+SED", "Linear Model VIS+NISP+SED"]


# Metric labels and data keys
metrics = [("accuracy", r"$\rm Accuracy$"),
           ("precision", r"$\rm Precision$"),
           ("recall", r"recall")]

# Function to plot data for a given set of methods
def plot_data(ax, methods, metric, ylabel, colors, linestyles, labels, xlabel=None):
    """
    Plot data for a given set of methods on the specified axis.
    """
    ax.tick_params(axis='both', which='major', length=10, width=1.5, direction='in', labelsize=20)
    ax.tick_params(axis='both', which='minor', length=6, width=1.5, direction='in')
    ax.tick_params(top=True, bottom=True, left=True, right=True)  # Enable ticks on all four sides

    for method, label in zip(methods, labels):
        try:
            ax.errorbar(data[method]["Percentage"], data[method][metric], 
                        yerr=data[method][f"{metric} Error"], label=label, 
                        marker='o', markersize=12, color=colors[method], linestyle=linestyles.get(method, "-"))
        except ValueError as e:
            print(f"Skipping {method} for {metric} due to mismatched data sizes.")
            continue  # Skip to the next plot

    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
    #ax.set_ylabel(ylabel, fontsize=28)
    plt.locator_params(axis='y', tight=True, nbins=3)
    ax.tick_params(axis='both', which='major', labelsize=28)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=32)
    if metric == "accuracy":
        ax.set_ylim(0.4, 1)  # Set y-axis range for Bias
    elif metric == "precision":
        ax.set_ylim(0.4,1)
    else:
        ax.set_ylim(0.4,1)        
    return ax

# Plot accuracy (first row)
for i, (metric, ylabel) in enumerate([("accuracy", r"$\rm Accuracy$")]):
    ax = plot_data(axes[0, 0], vis_methods, metric, ylabel, colors, linestyles, methods)
    #ax.legend(fontsize=23, loc='best', frameon=False, ncol=2)
    # Custom legend for the first panel (VIS)
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        # Filter legend handles and labels to include only Euclid photo-z_NNPZ and Supervised CNN
        custom_handles = [h for h, l in zip(handles, labels) if "Euclid" in l or "Supervised" in l]
        custom_labels = [l for l in labels if "Euclid" in l or "Supervised" in l]
        ax.legend(custom_handles, custom_labels, fontsize=26, loc='best', frameon=False, ncol=1)

    ax.set_ylabel(ylabel, fontsize=32)
    ax.text(0.6, 0.1, r'$\rm VIS$', fontsize=28, transform=ax.transAxes)
    ax = plot_data(axes[0, 1], vis_nisp_methods, metric, ylabel, colors, linestyles, methods)
    plt.setp(ax.get_yticklabels(), visible=False)
    # Custom legend for the second panel (VIS+NISP)
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        # Filter legend handles and labels to include only Linear Model and MLP Model
        custom_handles = [h for h, l in zip(handles, labels) if "Linear Model" in l or "MLP Model" in l]
        custom_labels = [l for l in labels if "Linear Model" in l or "MLP Model" in l]
        ax.legend(custom_handles, custom_labels, fontsize=26, loc='upper left', frameon=False, ncol=1)
    
    ax.text(0.6, 0.1, r'$\rm VIS+NISP$', fontsize=28, transform=ax.transAxes)
    ax = plot_data(axes[0, 2], sed_methods, metric, ylabel, colors, linestyles, sed_methods)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.text(0.5, 0.1, r'$\rm VIS+NISP+SED$', fontsize=28, transform=ax.transAxes)
    # Custom legend for the last panel (Bias)
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        custom_handles = [h for h, l in zip(handles, labels) if "VIS+NISP+SED" in l or "SED" in l]
        custom_labels = ["VIS+NISP+SED" if "VIS+NISP+SED" in l else "SED" for l in labels if "VIS+NISP+SED" in l or "SED" in l]        
        #custom_labels = [l for l in labels if "VIS+NISP+SED" in l or "SED" in l]
        #ax.legend(custom_handles, custom_labels, fontsize=26, loc='best', frameon=False, ncol=2)


# Plot Precision (second row)
for i, (metric, ylabel) in enumerate([("precision", r"$\rm Precision$")]):
    ax = plot_data(axes[1, 0], vis_methods, metric, ylabel, colors, linestyles, methods)
    ax.set_ylabel(ylabel, fontsize=32)
    ax.text(0.6, 0.8, r'$\rm VIS$', fontsize=28, transform=ax.transAxes)
    ax = plot_data(axes[1, 1], vis_nisp_methods, metric, ylabel, colors, linestyles, methods)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.text(0.6, 0.8, r'$\rm VIS+NISP$', fontsize=28, transform=ax.transAxes)
    ax = plot_data(axes[1, 2], sed_methods, metric, ylabel, colors, linestyles, sed_methods)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.text(0.5, 0.8, r'$\rm VIS+NISP+SED$', fontsize=28, transform=ax.transAxes)

# Plot recall(third row)
for i, (metric, ylabel) in enumerate([("recall", r"Recall")]):
    ax = plot_data(axes[2, 0], vis_methods, metric, ylabel, colors, linestyles, methods, xlabel=r"Percentage of Labels (\%)")
    ax.set_ylabel(ylabel, fontsize=32)
    ax.text(0.6, 0.8, r'$\rm VIS$', fontsize=28, transform=ax.transAxes)
    ax = plot_data(axes[2, 1], vis_nisp_methods, metric, ylabel, colors, linestyles, methods, xlabel=r"Percentage of Labels (\%)")
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.text(0.6, 0.8, r'$\rm VIS+NISP$', fontsize=28, transform=ax.transAxes)
    ax = plot_data(axes[2, 2], sed_methods, metric, ylabel, colors, linestyles, sed_methods, xlabel=r"Percentage of Labels (\%)")
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.text(0.5, 0.8, r'$\rm VIS+NISP+SED$', fontsize=28, transform=ax.transAxes)

# Save and show plot
#plt.tight_layout()
output_png_path = "Smooth_comparison.png"
plt.savefig(output_png_path, dpi=300)
#plt.show()
