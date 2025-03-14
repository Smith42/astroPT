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
# SupervisedCNN/VIS/logM_Supervised_VIS_{}.txt
percentages = [1, 5, 10, 20, 50, 80, 100]
methods = {
    "Supervised VIS": ["VIS/logM_VIS_{}_SupervisedCNN.txt".format(p) for p in percentages],
    "MLP Model VIS": ["VIS/logM_VIS_{}_MLP.txt".format(p) for p in percentages],
    "Linear Model VIS": ["VIS/logM_VIS_{}_LR.txt".format(p) for p in percentages],
    "Supervised VIS+NISP": ["VIS_NISP/logM_VIS_NISP_{}_SupervisedCNN.txt".format(p) for p in percentages],
    "MLP Model VIS+NISP": ["VIS_NISP/logM_VIS_NISP_{}_MLP.txt".format(p) for p in percentages],
    "Linear Model VIS+NISP": ["VIS_NISP/logM_VIS_NISP_{}_LR.txt".format(p) for p in percentages]
}

logM_Euclid = {
    "SED": ["Euclid_NNPZ/logM_Euclid_NNPZ.txt"],
}

# Initialize data storage
data = {method: {"Percentage": [], "Bias": [], "Bias Error": [], "NMAD": [], "NMAD Error": [], "Outlier Fraction": [], "Outlier Fraction Error": []} for method in methods.keys()}

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
        if "Bias" in stats:
            data[method]["Bias"].append(stats["Bias"][0])
            data[method]["Bias Error"].append(stats["Bias"][1])
        elif "bias" in stats:
            data[method]["Bias"].append(stats["bias"][0])
            data[method]["Bias Error"].append(stats["bias"][1])
        else:
            print(f"Missing 'Bias' in stats for {file}")
        
        if "NMAD" in stats:
            data[method]["NMAD"].append(stats["NMAD"][0])
            data[method]["NMAD Error"].append(stats["NMAD"][1])
        elif "nmad" in stats:
            data[method]["NMAD"].append(stats["nmad"][0])
            data[method]["NMAD Error"].append(stats["nmad"][1])
        else:
            print(f"Missing 'NMAD' in stats for {file}")
        
        if "Outlier Fraction" in stats:
            data[method]["Outlier Fraction"].append(stats["Outlier Fraction"][0])
            data[method]["Outlier Fraction Error"].append(stats["Outlier Fraction"][1])
        elif "outlier_fraction" in stats:
            data[method]["Outlier Fraction"].append(stats["outlier_fraction"][0])
            data[method]["Outlier Fraction Error"].append(stats["outlier_fraction"][1])
        else:
            print(f"Missing 'Outlier Fraction' in stats for {file}")


# Load single-point data from logM_Euclid
single_point_data = {}
for method, files in logM_Euclid.items():
    print(f"Processing method: {method}")
    for file in files:
        print(f"Processing file: {file}")
        stats = load_statistics(file)
        print(f"Stats: {stats}")
        single_point_data[method] = {
            "Bias": (stats["bias"], 0),  # Assume zero error for single-point data
            "NMAD": (stats["nmad"], 0),
            "Outlier Fraction": (stats["outlier_fraction"], 0),
        }
    print(f"Single-point data for {method}: {single_point_data[method]}")

# Plotting setup
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, axes = plt.subplots(3, 2, figsize=(22, 18), sharex=True)
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.0, wspace=0.0)  # Set wspace=0 for no space between columns

# Define colors and linestyles
colors = {
    "Supervised VIS": "blue",
    "Linear Model VIS": "green",
    "MLP Model VIS": "red",
    "Supervised VIS+NISP": "blue",
    "Linear Model VIS+NISP": "green",
    "MLP Model VIS+NISP": "red",
}

linestyles = {
    "Supervised VIS": "-",
    "Linear Model VIS": "-",
    "MLP Model VIS": "-",
    "Supervised VIS+NISP": "-",
    "Linear Model VIS+NISP": "-",
    "MLP Model VIS+NISP": "-",
}

# Define labels
methods_labels = ["Supervised CNN", "Linear Model", "MLP Model"]
vis_methods = ["Supervised VIS", "Linear Model VIS", "MLP Model VIS"]
vis_nisp_methods = ["Supervised VIS+NISP", "Linear Model VIS+NISP", "MLP Model VIS+NISP"]

# Metric labels and data keys
metrics = [("Bias", r"$\rm Bias$"),
           ("NMAD", r"$\rm NMAD$"),
           ("Outlier Fraction", r"Outlier Fraction (\%)")]

# Function to plot data for a given set of methods
def plot_data(ax, methods, metric, ylabel, single_point_data, colors, linestyles, labels, xlabel=None):
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
                        marker='o', markersize=12, color=colors[method], linestyle=linestyles[method])
        except ValueError as e:
            print(f"Skipping {method} for {metric} due to mismatched data sizes.")
            continue  # Skip to the next plot

    single_point = single_point_data["SED"]
    ax.scatter([100], [single_point[metric][0]], 
                label=r"$\rm Euclid\mbox{ } log(M_{star}/M_{\odot})_{NNPZ}$", marker='s', s=400, color='black', linestyle='')
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
    ax.tick_params(axis='both', which='major', labelsize=28)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=32)
    if metric == "Bias":
        ax.set_ylim(-0.1, 0.1)  # Set y-axis range for Bias
    elif metric == "NMAD":
        ax.set_ylim(0.01, 0.15)
    else:
        ax.set_ylim(0.1, 30)
        ax.set_yscale('log')
    return ax

# Plot Bias (first row)
for i, (metric, ylabel) in enumerate([("Bias", r"$\rm Bias$")]):
    ax = plot_data(axes[0, 0], vis_methods, metric, ylabel, single_point_data, colors, linestyles, methods_labels)
    ax.set_ylabel(ylabel, fontsize=32)
    ax.text(0.6, 0.1, r'$\rm VIS$', fontsize=28, transform=ax.transAxes)
    # Custom legend for the first panel (VIS)
    handles, labels = ax.get_legend_handles_labels()
    custom_handles = [h for h, l in zip(handles, labels) if "Euclid" in l or "Supervised" in l or "Linear Model" in l]
    custom_labels = [l for l in labels if "Euclid" in l or "Supervised" in l or "Linear Model" in l]
    ax.legend(custom_handles, custom_labels, fontsize=26, loc='best', frameon=False, ncol=2)

    ax = plot_data(axes[0, 1], vis_nisp_methods, metric, ylabel, single_point_data, colors, linestyles, methods_labels)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.text(0.6, 0.1, r'$\rm VIS+NISP$', fontsize=28, transform=ax.transAxes)
    # Custom legend for the second panel (VIS+NISP)
    handles, labels = ax.get_legend_handles_labels()
    custom_handles = [h for h, l in zip(handles, labels) if "MLP Model" in l]
    custom_labels = [l for l in labels if "MLP Model" in l]
    ax.legend(custom_handles, custom_labels, fontsize=26, loc='upper left', frameon=False, ncol=1)

# Plot NMAD (second row)
for i, (metric, ylabel) in enumerate([("NMAD", r"$\rm NMAD$")]):
    ax = plot_data(axes[1, 0], vis_methods, metric, ylabel, single_point_data, colors, linestyles, methods_labels)
    ax.set_ylabel(ylabel, fontsize=32)
    ax.text(0.6, 0.8, r'$\rm VIS$', fontsize=28, transform=ax.transAxes)
    ax = plot_data(axes[1, 1], vis_nisp_methods, metric, ylabel, single_point_data, colors, linestyles, methods_labels)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.text(0.6, 0.8, r'$\rm VIS+NISP$', fontsize=28, transform=ax.transAxes)

# Plot Outlier Fraction (third row)
for i, (metric, ylabel) in enumerate([("Outlier Fraction", r"Outlier Fraction (\%)")]):
    ax = plot_data(axes[2, 0], vis_methods, metric, ylabel, single_point_data, colors, linestyles, methods_labels, xlabel=r"Percentage of Labels (\%)")
    ax.set_ylabel(ylabel, fontsize=32)
    ax.text(0.6, 0.8, r'$\rm VIS$', fontsize=28, transform=ax.transAxes)
    ax = plot_data(axes[2, 1], vis_nisp_methods, metric, ylabel, single_point_data, colors, linestyles, methods_labels, xlabel=r"Percentage of Labels (\%)")
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.text(0.6, 0.8, r'$\rm VIS+NISP$', fontsize=28, transform=ax.transAxes)

# Save and show plot
output_png_path = "logM_comparison.png"
plt.savefig(output_png_path, dpi=300)
#plt.show()
