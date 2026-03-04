"""
AstroPT Training Metrics Plotter.

Reads the metrics.csv generated during training and produces a 
2x2 dashboard visualization of the run.

Features:
- Loss Curves (Train smoothed + Val raw)
- Gradient Norm & Clipping events
- Learning Rate Schedule
- System Metrics (MFU & VRAM)

Author: Victor Alonso Rodriguez
Date: January 2026
"""

import argparse
import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Plotting Global Configuration
plt.rcParams['text.latex.preamble'] = r'''
            \usepackage{siunitx}
            \usepackage{bm}
            \usepackage{amsmath} 
            \sisetup{
            detect-family,
            separate-uncertainty=true,
            output-decimal-marker={.},
            exponent-product=\cdot,
            inter-unit-product=\cdot,
            }
            \DeclareSIUnit{\cts}{cts}
            '''
plt.rc('text', usetex=True)
plt.rc('font', family='serif', weight='bold') 

plt.rcParams.update({
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'font.size': 14,          
    'axes.labelsize': 19,     
    'axes.titlesize': 21,     
    'xtick.labelsize': 19,   
    'ytick.labelsize': 19,
    'legend.fontsize': 16,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titlesize': 24,
    'figure.titleweight': 'bold',
})


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    
    parser = argparse.ArgumentParser(description="Plot AstroPT Training Metrics")
    
    # Parsing Arguments
    parser.add_argument("--out_dir", type=str, required=True, help="Directory containing metrics.csv")
    parser.add_argument("--csv_name", type=str, default="training_metrics.csv", help="Name of the CSV file")
    parser.add_argument("--save_name", type=str, default="training_metrics.png", help="Output image name")
    parser.add_argument("--smooth_window", type=int, default=10, help="Smoothing window for training loss")
    parser.add_argument("--train_name", type=str, default=None, help="Custom title for the plot (defaults to folder name)")
    
    return parser.parse_args()


def smooth_data(series: pd.Series, window: int) -> pd.Series:
    """Applies exponential moving average for smoother visualization."""
    return series.ewm(span=window, adjust=False).mean()


def main():

    # Parsing argumentss
    args = parse_args()
    
    # Load CSV
    csv_path = os.path.join(args.out_dir, args.csv_name)
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        sys.exit(1)
        
    print(f"Loading metrics from {csv_path}...")
    
    # Load as a Pandas Dataframe
    df = pd.read_csv(csv_path)

    # Separate Train and Val rows
    val_df = df.dropna(subset=['val_loss'])
    train_df = df.dropna(subset=['train_loss'])


    # PLOTTING SETUP
    
    fig, axs = plt.subplots(2, 2, figsize=(18, 17))
    
    # Customizing the title
    config_path = os.path.join(args.out_dir, "config.json")
    json_name = None
    
    # Reading the json file
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                json_name = config.get("train_name", None)
        except Exception:
            pass
    
    # Subtitle priority
    if args.train_name:
        subtitle = args.train_name
    elif json_name:
        subtitle = json_name
    else:
        subtitle = os.path.basename(os.path.normpath(args.out_dir))
    
    fig.suptitle(r"\textbf{AstroPT Training Dashboard}" + f"\n[{subtitle}]", fontsize=22, y=0.95)
    
    # Epochs compute
    iter_max = df['iter'].max()
    has_epoch = 'epoch' in df.columns
    
    if has_epoch:
        # Avoid zero division if run just started
        epoch_max = df['epoch'].max()
        if epoch_max == 0: epoch_max = 1 
        
        iters_per_epoch = iter_max / epoch_max
        
    else:
        iters_per_epoch = iter_max 
        
        print("Warning: 'epoch' column not found. Top axis will match iterations.")

    # Convert functions for secondary axis
    def iter_to_epoch(x): return x / iters_per_epoch
    def epoch_to_iter(x): return x * iters_per_epoch

    # PLOT 1: LOSS CURVES
    ax1 = axs[0, 0]
    
    # Grid alignment
    target_ticks = 6 
    margin_fraction = 0.1 
    
    # Train Loss
    t_min = train_df['train_loss'].min()
    t_max = train_df['train_loss'].max()
    t_range = t_max - t_min
    if t_range == 0: t_range = t_max * 0.1
    
    ax1.plot(train_df['iter'], train_df['train_loss'], color='black', alpha=0.35, label='Train (Raw)')
    ax1.plot(train_df['iter'], smooth_data(train_df['train_loss'], args.smooth_window), 
             color='dodgerblue', label=f'Train (Smooth {args.smooth_window})')
    
    ax1.set_ylabel(r'\textbf{Train Loss}', color='dodgerblue')
    ax1.tick_params(axis='y', labelcolor='dodgerblue')
    ax1.set_xlabel(r'\textbf{Iterations}')
    
    # Left ticks
    yticks_train = np.linspace(t_min, t_max, target_ticks)
    ax1.set_yticks(yticks_train)
    pad_train = t_range * margin_fraction
    ax1.set_ylim(t_min - pad_train, t_max + pad_train)
    
    # Val Losss
    ax1r = ax1.twinx()
    
    # Compute Limits
    if not val_df.empty:
        v_min = val_df['val_loss'].min()
        v_max = val_df['val_loss'].max()
        v_range = v_max - v_min
        if v_range == 0: v_range = v_max * 0.1
        
        # Fix ticks for the grid
        yticks_val = np.linspace(v_min, v_max, target_ticks)
        ax1r.set_yticks(yticks_val)
        pad_val = v_range * margin_fraction
        ax1r.set_ylim(v_min - pad_val, v_max + pad_val)

    ax1r.plot(val_df['iter'], val_df['val_loss'], color='red', marker='o', linestyle='--', 
             linewidth=1, markersize=4, label='Validation')
    ax1r.set_ylabel(r'\textbf{Validation Loss}', color='red')
    ax1r.tick_params(axis='y', labelcolor='red')
    
    ax1r.grid(False) 
    
    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)
    
    # Title
    ax1.set_title(r'\textbf{Convergence}', pad=15)
    
    if has_epoch:
        secax1 = ax1.secondary_xaxis('top', functions=(iter_to_epoch, epoch_to_iter))
        secax1.set_xlabel(r'\textbf{Epochs}', labelpad=10)

    # PLOT 2: GRADIENT DYNAMICS
    ax2 = axs[0, 1]
    color_grad = 'royalblue'
    color_clip = 'deeppink'
    
    # Grad Norm
    ax2.plot(train_df['iter'], train_df['grad_norm'], color=color_grad, lw=1.5, label='Grad Norm')
    ax2.set_ylabel(r'\textbf{Gradient Norm}', color=color_grad)
    ax2.tick_params(axis='y', labelcolor=color_grad)
    
    # Clipping Events 
    clipped_iter = train_df[train_df['clipped'] == 1]['iter']
    
    lines, labels = ax2.get_legend_handles_labels()
    if not clipped_iter.empty:
        
        ax2r = ax2.twinx()
        ax2r.scatter(clipped_iter, [1]*len(clipped_iter), color=color_clip, marker='|', s=80, label='Clipped')
        ax2r.set_ylabel(r'\textbf{Clipping Event}', color=color_clip)
        ax2r.set_ylim(0, 1.1)
        ax2r.set_yticks([]) 
        
        # Merge legends
        lines2, labels2 = ax2r.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    
    ax2.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
    
    # If no clipping
    if clipped_iter.empty:
        ax2.text(0.5, 0.9, "No Clipping Events", transform=ax2.transAxes, ha='center', color='green', fontweight='bold')

    ax2.set_title(r'\textbf{Gradient Stability}', pad=15)
    ax2.set_xlabel(r'\textbf{Iterations}')
    
    if has_epoch:
        secax2 = ax2.secondary_xaxis('top', functions=(iter_to_epoch, epoch_to_iter))
        secax2.set_xlabel(r'\textbf{Epochs}', labelpad=10)

    # PLOT 3: LEARNING RATE
    ax3 = axs[1, 0]
    ax3.plot(train_df['iter'], train_df['lr'], color='black', lw=2)
    ax3.set_ylabel(r'\textbf{Learning Rate}')
    ax3.set_xlabel(r'\textbf{Iterations}')
    ax3.set_title(r'\textbf{LR Schedule}', pad=15)
    
    # Dynamic Offset Logic
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0)) # Force scientific notation
    ax3.yaxis.set_major_formatter(formatter)
    
    fig.canvas.draw() 
    offset_text = ax3.yaxis.get_offset_text().get_text()
    
    if offset_text:
        ax3.set_ylabel(r'\textbf{Learning Rate} (' + offset_text + ')')
        ax3.yaxis.get_offset_text().set_visible(False)
    else:
        ax3.set_ylabel(r'\textbf{Learning Rate}')
    
    if has_epoch:
        secax3 = ax3.secondary_xaxis('top', functions=(iter_to_epoch, epoch_to_iter))
        secax3.set_xlabel(r'\textbf{Epochs}', labelpad=10)


    # PLOT 4: SYSTEM RESOURCES
    ax4 = axs[1, 1]
    color_mfu = 'darkorange'
    color_mem = 'blueviolet'
    
    target_ticks = 6 
    margin_fraction = 0.1 
    
    # Plot MFU (Left Axis)
    if 'mfu' in df.columns:
        
        mfu_data = train_df[train_df['mfu'] > 0]
        
        max_mfu = mfu_data['mfu'].max()
        min_mfu = mfu_data['mfu'].min()
        range_mfu = max_mfu - min_mfu
        
        if range_mfu == 0: range_mfu = max_mfu * 0.1
        
        ax4.plot(mfu_data['iter'], mfu_data['mfu'], color=color_mfu, lw=2, label='MFU (\\%)')
        ax4.set_ylabel(r'\textbf{MFU (\%)}', color=color_mfu)
        ax4.tick_params(axis='y', labelcolor=color_mfu)
        
        yticks_mfu = np.linspace(min_mfu, max_mfu, target_ticks)
        ax4.set_yticks(yticks_mfu)
        
        pad = range_mfu * margin_fraction
        ax4.set_ylim(min_mfu - pad, max_mfu + pad)
    
    # Plot VRAM (Right Axis)
    if 'mem_gb' in df.columns:
        ax4r = ax4.twinx()
        max_mem = train_df['mem_gb'].max()
        min_mem = train_df['mem_gb'].min()
        range_mem = max_mem - min_mem
        
        if range_mem == 0: range_mem = max_mem * 0.1
        
        ax4r.plot(train_df['iter'], train_df['mem_gb'], color=color_mem, linestyle='--', lw=2, label='VRAM (GB)')
        ax4r.set_ylabel(r'\textbf{VRAM (GB)}', color=color_mem)
        ax4r.tick_params(axis='y', labelcolor=color_mem)
        
        yticks_mem = np.linspace(min_mem, max_mem, target_ticks)
        ax4r.set_yticks(yticks_mem)
        
        pad_mem = range_mem * margin_fraction
        ax4r.set_ylim(min_mem - pad_mem, max_mem + pad_mem)
        
        ax4r.grid(False) 
    
    ax4.set_title(r'\textbf{Efficiency \& Resources}', pad=15)
    ax4.set_xlabel(r'\textbf{Iterations}')
    
    # Legends
    lines, labels = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4r.get_legend_handles_labels() if 'mem_gb' in df.columns else ([], [])
    ax4.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)

    if has_epoch:
        secax4 = ax4.secondary_xaxis('top', functions=(iter_to_epoch, epoch_to_iter))
        secax4.set_xlabel(r'\textbf{Epochs}', labelpad=10)

    # SAVING
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], w_pad=3.0, h_pad=0.5)
    
    save_path = os.path.join(args.out_dir, args.save_name)
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    print(f" --> Dashboard saved to: {save_path}")

if __name__ == "__main__":
    main()