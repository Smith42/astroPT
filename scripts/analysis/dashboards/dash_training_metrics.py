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
Date: March 2026
"""

import argparse
import json
import sys
import pandas as pd
from pathlib import Path
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
    'lines.linewidth': 1,
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
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing training weigths")
    parser.add_argument("--logs_dir", type=str, required=False, help="Directory containing metrics.csv (defaults to weights_dir.parent/logs)")
    parser.add_argument("--save_dir", type=str, required=False, help="Plot Saving Directory (defaults to weights_dir.parent/plots)")
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
    
    # Required paths
    weights_dir = Path(args.weights_dir)
    logs_dir = Path(args.logs_dir) if args.logs_dir else weights_dir.parent / "logs"
    save_dir = Path(args.save_dir) if args.save_dir else weights_dir.parent / "plots"
    
    # Ensure save_dir exists
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CSV
    csv_path =  logs_dir / args.csv_name
    
    if not csv_path.is_file():
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
    config_path = weights_dir / "config.json"
    json_name = None
    
    # Reading the json file
    if config_path.is_file():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                json_name = config.get("train_name", None)
        except Exception:
            pass
    
    # Subtitle priority
    subtitle = args.train_name or json_name or weights_dir.parent.name    
    subtitle = subtitle.replace('_', r'\_')
    
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
    
    ax1.plot(train_df['iter'], train_df['train_loss'], color='black', lw=1.5, alpha=0.35, label='Train (Raw)')
    ax1.plot(train_df['iter'], smooth_data(train_df['train_loss'], args.smooth_window), 
             color='dodgerblue', lw=1, label=f'Train (Smooth {args.smooth_window})')
             
    loss_cols = [c for c in train_df.columns if c.startswith('loss_') and c not in ['train_loss', 'val_loss']]
    colors_loss = ['green', 'orange', 'purple', 'brown']
    for i, c in enumerate(loss_cols):
        ax1.plot(train_df['iter'], smooth_data(train_df[c], args.smooth_window), 
                 color=colors_loss[i % len(colors_loss)], linestyle=':', alpha=0.7, label=f'Train {c.split("loss_")[1]} (Smooth)')

    cross_cols = [c for c in train_df.columns if c.startswith('cross_loss_')]
    colors_cross = ['darkgreen', 'saddlebrown', 'navy', 'crimson']
    for i, c in enumerate(cross_cols):
        valid_cross = train_df.dropna(subset=[c])
        if not valid_cross.empty:
            ax1.plot(valid_cross['iter'], smooth_data(valid_cross[c], args.smooth_window), 
                     color=colors_cross[i % len(colors_cross)], linestyle='--', alpha=0.8, label=f'Cross {c.split("cross_loss_")[1]} (Smooth)')
    
    # V4: CLIP & Reconstruction components
    if 'clip_loss' in train_df.columns and not train_df['clip_loss'].isna().all():
        ax1.plot(train_df['iter'], smooth_data(train_df['clip_loss'], args.smooth_window), 
                 color='forestgreen', lw=1.2, linestyle='-', label='Train CLIP (Smooth)')
    if 'recon_loss' in train_df.columns and not train_df['recon_loss'].isna().all():
        ax1.plot(train_df['iter'], smooth_data(train_df['recon_loss'], args.smooth_window), 
                 color='darkorange', lw=1.2, linestyle='-', label='Train Recon (Smooth)')
    
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
        # Determine min and max considering all val curves
        val_cols_to_check = ['val_loss']
        val_loss_specific = [c for c in val_df.columns if c.startswith('val_loss_')]
        val_cols_to_check.extend(val_loss_specific)
        
        v_min = val_df[val_cols_to_check].min().min()
        v_max = val_df[val_cols_to_check].max().max()
        v_range = v_max - v_min
        if v_range == 0: v_range = v_max * 0.1
        if pd.isna(v_range): v_range = 0.1
        
        if not pd.isna(v_min) and not pd.isna(v_max):
            # Fix ticks for the grid
            yticks_val = np.linspace(v_min, v_max, target_ticks)
            ax1r.set_yticks(yticks_val)
            pad_val = v_range * margin_fraction
            ax1r.set_ylim(v_min - pad_val, v_max + pad_val)

    ax1r.plot(val_df['iter'], val_df['val_loss'], color='red', marker='o', linestyle='-.', 
             lw=1, markersize=4, label='Validation')
             
    colors_val = ['darkorchid', 'darkturquoise', 'coral', 'gold']
    val_loss_specific = [c for c in val_df.columns if c.startswith('val_loss_') and c != 'val_loss']
    for i, c in enumerate(val_loss_specific):
        if not val_df[c].isna().all():
            name = c.replace('val_loss_', '')
            ax1r.plot(val_df['iter'], val_df[c], color=colors_val[i % len(colors_val)], marker='x', linestyle='--', 
                     lw=0.7, markersize=4, label=f'Zero-Shot: {name}')                 

    # V4: Val CLIP & Recon components
    if 'val_clip_loss' in val_df.columns and not val_df['val_clip_loss'].isna().all():
         ax1r.plot(val_df['iter'], val_df['val_clip_loss'], color='green', marker='s', linestyle=':', 
                  lw=0.8, markersize=3, label='Val CLIP')
    if 'val_recon_loss' in val_df.columns and not val_df['val_recon_loss'].isna().all():
         ax1r.plot(val_df['iter'], val_df['val_recon_loss'], color='orange', marker='d', linestyle=':', 
                  lw=0.8, markersize=3, label='Val Recon')                 
    ax1r.set_ylabel(r'\textbf{Validation Loss}', color='red')
    ax1r.tick_params(axis='y', labelcolor='red')
    
    ax1r.grid(False) 
    
    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
    
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
    ax2.plot(train_df['iter'], train_df['grad_norm'], color=color_grad, lw=1.5, label='Grad Norm (Total)')
    
    grad_cols = [c for c in train_df.columns if c.startswith('grad_') and c not in ['grad_norm', 'grad_backbone', 'grad_total']]
    colors_grad = ['green', 'orange', 'purple', 'brown']
    
    for i, c in enumerate(grad_cols):
        ax2.plot(train_df['iter'], train_df[c], color=colors_grad[i % len(colors_grad)], alpha=0.6, lw=1.0, linestyle='--', label=f'Grad {c.split("grad_")[1]}')
        
    if 'grad_backbone' in train_df.columns:
        ax2.plot(train_df['iter'], train_df['grad_backbone'], color='gray', alpha=0.6, lw=1.0, linestyle='--', label='Grad Backbone')
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
    
    ax2.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)
    
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
    
    ax3.plot(train_df['iter'], train_df['lr'], color='black', lw=1.5, label='Base LR')
    lr_cols = [c for c in train_df.columns if c.startswith('lr_') and c not in ['lr', 'lr_base', 'lr_backbone']]
    colors_lr = ['green', 'orange', 'purple', 'brown']
    
    for i, c in enumerate(lr_cols):
        ax3.plot(train_df['iter'], train_df[c], color=colors_lr[i % len(colors_lr)], lw=1.0, linestyle='-.', label=f'LR {c.split("lr_")[1]}')
        
    if 'lr_backbone' in train_df.columns:
        ax3.plot(train_df['iter'], train_df['lr_backbone'], color='gray', lw=1.0, linestyle='-.', label='LR Backbone')
        
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)

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
        
        ax4.plot(mfu_data['iter'], mfu_data['mfu'], color=color_mfu, lw=1.5, label='MFU (\\%)')
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
        
        ax4r.plot(train_df['iter'], train_df['mem_gb'], color=color_mem, linestyle='--', lw=1.5, label='VRAM (GB)')
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
    
    save_path = save_dir / args.save_name
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    print(f" --> Dashboard saved to: {save_path}")

if __name__ == "__main__":
    main()