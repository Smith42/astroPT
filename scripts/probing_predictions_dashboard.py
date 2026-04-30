"""
AstroPT Prediction Density Maps Dashboard.

This script reads the saved .npz predictions from `probing_downstream.py`
and generates a 3x3 density grid (KNN, LP, MLP x Images, Spectra, Joint)
for each regression target to visualize the predicted vs real distribution.

Features:
- Hexbin density plots for clear visualization of overlapping points.
- Calculates and displays R^2 for each subplot.
- y=x reference line.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import logging
import sys

# Logger Configuration
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-PredDash")

# Plotting Global Configuration
plt.rcParams['text.latex.preamble'] = r'''
            \usepackage{siunitx}
            \usepackage{bm}
            \usepackage{amsmath} 
            '''
plt.rc('text', usetex=True)
plt.rc('font', family='serif', weight='bold') 

plt.rcParams.update({
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'font.size': 14,          
    'axes.labelsize': 16,     
    'axes.titlesize': 18,     
    'xtick.labelsize': 14,   
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titlesize': 22,
    'figure.titleweight': 'bold',
})

PROBES = ['knn', 'lp', 'mlp']
MODES = ['images', 'spectra', 'joint']

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Prediction Density Maps Dashboard.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing the .npz prediction files.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the PNG plots.")
    parser.add_argument("--targets", nargs='+', default=None, help="Specific targets to plot. If None, plots all found.")
    return parser.parse_args()

def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    
    if not pred_dir.exists():
        logger.error(f"Prediction directory not found: {pred_dir}")
        sys.exit(1)
        
    save_dir = Path(args.save_dir) if args.save_dir else pred_dir.parent / "prediction_maps"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-discover targets from files
    # Format: preds_{target}_{mode}_{probe}.npz
    all_files = list(pred_dir.glob("preds_*.npz"))
    
    if not all_files:
        logger.error(f"No .npz prediction files found in {pred_dir}")
        sys.exit(1)
        
    found_targets = set()
    for f in all_files:
        # File name format assumes preds_{target}_{mode}_{probe}.npz
        # Because target name could contain underscores (e.g. HALPHA_FLUX),
        # we parse backwards.
        parts = f.stem.split('_')
        if len(parts) >= 4:
            probe = parts[-1]
            mode = parts[-2]
            target = "_".join(parts[1:-2])
            found_targets.add(target)
            
    if args.targets:
        targets_to_plot = [t for t in args.targets if t in found_targets]
    else:
        targets_to_plot = list(found_targets)
        
    logger.info(f"Found {len(targets_to_plot)} targets to plot: {targets_to_plot}")
    
    for target in targets_to_plot:
        logger.info(f"Generating density maps for target: {target}")
        
        fig, axes = plt.subplots(3, 3, figsize=(16, 16))
        fig.suptitle(f'Prediction Density Map: {target.replace("_", r"\_")}', y=0.95)
        
        # Determine global min/max for the target to set unified axes
        global_min = float('inf')
        global_max = float('-inf')
        
        # First pass to find limits
        for i, probe in enumerate(PROBES):
            for j, mode in enumerate(MODES):
                file_path = pred_dir / f"preds_{target}_{mode}_{probe}.npz"
                if file_path.exists():
                    data = np.load(file_path)
                    true_vals = data['true_vals']
                    preds = data['preds']
                    all_vals = np.concatenate([true_vals, preds])
                    global_min = min(global_min, np.min(all_vals))
                    global_max = max(global_max, np.max(all_vals))
                    
        # Add a small margin
        margin = (global_max - global_min) * 0.05
        axis_min = global_min - margin
        axis_max = global_max + margin

        # Valid limits check
        if np.isinf(axis_min) or np.isinf(axis_max):
            logger.warning(f"Skipping target {target} due to invalid limits (no valid data).")
            plt.close(fig)
            continue
        
        for i, probe in enumerate(PROBES):
            for j, mode in enumerate(MODES):
                ax = axes[i, j]
                file_path = pred_dir / f"preds_{target}_{mode}_{probe}.npz"
                
                if file_path.exists():
                    data = np.load(file_path)
                    true_vals = data['true_vals']
                    preds = data['preds']
                    
                    # Calculate R2
                    r2 = r2_score(true_vals, preds)
                    
                    # Density hexbin plot
                    hb = ax.hexbin(true_vals, preds, gridsize=50, cmap='inferno', bins='log', mincnt=1)
                    
                    # Reference line y = x
                    ax.plot([axis_min, axis_max], [axis_min, axis_max], 'w--', alpha=0.6, linewidth=2)
                    
                    ax.set_title(f"{mode.upper()} - {probe.upper()} ($R^2 = {r2:.3f}$)")
                    ax.set_xlim(axis_min, axis_max)
                    ax.set_ylim(axis_min, axis_max)
                    
                    # Colorbar (optional, but good for density)
                    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
                    cb.set_label('log10(N)')
                    
                else:
                    ax.text(0.5, 0.5, "Data Not Available", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{mode.upper()} - {probe.upper()}")
                    ax.set_xticks([])
                    ax.set_yticks([])
                
                # Labels
                if i == 2:
                    ax.set_xlabel('True Values')
                if j == 0:
                    ax.set_ylabel('Predicted Values')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        safe_target = target.replace('/', '_').replace(' ', '_')
        save_path = save_dir / f"pred_density_map_{safe_target}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f" -> Saved: {save_path}")

if __name__ == "__main__":
    main()
