"""
AstroPT Prediction Density Maps Dashboard.

This script reads the saved .npz predictions from `probing_downstream.py`
and generates a single large plot with N rows (one per target, sorted alphabetically)
and 9 columns (KNN, LP, MLP x Images, Spectra, Joint) to visualize predictions
for all targets in one comprehensive visualization.

Features:
- Hexbin density plots for clear visualization of overlapping points.
- Calculates and displays R^2 for each subplot in the legend.
- y=x reference line.
- Reads config.json from the run directory for figure title.
- Targets sorted alphabetically (uppercase first, then lowercase).
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titlesize': 18,
    'figure.titleweight': 'bold',
})

PROBES = ['knn', 'lp', 'mlp']
MODES = ['images', 'spectra', 'joint']

CLASSIFICATION_TARGETS = ['SPECTYPE', 'data_set_release']
FLUX_TARGETS = ['flux_detection_total', 'HALPHA_EW', 'HALPHA_FLUX', 'NII_6584_FLUX', 'OIII_5007_FLUX', 'HBETA_FLUX']

def apply_scaling(vals: np.ndarray, target: str) -> np.ndarray:
    """Apply asinh scaling for flux/EW targets to improve visualization of long tails."""
    if target in FLUX_TARGETS:
        return np.arcsinh(vals)
    return vals

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Prediction Density Maps Dashboard.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing the .npz prediction files.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the PNG plot.")
    parser.add_argument("--targets", nargs='+', default=None, help="Specific targets to plot. If None, plots all found.")
    return parser.parse_args()


def find_config_json(pred_dir: Path) -> Dict:
    """
    Search for config.json in parent directories of pred_dir.
    Looks up to 10 levels up from pred_dir to ensure it finds the run root.
    """
    search_dir = pred_dir
    for _ in range(10):
        # Check current dir, then weights child
        config_candidates = [
            search_dir / "config.json",
            search_dir / "weights" / "config.json"
        ]
        
        for config_path in config_candidates:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        logger.info(f"Found config.json at: {config_path}")
                        return config
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse config.json at {config_path}: {e}")
        
        if search_dir == search_dir.parent:
            break
        search_dir = search_dir.parent

    logger.warning("config.json not found in parent directories.")
    return {}


def get_figure_title(config: Dict) -> str:
    """
    Extract title from config matching the umaps style.
    """
    train_name = 'Unknown Run'
    for key in ['train_name', 'TITLE', 'title', 'name', 'exp_name']:
        if key in config:
            train_name = str(config[key])
            break
    
    safe_train_name = train_name.replace('_', r'\_')
    title_text = rf"\textbf{{AstroPT Downstream Probing - Prediction Density Maps}}" + f"\n[{safe_train_name}]"
    return title_text


def sort_targets(targets: List[str]) -> List[str]:
    """
    Sort targets alphabetically with uppercase first, then lowercase.
    """
    uppercase = sorted([t for t in targets if t[0].isupper()])
    lowercase = sorted([t for t in targets if t[0].islower()])
    return uppercase + lowercase


def get_target_axis_limits(pred_dir: Path, target: str) -> Tuple[float, float]:
    """
    Compute min/max across all modes/probes for a SINGLE target.
    """
    target_min = float('inf')
    target_max = float('-inf')

    for mode in MODES:
        for probe in PROBES:
            file_path = pred_dir / f"preds_{target}_{mode}_{probe}.npz"
            if file_path.exists():
                try:
                    data = np.load(file_path)
                    true_vals = apply_scaling(data['true_vals'], target)
                    preds = apply_scaling(data['preds'], target)
                    all_vals = np.concatenate([true_vals, preds])
                    target_min = min(target_min, np.nanmin(all_vals))
                    target_max = max(target_max, np.nanmax(all_vals))
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")
                    continue

    if np.isinf(target_min) or np.isinf(target_max):
        return 0.0, 1.0

    # Add margin
    margin = (target_max - target_min) * 0.05
    axis_min = target_min - margin
    axis_max = target_max + margin

    return axis_min, axis_max


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
        targets_to_plot = [t for t in args.targets if t in found_targets and t not in CLASSIFICATION_TARGETS]
    else:
        targets_to_plot = [t for t in found_targets if t not in CLASSIFICATION_TARGETS]

    # Sort targets: uppercase first, then lowercase
    targets_to_plot = sort_targets(targets_to_plot)

    logger.info(f"Found {len(targets_to_plot)} targets to plot: {targets_to_plot}")

    # Get config for title
    config = find_config_json(pred_dir)
    figure_title = get_figure_title(config)

    # Create figure with N rows and 9 columns
    n_targets = len(targets_to_plot)
    fig, axes = plt.subplots(n_targets, 9, figsize=(27, 3 * n_targets))

    # Handle single target case (axes is 1D, not 2D)
    if n_targets == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(figure_title, fontsize=24, y=1.02)

    logger.info("Generating unified density map dashboard...")

    for row_idx, target in enumerate(targets_to_plot):
        logger.info(f"Processing target: {target}")
        
        # Compute axis limits specific to this target
        axis_min, axis_max = get_target_axis_limits(pred_dir, target)

        for col_idx, (mode, probe) in enumerate([(m, p) for m in MODES for p in PROBES]):
            ax = axes[row_idx, col_idx]
            file_path = pred_dir / f"preds_{target}_{mode}_{probe}.npz"

            if file_path.exists():
                try:
                    data = np.load(file_path)
                    true_vals = apply_scaling(data['true_vals'], target)
                    preds = apply_scaling(data['preds'], target)

                    # Calculate R2
                    r2 = r2_score(data['true_vals'], data['preds']) # Calculate R2 on original scale

                    # Density hexbin plot
                    hb = ax.hexbin(true_vals, preds, gridsize=120, cmap='Blues',
                                  bins='log', mincnt=1)

                    # Reference line y = x
                    ax.plot([axis_min, axis_max], [axis_min, axis_max], 'r--',
                           alpha=0.8, linewidth=1.0, label=r'$y=x$')

                    # Set labels and limits
                    ax.set_xlim(axis_min, axis_max)
                    ax.set_ylim(axis_min, axis_max)
                    ax.set_aspect('equal')

                    # Title with mode and probe (no R2 in title)
                    title_text = f"{mode.capitalize()}\n{probe.upper()}"
                    ax.set_title(title_text, fontsize=12, pad=8)

                    # Add legend with R2
                    legend_label = rf'$R^2 = {r2:.3f}$'
                    ax.legend([Patch(visible=False)], [legend_label], loc='upper left', 
                              fontsize=10, framealpha=0.0, handlelength=0)

                    # Remove colorbar to save space (or add if needed)
                    # cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
                    # cb.set_label('log10(N)', fontsize=9)

                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
                    ax.text(0.5, 0.5, "Error\nLoading Data", ha='center', va='center',
                           transform=ax.transAxes, fontsize=11, style='italic', color='red')
                    ax.set_xticks([])
                    ax.set_yticks([])

            else:
                ax.text(0.5, 0.5, "Data Not\nAvailable", ha='center', va='center',
                       transform=ax.transAxes, fontsize=11, style='italic', color='gray')
                ax.set_xticks([])
                ax.set_yticks([])

            # Add target name on the left side (row label)
            if col_idx == 0:
                ax.text(-0.45, 0.5, target.replace("_", r"\_"), transform=ax.transAxes,
                       fontsize=12, va='center', ha='right', fontweight='bold',
                       rotation=0)

            # X and Y axis labels only on outer edges
            if row_idx == n_targets - 1:  # Bottom row
                ax.set_xlabel('True Values', fontsize=11, fontweight='bold')

            if col_idx == 0:  # Left column
                ax.set_ylabel('Predicted Values', fontsize=11, fontweight='bold')

    plt.tight_layout(rect=[0.04, 0, 1, 0.99])

    # Save the plot
    save_path = save_dir / "prediction_density_dashboard.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved unified dashboard to: {save_path}")
    plt.close()


if __name__ == "__main__":
    main()