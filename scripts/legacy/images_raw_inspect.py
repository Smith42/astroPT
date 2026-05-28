"""
AstroPT Split Comparator.

Diagnoses why Validation Loss might be lower than Training Loss.
Analyzes distributions of Redshift, Flux, and Missing Data across splits.
"""

import argparse
import logging
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow

# Logger
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("Split-Check")

class DummyRegistry:
    def get_config(self, name): return None

def parse_args():
    parser = argparse.ArgumentParser(description="Compare Train vs Test distributions")
    parser.add_argument("--data_dir", type=str, required=True, help="Arrow data root")
    parser.add_argument("--num_samples", type=int, default=2000, help="Samples per split to analyze")
    parser.add_argument("--out_dir", type=str, default="plots_analysis", help="Where to save histograms")
    return parser.parse_args()

def get_split_stats(data_dir, split, num_samples):
    logger.info(f"--- Analyzing Split: {split.upper()} ---")
    
    try:
        ds = EuclidDESIDatasetArrow(
            arrow_folder_root=data_dir,
            split=split,
            modality_registry=DummyRegistry(),
            transform=None
        )
    except Exception as e:
        logger.error(f"Could not load {split}: {e}")
        return None

    total = len(ds)
    indices = np.random.choice(total, min(num_samples, total), replace=False)
    
    stats = {
        "z": [],
        "vis_flux": [],
        "nisp_flux": [],
        "zeros_count": 0,  # Images totally black/missing
        "missing_nisp": 0  # Only NISP missing
    }
    
    for idx in indices:
        try:
            # Direct raw access
            rec = ds.ds[int(idx)]
            
            # 1. Redshift
            z = rec.get('redshift', 0.0)
            stats["z"].append(z)
            
            # 2. Flux Stats
            vis = np.array(rec['image_vis'])
            # NISP stack
            y = np.array(rec['image_nisp_y'])
            j = np.array(rec['image_nisp_j'])
            h = np.array(rec['image_nisp_h'])
            
            # Check for empty/zeros
            vis_mean = np.mean(vis) if vis.size > 0 else 0
            stats["vis_flux"].append(vis_mean)
            
            nisp_stack = np.concatenate([y, j, h]) if (y.size and j.size and h.size) else np.array([])
            
            if nisp_stack.size > 0:
                nisp_mean = np.mean(nisp_stack)
                stats["nisp_flux"].append(nisp_mean)
                
                # Check for "Missing NISP" (Dropout like sample 3 we saw)
                if np.max(nisp_stack) == 0:
                    stats["missing_nisp"] += 1
            else:
                stats["missing_nisp"] += 1
                stats["nisp_flux"].append(0)

            # Check for total darkness
            if vis_mean == 0 and (nisp_stack.size == 0 or np.mean(nisp_stack) == 0):
                stats["zeros_count"] += 1
                
        except Exception as e:
            continue

    return stats

def plot_comparisons(train_stats, test_stats, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    metrics = [
        ("z", "Redshift Distribution", "Redshift (z)"),
        ("vis_flux", "VIS Flux Mean Distribution", "Flux (Nanomaggies)"),
        ("nisp_flux", "NISP Flux Mean Distribution", "Flux (Nanomaggies)")
    ]
    
    for key, title, xlabel in metrics:
        plt.figure(figsize=(10, 6))
        
        # Histograms
        plt.hist(train_stats[key], bins=50, alpha=0.5, label='Train', density=True, color='blue')
        plt.hist(test_stats[key], bins=50, alpha=0.5, label='Test', density=True, color='orange')
        
        # Means
        mean_tr = np.mean(train_stats[key])
        mean_te = np.mean(test_stats[key])
        plt.axvline(mean_tr, color='blue', linestyle='dashed', linewidth=1)
        plt.axvline(mean_te, color='orange', linestyle='dashed', linewidth=1)
        
        plt.title(f"{title}\nTrain $\mu$={mean_tr:.3f} | Test $\mu$={mean_te:.3f}")
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(alpha=0.3)
        
        save_path = os.path.join(out_dir, f"comp_{key}.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved plot: {save_path}")

def main():
    args = parse_args()
    
    # 1. Get Stats
    s_train = get_split_stats(args.data_dir, "train", args.num_samples)
    s_test = get_split_stats(args.data_dir, "test", args.num_samples)
    
    if not s_train or not s_test:
        logger.error("Failed to gather stats.")
        return

    # 2. Print Report
    print("\n" + "="*40)
    print("      DATA SPLIT DIAGNOSTICS      ")
    print("="*40)
    
    def print_metric(name, tr_val, te_val):
        diff = ((te_val - tr_val) / (tr_val + 1e-9)) * 100
        print(f"{name:<20} | Train: {tr_val:.4f} | Test: {te_val:.4f} | Diff: {diff:+.1f}%")

    # Means
    print("\n--- Averages (Physics) ---")
    print_metric("Mean Redshift", np.mean(s_train["z"]), np.mean(s_test["z"]))
    print_metric("Mean VIS Flux", np.mean(s_train["vis_flux"]), np.mean(s_test["vis_flux"]))
    print_metric("Mean NISP Flux", np.mean(s_train["nisp_flux"]), np.mean(s_test["nisp_flux"]))
    
    # Quality
    print("\n--- Data Quality (Per 100 samples) ---")
    n_tr = len(s_train["z"])
    n_te = len(s_test["z"])
    
    tr_miss = (s_train["missing_nisp"] / n_tr) * 100
    te_miss = (s_test["missing_nisp"] / n_te) * 100
    print(f"Missing NISP Bands   | Train: {tr_miss:.1f}%   | Test: {te_miss:.1f}%")
    
    tr_zero = (s_train["zeros_count"] / n_tr) * 100
    te_zero = (s_test["zeros_count"] / n_te) * 100
    print(f"Totally Empty Inputs | Train: {tr_zero:.1f}%   | Test: {te_zero:.1f}%")
    
    print("\n" + "="*40)
    
    # 3. Plots
    plot_comparisons(s_train, s_test, args.out_dir)

if __name__ == "__main__":
    main()