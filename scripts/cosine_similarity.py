"""
AstroPT Alignment Analyst.

This script loads pre-computed embeddings from disk (Numpy Memmap) and 
performs a full alignment analysis without loading the model.

Pipeline:
1. Loads images.npy and spectra.npy (Memmap).
2. Computes diagonal Cosine Similarity (Self-Similarity).
3. Computes Retrieval Metrics (Top-1, Top-5) using batched GPU operations.
4. Generates publication-quality plots.

Author: Victor Alonso Rodriguez
Date: January 2026
"""

import argparse
import glob
import json
import logging
import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Logger Configuration
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-Analyst")

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
    parser = argparse.ArgumentParser(description="AstroPT Alignment Analyst")
    
    parser.add_argument("--out_dir", type=str, required=True, help="Directory containing the checkpoint (e.g., logs/train_name)")
    parser.add_argument("--emb_dir", type=str, required=True, 
                        help="Directory containing the .npy files (e.g., embeddings_runX_ckptY)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for matrix operations")
    parser.add_argument("--batch_size", type=int, default=1000, 
                        help="Batch size for retrieval calculation (adjust based on GPU VRAM)")
    parser.add_argument("--train_name", type=str, default=None, help="Custom title for the plot (defaults to folder name)")
    
    return parser.parse_args()

def calculate_retrieval_metrics(
    img_mmap: np.memmap, 
    spec_mmap: np.memmap, 
    device: torch.device,
    batch_size: int = 1000
) -> Tuple[float, float]:
    """
    Calculates Top-1 and Top-5 retrieval accuracy using batched processing.
    We iterate through the query images in batches, but we need to compare 
    against ALL spectra (the gallery).
    """
    n_samples = img_mmap.shape[0]
    top1_correct = 0
    top5_correct = 0
    
    logger.info(f"Computing Retrieval Metrics for {n_samples} samples...")
    
    # Pre-load ALL spectra to GPU if VRAM
    try:
        # Normalize Spectra
        logger.info("Loading Spectra to GPU...")
        gallery_spec = torch.from_numpy(spec_mmap[:]).to(device)
        gallery_spec = F.normalize(gallery_spec, p=2, dim=1)
    except RuntimeError:
        logger.warning("VRAM full loading Spectra. Falling back to CPU-heavy batched search (Slower).")
        gallery_spec = None

    if gallery_spec is not None:
        for i in tqdm(range(0, n_samples, batch_size), desc="Retrieval"):
            end = min(i + batch_size, n_samples)
            
            # Load Query Batch (Images)
            img_batch = torch.from_numpy(img_mmap[i:end]).to(device)
            img_batch = F.normalize(img_batch, p=2, dim=1)
            
            # Similarity Matrix
            sim_matrix = torch.matmul(img_batch, gallery_spec.T)
            
            # Get Top-K
            _, top_indices = sim_matrix.topk(5, dim=1)
            
            # Ground Truth (GT) targets for this batch
            targets = torch.arange(i, end, device=device).view(-1, 1)
            
            # Check hits
            top1_correct += (top_indices[:, 0].unsqueeze(1) == targets).sum().item()
            top5_correct += (top_indices == targets).sum().item()
            
            # Free VRAM
            del img_batch, sim_matrix, targets, top_indices

    return top1_correct / n_samples, top5_correct / n_samples

def compute_diagonal_stats(
    img_mmap: np.memmap, 
    spec_mmap: np.memmap, 
    device: torch.device,
    batch_size: int = 5000
) -> np.ndarray:
    """
    Computes the cosine similarity between the image and ITS OWN spectrum (diagonal).
    Low memory footprint.
    """
    n_samples = img_mmap.shape[0]
    all_sims = []
    
    logger.info("Computing Self-Similarity Statistics...")
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        
        # Load small chunks
        img_chunk = torch.from_numpy(img_mmap[i:end]).to(device)
        spec_chunk = torch.from_numpy(spec_mmap[i:end]).to(device)
        
        # Normalize
        img_chunk = F.normalize(img_chunk, p=2, dim=1)
        spec_chunk = F.normalize(spec_chunk, p=2, dim=1)
        
        # Compute Cosine Similarity (Element-wise)
        sims = F.cosine_similarity(img_chunk, spec_chunk, dim=1)
        
        all_sims.append(sims.cpu().numpy())
        
    return np.concatenate(all_sims)

def plot_histogram(sim_values: np.ndarray, out_dir: str, stats: dict, suffix: str = "", retrieval_stats: dict = None):
    """Generates the distribution plot."""
    logger.info("Generating plot...")
    plt.figure(figsize=(18, 6))
    
    # Plotting data
    plt.hist(sim_values, bins=200, color='royalblue', edgecolor='black', linewidth=0.2, alpha=1, range=(-1, 1), zorder=3)
    
    # Plotting mean and median lines
    plt.axvline(stats['mean'], color='crimson', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.2f}", zorder=4)
    plt.axvline(stats['median'], color='darkorange', linestyle=':', linewidth=2, label=f"Median: {stats['median']:.2f}", zorder=4)
    
    # Top 1 and Top 5 information
    if retrieval_stats is not None:
        
        textstr = '\n'.join((
            r"\textbf{Retrieval Metrics}",
            fr"Top-1: {retrieval_stats['top1']*100:.5f}\%",
            fr"Top-5: {retrieval_stats['top5']*100:.5f}\%",
            fr"Random: {retrieval_stats['random']*100:.5f}\%"
        ))
        
        # Text box properties
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5)
        
        # Text boz location
        plt.gca().text(0.98, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
                       verticalalignment='top', horizontalalignment='right', bbox=props, zorder=5)

    # Cutomizing the plot
    plt.title(r"\textbf{Image-Spectrum Alignment Distribution}" + f"\n[{suffix}]", fontsize=16)
    plt.xlabel(r"Cosine Similarity ($\cos\theta$)", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3, zorder='1')
    
    # Saving the plot
    plot_path = os.path.join(out_dir, "alignment_histogram.png")
    plt.savefig(plot_path, format='png', dpi=300, bbox_inches='tight')
    logger.info(f"-> Plot saved to {plot_path}")

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Locate Files from .npy
    img_path = os.path.join(args.emb_dir, "images.npy")
    spec_path = os.path.join(args.emb_dir, "spectra.npy")
    
    img_data = None
    spec_data = None
        
    # Load with Memmap 
    if os.path.exists(img_path) and os.path.exists(spec_path):
        logger.info(f"Found unpacked .npy files in {args.emb_dir}")
        logger.info("-> Using Memory Mapping (Low RAM mode)")
        img_data = np.load(img_path, mmap_mode='r')
        spec_data = np.load(spec_path, mmap_mode='r')

    # Fallback to .npz file
    else:
        logger.warning(f"Memmap files (.npy) not found in {args.emb_dir}")
        logger.info("-> Searching for compressed .npz fallback...")
        
        npz_files = glob.glob(os.path.join(args.emb_dir, "*.npz"))
        
        if not npz_files:
            logger.error(f"CRITICAL: No .npy files AND no .npz files found in {args.emb_dir}")
            sys.exit(1)
            
        # Taking the first file
        npz_target = npz_files[0]
        logger.info(f"-> Loading backup archive: {os.path.basename(npz_target)}")
        logger.warning("-> CAUTION: Loading fully into RAM (Legacy Mode).")
        
        try:
            # Load file
            archive = np.load(npz_target)
            
            # Verify keys
            if 'images' in archive and 'spectra' in archive:
                img_data = archive['images']
                spec_data = archive['spectra']
            else:
                logger.error(f"The .npz file exists but keys are missing. Found: {list(archive.keys())}")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Failed to read .npz file: {e}")
            sys.exit(1)
    
    # Validate dimensions
    n_samples = img_data.shape[0]
    dim = img_data.shape[1]
    
    logger.info("-" * 40)
    logger.info(f"Data Loaded Successfully")
    logger.info(f"Samples:    {n_samples}")
    logger.info(f"Dimensions: {dim}")
    logger.info(f"Backend:    {'Numpy Memmap' if isinstance(img_data, np.memmap) else 'Numpy Array (RAM)'}")
    logger.info("-" * 40)
    
    # Compute Diagonal Stats
    self_sims = compute_diagonal_stats(img_data, spec_data, device)
    
    mean_sim = np.mean(self_sims)
    median_sim = np.median(self_sims)
    std_sim = np.std(self_sims)
    
    logger.info("-" * 40)
    logger.info(f"Mean Cosine Similarity:   {mean_sim:.4f}")
    logger.info(f"Median Cosine Similarity: {median_sim:.4f}")
    logger.info(f"Std Dev:                  {std_sim:.4f}")
    logger.info("-" * 40)
    
    # Compute Retrieval Top Accuracy
    if n_samples > 100:
        top1, top5 = calculate_retrieval_metrics(img_data, spec_data, device, args.batch_size)
        random_chance = 1.0 / n_samples
        
        # Saving the results
        retrieval_stats = {
            'top1': top1,
            'top5': top5,
            'random': random_chance
        }
        
        logger.info(f"Top-1 Accuracy: {top1:.2%} (Random: {random_chance:.5%})")
        logger.info(f"Top-5 Accuracy: {top5:.2%}")
        
        diagnosis = "POOR"
        if top1 > 0.5: diagnosis = "EXCELLENT"
        elif top1 > 0.1: diagnosis = "MODERATE"
        logger.info(f"Diagnosis: {diagnosis}")
        logger.info("-" * 40)
    
    # Saving plot
    save_dir = args.out_dir if hasattr(args, 'out_dir') and args.out_dir else args.emb_dir
    stats = {'mean': mean_sim, 'median': median_sim}
    
    # TITLE LOGIC
    config_path = os.path.join(save_dir, "config.json")
    json_name = None
    
    # Reading config.json
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                json_name = config.get("train_name", None)
        except Exception:
            pass 

    # Select ID: CLI > JSON > Folder
    if args.train_name:
        train_name = args.train_name
    elif json_name:
        train_name = json_name
    else:
        train_name = os.path.basename(os.path.normpath(save_dir))
    
    # Plotting the histogram
    plot_histogram(self_sims, save_dir, stats, suffix=train_name, retrieval_stats=retrieval_stats)

if __name__ == "__main__":
    main()