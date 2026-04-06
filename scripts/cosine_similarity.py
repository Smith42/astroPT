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
import json
import logging
import sys
from pathlib import Path
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
    
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing training weights")
    parser.add_argument("--emb_dir", type=str, required=True, help="Directory containing the .npy files (e.g., embeddings_runX_ckptY)")
    parser.add_argument("--save_dir", type=str, default=None, help="Plot Saving Directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device for matrix operations")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for retrieval calculation (adjust based on GPU VRAM)")
    parser.add_argument("--train_name", type=str, default=None, help="Custom title for the plot (defaults to folder name)")
    
    return parser.parse_args()

def calculate_retrieval_and_advanced_metrics(
    img_mmap: np.memmap, 
    spec_mmap: np.memmap, 
    device: torch.device,
    batch_size: int = 1000
) -> dict:
    """
    Calculates Top-1, Top-5 retrieval accuracy, and Hubness (Skewness).
    """
    n_samples = img_mmap.shape[0]
    top1_correct = 0
    top5_correct = 0
    
    # Track how many times each spectrum is retrieved as Top-1
    hub_counts_spec = torch.zeros(n_samples, dtype=torch.long, device=device)
    
    logger.info(f"Computing Retrieval & Hubness for {n_samples} samples...")
    
    try:
        # Pre-load ALL spectra to GPU
        logger.info("Loading Spectra to GPU...")
        gallery_spec = torch.from_numpy(spec_mmap[:]).to(device)
        gallery_spec = F.normalize(gallery_spec, p=2, dim=1)
    except RuntimeError:
        logger.warning("VRAM full loading Spectra. Retrieval requires Gallery in VRAM.")
        return {'top1': 0.0, 'top5': 0.0, 'hubness_skew': 0.0, 'hub_array': None}

    for i in tqdm(range(0, n_samples, batch_size), desc="Retrieval & Hubness"):
        end = min(i + batch_size, n_samples)
        
        # Load Query Batch (Images)
        img_batch = torch.from_numpy(img_mmap[i:end]).to(device)
        img_batch = F.normalize(img_batch, p=2, dim=1)
        
        # Similarity Matrix
        sim_matrix = torch.matmul(img_batch, gallery_spec.T)
        
        # Get Top-K
        _, top_indices = sim_matrix.topk(5, dim=1)
        
        # Track hits for Hubness (Top 1)
        top1_indices = top_indices[:, 0]
        unique_idx, counts = torch.unique(top1_indices, return_counts=True)
        hub_counts_spec[unique_idx] += counts
        
        # Ground Truth (GT) targets for this batch
        targets = torch.arange(i, end, device=device).view(-1, 1)
        
        # Check hits
        top1_correct += (top_indices[:, 0].unsqueeze(1) == targets).sum().item()
        top5_correct += (top_indices == targets).sum().item()
        
        del img_batch, sim_matrix, targets, top_indices

    top1 = top1_correct / n_samples
    top5 = top5_correct / n_samples
    
    # Compute Hubness Skewness
    hub_array = hub_counts_spec.float().cpu()
    mean_hubs = hub_array.mean()
    std_hubs = hub_array.std()
    hubness_skew = (((hub_array - mean_hubs) / std_hubs)**3).mean().item() if std_hubs > 0 else 0.0
    
    return {
        'top1': top1, 
        'top5': top5, 
        'hubness_skew': hubness_skew, 
        'hub_array': hub_array.numpy()
    }

def calculate_cka(img_mmap: np.memmap, spec_mmap: np.memmap, device: torch.device, batch_size: int = 5000) -> float:
    """Computes Linear Feature CKA incrementally over mini-batches to save memory."""
    n_samples, dim = img_mmap.shape
    logger.info("Computing Minibatch Feature CKA...")
    
    mean_x = torch.zeros(dim, device=device, dtype=torch.float64)
    mean_y = torch.zeros(dim, device=device, dtype=torch.float64)
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        mean_x += torch.from_numpy(img_mmap[i:end]).to(device).double().sum(dim=0)
        mean_y += torch.from_numpy(spec_mmap[i:end]).to(device).double().sum(dim=0)
        
    mean_x /= n_samples
    mean_y /= n_samples
    
    cov_xx = torch.zeros((dim, dim), device=device, dtype=torch.float64)
    cov_yy = torch.zeros((dim, dim), device=device, dtype=torch.float64)
    cov_xy = torch.zeros((dim, dim), device=device, dtype=torch.float64)
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        x_batch = torch.from_numpy(img_mmap[i:end]).to(device).double() - mean_x
        y_batch = torch.from_numpy(spec_mmap[i:end]).to(device).double() - mean_y
        
        cov_xx += x_batch.T @ x_batch
        cov_yy += y_batch.T @ y_batch
        cov_xy += x_batch.T @ y_batch
        
    num = torch.norm(cov_xy, p='fro')**2
    denom = torch.norm(cov_xx, p='fro') * torch.norm(cov_yy, p='fro')
    
    return (num / denom).item() if denom > 0 else 0.0

def calculate_uniformity(img_mmap: np.memmap, spec_mmap: np.memmap, device: torch.device, sample_size: int = 5000) -> Tuple[float, float]:
    """Calculates Wang & Isola Uniformity (Log Expected Pairwise Distance) over a random subset."""
    n_samples = img_mmap.shape[0]
    idx = np.random.choice(n_samples, min(sample_size, n_samples), replace=False)
    
    logger.info(f"Computing Uniformity on a {len(idx)} sample subset...")
    x = F.normalize(torch.from_numpy(img_mmap[idx]).to(device), p=2, dim=1)
    y = F.normalize(torch.from_numpy(spec_mmap[idx]).to(device), p=2, dim=1)
    
    sqdist_x = 2.0 - 2.0 * (x @ x.T)
    sqdist_y = 2.0 - 2.0 * (y @ y.T)
    
    unif_x = torch.log(torch.mean(torch.exp(-2.0 * sqdist_x))).item()
    unif_y = torch.log(torch.mean(torch.exp(-2.0 * sqdist_y))).item()
    
    return unif_x, unif_y

def calculate_neighborhood_overlap(img_mmap: np.memmap, spec_mmap: np.memmap, device: torch.device, sample_size: int = 2000, K: int = 10) -> float:
    """Calculates Latent R-Precision proxy: Intersecting neighbors between Modalities."""
    n_samples = img_mmap.shape[0]
    idx = np.random.choice(n_samples, min(sample_size, n_samples), replace=False)
    
    logger.info(f"Computing Neighborhood Overlap (K={K}) on a {len(idx)} subset...")
    try:
        gallery_x = F.normalize(torch.from_numpy(img_mmap[:]).to(device), p=2, dim=1)
        gallery_y = F.normalize(torch.from_numpy(spec_mmap[:]).to(device), p=2, dim=1)
    except:
        logger.warning("VRAM full. Skipping Neighborhood Overlap.")
        return 0.0
        
    query_x = gallery_x[idx]
    query_y = gallery_y[idx]
    
    sim_x = query_x @ gallery_x.T
    _, top_x = sim_x.topk(K + 1, dim=1) 
    top_x = top_x[:, 1:] # Drop self
    
    sim_y = query_y @ gallery_y.T
    _, top_y = sim_y.topk(K + 1, dim=1)
    top_y = top_y[:, 1:]
    
    overlaps = []
    for i in range(len(idx)):
        set_x = set(top_x[i].cpu().numpy())
        set_y = set(top_y[i].cpu().numpy())
        intersect = len(set_x.intersection(set_y))
        union = len(set_x.union(set_y))
        overlaps.append(intersect / union if union > 0 else 0.0)
        
    return float(np.mean(overlaps))

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

def plot_latent_geometry_dashboard(
    sim_values: np.ndarray,
    hub_array: np.ndarray,
    save_dir: str | Path,
    metrics: dict,
    suffix: str = ""
):
    """Generates a comprehensive latent geometry dashboard."""
    logger.info("Generating Latent Geometry Dashboard...")
    fig, axs = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [2, 1.5]})
    
    # 1. Cosine Similarity Distribution
    ax = axs[0]
    ax.hist(sim_values, bins=200, color='royalblue', edgecolor='black', linewidth=0.2, alpha=1, range=(-1, 1), zorder=3)
    ax.axvline(metrics['cosine_mean'], color='crimson', linestyle='--', linewidth=2, label=f"Mean: {metrics['cosine_mean']:.2f}", zorder=4)
    ax.axvline(metrics['cosine_median'], color='darkorange', linestyle=':', linewidth=2, label=f"Median: {metrics['cosine_median']:.2f}", zorder=4)
    
    textstr = '\n'.join((
        r"\textbf{Geometric Metrics}",
        fr"Alignment $\downarrow$: {metrics['alignment']:.3f}",
        fr"Uniformity (I) $\downarrow$: {metrics['uniformity_img']:.2f}",
        fr"Uniformity (S) $\downarrow$: {metrics['uniformity_spec']:.2f}",
        fr"Feat CKA $\uparrow$: {metrics['cka']:.3f}"
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5)
    ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', horizontalalignment='right', bbox=props, zorder=5)
                   
    ax.set_title(r"\textbf{Modal Alignment Distribution}", fontsize=16)
    ax.set_xlabel(r"Cosine Similarity ($\cos\theta$)", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3, zorder=1)
    
    # 2. Hubness Distribution
    ax = axs[1]
    if hub_array is not None:
        sorted_hubs = np.sort(hub_array)[::-1]
        ax.plot(sorted_hubs, color='darkviolet', linewidth=2.5, zorder=3)
        ax.fill_between(range(len(sorted_hubs)), sorted_hubs, color='darkviolet', alpha=0.2, zorder=2)
        
        info_str = '\n'.join((
            r"\textbf{Retrieval Performance}",
            fr"Top-1 $\uparrow$: {metrics.get('top1', 0)*100:.3f}\%",
            fr"Top-5 $\uparrow$: {metrics.get('top5', 0)*100:.3f}\%",
            fr"N-IOU $\uparrow$: {metrics['neighborhood_overlap']*100:.2f}\%",
            fr"Hub Skew $\downarrow$: {metrics.get('hubness_skew', 0):.2f}"
        ))
        ax.text(0.95, 0.95, info_str, transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', horizontalalignment='right', bbox=props, zorder=5)
                       
        ax.set_title(r"\textbf{Gallery Hubness Profile}", fontsize=16)
        ax.set_xlabel("Sorted Gallery Spectra Rank", fontsize=14)
        ax.set_ylabel("Times Retrieved as Nearest", fontsize=14)
        ax.set_yscale('symlog')
        ax.grid(True, alpha=0.3, zorder=1)
    else:
        ax.text(0.5, 0.5, "Hubness data not available\n(Gallery VRAM limit)", ha='center', va='center', fontsize=14)
        ax.set_title(r"\textbf{Gallery Hubness Profile}", fontsize=16)
        
    fig.suptitle(r"\textbf{Latent Representation Geometry}" + f"\n{suffix}", fontsize=20, y=1.05)
    
    plot_path = Path(save_dir) / "latent_geometry_dashboard.png"
    plt.savefig(plot_path, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f" --> Geometry Dashboard saved to {plot_path}")

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Required paths
    weights_dir = Path(args.weights_dir)
    emb_dir = Path(args.emb_dir)
    
    # Check unimodal early exit
    config_path = weights_dir / "config.json"
    json_train_name = None
    if config_path.is_file():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                json_train_name = config.get("train_name", None)
                img_train = config.get("images_train", config.get("img_train", True))
                spec_train = config.get("spectra_train", config.get("spec_train", True))
                
                if not img_train or not spec_train:
                    logger.warning("Unimodal architecture detected in config.json.")
                    logger.info("Cosine Similarity requires Cross-Modal data. Exiting cleanly.")
                    sys.exit(0)
        except Exception as e:
            logger.warning(f"Failed to read config.json: {e}")
            
    # Create output directory
    save_dir = Path(args.save_dir) if args.save_dir else emb_dir / "similarity_metrics"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Locate Files from .npy
    img_path = emb_dir / "images.npy"
    spec_path = emb_dir / "spectra.npy"
    
    img_data = None
    spec_data = None
        
    # Load with Memmap 
    if img_path.is_file() and spec_path.is_file():
        logger.info(f"Found unpacked .npy files in {emb_dir}")
        logger.info(" --> Using Memory Mapping (Low RAM mode)")
        img_data = np.load(img_path, mmap_mode='r')
        spec_data = np.load(spec_path, mmap_mode='r')

    # Fallback to .npz file
    else:
        logger.warning(f"Memmap files (.npy) not found in {emb_dir}")
        logger.info(" --> Searching for compressed .npz fallback...")
        
        npz_files = list(emb_dir.glob("*.npz"))
        
        if not npz_files:
            logger.error(f"CRITICAL: No .npy files AND no .npz files found in {emb_dir}")
            sys.exit(1)
            
        # Taking the first file
        npz_target = npz_files[0]
        logger.warning(" --> CAUTION: Loading fully into RAM.")
        
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
    
    # Compute Diagonal Stats (Alignment is derived from this)
    self_sims = compute_diagonal_stats(img_data, spec_data, device)
    
    mean_sim = np.mean(self_sims)
    median_sim = np.median(self_sims)
    std_sim = np.std(self_sims)
    alignment_score = 2.0 - 2.0 * mean_sim
    
    logger.info("-" * 40)
    logger.info(f"Mean Cosine Similarity:   {mean_sim:.4f}")
    logger.info(f"Median Cosine Similarity: {median_sim:.4f}")
    logger.info(f"Std Dev:                  {std_sim:.4f}")
    logger.info(f"Alignment Score (L2):     {alignment_score:.4f}")
    logger.info("-" * 40)
    
    # Compute Advanced Geometric Metrics
    unif_img, unif_spec = calculate_uniformity(img_data, spec_data, device)
    cka_score = calculate_cka(img_data, spec_data, device)
    overlap_score = calculate_neighborhood_overlap(img_data, spec_data, device)
    
    logger.info(f"Uniformity (Images):      {unif_img:.4f}")
    logger.info(f"Uniformity (Spectra):     {unif_spec:.4f}")
    logger.info(f"Feature CKA (Alignment):  {cka_score:.4f}")
    logger.info(f"Neighborhood Overlap:     {overlap_score:.4f} (R-Precision proxy)")
    logger.info("-" * 40)
    
    retrieval_stats = None

    # Compute Retrieval Top Accuracy & Hubness
    if n_samples > 100:
        retrieval_res = calculate_retrieval_and_advanced_metrics(img_data, spec_data, device, args.batch_size)
        random_chance = 1.0 / n_samples
        
        # Saving the results
        retrieval_stats = {
            'top1': retrieval_res['top1'],
            'top5': retrieval_res['top5'],
            'hubness_skew': retrieval_res['hubness_skew'],
            'random': random_chance,
            'hub_array': retrieval_res['hub_array']
        }
        
        logger.info(f"Top-1 Accuracy:   {retrieval_res['top1']:.2%} (Random: {random_chance:.5%})")
        logger.info(f"Top-5 Accuracy:   {retrieval_res['top5']:.2%}")
        logger.info(f"Hubness Skewness: {retrieval_res['hubness_skew']:.4f}")
        
        diagnosis = "POOR"
        if retrieval_res['top1'] > 0.5: diagnosis = "EXCELLENT"
        elif retrieval_res['top1'] > 0.1: diagnosis = "MODERATE"
        logger.info(f"Diagnosis: {diagnosis}")
        logger.info("-" * 40)
    
    # Save machine-readable metrics for cross-experiment ranking.
    metrics_payload = {
        'n_samples': int(n_samples),
        'embedding_dim': int(dim),
        'cosine_mean': float(mean_sim),
        'cosine_median': float(median_sim),
        'cosine_std': float(std_sim),
        'alignment': float(alignment_score),
        'uniformity_img': float(unif_img),
        'uniformity_spec': float(unif_spec),
        'cka': float(cka_score),
        'neighborhood_overlap': float(overlap_score),
        'top1': float(retrieval_stats['top1']) if retrieval_stats is not None else None,
        'top5': float(retrieval_stats['top5']) if retrieval_stats is not None else None,
        'hubness_skew': float(retrieval_stats['hubness_skew']) if retrieval_stats is not None else None,
        'random_top1': float(retrieval_stats['random']) if retrieval_stats is not None else None,
    }
    metrics_json_path = save_dir / 'cosine_metrics.json'
    try:
        with open(metrics_json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_payload, f, indent=2)
        logger.info(f" --> Metrics JSON saved to {metrics_json_path}")
    except Exception as e:
        logger.warning(f"Could not write cosine_metrics.json: {e}")
    # Select ID: CLI > JSON > Folder
    train_name = args.train_name or json_train_name or weights_dir.parent.name
    raw_emb_name = save_dir.name
    emb_parts = [p.upper() for p in raw_emb_name.split('_')]
    embedding_method = " + ".join(emb_parts)
    
    title_suffix = f"[{train_name} - {embedding_method}]"
    title_suffix = title_suffix.replace('_', r'\_')
    
    # Plotting the dashboard
    hub_array = retrieval_stats['hub_array'] if retrieval_stats is not None else None
    plot_latent_geometry_dashboard(self_sims, hub_array, save_dir, metrics_payload, suffix=title_suffix)

if __name__ == "__main__":
    main()