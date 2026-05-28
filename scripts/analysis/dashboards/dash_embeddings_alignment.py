"""
AstroPT Multimodal Alignment Analyst.

This script loads pre-computed embeddings from disk (Numpy Memmap or NPZ) and 
performs a full alignment analysis (Cosine Similarity, CKA, Uniformity, Retrieval).

Pipeline:
1. Loads embeddings for two specified modalities.
2. Computes diagonal Cosine Similarity (Self-Similarity).
3. Computes Retrieval Metrics (Top-1, Top-5) using batched GPU operations.
4. Calculates Geometric Metrics: CKA, Uniformity, and Neighborhood Overlap.
5. Generates publication-quality dashboards.

Author: Victor Alonso Rodriguez
Date: May 2026
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

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
logger = logging.getLogger("AstroPT-Alignment")

# Plotting Global Configuration
try:
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = r'''
                \usepackage{siunitx}
                \usepackage{bm}
                \usepackage{amsmath} 
                '''
except:
    logger.warning("LaTeX not found. Falling back to standard fonts.")
    plt.rc('text', usetex=False)

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
    parser = argparse.ArgumentParser(description="AstroPT Multimodal Alignment Analyst")
    
    parser.add_argument("--emb_dir", type=str, required=True, help="Directory containing the .npy/.npz files")
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing model config (optional)")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--mod1", type=str, default=None, help="First modality name (e.g. EuclidImage)")
    parser.add_argument("--mod2", type=str, default=None, help="Second modality name (e.g. DESISpectrum)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for matrix operations")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for retrieval calculation")
    parser.add_argument("--train_name", type=str, default=None, help="Custom title for the plot")
    
    return parser.parse_args()

def calculate_retrieval_metrics(
    img_mmap: np.ndarray, 
    spec_mmap: np.ndarray, 
    device: torch.device,
    batch_size: int = 1000
) -> dict:
    """Calculates Top-1, Top-5 retrieval accuracy and Hubness."""
    n_samples = img_mmap.shape[0]
    top1_correct = 0
    top5_correct = 0
    hub_counts = torch.zeros(n_samples, dtype=torch.long, device=device)
    
    logger.info(f"Computing Retrieval for {n_samples} samples...")
    
    # Pre-load Gallery to GPU
    gallery = torch.from_numpy(spec_mmap[:].copy()).to(device)
    gallery = F.normalize(gallery, p=2, dim=1)

    for i in tqdm(range(0, n_samples, batch_size), desc="Retrieval"):
        end = min(i + batch_size, n_samples)
        queries = torch.from_numpy(img_mmap[i:end].copy()).to(device)
        queries = F.normalize(queries, p=2, dim=1)
        
        sim_matrix = torch.matmul(queries, gallery.T)
        _, top_indices = sim_matrix.topk(5, dim=1)
        
        # Track hubness (Top-1 hits)
        unique_idx, counts = torch.unique(top_indices[:, 0], return_counts=True)
        hub_counts[unique_idx] += counts
        
        targets = torch.arange(i, end, device=device).view(-1, 1)
        top1_correct += (top_indices[:, 0].unsqueeze(1) == targets).sum().item()
        top5_correct += (top_indices == targets).sum().item()
        
    hub_array = hub_counts.float().cpu()
    skew = (((hub_array - hub_array.mean()) / hub_array.std())**3).mean().item() if hub_array.std() > 0 else 0.0
    
    return {
        'top1': top1_correct / n_samples, 
        'top5': top5_correct / n_samples, 
        'hubness_skew': skew, 
        'hub_array': hub_array.numpy()
    }

def calculate_cka(img_mmap: np.ndarray, spec_mmap: np.ndarray, device: torch.device, batch_size: int = 5000) -> float:
    """Computes Linear Feature CKA incrementally."""
    n_samples, dim = img_mmap.shape
    mean_x = torch.zeros(dim, device=device, dtype=torch.float64)
    mean_y = torch.zeros(dim, device=device, dtype=torch.float64)
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        mean_x += torch.from_numpy(img_mmap[i:end].copy()).to(device).double().sum(dim=0)
        mean_y += torch.from_numpy(spec_mmap[i:end].copy()).to(device).double().sum(dim=0)
    mean_x /= n_samples
    mean_y /= n_samples
    
    cov_xx = torch.zeros((dim, dim), device=device, dtype=torch.float64)
    cov_yy = torch.zeros((dim, dim), device=device, dtype=torch.float64)
    cov_xy = torch.zeros((dim, dim), device=device, dtype=torch.float64)
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        x = torch.from_numpy(img_mmap[i:end].copy()).to(device).double() - mean_x
        y = torch.from_numpy(spec_mmap[i:end].copy()).to(device).double() - mean_y
        cov_xx += x.T @ x
        cov_yy += y.T @ y
        cov_xy += x.T @ y
        
    num = torch.norm(cov_xy, p='fro')**2
    denom = torch.norm(cov_xx, p='fro') * torch.norm(cov_yy, p='fro')
    return (num / denom).item() if denom > 0 else 0.0

def calculate_uniformity(data: np.ndarray, device: torch.device, sample_size: int = 5000) -> float:
    """Calculates Wang & Isola Uniformity."""
    n_samples = data.shape[0]
    idx = np.random.choice(n_samples, min(sample_size, n_samples), replace=False)
    x = F.normalize(torch.from_numpy(data[idx].copy()).to(device), p=2, dim=1)
    sqdist = 2.0 - 2.0 * (x @ x.T)
    return torch.log(torch.mean(torch.exp(-2.0 * sqdist))).item()

def calculate_neighborhood_overlap(img_mmap: np.ndarray, spec_mmap: np.ndarray, device: torch.device, sample_size: int = 2000, K: int = 10) -> float:
    """Calculates Latent R-Precision proxy: Intersecting neighbors between Modalities."""
    n_samples = img_mmap.shape[0]
    idx = np.random.choice(n_samples, min(sample_size, n_samples), replace=False)
    
    logger.info(f"Computing Neighborhood Overlap (K={K}) on a {len(idx)} subset...")
    try:
        # We need the full galleries for search
        gallery_x = F.normalize(torch.from_numpy(img_mmap[:].copy()).to(device), p=2, dim=1)
        gallery_y = F.normalize(torch.from_numpy(spec_mmap[:].copy()).to(device), p=2, dim=1)
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
    img_mmap: np.ndarray, 
    spec_mmap: np.ndarray, 
    device: torch.device,
    batch_size: int = 5000
) -> np.ndarray:
    """Computes element-wise cosine similarity (alignment)."""
    n_samples = img_mmap.shape[0]
    all_sims = []
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        img = F.normalize(torch.from_numpy(img_mmap[i:end].copy()).to(device), p=2, dim=1)
        spec = F.normalize(torch.from_numpy(spec_mmap[i:end].copy()).to(device), p=2, dim=1)
        sims = F.cosine_similarity(img, spec, dim=1)
        all_sims.append(sims.cpu().numpy())
    return np.concatenate(all_sims)

def plot_alignment_dashboard(sim_values: np.ndarray, hub_array: np.ndarray, save_dir: Path, metrics: dict, title: str, m1: str, m2: str, filename: str = "alignment_analysis.png"):
    """Generates the final analysis dashboard."""
    fig, axs = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [2, 1.5]})
    
    # 1. Similarity Distribution
    ax = axs[0]
    ax.hist(sim_values, bins=200, color='royalblue', edgecolor='black', linewidth=0.2, alpha=1, range=(-1, 1), zorder=3)
    ax.axvline(metrics['cosine_mean'], color='crimson', linestyle='--', linewidth=2, label=f"Mean: {metrics['cosine_mean']:.2f}", zorder=4)
    ax.axvline(metrics['cosine_median'], color='darkorange', linestyle=':', linewidth=2, label=f"Median: {metrics['cosine_median']:.2f}", zorder=4)
    
    # Label formatting for Uniformity
    m1_tag = m1[:3] if len(m1) > 3 else m1
    m2_tag = m2[:3] if len(m2) > 3 else m2

    stats_text = '\n'.join((
        r"\textbf{Geometric Metrics}",
        fr"Alignment $\downarrow$: {metrics['alignment']:.3f}",
        fr"Unif ({m1_tag}) $\downarrow$: {metrics['unif1']:.2f}",
        fr"Unif ({m2_tag}) $\downarrow$: {metrics['unif2']:.2f}",
        fr"Feat CKA $\uparrow$: {metrics['cka']:.3f}"
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5)
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=12, va='top', ha='right', bbox=props, zorder=5)
    ax.set_title(r"\textbf{Modal Alignment Distribution}", fontsize=16)
    ax.set_xlabel(r"Cosine Similarity ($\cos\theta$)", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3, zorder=1)

    # 2. Hubness Profile
    ax = axs[1]
    sorted_hubs = np.sort(hub_array)[::-1]
    ax.plot(sorted_hubs, color='darkviolet', linewidth=2.5, zorder=3)
    ax.fill_between(range(len(sorted_hubs)), sorted_hubs, color='darkviolet', alpha=0.2, zorder=2)
    
    ret_text = '\n'.join((
        r"\textbf{Retrieval Performance}",
        fr"Top-1 $\uparrow$: {metrics['top1']*100:.3f}\%",
        fr"Top-5 $\uparrow$: {metrics['top5']*100:.3f}\%",
        fr"N-IOU $\uparrow$: {metrics['overlap']*100:.2f}\%",
        fr"Hub Skew $\downarrow$: {metrics['hub_skew']:.2f}"
    ))
    ax.text(0.95, 0.95, ret_text, transform=ax.transAxes, fontsize=12, va='top', ha='right', bbox=props, zorder=5)
    ax.set_title(r"\textbf{Gallery Hubness Profile}", fontsize=16)
    ax.set_xlabel("Sorted Gallery Spectra Rank", fontsize=14)
    ax.set_ylabel("Times Retrieved as Nearest", fontsize=14)
    ax.set_yscale('symlog')
    ax.grid(True, alpha=0.3, zorder=1)

    fig.suptitle(r"\textbf{Latent Representation Geometry}" + f"\n{title}", fontsize=20, y=1.05)
    plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    emb_dir = Path(args.emb_dir)
    weights_dir = Path(args.weights_dir) if args.weights_dir else None
    save_dir = Path(args.save_dir) if args.save_dir else emb_dir / "alignment_analysis"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    data = {}
    # Try .npy files first
    for f in emb_dir.glob("*.npy"):
        if f.stem != 'ids':
            data[f.stem] = np.load(f, mmap_mode='r')
    
    # Try .npz fallback
    if not data:
        npz_files = list(emb_dir.glob("*.npz"))
        if npz_files:
            archive = np.load(npz_files[0])
            data = {k: archive[k] for k in archive.keys() if k != 'ids'}

    if len(data) < 2:
        logger.error("At least 2 modalities required for alignment analysis.")
        sys.exit(1)

    # Check if the model uses contrastive alignment (CLIP/V4)
    use_contrastive = False
    if weights_dir:
        config_path = weights_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    cfg_dict = json.load(f)
                    use_contrastive = cfg_dict.get("use_contrastive_alignment", False)
            except Exception as e:
                logger.warning(f"Could not load config.json: {e}")

    # Reusable comparison runner
    def run_comparison(m1_name: str, m2_name: str, suffix: str = ""):
        logger.info("=" * 50)
        logger.info(f"RUNNING ALIGNMENT COMPARISON: {m1_name} <-> {m2_name}")
        logger.info("=" * 50)
        
        if m1_name not in data or m2_name not in data:
            logger.warning(f"Modalities '{m1_name}' or '{m2_name}' not found in data. Skipping this comparison.")
            return

        emb1, emb2 = data[m1_name], data[m2_name]
        
        # 2. Compute Metrics
        sims = compute_diagonal_stats(emb1, emb2, device)
        ret = calculate_retrieval_metrics(emb1, emb2, device, args.batch_size)
        cka = calculate_cka(emb1, emb2, device)
        unif1 = calculate_uniformity(emb1, device)
        unif2 = calculate_uniformity(emb2, device)
        overlap = calculate_neighborhood_overlap(emb1, emb2, device)

        metrics = {
            'cosine_mean': float(np.mean(sims)),
            'cosine_median': float(np.median(sims)),
            'alignment': float(2.0 - 2.0 * np.mean(sims)),
            'cka': float(cka),
            'unif1': float(unif1),
            'unif2': float(unif2),
            'overlap': float(overlap),
            'top1': float(ret['top1']),
            'top5': float(ret['top5']),
            'hub_skew': float(ret['hubness_skew'])
        }

        # 3. Output
        metrics_filename = f"alignment_metrics_{suffix}.json" if suffix else "alignment_metrics.json"
        with open(save_dir / metrics_filename, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Title logic
        train_name = args.train_name or (weights_dir.parent.name if weights_dir else "AstroPT")
        raw_emb_name = emb_dir.name
        embedding_method = " + ".join([p.upper() for p in raw_emb_name.split('_')])
        
        suffix_title = f" - {suffix.upper()}" if suffix else ""
        title_suffix = f"[{train_name}{suffix_title} - {embedding_method}]".replace('_', r'\_')
        
        plot_filename = f"alignment_analysis_{suffix}.png" if suffix else "alignment_analysis.png"
        plot_alignment_dashboard(sims, ret['hub_array'], save_dir, metrics, title_suffix, m1_name, m2_name, plot_filename)
        logger.info(f"Comparison {suffix if suffix else 'GENERAL'} complete. Saved to: {metrics_filename} and {plot_filename}")

    # Determine execution plan
    if args.mod1 or args.mod2:
        # User specified explicit modalities: run only that comparison
        m1 = args.mod1 or next((k for k in data.keys() if "image" in k.lower() or k == 'images'), list(data.keys())[0])
        m2 = args.mod2 or next((k for k in data.keys() if ("spec" in k.lower() or k == 'spectra') and k != m1), list(data.keys())[1])
        run_comparison(m1, m2)
    else:
        # Default: auto-detect modalities
        m1_base = next((k for k in data.keys() if ("image" in k.lower() or k == 'images') and "_phase" not in k), None)
        m2_base = next((k for k in data.keys() if ("spec" in k.lower() or k == 'spectra') and k != m1_base and "_phase" not in k), None)
        
        if not m1_base or not m2_base:
            # Fallback to absolute first two
            m1_base = list(data.keys())[0]
            m2_base = list(data.keys())[1]

        if use_contrastive:
            logger.info("Contrastive Alignment (CLIP) detected in config! Generating 3 alignment dashboards...")
            # 1. General Dashboard
            run_comparison(m1_base, m2_base, suffix="")
            # 2. Phase 1 (Unimodal Experts) Dashboard
            run_comparison(f"{m1_base}_phase1", f"{m2_base}_phase1", suffix="phase1")
            # 3. Phase 2 (Multimodal Experts) Dashboard
            run_comparison(f"{m1_base}_phase2", f"{m2_base}_phase2", suffix="phase2")
        else:
            # Standard model (No CLIP): run single general comparison
            run_comparison(m1_base, m2_base, suffix="")

    logger.info(f"Analysis complete. Results in: {save_dir}")

if __name__ == "__main__":
    main()
