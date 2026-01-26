"""
AstroPT Alignment Validator.

This script performs a full validation of the cross-modal alignment capabilities
of a trained AstroPT model. It executes the following pipeline:

1. Loads Model & Dataset (Test Split).
2. Generates embeddings for Images and Spectra on-the-fly.
3. Calculates Cosine Similarity statistics (Mean, Median).
4. Performs Retrieval Analysis (Top-1 and Top-5 Accuracy).
5. Generates a distribution histogram.

Author: Victor Alonso Rodriguez
Date: January 2026
"""

import argparse
import logging
import os
import sys
from typing import List, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow
from astropt.model_utils import load_local_model

# --- Logging Configuration ---
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-Validator")

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
    parser = argparse.ArgumentParser(description="AstroPT Alignment Validation")
    
    parser.add_argument("--out_dir", type=str, required=True, help="Directory containing the checkpoint")
    parser.add_argument("--ckpt_name", type=str, default="ckpt_best.pt", help="Checkpoint filename")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save plots")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--limit_samples", type=int, default=None, help="Limit number of samples for quick testing")
    
    return parser.parse_args()


def calculate_retrieval_accuracy(
    img_emb: torch.Tensor, 
    spec_emb: torch.Tensor, 
    batch_size: int = 1000
) -> Tuple[float, float]:
    """
    Calculates Top-1 and Top-5 retrieval accuracy.
    
    Args:
        img_emb: Normalized image embeddings [N, D]
        spec_emb: Normalized spectra embeddings [N, D]
        batch_size: Batch size for matrix multiplication to avoid OOM.
        
    Returns:
        Tuple (Top-1 Acc, Top-5 Acc)
    """
    n_samples = img_emb.shape[0]
    top1_correct = 0
    top5_correct = 0
    device = img_emb.device

    logger.info(f"Calculating Retrieval Metrics for {n_samples} samples...")
    
    # We iterate in matrix chunks
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        
        # Current batch of images [B, D]
        img_batch = img_emb[i:end]
        
        # Compare against ALL spectra
        sim_matrix = torch.matmul(img_batch, spec_emb.T)
        
        # Get Top-5 indices 
        _, top_indices = sim_matrix.topk(5, dim=1)
        
        # Correct targets for this batch
        targets = torch.arange(i, end, device=device).view(-1, 1)
        
        # Check matches
        # Top-1: First column matches target
        top1_correct += (top_indices[:, 0].unsqueeze(1) == targets).sum().item()
        
        # Top-5: Target appears anywhere in the top 5 columns
        top5_correct += (top_indices == targets).sum().item()
        
    return top1_correct / n_samples, top5_correct / n_samples


def main():
    
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Setup Output
    save_dir = args.save_dir if args.save_dir else os.path.join(args.out_dir, "validation_plots")
    os.makedirs(save_dir, exist_ok=True)

    # Load Model
    ckpt_path = os.path.join(args.out_dir, args.ckpt_name)
    try:
        model, config, registry, raw_config = load_local_model(ckpt_path, device)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Setup Data
    data_dir = args.data_dir if args.data_dir else raw_config.get('data_dir', "")
    logger.info(f"Loading Test Data from: {data_dir}")

    transforms = EuclidDESIDatasetArrow.data_transforms(
        norm_type_img=raw_config.get('img_norm_type', 'constant'),
        norm_const_img=raw_config.get('img_norm_const', 1.0),
        norm_type_spec=raw_config.get('spectra_norm_type', 'constant'),
        norm_const_spec=raw_config.get('spectra_norm_const', 1.0)
    )

    ds = EuclidDESIDatasetArrow(
        arrow_folder_root=data_dir,
        split="test",
        modality_registry=registry,
        spiral=False,
        transform=transforms
    )
    
    # Optional subset for debugging
    if args.limit_samples and len(ds) > args.limit_samples:
        logger.warning(f"Limiting validation to first {args.limit_samples} samples.")
        indices = list(range(args.limit_samples))
        ds = torch.utils.data.Subset(ds, indices)

    # Create the dataloader
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Inference Loop
    logger.info(f"Starting inference on {len(ds)} samples...")
    
    img_embeddings_list = []
    spec_embeddings_list = []
    
    # Mixed precision for speed
    ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    with torch.no_grad(), ctx:
        for batch in tqdm(dl, desc="Inference"):
            if batch is None: continue
            
            # Move to device
            X = {k: v.to(device) for k, v in batch.items() if k in ['images', 'spectra', 'images_positions', 'spectra_positions']}
            
            # Get Embeddings
            emb_dict = model.get_embeddings(X, draw_from_centre=True)
            
            if 'images' in emb_dict and 'spectra' in emb_dict:
                
                # Mean Pooling
                img_pool = emb_dict['images'].mean(dim=1).float()
                spec_pool = emb_dict['spectra'].mean(dim=1).float()
                
                img_embeddings_list.append(img_pool)
                spec_embeddings_list.append(spec_pool)

    # Concatenate results
    if not img_embeddings_list:
        logger.error("No embeddings extracted. Check dataset/model keys.")
        sys.exit(1)

    all_img = torch.cat(img_embeddings_list, dim=0)
    all_spec = torch.cat(spec_embeddings_list, dim=0)
    
    # Analysis
    logger.info("Computing metrics...")
    
    # Normalize 
    all_img = F.normalize(all_img, p=2, dim=1)
    all_spec = F.normalize(all_spec, p=2, dim=1)
    
    # Cosine Similarity 
    self_sim = F.cosine_similarity(all_img, all_spec, dim=1).cpu().numpy()
    
    mean_sim = np.mean(self_sim)
    median_sim = np.median(self_sim)
    std_sim = np.std(self_sim)
    
    logger.info("-" * 40)
    logger.info(f"Mean Cosine Similarity:   {mean_sim:.4f}")
    logger.info(f"Median Cosine Similarity: {median_sim:.4f}")
    logger.info(f"Std Dev:                  {std_sim:.4f}")
    logger.info("-" * 40)
    
    # Retrieval Accuracy 
    if len(ds) > 100:
        top1, top5 = calculate_retrieval_accuracy(all_img, all_spec)
        random_chance = 1.0 / len(ds)
        
        logger.info(f"Top-1 Accuracy: {top1:.2%} (Random: {random_chance:.5%})")
        logger.info(f"Top-5 Accuracy: {top5:.2%}")
        
        diagnosis = "POOR"
        if top1 > 0.5: diagnosis = "EXCELLENT"
        elif top1 > 0.1: diagnosis = "MODERATE"
        
        logger.info(f"Diagnosis: {diagnosis}")
        logger.info("-" * 40)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(self_sim, bins=50, color='royalblue', edgecolor='black', alpha=0.7, range=(-1, 1))
    
    plt.axvline(mean_sim, color='crimson', linestyle='--', linewidth=2, label=f'Mean: {mean_sim:.2f}')
    plt.axvline(median_sim, color='orange', linestyle=':', linewidth=2, label=f'Median: {median_sim:.2f}')
    
    plt.title(r"\textbf{Image-Spectrum Alignment Distribution}", fontsize=16)
    plt.xlabel(r"Cosine Similarity ($\cos\theta$)", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    out_file = os.path.join(save_dir, "alignment_histogram.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to {out_file}")

if __name__ == "__main__":
    main()