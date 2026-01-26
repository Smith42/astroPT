"""
AstroPT Embedding Extraction Script.

This script loads a trained AstroPT model and extracts the embeddings using 
the model's specific `get_embeddings` method (preserving center logic).

It aligns with the training pipeline by using the same data transforms and 
configuration loading strategies.

Author: Victor Alonso Rodriguez
Date: January 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow
from astropt.model_utils import load_local_model

# Configure logging
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-Extract")


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    
    parser = argparse.ArgumentParser(description="AstroPT Embedding Extractor")
    
    # Parsing Arguments
    parser.add_argument("--out_dir", type=str, required=True, help="Directory containing the checkpoint")
    parser.add_argument("--ckpt_name", type=str, default="ckpt_best.pt", help="Checkpoint filename")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory if needed")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    return parser.parse_args()

def main():
    
    # Parsing argumentss
    args = parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(args.out_dir, args.ckpt_name)
    
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    logger.info(f"Loading checkpoint from {ckpt_path}...")

    # Load Config and Registry
    try:
        model, config, registry, raw_config_dict = load_local_model(ckpt_path, device)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Setup Data
    data_dir = args.data_dir if args.data_dir else raw_config_dict.get('data_dir', "/default/path")
    logger.info(f"Loading Test Data from: {data_dir}")

    # Data transformations
    data_tf = EuclidDESIDatasetArrow.data_transforms(
        norm_type_img=raw_config_dict.get('img_norm_type', 'constant'),
        norm_const_img=raw_config_dict.get('img_norm_const', 1.0),
        norm_type_spec=raw_config_dict.get('spectra_norm_type', 'constant'),
        norm_const_spec=raw_config_dict.get('spectra_norm_const', 1.0)
    )
    
    # Creating the dataset
    ds = EuclidDESIDatasetArrow(
        arrow_folder_root=data_dir,
        split="test",
        modality_registry=registry,
        spiral=False,      
        stochastic=False,  
        transform=data_tf
    )
    
    # Creating the dataloader
    dl = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    # Extraction Loop
    logger.info(f"Starting extraction on {len(ds)} samples...")
    
    target_mods = ['images', 'spectra']
    all_embeddings = {mod: [] for mod in target_mods}
    all_ids = []

    # Mixed Precision Context
    ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    with torch.no_grad(), ctx:
        for batch in tqdm(dl, desc="Extracting"):
            if batch is None: continue
            
            # Prepare Input Dict
            X = {}
            for mod in target_mods:
                if mod in batch:
                    X[mod] = batch[mod].to(device)
                    pos_key = f"{mod}_positions"
                    if pos_key in batch:
                        X[pos_key] = batch[pos_key].to(device)
            

            emb_dict = model.get_embeddings(X, draw_from_centre=True)
            
            # Extracting modalities
            for mod, tensor in emb_dict.items():
                if tensor.ndim == 3:
                    # [Batch, Seq, Dim] -> [Batch, Dim]
                    pooled = tensor.mean(dim=1).float().cpu().numpy()
                else:
                    pooled = tensor.float().cpu().numpy()
            
            if mod in all_embeddings:
                all_embeddings[mod].append(pooled)
            
            # Joint embeddings
            valid_tensors = [t for t in emb_dict.values() if t.ndim == 3]
            
            if valid_tensors:
                # [Batch, Total_Seq_Len, Dim]
                joint_seq = torch.cat(valid_tensors, dim=1)
                
                # Global Mean Pooling
                joint_pooled = joint_seq.mean(dim=1).float().cpu().numpy()
                
                all_embeddings['joint'].append(joint_pooled)

            # Store IDs
            if 'targetid' in batch:
                all_ids.extend(batch['targetid'].numpy())


    # Saving the embeddigns
    logger.info("Concatenating and saving...")
    final_arrays = {k: np.concatenate(v, axis=0) for k, v in all_embeddings.items() if len(v) > 0}
    
    if all_ids:
        final_arrays['targetid'] = np.array(all_ids)
    
    # Save Name: embeddings_{ckpt_name}.npz
    out_name = f"embeddings_{args.ckpt_name.replace('.pt', '')}.npz"
    out_path = os.path.join(args.out_dir, out_name)
    
    np.savez_compressed(out_path, **final_arrays)
    logger.info(f"Saved successfully to {out_path}")
    logger.info(f"Contains keys: {list(final_arrays.keys())}")

if __name__ == "__main__":
    main()