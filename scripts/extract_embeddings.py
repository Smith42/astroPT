"""
AstroPT Production Embedding Extractor.

Features:
- Uses Numpy Memmap to write directly to disk (Low RAM usage).
- Dynamic folder naming based on run and checkpoint.
- Extracts: Images, Spectra, Joint, and TargetIDs.
- Generates both individual .npy files (fast access) and .npz (portable).

Author: Victor Alonso Rodriguez
Date: January 2026
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import gc

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
logger = logging.getLogger("AstroPT-ProdExtract")


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="AstroPT Production Embedding Extractor")
    
    parser.add_argument("--out_dir", type=str, required=True, help="Directory containing the checkpoint (e.g., logs/run_name)")
    parser.add_argument("--ckpt_name", type=str, default="ckpt_best.pt", help="Checkpoint filename")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory if needed")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    return parser.parse_args()

def get_output_folder_name(out_dir: str, ckpt_name: str) -> str:
    """Generates the folder name: embeddings_{parent_suffix}_{ckpt_suffix}"""
    # Get directory suffix
    parent_suffix = os.path.basename(os.path.normpath(out_dir))
    
    # Get Checkpoint Suffix
    ckpt_suffix = ckpt_name.replace(".pt", "").replace("ckpt_", "")
    
    folder_name = f"embeddings_{parent_suffix}_{ckpt_suffix}"
    return folder_name

def main():
    args = parse_args()
    
    # Load Model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(args.out_dir, args.ckpt_name)
    
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    logger.info(f"Loading checkpoint: {ckpt_path}")
    
    try:
        model, config, registry, raw_config_dict = load_local_model(ckpt_path, device)
        model.eval() # Enforcement
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Arrow data directory
    data_dir = args.data_dir if args.data_dir else raw_config_dict.get('data_dir')
    assert data_dir is not None, "data_dir cannot be None. Check your config.json or --data_dir argument."
    
    # Retrieve Normalization Constants
    norm_type_img = raw_config_dict.get('images_norm_type', raw_config_dict.get('img_norm_type', 'asinh'))
    norm_scaler_img=raw_config_dict.get('images_norm_scaler',raw_config_dict.get('img_norm_scaler',1.0))
    norm_const_img = raw_config_dict.get('images_norm_const',raw_config_dict.get('img_norm_const',1.0))
    norm_type_spec = raw_config_dict.get('spectra_norm_type', 'constant')
    norm_scaler_spec = raw_config_dict.get('spectra_norm_scaler',1.0)
    norm_const_spec = raw_config_dict.get('spectra_norm_const', 1.0)
    
    # Aplying tranformations
    data_tf = EuclidDESIDatasetArrow.data_transforms(
        norm_type_img=norm_type_img,
        norm_scaler_img=norm_scaler_img,
        norm_const_img=norm_const_img,
        norm_type_spec=norm_type_spec,
        norm_scaler_spec=norm_scaler_spec,
        norm_const_spec=norm_const_spec,
    )
    
    logger.info(f"Initializing Dataset from: {data_dir}")
    
    ds = EuclidDESIDatasetArrow(
        arrow_folder_root=data_dir,
        split="test",
        modality_registry=registry,
        spiral=True,      
        stochastic=False,  
        transform=data_tf
    )
    
    dl = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    total_samples = len(ds)
    emb_dim = config.n_embd
    logger.info(f"Total samples to process: {total_samples}")
    logger.info(f"Embedding dimension: {emb_dim}")

    # MEMMAP Format
    folder_name = get_output_folder_name(args.out_dir, args.ckpt_name)
    save_path = os.path.join(args.out_dir, folder_name)
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Output directory created: {save_path}")

    # Initialize Memmap files
    
    # Images
    mmap_images = np.lib.format.open_memmap(
        os.path.join(save_path, 'images.npy'), mode='w+', dtype='float32', shape=(total_samples, emb_dim)
    )
    # Spectra
    mmap_spectra = np.lib.format.open_memmap(
        os.path.join(save_path, 'spectra.npy'), mode='w+', dtype='float32', shape=(total_samples, emb_dim)
    )
    # Joint
    mmap_joint = np.lib.format.open_memmap(
        os.path.join(save_path, 'joint.npy'), mode='w+', dtype='float32', shape=(total_samples, emb_dim)
    )
    # Target IDs
    mmap_ids = np.lib.format.open_memmap(
        os.path.join(save_path, 'ids.npy'), mode='w+', dtype='int64', shape=(total_samples,)
    )

    # Extraction loop
    ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype) # type: ignore
    
    start_idx = 0
    target_mods = ['images', 'spectra']

    with torch.no_grad(), ctx:
        for batch in tqdm(dl, desc="Extracting to Disk"):
            if batch is None: continue
            
            # Batch size
            batch_len = len(batch['targetid'])
            end_idx = start_idx + batch_len

            # Prepare Batch
            X = EuclidDESIDatasetArrow.prepare_batch(
                batch_data=batch,
                modality_registry=registry,
                device=device
            )

            # Getting mebeddings
            emb_dict = model.get_embeddings(X, draw_from_centre=True)

            # Write Images
            if 'images' in emb_dict:
                # Mean Pool: [B, Seq, Dim] -> [B, Dim]
                img_pool = emb_dict['images'].mean(dim=1).float().cpu().numpy()
                mmap_images[start_idx:end_idx] = img_pool
            else:
                # Fallback zero fill if missing (unlikely in test set but safe)
                mmap_images[start_idx:end_idx] = np.zeros((batch_len, emb_dim), dtype='float32')

            # Write Spectra
            if 'spectra' in emb_dict:
                spec_pool = emb_dict['spectra'].mean(dim=1).float().cpu().numpy()
                mmap_spectra[start_idx:end_idx] = spec_pool
            else:
                mmap_spectra[start_idx:end_idx] = np.zeros((batch_len, emb_dim), dtype='float32')

            # Write Joint
            valid_seqs = [emb_dict[m] for m in target_mods if m in emb_dict and emb_dict[m].ndim == 3]
            if valid_seqs:
                
                # Concat sequence length -> Mean Pool
                joint_seq = torch.cat(valid_seqs, dim=1)
                joint_pool = joint_seq.mean(dim=1).float().cpu().numpy()
                mmap_joint[start_idx:end_idx] = joint_pool
            else:
                 mmap_joint[start_idx:end_idx] = np.zeros((batch_len, emb_dim), dtype='float32')

            # Write IDs
            mmap_ids[start_idx:end_idx] = batch['targetid'].numpy().astype('int64')

            # Update pointer
            start_idx = end_idx


    # Flush changes to disk
    mmap_images.flush()
    mmap_spectra.flush()
    mmap_joint.flush()
    mmap_ids.flush()
    
    # Deleting the variables
    del mmap_images, mmap_spectra, mmap_joint, mmap_ids
    gc.collect()

    logger.info("Raw .npy extraction complete.")

    # Create a portable .npz file
    logger.info("Generating portable .npz file from disk data...")
    
    npz_path = os.path.join(save_path, "embeddings_all.npz")
    
    # Reading .npy to convert them to .npz
    try:
        arr_img = np.load(os.path.join(save_path, 'images.npy'), mmap_mode='r')
        arr_spec = np.load(os.path.join(save_path, 'spectra.npy'), mmap_mode='r')
        arr_joint = np.load(os.path.join(save_path, 'joint.npy'), mmap_mode='r')
        arr_ids = np.load(os.path.join(save_path, 'ids.npy'), mmap_mode='r')
        
        np.savez_compressed(
            npz_path,
            images=arr_img,
            spectra=arr_spec,
            joint=arr_joint,
            targetid=arr_ids
        )
        logger.info(f"Compressed archive created: {npz_path}")
        
    except Exception as e:
        logger.error(f"Error creating .npz: {e}")

    logger.info("Extraction Pipeline Finished Successfully.")
    logger.info(f"Results stored in: {save_path}")

if __name__ == "__main__":
    main()