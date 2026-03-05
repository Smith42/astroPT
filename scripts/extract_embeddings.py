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
import sys
import gc
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple


from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow
from astropt.model_utils import load_local_model
from sklearn.decomposition import IncrementalPCA

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
    
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing training weights")
    parser.add_argument("--data_dir", type=str, required=True, help="Arrow data root directory")
    parser.add_argument("--save_dir", type=str, required=True, help="Plot Saving Directory")
    parser.add_argument("--ckpt_name", type=str, default="ckpt_best.pt", help="Checkpoint filename")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--pool_method_img", type=str, default="mean", help="Pooling method to reduce dimensionality for images")
    parser.add_argument("--pool_method_spec", type=str, default="rank", help="Pooling method to reduce dimensionality for spectra")
    parser.add_argument("--pca_dim", type=int, default=0, help="Applying PCA analysis with the resulting pca_dim dimension. 0 means disabled")
    
    return parser.parse_args()


def apply_pooling(tensor: torch.Tensor,
                  method: str = "mean", 
                  p_value: float = 2.0, 
                  mixed_lambda: float = 0.5
    ) -> torch.Tensor:
    """
    Applying different pooling methods.
    Reference: Gholamalinezhad & Khosravi (2020).
    """
    if method == 'mean':
        # Average Pooling
        return tensor.mean(dim=1)
    
    elif method == 'max':
        # Max Pooling
        return tensor.max(dim=1)[0]
    
    elif method == 'mixed':
        # Mixed Pooling
        max_p = tensor.max(dim=1)[0]
        avg_p = tensor.mean(dim=1)
        return mixed_lambda * max_p + (1 - mixed_lambda) * avg_p
    
    elif method == 'lp':
        # L_p Pooling
        return (torch.mean(tensor.abs()**p_value, dim=1))**(1.0 / p_value)
    
    elif method == 'rank':
        # Rank-based Average Pooling (RAP)
        sorted_tensor, _ = torch.sort(tensor, dim=1, descending=True)
        k = max(1, int(sorted_tensor.size(1) * 0.1))
        return sorted_tensor[:, :k, :].mean(dim=1)
    
    else:
        raise ValueError(f"Invalid pooling method '{method}'.")

def apply_pca_to_memmap(
        file_path: str | Path, 
        original_shape: Tuple, 
        n_components: int = 0, 
        batch_size: int = 1024
    ) -> Path:
    """
    Applying PCA to memmaps files
    """
    file_path = Path(file_path)
    logger.info(f"Starting PCA {file_path} (Target: {n_components} dims)")
    
    ipca = IncrementalPCA(n_components=n_components)
    
    arr_mmap = np.load(file_path, mmap_mode='r')
    for i in range(0, original_shape[0], batch_size):
        chunk = arr_mmap[i:i + batch_size]
        ipca.partial_fit(chunk)
    
    pca_path = file_path.with_name(f"{file_path.stem}_pca{n_components}.npy")
    new_mmap = np.lib.format.open_memmap(
        pca_path, mode='w+', dtype='float32', shape=(original_shape[0], n_components)
    )
    
    for i in range(0, original_shape[0], batch_size):
        chunk = arr_mmap[i:i + batch_size]
        new_mmap[i:i + batch_size] = ipca.transform(chunk)
    
    new_mmap.flush()
    logger.info(f"PCA completed. New file: {pca_path}")
    return pca_path

def get_output_folder_name(ckpt_name: str, pool_name: str, pca_dim: int) -> Path:
    """Generates a clean subfolder name: ckpt_pool_pca"""
    ckpt_suffix = ckpt_name.replace(".pt", "").replace("ckpt_", "")
    
    name = f"{ckpt_suffix}_{pool_name}"
    if pca_dim > 0:
        name += f"_pca{pca_dim}"
        
    return Path(name)

def main():
    args = parse_args()
    
    # Load Model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Required paths
    weights_dir = Path(args.weights_dir)
    save_dir = Path(args.save_dir)
    
    logger.info("Analysis Directories:")
    logger.info(f" --> [Weights]:   {weights_dir}")
    logger.info(f" --> [Saving]:    {save_dir}")
    
    # Loading weights
    ckpt_path = weights_dir / args.ckpt_name
    
    if not ckpt_path.is_file():
        logger.error(f"Checkpoint not found: {ckpt_path}. Starting smart search.")

        all_ckpts = list(weights_dir.glob("*.pt"))
        if not all_ckpts:
            logger.error(f"FATAL: No .pt file found in {weights_dir}")
            sys.exit(1)

        best_matches = [c for c in all_ckpts if "best" in c.name]
        last_matches = [c for c in all_ckpts if "last" in c.name]
        
        if best_matches:
            ckpt_path = sorted(best_matches, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            logger.info(f"Selected by priority [BEST]: {ckpt_path.name}")
            
        elif last_matches:
            ckpt_path = sorted(last_matches, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            logger.info(f"Selected by priority [LAST]: {ckpt_path.name}")
        
        else:
            ckpt_path = sorted(all_ckpts, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            logger.info(f"Selected by date [RECENT]: {ckpt_path.name}")

    logger.info(f"Loading checkpoint: {ckpt_path}")
    
    try:
        model, config, registry, raw_config_dict = load_local_model(ckpt_path, device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Arrow data directory
    data_dir = args.data_dir if args.data_dir else raw_config_dict.get('data_dir')
    data_dir = Path(data_dir)
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
    
    if args.pool_method_img != args.pool_method_spec:
        folder_pool_name = args.pool_method_img + args.pool_method_spec
    else:
        folder_pool_name = args.pool_method_img
    
    # Specific embeddings folder
    folder_name = get_output_folder_name(args.ckpt_name, folder_pool_name, args.pca_dim)
    save_path = save_dir / folder_name
    save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Embeddings directory created: {save_path}")

    # Initialize Memmap files
    
    # Images
    mmap_images = np.lib.format.open_memmap(
        save_path / 'images.npy', mode='w+', dtype='float32', shape=(total_samples, emb_dim)
    )
    # Spectra
    mmap_spectra = np.lib.format.open_memmap(
        save_path / 'spectra.npy', mode='w+', dtype='float32', shape=(total_samples, emb_dim)
    )
    # Joint
    mmap_joint = np.lib.format.open_memmap(
        save_path / 'joint.npy', mode='w+', dtype='float32', shape=(total_samples, emb_dim)
    )
    # Target IDs
    mmap_ids = np.lib.format.open_memmap(
        save_path / 'ids.npy', mode='w+', dtype='int64', shape=(total_samples,)
    )

    # Extraction loop
    ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype) # type: ignore
    
    start_idx = 0
    target_mods = ['images', 'spectra']

    with torch.no_grad(), ctx:
        for batch in tqdm(dl, desc="Extracting Embeddings"):
            if batch is None:
                continue
                
            # 1. Coordinate batch indices
            current_batch_size = len(batch['targetid'])
            end_idx = start_idx + current_batch_size

            X = EuclidDESIDatasetArrow.prepare_batch(
                batch_data=batch,
                modality_registry=registry,
                device=device
            )

            # Generate raw embeddings
            # Expected shape per modality: (batch_size, sequence_length, embedding_dim)
            embeddings = model.get_embeddings(X, draw_from_centre=True)
            
            # Buffers for late fusion
            img_pooled = None
            spec_pooled = None
            
            # Process individual modalities with Hybrid Pooling
            for modality in ['images', 'spectra']:
                if modality in embeddings:
                    # Select specific pooling method based on modality nature
                    method = args.pool_method_img if modality == 'images' else args.pool_method_spec
                    
                    pooled_tensor = apply_pooling(embeddings[modality], method=method)
                    pooled_numpy = pooled_tensor.float().cpu().numpy()
                    
                    if modality == 'images':
                        img_pooled = pooled_tensor 
                        mmap_images[start_idx:end_idx] = pooled_numpy
                    else:
                        spec_pooled = pooled_tensor
                        mmap_spectra[start_idx:end_idx] = pooled_numpy
                else:
                    # Fallback: Zero-padding for missing modalities
                    padding = np.zeros((current_batch_size, emb_dim), dtype='float32')
                    if modality == 'images':
                        mmap_images[start_idx:end_idx] = padding
                    else:
                        mmap_spectra[start_idx:end_idx] = padding

            # Hybrid Joint Embedding
            if img_pooled is not None and spec_pooled is not None:
                joint_tensor = (img_pooled + spec_pooled) / 2.0
                mmap_joint[start_idx:end_idx] = joint_tensor.float().cpu().numpy()
            elif img_pooled is not None:
                mmap_joint[start_idx:end_idx] = img_pooled.float().cpu().numpy()
            elif spec_pooled is not None:
                mmap_joint[start_idx:end_idx] = spec_pooled.float().cpu().numpy()
            else:
                mmap_joint[start_idx:end_idx] = np.zeros((current_batch_size, emb_dim), dtype='float32')

            mmap_ids[start_idx:end_idx] = batch['targetid'].numpy().astype('int64')
            start_idx = end_idx


    # Flush changes to disk
    mmap_images.flush()
    mmap_spectra.flush()
    mmap_joint.flush()
    mmap_ids.flush()
    
    # Deleting the variables to close files
    del mmap_images, mmap_spectra, mmap_joint, mmap_ids
    gc.collect()

    logger.info("Raw .npy extraction complete.")

    # Create a portable .npz file
    logger.info("Generating portable .npz file from disk data...")
    
    final_paths = {
        'images': save_path / "images.npy",
        'spectra': save_path / "spectra.npy",
        'joint': save_path / "joint.npy",
        'ids': save_path / "ids.npy"
    }

    if args.pca_dim > 0:
        logger.info(f"Applying PCA reduction to {args.pca_dim} dimensions...")
        for key in ['images', 'spectra', 'joint']:

            final_paths[key] = apply_pca_to_memmap(
                final_paths[key], 
                (total_samples, emb_dim), 
                args.pca_dim
            )
            
    logger.info("Generating portable .npz file...")
    npz_path = save_path / "embeddings_all.npz"
    
    try:
        
        np.savez_compressed(
            npz_path,
            images=np.load(final_paths['images'], mmap_mode='r'),
            spectra=np.load(final_paths['spectra'], mmap_mode='r'),
            joint=np.load(final_paths['joint'], mmap_mode='r'),
            targetid=np.load(final_paths['ids'], mmap_mode='r')
        )
        logger.info(f"Compressed archive created: {npz_path}")
        
    except Exception as e:
        logger.error(f"Error creating .npz: {e}")

    logger.info("Extraction Pipeline Finished Successfully.")

if __name__ == "__main__":
    main()