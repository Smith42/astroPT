"""
AstroPT Arrow Dataset Interpolation (Re-centering & Resizing)

This script applies the zooming/interpolation formula from Marc Huertas-Company
to ensure that all galaxies have a consistent physical/angular size across the dataset.
It dynamically calculates the bounding box based on the Sersic radius and ellipticity,
crops (or pads) the image, and uses skimage to resize it back to 224x224.

Usage example:
python arrow_dataset_interpolation.py \
    --data_dir /path/to/original_arrow \
    --save_dir /path/to/resized_arrow \
    --metadata_path /path/to/catalog.fits \
    --col_radius sersic_sersic_vis_radius \
    --col_q_ratio sersic_sersic_vis_axis_ratio
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from astropy.table import Table
from datasets import load_from_disk, concatenate_datasets
from skimage.transform import resize

logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-Resizer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize galaxies based on Sersic radius.")
    
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory of the existing Arrow dataset")
    parser.add_argument("--save_dir", type=str, required=True, 
                        help="Directory to save the new resized dataset")
    parser.add_argument("--metadata_path", type=str, required=True, 
                        help="Path to the .fits catalog")
    parser.add_argument("--col_radius", type=str, default="sersic_sersic_vis_radius", 
                        help="Column name for Sersic half-light radius (re)")
    parser.add_argument("--col_q_ratio", type=str, default="sersic_sersic_vis_axis_ratio", 
                        help="Column name for axis ratio (q)")
    parser.add_argument("--re_multiplier", type=float, default=5.0,
                        help="Multiplier for the effective radius (Re) to define the crop window (default: 5.0)")
    parser.add_argument("--num_workers", type=int, default=8, 
                        help="Number of CPU cores to use for Arrow mapping")
    
    return parser.parse_args()


def load_metadata_dict(metadata_path: Path, col_r: str, col_q: str) -> Dict[int, Tuple[float, float]]:
    logger.info(f"Loading metadata from {metadata_path}...")
    try:
        catalog = Table.read(metadata_path)
        df = catalog.to_pandas()
    except Exception as e:
        logger.error(f"Failed to read catalog: {e}")
        sys.exit(1)
        
    target_col = 'TARGETID' if 'TARGETID' in df.columns else 'targetid'
    
    # Drop NaNs in required columns
    df_clean = df.dropna(subset=[target_col, col_r, col_q]).copy()
    
    # Set Axis Ratio (q)
    df_clean['q_ratio'] = df_clean[col_q]
        
    # Dictionary mapping TARGETID -> (re, q) for O(1) lookups
    meta_dict = {}
    for _, row in df_clean.iterrows():
        meta_dict[int(row[target_col])] = (float(row[col_r]), float(row['q_ratio']))
        
    logger.info(f"Loaded {len(meta_dict)} valid galaxies with (re, q) metadata.")
    return meta_dict


def crop_and_resize(image: np.ndarray, target_half_size: float, out_shape: int = 224) -> np.ndarray:
    """
    Crops (or pads) a window of size 2*hw centered in the image, 
    then resizes it back to out_shape x out_shape.
    """
    in_shape = image.shape[0] # assuming square input
    hw = int(np.round(target_half_size))
    
    if hw < 1:
        hw = 1
        
    cx, cy = in_shape // 2, in_shape // 2
    
    x0, x1 = cx - hw, cx + hw
    y0, y1 = cy - hw, cy + hw
    
    pad_x0, pad_x1 = max(0, -x0), max(0, x1 - in_shape)
    pad_y0, pad_y1 = max(0, -y0), max(0, y1 - in_shape)
    
    safe_x0, safe_x1 = max(0, x0), min(in_shape, x1)
    safe_y0, safe_y1 = max(0, y0), min(in_shape, y1)
    
    extracted = image[safe_y0:safe_y1, safe_x0:safe_x1]
    
    # If the bounding box is larger than the 224 image, pad with background (0)
    if pad_x0 > 0 or pad_y0 > 0 or pad_x1 > 0 or pad_y1 > 0:
        extracted = np.pad(extracted, ((pad_y0, pad_y1), (pad_x0, pad_x1)), mode='constant', constant_values=0)
        
    # Resize to original shape using skimage (anti-aliasing is enabled by default for downsampling)
    out_image = resize(extracted, output_shape=(out_shape, out_shape), 
                       anti_aliasing=True, preserve_range=True).astype(np.float32)
    return out_image


def get_processor(metadata_dict: Dict[int, Tuple[float, float]], re_multiplier: float):
    """Returns the map function loaded with the metadata dictionary."""
    
    def process_batch(batch):
        targetids = batch['targetid']
        batch_size = len(targetids)
        
        new_vis, new_y, new_j, new_h = [], [], [], []
        
        for i in range(batch_size):
            tid = int(targetids[i])
            re, q = metadata_dict.get(tid, (None, None))
            
            img_vis = np.array(batch['image_vis'][i])
            img_y = np.array(batch['image_nisp_y'][i])
            img_j = np.array(batch['image_nisp_j'][i])
            img_h = np.array(batch['image_nisp_h'][i])
            
            # If metadata missing or invalid, keep original unchanged
            if re is None or q is None or not np.isfinite(re) or not np.isfinite(q):
                new_vis.append(img_vis)
                new_y.append(img_y)
                new_j.append(img_j)
                new_h.append(img_h)
                continue
                
            # --- MARC'S FORMULA ADAPTED FOR EUCLID VIS (0.1"/pix) ---
            # 're' is in arcseconds, 'q' is dimensionless axis ratio.
            # We want to crop a window that scales with the galaxy size.
            
            # arcsec_cut calculation (physical size in arcseconds for the half-window)
            # We use the user-defined multiplier (default 5.0) to ensure the galaxy is well-framed.
            arcsec_cut = re_multiplier * re * np.sqrt(q)
            
            # Convert arcseconds to pixels using Euclid VIS scale (0.1 arcsec/pixel)
            hw = arcsec_cut / 0.1
            
            # SAFEGUARD: Prevent OOM by capping maximum half-window size.
            hw = float(np.minimum(hw, 1000.0))
            
            new_vis.append(crop_and_resize(img_vis, hw, 224))
            new_y.append(crop_and_resize(img_y, hw, 224))
            new_j.append(crop_and_resize(img_j, hw, 224))
            new_h.append(crop_and_resize(img_h, hw, 224))
            
        batch['image_vis'] = new_vis
        batch['image_nisp_y'] = new_y
        batch['image_nisp_j'] = new_j
        batch['image_nisp_h'] = new_h
        return batch
        
    return process_batch


def main():
    args = parse_args()
    data_dir, save_dir = Path(args.data_dir), Path(args.save_dir)
    
    if not data_dir.exists():
        logger.error(f"Input directory not found: {data_dir}")
        sys.exit(1)
        
    save_dir.mkdir(parents=True, exist_ok=True)
    meta_dict = load_metadata_dict(Path(args.metadata_path), args.col_radius, args.col_q_ratio)
    processor = get_processor(meta_dict, args.re_multiplier)
    
    for split in ["train", "test"]:
        logger.info(f"\n#--- PROCESSING SPLIT: {split.upper()} ---#")
        split_dirs = sorted(data_dir.glob(f"{split}_*"))
        if not split_dirs:
            continue
            
        ds = concatenate_datasets([load_from_disk(str(p)) for p in split_dirs])
        logger.info(f"Loaded {split} dataset ({len(ds)} samples). Processing...")
        
        # Apply the mapping function using HuggingFace's multi-processing
        filtered_ds = ds.map(
            processor,
            batched=True,
            batch_size=200, # Small batch for image heavy operations
            num_proc=args.num_workers,
            desc=f"Resizing {split} images"
        )
        
        split_save_path = save_dir / f"{split}_0"
        logger.info(f"Saving {split} to {split_save_path}...")
        filtered_ds.save_to_disk(str(split_save_path), max_shard_size="500MB")
        logger.info(f"Completed {split}.")

if __name__ == "__main__":
    main()
