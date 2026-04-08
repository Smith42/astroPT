"""
AstroPT Arrow Dataset Filter (Metadata-Driven).

This script performs a high-performance, out-of-core filtration of the Euclid-DESI 
multimodal Arrow dataset. It uses a metadata catalog (.fits) to identify the largest 
galaxies based on their Sersic VIS radius and creates a new, lightweight Arrow dataset 
containing only these high-quality targets.

Features:
- Zero-copy Arrow transformations via HuggingFace `datasets`.
- Multi-core processing for extreme speed.
- Native 500MB sharding to maintain dataloader compatibility.

Author: Victor Alonso Rodriguez
Date: March 2026
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Set, List, Optional

import numpy as np
import pandas as pd
from astropy.table import Table
from datasets import load_from_disk, concatenate_datasets, Dataset

# Configure production-level logging
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-ArrowFilter")


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Filter Arrow dataset for largest galaxies.")
    
    parser.add_argument("--data_dir", type=str, 
                        default="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow", 
                        help="Root directory of the existing Arrow dataset")
    parser.add_argument("--save_dir", type=str, required=True, 
                        help="Directory to save the new filtered dataset")
    parser.add_argument("--metadata_path", type=str, default=None, 
                        help="Path to the .fits catalog containing galaxy properties (optional)")
    parser.add_argument("--size_col", type=str, default="sersic_sersic_vis_radius", 
                        help="Name of the column containing galaxy size")
    parser.add_argument("--percentile", type=float, default=75.0, 
                        help="Percentile threshold (e.g., 75 keeps the top 25% largest)")
    parser.add_argument("--num_workers", type=int, default=8, 
                        help="Number of CPU cores to use for Arrow filtering")
    parser.add_argument("--filter_corrupt", action="store_true",
                        help="Clean rows that crash PyArrow when accessed due to size/shape corruption")
    
    return parser.parse_args()


def extract_large_galaxy_ids(metadata_path: Path, size_col: str, percentile: float) -> Set[int]:
    """
    Reads the metadata catalog, computes the size threshold, and returns a hashed set 
    of TARGETIDs that pass the criteria.
    
    Args:
        metadata_path (Path): Path to the .fits metadata file.
        size_col (str): The column name containing the size metric.
        percentile (float): The percentile threshold to apply.
        
    Returns:
        Set[int]: A set of valid TARGETIDs for O(1) lookup performance.
    """
    logger.info(f"Loading metadata from: {metadata_path}")
    
    try:
        catalog = Table.read(metadata_path)
        df = catalog.to_pandas()
    except Exception as e:
        logger.error(f"Failed to read metadata catalog: {e}")
        sys.exit(1)
        
    # Ensure correct column names (case-insensitivity handling)
    target_col = 'TARGETID' if 'TARGETID' in df.columns else 'targetid'
    
    if size_col not in df.columns:
        logger.error(f"Size column '{size_col}' not found in metadata. Available: {df.columns.tolist()}")
        sys.exit(1)
        
    # Clean and compute threshold
    size_vals = df[size_col].values
    valid_mask = np.isfinite(size_vals)
    clean_sizes = size_vals[valid_mask]
    
    threshold = np.percentile(clean_sizes, percentile)
    logger.info(f"Computed P{percentile} threshold for '{size_col}': {threshold:.4f}")
    
    # Filter dataframe
    large_df = df[(valid_mask) & (df[size_col] > threshold)]
    
    # Extract IDs into a Set
    valid_ids = set(large_df[target_col].astype('int64').tolist())
    
    logger.info(f"Selected {len(valid_ids)} / {len(df)} targets ({len(valid_ids)/len(df):.2%} retention)")
    
    return valid_ids


def process_split(split: str, data_dir: Path, save_dir: Path, valid_ids: Optional[Set[int]], num_workers: int, filter_corrupt: bool) -> None:
    """
    Loads, filters, and saves a specific data split utilizing HuggingFace's 
    multi-threaded filtering.
    
    Args:
        split (str): The dataset split to process (e.g., 'train', 'test').
        data_dir (Path): Source Arrow directory.
        save_dir (Path): Destination Arrow directory.
        valid_ids (Set[int]): Hashed set of valid TARGETIDs.
        num_workers (int): CPU cores allocated for mapping.
    """
    logger.info(f"\n#--- PROCESSING SPLIT: {split.upper()} ---#")
    
    # 1. Locate parts for the current split
    split_dirs = sorted(data_dir.glob(f"{split}_*"))
    if not split_dirs:
        logger.warning(f"No Arrow directories found for split '{split}' in {data_dir}. Skipping.")
        return
        
    logger.info(f"Found {len(split_dirs)} parts. Loading dataset...")
    
    # 2. Load and concatenate
    try:
        ds = concatenate_datasets([load_from_disk(str(p)) for p in split_dirs])
        logger.info(f"Original '{split}' size: {len(ds)} samples.")
    except Exception as e:
        logger.error(f"Failed to load datasets for {split}: {e}")
        return

    # Optional pre-filter to detect corrupted array shapes matching TARGETIDS
    if filter_corrupt:
        logger.info(f"Scanning for data corruption in {split}... This might take a bit.")
        ds = ds.with_format("numpy")
        corrupt_indices = set()
        
        for i in range(len(ds)):
            try:
                # Triggers the access error if shape/bytes mismatch exists
                _ = ds[i]
            except Exception as e:
                corrupt_indices.add(i)
                
        if corrupt_indices:
            logger.warning(f"Found {len(corrupt_indices)} corrupted records in {split}. Removing them.")
            ds = ds.select([i for i in range(len(ds)) if i not in corrupt_indices])
        else:
            logger.info(f"No corruption found in {split}.")
            
        ds = ds.with_format(None) # Revert to pyarrow default for safe processing

    # 3. Apply high-performance filtration
    if valid_ids is not None:
        logger.info(f"Applying target ID filter using {num_workers} workers...")
        
        # The lambda function checks O(1) against the set
        filtered_ds = ds.filter(
            lambda targetids: [int(tid) in valid_ids for tid in targetids],
            input_columns=["targetid"],
            batched=True,
            batch_size=1000,
            num_proc=num_workers,
            desc=f"Filtering {split}"
        )
        logger.info(f"Filtered '{split}' size: {len(filtered_ds)} samples.")
    else:
        logger.info(f"Skipping target ID size filter (no metadata provided).")
        filtered_ds = ds

    # 4. Save to disk with explicit sharding
    # We save into a single folder (e.g., train_0) to maintain compatibility with 
    # the dataloader's `f"{split}_*"` pattern search.
    split_save_path = save_dir / f"{split}_0"
    
    logger.info(f"Saving to {split_save_path} (Shard size: 500MB)...")
    filtered_ds.save_to_disk(
        str(split_save_path), 
        max_shard_size="500MB"
    )
    
    logger.info(f"Split '{split}' completed successfully.")


def main():
    args = parse_args()
    
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    
    # Validation
    if not data_dir.exists():
        logger.error(f"Input Data directory not found: {data_dir}")
        sys.exit(1)
        
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Phase 1: Metadata extraction
    valid_target_ids = None
    if args.metadata_path is not None:
        valid_target_ids = extract_large_galaxy_ids(
            metadata_path=Path(args.metadata_path), 
            size_col=args.size_col, 
            percentile=args.percentile
        )
        
        if not valid_target_ids:
            logger.error("No valid IDs extracted. Check your threshold and column name.")
            sys.exit(1)
    else:
        logger.info("No metadata path provided. Skipping size filtration and proceeding to Arrow processing.")

    # Phase 2: Arrow filtration
    for split in ["train", "test"]:
        process_split(
            split=split,
            data_dir=data_dir,
            save_dir=save_dir,
            valid_ids=valid_target_ids,
            num_workers=args.num_workers,
            filter_corrupt=args.filter_corrupt
        )
        
    logger.info("\nDataset filtration pipeline completed successfully.")


if __name__ == "__main__":
    main()