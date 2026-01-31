"""
Robust P99 Statistics Calculator using PyArrow Compute.

This script uses PyArrow's compute kernel to safely cast and filter data 
before moving it to Python/Numpy. This prevents type errors with nested 
lists or null values.

Author: Victor Alonso Rodríguez
Date: January 2026
"""

import argparse
import glob
import logging
import os
import random
import sys
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc 
from tqdm import tqdm
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate P99 after ASINH transformation")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--sample_prob", type=float, default=1.0, help="File subsample percent to be process")
    parser.add_argument("--pixels_per_file", type=int, default=50_000)
    parser.add_argument("--asinh_a_img", type=float, default=0.01, help="Scaler 'a' for images")
    parser.add_argument("--asinh_a_spec", type=float, default=0.01, help="Scaler 'a' for spectra")
    return parser.parse_args()

def get_arrow_files(root_dir: str) -> List[str]:
    """
    Recursively finds all Arrow IPC files within a directory structure.

    Args:
        root_dir (str): The base directory to search in (e.g., containing 'train_00', 'test_01').

    Returns:
        List[str]: A sorted list of absolute file paths ending in '.arrow'.
    """
    
    pattern = os.path.join(root_dir, "**", "*.arrow")
    files = sorted(glob.glob(pattern, recursive=True))
    
    return files

def extract_valid_values(
    arrow_array: pa.Array, 
    max_values: int
) -> np.ndarray:
    """
    Flattens, casts, and filters a PyArrow Array using compute kernels.

    This function handles nested list types (recursively flattening them),
    casts the data to float32 using PyArrow's safe casting (handling nulls),
    and filters out non-finite values (NaNs/Infs) before converting to NumPy.
    It also performs random subsampling if the resulting array exceeds `max_values`.

    Args:
        arrow_array (pa.Array): A chunk of data from an Arrow Table column. 
                                Can be nested (ListArray) or flat.
        max_values (int): The maximum number of elements to return. If the valid 
                          data exceeds this, a random subsample is returned.

    Returns:
        np.ndarray: A 1D NumPy array of type float32 containing valid, finite values.
                    Returns an empty array if the input type is not numeric or on error.
    """
    try:
        
        # Recursive Flattening using Arrow checks
        current = arrow_array
        while isinstance(current.type, (pa.ListType, pa.LargeListType, pa.FixedSizeListType)):
            current = current.flatten()
            
        # Check compatibility
        if not (pa.types.is_floating(current.type) or pa.types.is_integer(current.type)):
            return np.array([])

        # Safe Cast to Float32 
        float_data = pc.cast(current, pa.float32())

        # Filter Finite Values 
        is_finite = pc.is_finite(float_data)
        is_finite = pc.fill_null(is_finite, False)
        
        valid_data = pc.filter(float_data, is_finite)

        # Convert to Numpy
        data_np = valid_data.to_numpy(zero_copy_only=False)

        # Subsample if too large
        if len(data_np) > max_values:
            data_np = np.random.choice(data_np, max_values, replace=False)
            
        return data_np

    except Exception as e:
        # Debug log only if needed, usually just return empty to keep moving
        # logger.debug(f"Extraction error: {e}")
        return np.array([])

def process_arrow_file(
    file_path: str, 
    max_values: int = 100_000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads a single Arrow IPC file and extracts sampled statistics for images and spectra.

    Opens the file using memory mapping for efficiency and iterates through
    relevant columns ('image_vis', 'image_nisp_*', 'spectrum_flux').
    Aggregates valid pixels/flux points across all chunks in the file.

    Args:
        file_path (str): Path to the .arrow file.
        max_values (int, optional): Maximum number of pixels/flux points to keep 
                                    per file/chunk to control memory usage. Defaults to 100,000.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two 1D float32 arrays:
            - element 0: Aggregated valid image pixels (VIS + NISP).
            - element 1: Aggregated valid spectrum flux values.
            Returns empty arrays if the file cannot be read or contains no valid data.
    """
    img_buffer = []
    spec_buffer = []
    
    try:
        with pa.memory_map(file_path, 'r') as source:
            try:
                reader = pa.ipc.open_file(source)
            except pa.lib.ArrowInvalid:
                source.seek(0)
                reader = pa.ipc.open_stream(source)
                
            table = reader.read_all()
            
            # IMAGES
            img_cols = ['image_vis', 'image_nisp_h', 'image_nisp_j', 'image_nisp_y']
            for col_name in img_cols:
                if col_name in table.column_names:
                    col = table[col_name]
                    # Process entire column at once (faster than iterating chunks for Compute)
                    # But if file is huge, iterating chunks is safer for RAM. 
                    # Arrow tables usually have few chunks.
                    for chunk in col.chunks:
                        vals = extract_valid_values(chunk, max_values)
                        if len(vals) > 0:
                            img_buffer.append(vals)

            # SPECTRA
            if 'spectrum_flux' in table.column_names:
                col = table['spectrum_flux']
                for chunk in col.chunks:
                    vals = extract_valid_values(chunk, max_values)
                    if len(vals) > 0:
                        spec_buffer.append(vals)

    except Exception as e:
        logger.warning(f"Skipping {os.path.basename(file_path)}: {e}")
        return None, None

    final_img = np.concatenate(img_buffer) if img_buffer else np.array([])
    final_spec = np.concatenate(spec_buffer) if spec_buffer else np.array([])
    
    return final_img, final_spec

def main() -> None:
    args = parse_args()
    all_files = get_arrow_files(args.data_dir)
    
    num_files_to_process = max(1, int(len(all_files) * args.sample_prob))
    selected_files = random.sample(all_files, num_files_to_process)
    
    logger.info(f"Cumputing P99 with asinh(x/a). a_img={args.asinh_a_img}, a_spec={args.asinh_a_spec}")
    
    global_img = []
    global_spec = []
    
    for fpath in tqdm(selected_files, desc="Processing Arrows"):
        i_data, s_data = process_arrow_file(fpath, args.pixels_per_file)
        
        if i_data is not None and len(i_data) > 0:
            transformed_img = np.arcsinh(i_data / args.asinh_a_img)
            global_img.append(transformed_img)
            
        if s_data is not None and len(s_data) > 0:
            transformed_spec = np.arcsinh(s_data / args.asinh_a_spec)
            global_spec.append(transformed_spec)
            
    print("FINAL STATS (ASINH)")
    
    if global_img:
        full_img = np.concatenate(global_img)
        img_p99 = np.percentile(full_img, 99)
        print(f" -> GLOBAL CONSTANT IMAGES:     {img_p99:.6f}")
    
    if global_spec:
        full_spec = np.concatenate(global_spec)
        spec_p99 = np.percentile(full_spec, 99)
        print(f" -> GLOBAL CONSTANT SPECTRA:    {spec_p99:.6f}")
    
    print("-"*40)

if __name__ == "__main__":
    main()