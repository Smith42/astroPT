from datasets import Dataset, Features, Value, Array2D, Sequence
import gc
import glob
import logging
import numpy as np
import os
import pickle
from typing import Any, Iterator, List

# PATH Configuration
INPUT_DIR = "/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data"
OUTPUT_ARROW_DIR = "/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_numpy"

# Dataset configuration
SPLITS = ["train", "test"]
SAVE_SIZE = 50

# Logging messages configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def clean_array(
    arr: np.ndarray | list[Any] | None, 
    dtype: type = np.float32
) -> np.ndarray | None:
    """
    Prepares, normalizes, and optimizes a data structure for efficient storage 
    in Apache Arrow.

    Performs three critical operations:
    1. Null handling (None safety).
    2. Data type conversion (to reduce memory footprint).
    3. Memory reorganization (contiguity) for Zero-Copy operations.

    Args:
        arr (np.ndarray | list[Any] | None): Input data. Can be a Python list, 
            an existing NumPy array, or None.
        dtype (type, optional): Target numeric data type. 
            Defaults to np.float32.

    Returns:
        np.ndarray | None: A contiguous and typed NumPy array, or None 
            if the input was null.
    """
    # 1. Protection against missing data
    if arr is None:
        return None

    # 2. Ensure it is a NumPy object
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    # 3. Type conversion and memory reordering
    return np.ascontiguousarray(arr.astype(dtype))


def generator_logic(
        split_name: str,
        file_list: List[str]
    ) -> Iterator[dict[str, Any]]:
    """
    Stream-processes chunked Pickle files to yield single galaxy examples for Arrow conversion.

    This function implements a memory-efficient generator pattern. Instead of loading 
    the entire dataset into RAM, it iterates through pickle files one by one. 
    It extracts scientific data (images and spectra), cleans the arrays for 
    Arrow compatibility, and yields individual dictionaries. Crucially, it manages 
    memory manually by deleting processed batches and invoking the garbage collector.

    Args:
        split_name (str): The dataset split to process (e.g., 'train', 'test'). 
                          Used to construct file paths like 'batch_train_1.pkl'.
        file_list (str): Specific pickle files to be converted to save them in batches.

    Yields:
        Iterator[dict[str, Any]]: A generator yielding a dictionary for each galaxy 
                                  containing keys like 'targetid', 'image_vis', 
                                  'spectrum_flux', etc.
    """

    # 1. Outer Loop: Iterate over files
    for i, filepath in enumerate(file_list):
        # Log progress every 10 files
        if i % 5 == 0: 
            logging.info(f"[{split_name}] Processing file {i+1}/{len(file_list)}: {os.path.basename(filepath)}")
        
        try:
            # Load the entire batch (list of dicts) into memory
            with open(filepath, 'rb') as f:
                batch_data = pickle.load(f)
        except Exception as e:
            logging.error(f"Error reading file {filepath}: {e}")
            continue
    
        if i == 0 and len(batch_data) > 0:
            sample_keys = list(batch_data[0].keys())
            logging.info(f"DEBUG KEYS FOUND IN PICKLE: {sample_keys}")

        # 2. Inner Loop: Iterate over galaxies within the current pickle batch
        for item in batch_data:
            try:
                
                # A. Identifiers
                targetid = item.get("targetid", -1)
                
                # B. Images (Individual Bands)
                vis_raw = item.get('vis_image')
                if vis_raw is None:
                    vis_raw = item.get('VIS_image')
                vis = clean_array(vis_raw)
                
                def get_nisp(key_lower, key_upper):
                    val = item.get(key_lower)
                    if val is None: 
                        val = item.get(key_upper)
                    return clean_array(val)
                    
                nisp_y = get_nisp('nisp_y_image', 'NISP_Y_image')
                nisp_j = get_nisp('nisp_j_image', 'NISP_J_image')
                nisp_h = get_nisp('nisp_h_image', 'NISP_H_image')
                
                # Critical Validation: VIS image is mandatory. Skip if missing.
                if vis is None:
                    logging.warning(f"Skipping {targetid}: VIS image is None. Available keys: {list(item.keys())}")
                    continue

                # C. Spectra (Unpack dictionary)
                spec_data = item.get('spectrum')
                flux = None
                wave = None
                ivar = None
                mask = None
                
                # Check if spectrum exists and extract components
                if spec_data is not None and isinstance(spec_data, dict):
                    flux = clean_array(spec_data.get('flux'))
                    wave = clean_array(spec_data.get('wavelength'))
                    ivar = clean_array(spec_data.get('ivar'))
                    # Masks are typically integers (uint32)
                    mask = clean_array(spec_data.get('mask'), dtype=np.uint32)

                # Yield a clean dictionary. Arrow writes this to disk immediately.
                yield {
                    "targetid": targetid,
                    "healpix": item.get("healpix", -1),
                    "redshift": item.get("redshift", -1.0),
                    
                    # Images
                    "image_vis": vis,
                    "image_nisp_y": nisp_y,
                    "image_nisp_j": nisp_j,
                    "image_nisp_h": nisp_h,
                    
                    # Spectra
                    "spectrum_flux": flux,
                    "spectrum_wave": wave,
                    "spectrum_ivar": ivar,
                    "spectrum_mask": mask
                }

            except Exception as e:
                # Skip single corrupt samples without crashing the pipeline
                logging.debug(f"Skipping sample {targetid}: {e}") # Optional debug log
                continue
        
        # 3. Memory Management
        # Delete the heavy list of dicts from RAM
        del batch_data
        # Force Python's Garbage Collector to release memory back to the OS
        gc.collect()

if __name__ == "__main__":
    
    # Define the schema explicitly to optimize Arrow storage efficiency.
    features = Features({
        # Metadata fields
        "targetid": Value("int64"),
        "healpix": Value("int64"),
        "redshift": Value("float32"),
        
        # Scientific Images (Fixed Shape Optimization)
        "image_vis": Array2D(shape=(224, 224), dtype="float32"),
        "image_nisp_y": Array2D(shape=(224, 224), dtype="float32"),
        "image_nisp_j": Array2D(shape=(224, 224), dtype="float32"),
        "image_nisp_h": Array2D(shape=(224, 224), dtype="float32"),
        
        # Spectra Data (1D Sequences)
        "spectrum_flux": Sequence(Value("float32")),
        "spectrum_wave": Sequence(Value("float32")),
        "spectrum_ivar": Sequence(Value("float32")),
        "spectrum_mask": Sequence(Value("uint32")),
    })

    for split in SPLITS:
        logging.info(f"#--- STARTING CONVERSION FOR SPLIT: {split} ---#")
        
        # 1. FIND ALL FILES
        pattern = os.path.join(INPUT_DIR, f"batch_{split}_*.pkl")
        all_files = sorted(glob.glob(pattern))
        all_files = [f for f in all_files if "_info" not in f]
        
        if not all_files:
            logging.warning(f"No files found for {split}")
            continue
        
        logging.info(f"Found {len(all_files)} files. Processing in chunks of {SAVE_SIZE}...")
        
        # 2. CHUNK LOOP
        total_chunks = (len(all_files) + SAVE_SIZE - 1) // SAVE_SIZE
        
        for i in range(0, len(all_files), SAVE_SIZE):
            chunk_idx = i // SAVE_SIZE
            chunk_files = all_files[i : i + SAVE_SIZE]
            
            # Define output path: .../processed_data_arrow/train/part_0
            part_dir = os.path.join(OUTPUT_ARROW_DIR, f"{split}_{chunk_idx}")
            
            # RESUME LOGIC: Check if this part already exists
            if os.path.exists(part_dir):
                logging.info(f"Chunk {chunk_idx}/{total_chunks} already exists at {part_dir}. Skipping.")
                continue
            
            logging.info(f"--- Processing Chunk {chunk_idx}/{total_chunks} ({len(chunk_files)} files) ---")
            
            # Create Dataset from the specific file list
            ds = Dataset.from_generator(
                generator_logic, 
                gen_kwargs={"split_name":split,"file_list": chunk_files},
                features=features 
            )
            
            # Save IMMEDIATELY
            ds.save_to_disk(part_dir, max_shard_size="500MB")
            
            # Cleanup
            del ds
            gc.collect()

    print("Conversion completed successfully.")