import os
import argparse
import logging
import torch
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dataclasses import asdict
import datetime

# Add src to path to import astropt modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from astropt.aion_tokeniser import MultiprocessCodecManager
from aion.modalities import DESISpectrum
from fmb.models.aion.modalities import EuclidImage

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("Pretokeniser")

def process_file(file_path, output_path, codec_manager, batch_size, device):
    logger = logging.getLogger("Pretokeniser")
    
    # Read the arrow file
    try:
        table = pa.ipc.open_stream(file_path).read_all()
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return False

    num_rows = len(table)
    image_tokens_list = [None] * num_rows
    spectra_tokens_list = [None] * num_rows
    
    # Process in batches
    for i in range(0, num_rows, batch_size):
        batch_end = min(i + batch_size, num_rows)
        batch = table.slice(i, batch_end - i)
        
        # 1. Tokenize Images
        if 'image_vis' in table.column_names:
            vis_data = batch['image_vis'].to_pylist()
            y_data = batch['image_nisp_y'].to_pylist() if 'image_nisp_y' in table.column_names else [None]*len(vis_data)
            j_data = batch['image_nisp_j'].to_pylist() if 'image_nisp_j' in table.column_names else [None]*len(vis_data)
            h_data = batch['image_nisp_h'].to_pylist() if 'image_nisp_h' in table.column_names else [None]*len(vis_data)
            
            for j, (vis, y, j_band, h) in enumerate(zip(vis_data, y_data, j_data, h_data)):
                if vis is None:
                    continue
                # Some might be missing, pad with zeros matching vis
                vis_t = torch.from_numpy(np.array(vis)).to(torch.float32)
                y_t = torch.from_numpy(np.array(y)).to(torch.float32) if y is not None else torch.zeros_like(vis_t)
                j_t = torch.from_numpy(np.array(j_band)).to(torch.float32) if j_band is not None else torch.zeros_like(vis_t)
                h_t = torch.from_numpy(np.array(h)).to(torch.float32) if h is not None else torch.zeros_like(vis_t)
                
                img_tensor = torch.stack([vis_t, y_t, j_t, h_t], dim=0) # (4, H, W)
                
                # Convert to EuclidImage
                img_mod = EuclidImage(flux=img_tensor.unsqueeze(0), bands=["EUCLID-VIS", "EUCLID-Y", "EUCLID-J", "EUCLID-H"])
                tokens = codec_manager.encode(img_mod)
                # tokens['tok_image_hsc'] is a tensor of IDs from the transformed HSC image
                image_tokens_list[i + j] = tokens['tok_image_hsc'].squeeze(0).cpu().numpy().tolist()

        # 2. Tokenize Spectra
        if 'spectrum_flux' in table.column_names:
            flux_data = batch['spectrum_flux'].to_pylist()
            wave_data = batch['spectrum_wave'].to_pylist()
            
            for j, (flux, wave) in enumerate(zip(flux_data, wave_data)):
                if flux is None or wave is None:
                    continue
                f = torch.from_numpy(np.array(flux)).unsqueeze(0).float()
                # AION needs ivar and mask too, providing dummies if not in dataset
                ivar = torch.ones_like(f)
                mask = torch.zeros_like(f, dtype=torch.bool)
                
                spec_mod = DESISpectrum(
                    flux=f, 
                    wavelength=torch.from_numpy(np.array(wave)).unsqueeze(0).float(),
                    ivar=ivar,
                    mask=mask
                )
                tokens = codec_manager.encode(spec_mod)
                spectra_tokens_list[i + j] = tokens['tok_spectrum_desi'].squeeze(0).cpu().numpy().tolist()

    # Create new columns
    if any(image_tokens_list):
        if "image_tokens" in table.column_names:
            table = table.drop_columns(["image_tokens"])
        col_arr = pa.array(image_tokens_list, type=pa.list_(pa.int64()))
        table = table.append_column("image_tokens", col_arr)
    
    if any(spectra_tokens_list):
        if "spectra_tokens" in table.column_names:
            table = table.drop_columns(["spectra_tokens"])
        col_arr = pa.array(spectra_tokens_list, type=pa.list_(pa.int64()))
        table = table.append_column("spectra_tokens", col_arr)

    # Save to new file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pa.OSFile(str(output_path), 'wb') as f:
        with pa.ipc.new_stream(f, table.schema) as writer:
            writer.write_table(table)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize Euclid and DESI data from Arrow files using AION.")
    parser.add_argument("--data_dir", type=str, required=True, help="Input directory containing .arrow files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save tokenized .arrow files.")
    parser.add_argument("--unet_weights", type=str, default="logs/unet_adapter_weights/adapters_final.pt", help="Path to U-Net weights.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for tokenization.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu).")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel files to process (if using multiple GPUs/CPUs).")
    
    args = parser.parse_args()
    logger = setup_logging()
    
    device = torch.device(args.device)
    
    # Initialize Codec Manager (not multiprocess here to keep it simple, just sequential file processing)
    # But we use the one from aion_tokeniser.py because it has the U-Net logic
    from astropt.aion_tokeniser import MultiprocessCodecManager
    codec_manager = MultiprocessCodecManager(
        unet_weights_path=args.unet_weights,
        device=device,
        aion_image_size=112,
        aion_image_transform="resize"
    )
    
    input_root = Path(args.data_dir)
    output_root = Path(args.output_dir)
    
    # Find all arrow files
    all_files = list(input_root.rglob("*.arrow"))
    logger.info(f"Found {len(all_files)} arrow files in {args.data_dir}")
    
    for file_path in tqdm(all_files, desc="Processing files"):
        # Determine relative path for output
        rel_path = file_path.relative_to(input_root)
        output_path = output_root / rel_path
        
        logger.info(f"Processing {file_path} -> {output_path}")
        success = process_file(file_path, output_path, codec_manager, args.batch_size, device)
        if not success:
            logger.error(f"Failed to process {file_path}")

if __name__ == "__main__":
    main()
