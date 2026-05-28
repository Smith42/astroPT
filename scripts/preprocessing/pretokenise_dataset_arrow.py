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
import shutil

from astropt.aion_tokeniser import MultiprocessCodecManager
from aion.modalities import DESISpectrum

from astropt.resnet_adapter import EuclidImage

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("Pretokeniser")

def normalise_asinh(x, a=1.0, c=1.0):
    """Applies Inverse Hyperbolic Sine (asinh) transformation."""
    return torch.asinh(x.float() / a) / c

def process_file(file_path, output_path, codec_manager, batch_size, device, norm_params):
    logger = logging.getLogger("Pretokeniser")
    
    # Read the arrow file (Try File format first, then Stream)
    try:
        try:
            with pa.memory_map(str(file_path), 'rb') as source:
                table = pa.ipc.open_file(source).read_all()
        except Exception:
            with pa.ipc.open_stream(str(file_path)) as reader:
                table = reader.read_all()
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return False

    num_rows = len(table)
    image_tokens_list = [None] * num_rows
    
    # Check if spectra_tokens already exists to avoid re-tokenization
    has_spectra_tokens = 'spectra_tokens' in table.column_names
    if has_spectra_tokens:
        logger.info("  --> 'spectra_tokens' already exists. Skipping spectrum re-tokenization.")
        spectra_tokens_list = table['spectra_tokens'].to_pylist()
    else:
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
                
                # Apply asinh normalization
                vis_t = normalise_asinh(vis_t, norm_params['img_a'], norm_params['img_c'])
                y_t = normalise_asinh(y_t, norm_params['img_a'], norm_params['img_c'])
                j_t = normalise_asinh(j_t, norm_params['img_a'], norm_params['img_c'])
                h_t = normalise_asinh(h_t, norm_params['img_a'], norm_params['img_c'])

                img_tensor = torch.stack([vis_t, y_t, j_t, h_t], dim=0) # (4, H, W)
                
                # Convert to EuclidImage
                img_mod = EuclidImage(flux=img_tensor.unsqueeze(0), bands=["EUCLID-VIS", "EUCLID-Y", "EUCLID-J", "EUCLID-H"])
                
                tokens = codec_manager.encode(img_mod)
                # tokens['tok_image_hsc'] is a tensor of IDs from the transformed HSC image
                image_tokens_list[i + j] = tokens['tok_image_euclid'].squeeze(0).cpu().numpy().tolist()

        # 2. Tokenize Spectra (Only if not already present)
        if not has_spectra_tokens and 'spectrum_flux' in table.column_names:
            flux_data = batch['spectrum_flux'].to_pylist()
            wave_data = batch['spectrum_wave'].to_pylist()
            
            for j, (flux, wave) in enumerate(zip(flux_data, wave_data)):
                if flux is None or wave is None:
                    continue
                f = torch.from_numpy(np.array(flux)).unsqueeze(0).float()
                
                # Apply asinh normalization
                f = normalise_asinh(f, norm_params['spec_a'], norm_params['spec_c'])

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

    # Save to a temporary file first to avoid Bus Error if input == output
    # (caused by truncating a memory-mapped file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_output_path = f"{output_path}.tmp"
    
    try:
        with pa.OSFile(temp_output_path, 'wb') as f:
            with pa.ipc.new_stream(f, table.schema) as writer:
                writer.write_table(table)
        
        # Atomically move the temporary file to the final destination
        os.replace(temp_output_path, output_path)
    except Exception as e:
        logger.error(f"Failed to write output to {output_path}: {e}")
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize Euclid and DESI data from Arrow files using AION.")
    parser.add_argument("--data_dir", type=str, required=True, help="Input directory containing .arrow files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save tokenized .arrow files.")
    parser.add_argument("--resnet_weights", type=str, default="/home/valonso/iac18_mhuertas_shared/valonso/logs/resnet_adapter_weights/adapters_final.pt", help="Path to Resnet weights.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for tokenization.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu).")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel files to process (if using multiple GPUs/CPUs).")
    parser.add_argument("--max_files", type=int, default=None, help="Maximum number of files to process (useful for testing).")
    parser.add_argument("--resnet_hidden", type=int, default=128, help="Hidden dimension of the Resnet adapter.")
    
    # Normalization parameters
    parser.add_argument("--images_norm_scaler", type=float, default=1.0, help="Scaler for asinh normalization of images.")
    #parser.add_argument("--images_norm_const", type=float, default=7.603847, help="Constant for asinh normalization of images.")
    parser.add_argument("--images_norm_const", type=float, default=1, help="Constant for asinh normalization of images.")
    parser.add_argument("--spectra_norm_scaler", type=float, default=1.0, help="Scaler for asinh normalization of spectra.")
    parser.add_argument("--spectra_norm_const", type=float, default=1.0, help="Constant for asinh normalization of spectra.")

    args = parser.parse_args()
    logger = setup_logging()
    
    device = torch.device(args.device)
    
    norm_params = {
        'img_a': args.images_norm_scaler,
        'img_c': args.images_norm_const,
        'spec_a': args.spectra_norm_scaler,
        'spec_c': args.spectra_norm_const
    }

    # Initialize Codec Manager
    from astropt.aion_tokeniser import MultiprocessCodecManager
    codec_manager = MultiprocessCodecManager(
        resnet_weights_path=args.resnet_weights,
        device=device,
        aion_image_size=112,
        aion_image_transform="resize",
        resnet_hidden=args.resnet_hidden
    )
    
    input_root = Path(args.data_dir)
    output_root = Path(args.output_dir)
    
    # Find all arrow files
    all_files = list(input_root.rglob("*.arrow"))
    logger.info(f"Found {len(all_files)} arrow files in {args.data_dir}")
    
    if args.max_files is not None:
        all_files = all_files[:args.max_files]
        logger.info(f"Limiting to {len(all_files)} files as requested by --max_files")
    
    for file_path in tqdm(all_files, desc="Processing files"):
        # Determine relative path for output
        rel_path = file_path.relative_to(input_root)
        output_path = output_root / rel_path
        
        logger.info(f"Processing {file_path} -> {output_path}")
        success = process_file(file_path, output_path, codec_manager, args.batch_size, device, norm_params)
        if not success:
            logger.error(f"Failed to process {file_path}")

    # Preserve ALL non-arrow files from the original dataset (metadata, state, info, etc.)
    logger.info("Preserving all dataset metadata files...")
    for item in input_root.rglob("*"):
        if item.is_file() and item.suffix != ".arrow":
            rel_path = item.relative_to(input_root)
            out_item_path = output_root / rel_path
            os.makedirs(out_item_path.parent, exist_ok=True)
            if not out_item_path.exists(): # Avoid re-copying if already there
                shutil.copy2(item, out_item_path)
                logger.info(f"Preserved: {rel_path}")

if __name__ == "__main__":
    main()
