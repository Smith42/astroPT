"""
AION Frozen Embedding Extractor.

Features:
- Loads the frozen `aion-base` model from Hugging Face.
- Uses `EuclidDESIDatasetArrow` to read and tokenize on the fly:
  - Mapped Euclid to HSC space and tokenized as "tok_image_hsc".
  - Tokenized DESI spectra as "tok_spectrum_desi".
- Performs isolated passes for imagery and spectra, and a combined joint pass.
- Employs Numpy Memmap to write directly to disk with minimal RAM overhead.
- Packages output in format compatible with probing_downstream_benchmark.py.

Author: Victor Alonso Rodriguez / Antigravity
Date: May 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import gc
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
from contextlib import nullcontext

from astropt.model import ModalityConfig, ModalityRegistry
from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow

# Configure logging
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AION-Extract")

def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="AION Frozen Embedding Extractor")
    
    parser.add_argument("--data_dir", type=str, default="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated", help="Arrow data root directory")
    parser.add_argument("--save_dir", type=str, default="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/AION_freeze/embeddings/aion_embeddings", help="Output Saving Directory")
    parser.add_argument("--resnet_weights_path", type=str, default="/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/Euclid2AION_resnet_adapter_weights/adapters_final.pt", help="Path to ResNet adapter weights")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--pool_method_img", type=str, default="mean", help="Pooling method for images")
    parser.add_argument("--pool_method_spec", type=str, default="mean", help="Pooling method for spectra")
    parser.add_argument("--draw_from_centre", action="store_true", default=False, help="Use embeddings from middle layer instead of final layer")
    
    return parser.parse_args()


def apply_pooling(tensor: torch.Tensor,
                  method: str = "mean", 
                  p_value: float = 2.0, 
                  mixed_lambda: float = 0.5,
    ) -> torch.Tensor:
    """Applying different pooling methods to reduce token sequences to vectors."""
    if method == 'mean':
        return tensor.mean(dim=1)
    elif method == 'max':
        return tensor.max(dim=1)[0]
    elif method == 'mixed':
        return mixed_lambda * tensor.max(dim=1)[0] + (1 - mixed_lambda) * tensor.mean(dim=1)
    elif method == 'lp':
        return (torch.mean(tensor.abs()**p_value, dim=1))**(1.0 / p_value)
    elif method == 'rank':
        sorted_tensor, _ = torch.sort(tensor, dim=1, descending=True)
        k = max(1, int(sorted_tensor.size(1) * 0.1))
        return sorted_tensor[:, :k, :].mean(dim=1)
    else:
        raise ValueError(f"Invalid pooling method '{method}'.")


def run_aion_encoder(model, encoder_tokens, encoder_emb, encoder_mask, draw_from_centre=False):
    """Passes inputs through AION's encoder block-by-block, supporting middle-layer extraction."""
    x = encoder_tokens + encoder_emb
    centre_x = None
    
    for i, blk in enumerate(model.encoder):
        x = blk(x, mask=encoder_mask)
        if draw_from_centre and i == len(model.encoder) // 2:
            centre_x = x
            
    if draw_from_centre and centre_x is not None:
        x_final = model.encoder_norm(centre_x)
    else:
        x_final = model.encoder_norm(x)
        
    context = model.decoder_proj_context(x_final) + encoder_emb
    return context


def robust_collate(batch):
    """Filters out any sample that is missing either image or spectrum tokens."""
    valid_samples = []
    for s in batch:
        if s is not None and "aion_images" in s and "aion_spectra" in s:
            valid_samples.append(s)
            
    if not valid_samples:
        return None
        
    collated = {
        "targetid": torch.tensor([s["targetid"] for s in valid_samples], dtype=torch.long),
        "aion_images": torch.stack([s["aion_images"] for s in valid_samples]),
        "aion_spectra": torch.stack([s["aion_spectra"] for s in valid_samples])
    }
    return collated


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Targeting save directory: {args.save_dir}")
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 1. Load AION model
    logger.info("Loading pretrained AION-base model...")
    import os
    from aion import AION
    local_only = (os.environ.get("HF_HUB_OFFLINE") == "1") or (os.environ.get("HF_OFFLINE") == "1")
    if local_only:
        logger.info("HF_HUB_OFFLINE or HF_OFFLINE is set. Forcing local_files_only=True.")
    model = AION.from_pretrained("polymathic-ai/aion-base", local_files_only=local_only).to(device)
    model.eval()
    logger.info("AION-base model loaded successfully.")

    # 2. Setup Dataloader
    logger.info("Initializing ModalityRegistry and EuclidDESIDatasetArrow...")
    
    # Configure AION discrete tokenization modalities
    modalities = [
        ModalityConfig(
            name="aion_images",
            input_size=0,
            pos_input_size=0,
            patch_size=0,
            embed_pos=True,
        ),
        ModalityConfig(
            name="aion_spectra",
            input_size=0,
            pos_input_size=0,
            patch_size=0,
            embed_pos=True,
        )
    ]
    registry = ModalityRegistry(modalities)
    
    dataset = EuclidDESIDatasetArrow(
        arrow_folder_root=args.data_dir,
        split="test",  # Downstream probing tasks benchmark uses the test split
        modality_registry=registry,
        spiral=True,
        stochastic=False,
        transform={},
        resnet_weights_path=args.resnet_weights_path,
        aion_image_size=96,
        aion_image_transform="resize",
        use_pretokenized=False,
    )
    
    total_samples = len(dataset)
    emb_dim = 768  # AION-base dimension
    logger.info(f"Dataset successfully loaded. Total test samples: {total_samples}")
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,  # Crucial to include all samples
        collate_fn=robust_collate
    )

    # 3. Setup Memmap outputs directly on disk
    logger.info("Configuring memmapped files on disk...")
    mmaps = {
        'ids': np.lib.format.open_memmap(save_path / 'ids.npy', mode='w+', dtype='int64', shape=(total_samples,)),
        'aion_images': np.lib.format.open_memmap(save_path / 'aion_images.npy', mode='w+', dtype='float32', shape=(total_samples, emb_dim)),
        'aion_spectra': np.lib.format.open_memmap(save_path / 'aion_spectra.npy', mode='w+', dtype='float32', shape=(total_samples, emb_dim)),
        'joint': np.lib.format.open_memmap(save_path / 'joint.npy', mode='w+', dtype='float32', shape=(total_samples, emb_dim))
    }

    # Use Amp Autocast context if cuda is active
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    # 4. Extraction Loop
    write_idx = 0
    
    with torch.no_grad(), ctx:
        for batch in tqdm(loader, desc="Extracting AION embeddings"):
            if batch is None:
                continue
            curr_bs = len(batch['targetid'])
            end_idx = write_idx + curr_bs
            
            # Save IDs
            mmaps['ids'][write_idx:end_idx] = batch['targetid'].cpu().numpy()
            
            image_tokens = batch.get("aion_images")
            spectra_tokens = batch.get("aion_spectra")
            
            # 4a. Isolated Image pass
            if image_tokens is not None:
                image_tokens = image_tokens.to(device).long()
                # AION key: tok_image_hsc
                encoder_tokens, encoder_emb, encoder_mask, _ = model.embed_inputs(
                    {"tok_image_hsc": image_tokens}, mask=None, num_encoder_tokens=600
                )
                context_img = run_aion_encoder(model, encoder_tokens, encoder_emb, encoder_mask, args.draw_from_centre)
                p_img = apply_pooling(context_img, args.pool_method_img)
                mmaps['aion_images'][write_idx:end_idx] = p_img.float().cpu().numpy()
                
            # 4b. Isolated Spectrum pass
            if spectra_tokens is not None:
                spectra_tokens = spectra_tokens.to(device).long()
                # AION key: tok_spectrum_desi
                encoder_tokens, encoder_emb, encoder_mask, _ = model.embed_inputs(
                    {"tok_spectrum_desi": spectra_tokens}, mask=None, num_encoder_tokens=300
                )
                context_spec = run_aion_encoder(model, encoder_tokens, encoder_emb, encoder_mask, args.draw_from_centre)
                p_spec = apply_pooling(context_spec, args.pool_method_spec)
                mmaps['aion_spectra'][write_idx:end_idx] = p_spec.float().cpu().numpy()
                
            # 4c. Combined Joint pass
            if image_tokens is not None and spectra_tokens is not None:
                # AION keys: tok_image_hsc and tok_spectrum_desi
                encoder_tokens, encoder_emb, encoder_mask, _ = model.embed_inputs(
                    {
                        "tok_image_hsc": image_tokens,
                        "tok_spectrum_desi": spectra_tokens
                    },
                    mask=None,
                    num_encoder_tokens=900
                )
                context_joint = run_aion_encoder(model, encoder_tokens, encoder_emb, encoder_mask, args.draw_from_centre)
                p_joint = apply_pooling(context_joint, "mean")
                mmaps['joint'][write_idx:end_idx] = p_joint.float().cpu().numpy()
                
            write_idx = end_idx

    # 5. Flush and truncate memmapped files to actual valid samples count
    logger.info(f"Flushing memmapped files (valid samples: {write_idx}/{total_samples})...")
    for m in mmaps.values():
        m.flush()
        
    # Read, truncate and overwrite
    logger.info("Truncating memmapped files to match valid samples count...")
    for name in mmaps.keys():
        file_path = save_path / f"{name}.npy"
        data = np.load(file_path, mmap_mode='r')
        truncated_data = np.array(data[:write_idx])
        # Overwrite file
        np.save(file_path, truncated_data)

    final_data = {k: np.load(save_path / f"{k}.npy", mmap_mode='r') for k in mmaps.keys()}
    np.savez_compressed(save_path / "embeddings_all.npz", **final_data)
    
    # Save Metadata
    with open(save_path / "extraction_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
        
    logger.info(f"Extraction and packaging complete. Outputs saved under {save_path}.")

if __name__ == "__main__":
    main()
