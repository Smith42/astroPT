#!/usr/bin/env python3
"""
AstroPT U-Net Adapter Retraining Script.

Trains the EuclidToHSC (and optionally HSCToEuclid) U-Net adapter using
AstroPT's own EuclidDESIDatasetArrow dataset. The adapter is needed to
bridge Euclid imagery (4 bands: VIS, Y, J, H) to HSC format (5 bands:
g, r, i, z, y) before tokenization with AION's frozen ImageCodec.

The training uses a roundtrip loss:
    Euclid -> UNet_fwd -> (Codec encode -> decode) -> UNet_inv -> Euclid_rec
with a Straight-Through Estimator (STE) through the frozen codec.

Usage:
    python scripts/retrain_unet_adapter.py --epochs 20 --batch-size 32
    
Author: Victor Alonso Rodriguez
Date: April 2026
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# CRITICAL: Monkey-patch aion.modalities BEFORE any codec import
import aion.modalities
from aion.modalities import Image as AionImage, HSCImage as AionHSCImage
from fmb.models.aion.modalities import EuclidImage

aion.modalities.EuclidImage = EuclidImage

from aion.codecs.image import ImageCodec
from aion.codecs.config import HF_REPO_ID
from fmb.models.aion.model import (
    EUCLID_BANDS,
    HSC_BANDS,
    EuclidToHSC,
    HSCToEuclid,
)

# AstroPT imports
from astropt.model import ModalityConfig, ModalityRegistry
from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain EuclidToHSC U-Net adapter using AstroPT Arrow dataset."
    )
    
    # Data
    parser.add_argument(
        "--data-dir", type=str,
        default="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_filter_corrupt",
        help="Path to the Arrow dataset root"
    )
    
    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Model
    parser.add_argument("--hidden", type=int, default=16,
                        help="Hidden dimension of the U-Net adapter")
    
    # Output
    parser.add_argument("--output", type=str, default="logs/unet_adapter_weights",
                        help="Directory to save adapter weights")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Preprocessing
    parser.add_argument("--target-size", type=int, default=112,
                        help="Target spatial size for codec input")
    parser.add_argument("--transform-method", type=str, default="crop",
                        choices=["crop", "resize"],
                        help="Method to adjust image size: 'crop' or 'resize'")
    parser.add_argument("--max-abs", type=float, default=100.0,
                        help="Clamp pixel values to [-max_abs, max_abs]")
    
    return parser.parse_args()


from astropt.aion_tokeniser import load_frozen_image_codec


def codec_roundtrip(codec: ImageCodec, hsc_flux: torch.Tensor) -> torch.Tensor:
    """Run HSC flux through the codec (encode -> decode) to get a reconstruction."""
    hsc_obj = AionHSCImage(flux=hsc_flux, bands=HSC_BANDS)
    tokens = codec.encode(hsc_obj)
    hsc_rec = codec.decode(tokens, bands=HSC_BANDS)
    return hsc_rec.flux


def codec_bridge_ste(codec: ImageCodec, hsc_flux: torch.Tensor) -> torch.Tensor:
    """
    Straight-Through Estimator bridge through the frozen codec.
    Allows gradients to flow through the adapter while the codec is frozen.
    """
    with torch.no_grad():
        y = codec_roundtrip(codec, hsc_flux)
    # STE: forward uses codec output, backward uses identity gradient
    return hsc_flux + (y - hsc_flux).detach()


def main() -> None:
    args = parse_args()
    
    # Setup
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Output: {out_dir}")
    
    # ---- Data ----
    # Create a minimal ModalityRegistry with only images
    # We just need the dataloader to load images, no spectra
    registry = ModalityRegistry([
        ModalityConfig(
            name="images",
            input_size=32,  # Dummy, we'll use raw images not patches
            patch_size=8,
            pos_input_size=1,
            embed_pos=True,
        )
    ])
    
    # No transforms — we want raw pixel values for the U-Net
    train_dataset = EuclidDESIDatasetArrow(
        arrow_folder_root=args.data_dir,
        split="train",
        modality_registry=registry,
        spiral=False,
        stochastic=False,
        transform={},
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    print(f"Dataset: {len(train_dataset)} samples, {len(train_loader)} batches/epoch")
    
    # ---- Models ----
    codec = load_frozen_image_codec(device)
    
    euclid_to_hsc = EuclidToHSC(
        hidden=args.hidden, use_checkpointing=False
    ).to(device)
    
    hsc_to_euclid = HSCToEuclid(
        hidden=args.hidden, use_checkpointing=False
    ).to(device)
    
    optimizer = torch.optim.Adam(
        list(euclid_to_hsc.parameters()) + list(hsc_to_euclid.parameters()),
        lr=args.lr,
    )
    criterion = nn.MSELoss(reduction="mean")
    
    # AMP for faster training
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bfloat16 doesn't need scaling
    
    # Resume
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        euclid_to_hsc.load_state_dict(ckpt["euclid_to_hsc"])
        hsc_to_euclid.load_state_dict(ckpt["hsc_to_euclid"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        print(f"Resumed from {args.resume} (epoch {start_epoch})")
    
    # ---- Image Transform ----
    import torchvision.transforms.functional as TF
    
    # ---- Training Loop ----
    print(f"Starting training (epochs {start_epoch} to {args.epochs})...")
    
    for epoch in range(start_epoch, args.epochs + 1):
        euclid_to_hsc.train()
        hsc_to_euclid.train()
        epoch_losses = []
        
        for step, batch in enumerate(train_loader):
            # The dataloader returns a dict with "images" as patched tensor
            # We need the raw stacked galaxy images — they're assembled inside __getitem__
            # but we get patches. For the U-Net we need full images.
            # 
            # WORKAROUND: Since our dataloader patches images, we load raw data differently.
            # The "images" tensor has shape (B, num_patches, patch_size*patch_size*channels)
            # We cannot easily reverse the spiral. Instead, we'll access the raw dataset.
            #
            # For simplicity, we skip this batch if images are not available
            if "images" not in batch:
                continue
            
            # Get images tensor (B, num_patches, patch_dim) and reshape back to (B, C, H, W)
            # This only works correctly with non-spiral, non-augmented images
            images = batch["images"].to(device, dtype=torch.float32)
            B, num_patches, patch_dim = images.shape
            
            # Compute spatial dimensions
            # patch_dim = patch_size * patch_size * channels
            patch_size = 8  # Default from our config
            channels = 4    # VIS + NISP Y,J,H
            H_patches = W_patches = int(math.sqrt(num_patches))
            H = H_patches * patch_size
            W = W_patches * patch_size
            
            # Reshape: (B, H_p, W_p, ps, ps, C) -> (B, C, H, W) 
            images = images.reshape(B, H_patches, W_patches, patch_size, patch_size, channels)
            images = images.permute(0, 5, 1, 3, 2, 4).contiguous()
            images = images.reshape(B, channels, H, W)
            
            # Clamp extreme values
            if args.max_abs > 0:
                images = torch.clamp(images, -args.max_abs, args.max_abs)
            
            # Apply configured spatial transformation
            if args.transform_method == "crop":
                # Use RandomCrop during training for data augmentation
                images = TF.crop(images, 
                                 top=torch.randint(0, images.shape[2] - args.target_size + 1, (1,)).item(),
                                 left=torch.randint(0, images.shape[3] - args.target_size + 1, (1,)).item(),
                                 height=args.target_size, width=args.target_size)
            elif args.transform_method == "resize":
                images = TF.resize(images, size=[args.target_size, args.target_size], antialias=True)
            
            # Forward: Euclid -> HSC -> Codec roundtrip -> Euclid reconstruction
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                hsc_pred = euclid_to_hsc(images)
                hsc_decoded = codec_bridge_ste(codec, hsc_pred)
                euclid_rec = hsc_to_euclid(hsc_decoded)
                loss = criterion(euclid_rec, images)
            
            if not torch.isfinite(loss):
                print(f"[WARNING] Non-finite loss at step {step}, skipping")
                continue
            
            loss.backward()
            
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    list(euclid_to_hsc.parameters()) + list(hsc_to_euclid.parameters()),
                    args.grad_clip,
                )
            
            optimizer.step()
            epoch_losses.append(loss.item())
            
            if (step + 1) % 50 == 0:
                avg = np.mean(epoch_losses[-50:])
                print(f"  [Epoch {epoch}] Step {step+1}/{len(train_loader)} | Loss: {avg:.6f}")
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        print(f"[Epoch {epoch}] Avg MSE: {avg_loss:.6f}")
        
        # Save checkpoint
        ckpt_path = out_dir / f"adapters_epoch_{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "euclid_to_hsc": euclid_to_hsc.state_dict(),
            "hsc_to_euclid": hsc_to_euclid.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }, ckpt_path)
    
    # Final save
    torch.save({
        "euclid_to_hsc": euclid_to_hsc.state_dict(),
        "hsc_to_euclid": hsc_to_euclid.state_dict(),
        "args": vars(args),
    }, out_dir / "adapters_final.pt")
    
    print(f"Training complete. Weights saved to {out_dir}/adapters_final.pt")


if __name__ == "__main__":
    main()
