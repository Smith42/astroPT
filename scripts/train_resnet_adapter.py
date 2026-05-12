#!/usr/bin/env python3
"""
AstroPT ResNet Adapter Training Script.

Trains the EuclidToHSC_ResNet (and HSCToEuclid_ResNet) adapter using
AstroPT's EuclidDESIDatasetArrow. The adapter bridges Euclid imagery 
(4 bands: VIS, Y, J, H) to HSC format (5 bands: g, r, i, z, y) 
before tokenization with AION's frozen ImageCodec.

Author: Víctor Alonso
Date: May 2026
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

# CRITICAL: Monkey-patch aion.modalities BEFORE any codec import
import aion.modalities
from aion.modalities import Image as AionImage, HSCImage as AionHSCImage
from astropt.resnet_adapter import EuclidImage, EuclidToHSC_ResNet, HSCToEuclid_ResNet, EUCLID_BANDS, HSC_BANDS

aion.modalities.EuclidImage = EuclidImage

from aion.codecs.image import ImageCodec
from aion.codecs.config import HF_REPO_ID
from astropt.aion_tokeniser import load_frozen_image_codec

# AstroPT imports
from astropt.model import ModalityConfig, ModalityRegistry
from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow


def setup_logging(output_dir: Path):
    """Set up logging to both console and file."""
    log_file = output_dir / "train_adapter.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("TrainResNet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ResNet-based EuclidToHSC adapter using AstroPT Arrow dataset."
    )
    
    # Data
    parser.add_argument(
        "--data-dir", type=str,
        default="/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated",
        help="Path to the Arrow dataset root"
    )
    
    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate for cosine decay")
    parser.add_argument("--warmup-epochs", type=int, default=1, help="Number of warmup epochs")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=100 , help="Log every N steps")
    
    # Model
    parser.add_argument("--hidden", type=int, default=128,
                        help="Hidden dimension of the ResNet adapter")
    
    # Output
    parser.add_argument("--output", type=str, default="astroPT/logs/resnet_adapter_weights",
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


def codec_roundtrip(codec: ImageCodec, hsc_flux_linear: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Run HSC flux (linear scale) through the codec (encode -> decode) to get reconstruction and unique token count."""
    hsc_obj = AionHSCImage(flux=hsc_flux_linear.clone(), bands=HSC_BANDS)
    tokens = codec.encode(hsc_obj)
    
    # Calculate unique tokens in the batch
    num_unique = len(torch.unique(tokens))
    
    hsc_rec = codec.decode(tokens, bands=HSC_BANDS)
    return hsc_rec.flux, num_unique


def codec_bridge_ste(codec: ImageCodec, hsc_flux_arcsinh: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    Straight-Through Estimator bridge through the frozen codec.
    Translates from arcsinh (Adapter domain) to linear (AION domain) and back.
    Returns (reconstructed_arcsinh, unique_tokens_count)
    """
    with torch.no_grad():
        # Protection: Clamp values to prevent sinh explosion (max ~10 for safety)
        safe_arcsinh = torch.clamp(hsc_flux_arcsinh, min=-10.0, max=10.0)
        
        # Translate to linear domain for AION codec
        hsc_linear = torch.sinh(safe_arcsinh)
        
        # Pass through Codec roundtrip
        y_linear, unique_tokens = codec_roundtrip(codec, hsc_linear)
        
        # Translate back to arcsinh domain for MSE loss calculation
        y_arcsinh = torch.arcsinh(y_linear)
        
    # STE: forward uses codec output, backward uses identity gradient
    return hsc_flux_arcsinh + (y_arcsinh - hsc_flux_arcsinh).detach(), unique_tokens


# LR Scheduler Functions
def get_learning_rate(it: int, total_it: int, warmup_it: int, lr: float, min_lr: float) -> float:
    """Calculates the learning rate for the current iteration using Cosine Decay with Warmup."""
    # 1. Linear Warmup Phase
    if it < warmup_it:
        return lr * (it + 1) / (warmup_it + 1)
    
    # 2. Post-Decay Phase (Constant Min LR)
    if it > total_it:
        return min_lr
    
    # 3. Cosine Decay Phase
    decay_ratio = (it - warmup_it) / (total_it - warmup_it)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def main() -> None:
    args = parse_args()
    
    # Setup
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(out_dir)
    
    logger.info("="*50)
    logger.info("ResNet Adapter Training Initialized")
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    logger.info(f"Output Directory: {out_dir}")
    logger.info("="*50)
    
    # ---- Data ----
    registry = ModalityRegistry([
        ModalityConfig(
            name="images",
            input_size=32,
            patch_size=8,
            pos_input_size=1,
            embed_pos=True,
        )
    ])
    
    def euclid_norm(x):
        return torch.arcsinh(x)

    data_tf = {"images_norm": euclid_norm}

    logger.info(f"Loading dataset from: {args.data_dir}")
    train_dataset = EuclidDESIDatasetArrow(
        arrow_folder_root=args.data_dir,
        split="train",
        modality_registry=registry,
        spiral=False,
        stochastic=False,
        transform=data_tf,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    logger.info(f"Dataset size: {len(train_dataset)} samples")
    logger.info(f"Loader: {len(train_loader)} batches/epoch")
    
    # ---- Models ----
    logger.info("Loading frozen AION ImageCodec...")
    codec = load_frozen_image_codec(device)
    
    logger.info(f"Initializing ResNet Adapters (hidden_dim={args.hidden})...")
    euclid_to_hsc = EuclidToHSC_ResNet(hidden_dim=args.hidden, num_blocks=4).to(device)
    hsc_to_euclid = HSCToEuclid_ResNet(hidden_dim=args.hidden, num_blocks=4).to(device)
    
    optimizer = torch.optim.Adam(
        list(euclid_to_hsc.parameters()) + list(hsc_to_euclid.parameters()),
        lr=args.lr,
        weight_decay=1e-4, 
    )
    criterion = nn.MSELoss(reduction="mean")
    
    # AMP configuration
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    
    # Resume training
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        euclid_to_hsc.load_state_dict(ckpt["euclid_to_hsc"])
        hsc_to_euclid.load_state_dict(ckpt["hsc_to_euclid"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        logger.info(f"Resumed from {args.resume} (starting at epoch {start_epoch})")
    
    # ---- Training Loop ----
    logger.info(f"Starting training loop ({args.epochs} epochs)...")
    
    total_iters = args.epochs * len(train_loader)
    warmup_iters = args.warmup_epochs * len(train_loader)
    
    curr_iter = (start_epoch - 1) * len(train_loader)
    ema_loss = None
    alpha = 0.9  # For EMA
    
    for epoch in range(start_epoch, args.epochs + 1):
        euclid_to_hsc.train()
        hsc_to_euclid.train()
        
        epoch_start_time = time.time()
        step_start_time = time.time()
        
        for step, batch in enumerate(train_loader):
            if "images" not in batch:
                continue
            
            # Reshape patched images back to (B, C, H, W)
            images = batch["images"].to(device, dtype=torch.float32)
            B, num_patches, patch_dim = images.shape
            
            patch_size = 8
            channels = 4
            H_patches = W_patches = int(math.sqrt(num_patches))
            H = H_patches * patch_size
            W = W_patches * patch_size
            
            images = images.reshape(B, H_patches, W_patches, patch_size, patch_size, channels)
            images = images.permute(0, 5, 1, 3, 2, 4).contiguous()
            images = images.reshape(B, channels, H, W)
            
            if args.max_abs > 0:
                images = torch.clamp(images, -args.max_abs, args.max_abs)
            
            if args.transform_method == "crop":
                images = TF.crop(images, 
                                 top=torch.randint(0, images.shape[2] - args.target_size + 1, (1,)).item(),
                                 left=torch.randint(0, images.shape[3] - args.target_size + 1, (1,)).item(),
                                 height=args.target_size, width=args.target_size)
            elif args.transform_method == "resize":
                images = TF.resize(images, size=[args.target_size, args.target_size], antialias=True)
            
            # SET LEARNING RATE for this iteration
            lr = get_learning_rate(curr_iter, total_iters, warmup_iters, args.lr, args.min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                # 1. ResNet forward pass (arcsinh domain)
                hsc_pred_arcsinh = euclid_to_hsc(images)
                
                # Metric: Max value emitted by ResNet
                max_val = torch.max(torch.abs(hsc_pred_arcsinh)).item()
                
                # 2. STE bridge (domain translation -> Codec -> back to arcsinh)
                hsc_decoded_arcsinh, unique_tokens = codec_bridge_ste(codec, hsc_pred_arcsinh)
                
                # 3. ResNet inverse pass (arcsinh domain)
                euclid_rec = hsc_to_euclid(hsc_decoded_arcsinh)
                
                # 4. MSE Loss in arcsinh space
                loss = criterion(euclid_rec, images)
            
            if not torch.isfinite(loss):
                logger.warning(f"Non-finite loss at epoch {epoch}, step {step}. Skipping.")
                continue
            
            loss.backward()
            
            # Metric: Gradient Norm
            params = list(euclid_to_hsc.parameters()) + list(hsc_to_euclid.parameters())
            grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in params if p.grad is not None]), 2).item()

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    list(euclid_to_hsc.parameters()) + list(hsc_to_euclid.parameters()),
                    args.grad_clip,
                )
            
            optimizer.step()
            curr_iter += 1
            
            # Update EMA Loss
            curr_loss = loss.item()
            if ema_loss is None:
                ema_loss = curr_loss
            else:
                ema_loss = alpha * ema_loss + (1 - alpha) * curr_loss
            
            # Reporting
            if (step + 1) % args.log_interval == 0:
                step_end_time = time.time()
                dt = (step_end_time - step_start_time) / args.log_interval
                samples_per_sec = args.batch_size / dt
                
                # Progress and ETA
                percent_complete = (step + 1) / len(train_loader) * 100
                remaining_steps = len(train_loader) - (step + 1)
                eta_seconds = remaining_steps * dt
                eta_str = str(timedelta(seconds=int(eta_seconds)))
                
                mem_info = ""
                if device.type == "cuda":
                    mem = torch.cuda.max_memory_allocated() / 1024**2
                    mem_info = f" | Mem: {mem:.0f}MB"
                
                logger.info(
                    f"Epoch {epoch:02d} | Step {step+1:04d}/{len(train_loader)} "
                    f"({percent_complete:4.1f}%) | Loss: {ema_loss:.6f} | "
                    f"MaxVal: {max_val:5.1f} | UnqTok: {unique_tokens:3d} | GradN: {grad_norm:.2e} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                    f"{dt*1000:4.0f}ms/step | {samples_per_sec:4.1f}s/s | "
                    f"ETA: {eta_str}{mem_info}"
                )
                step_start_time = time.time()
        
        # Epoch Summary
        epoch_duration = str(timedelta(seconds=int(time.time() - epoch_start_time)))
        logger.info(f"--- Epoch {epoch} Finished in {epoch_duration} | Avg Loss: {ema_loss:.6f} ---")
        
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
    final_path = out_dir / "adapters_final.pt"
    torch.save({
        "euclid_to_hsc": euclid_to_hsc.state_dict(),
        "hsc_to_euclid": hsc_to_euclid.state_dict(),
        "args": vars(args),
    }, final_path)
    
    logger.info("="*50)
    logger.info(f"Training Complete! Final weights saved to: {final_path}")
    logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*50)


if __name__ == "__main__":
    main()
