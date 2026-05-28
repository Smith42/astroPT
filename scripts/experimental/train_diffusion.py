"""
Conditional DDPM Training Script for Spectrum Generation.

Trains a SpectrumDiffusionModel to generate astronomical spectra (1D flux vectors)
conditioned on pre-extracted image embeddings from AstroPT. The training pipeline
follows the same iteration-based, checkpoint-resumable architecture used across
all AstroPT training scripts.

Usage:
    python scripts/train_diffusion.py \\
        --embeddings_path /path/to/embeddings_all.npz \\
        --data_dir /path/to/processed_data_arrow \\
        --train_dir ./logs/diffusion_run \\
        --context_dim 512 \\
        --max_iters 100000

Author: Antigravity (Senior ML Research Engineer)
Date: May 2026
"""

from __future__ import annotations

import contextlib
import datetime
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import HfArgumentParser

from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow
from astropt.model import ModalityRegistry, ModalityConfig
from astropt.model_diffusion import (
    DiffusionModelConfig,
    SpectrumDiffusionModel,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DiffusionTrainingConfig:
    """
    Training configuration for the Spectrum Diffusion Model.

    Aligned with the AstroPT training infrastructure (iteration-based loop,
    cosine LR with warmup, checkpointing, CSV logging, SLURM time-limit).
    All arguments are parseable via HfArgumentParser / CLI.
    """

    # --- Metadata & Paths ---
    train_name: str = "spectrum_diffusion"
    train_date: Optional[str] = None
    train_dir: Optional[str] = None
    seed: int = 42

    # --- Data Sources ---
    embeddings_path: str = ""                       # .npz with 'images' and 'targetid' keys
    embeddings_key: str = "images"                  # Key inside .npz for the image embeddings
    data_dir: str = "/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated"

    # --- Model Architecture ---
    context_dim: int = 512                          # Image embedding dimension
    time_emb_dim: int = 256                         # Timestep embedding dimension
    channel_dims: str = "64,128,256,512"            # U-Net channel progression (comma-separated)
    num_groups: int = 8                             # GroupNorm groups
    use_mid_attention: bool = True                  # Self-attention in bottleneck

    # --- Diffusion Schedule ---
    num_timesteps: int = 1000                       # DDPM timesteps T
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # --- Spectrum Processing ---
    spectrum_length: int = 7781
    spectra_norm_type: str = "asinh"
    spectra_norm_scaler: float = 1.0
    spectra_norm_const: float = 1.0                 # asinh only, no P99 division
    use_standard_scaling: bool = True               # Apply z-score after asinh for DDPM stability
    scaler_num_samples: int = 5000                  # How many samples to estimate μ/σ

    # --- Training Hyperparameters ---
    batch_size: int = 32
    num_workers: int = 4
    max_iters: int = 100_000
    init_from: str = "scratch"                      # "scratch" or "resume"
    learning_rate: float = 2e-4
    lr_min: float = 2e-5
    lr_warmup_iters: int = 5_000
    lr_decay_iters: int = 90_000
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0

    # --- Logging & Checkpointing ---
    eval_interval: int = 1_000
    eval_batches: int = 100
    log_interval: int = 200
    checkpoint_interval: int = 1_000
    early_stop_patience: int = 15
    early_stopping_min_iters: int = 55_000          # Minimum iterations before starting to count patience

    # --- Sampling ---
    num_samples_at_end: int = 8                     # How many spectra to generate after training

    # --- System ---
    max_run_hours: Optional[str] = None             # Time limit "HH:MM:SS"

    def __post_init__(self):
        if self.train_date is None:
            self.train_date = datetime.datetime.now().strftime("%Y%m%d")

        if self.train_dir is None:
            clean_name = re.sub(r"[^a-z0-9]", " ", self.train_name.lower())
            suffix = "_".join(clean_name.split())
            self.train_dir = f"./logs/{self.train_date}_{suffix}"


# ---------------------------------------------------------------------------
# Utility Functions (aligned with AstroPT conventions)
# ---------------------------------------------------------------------------

def parse_time_to_seconds(time_str: str) -> float:
    """Converts 'HH:MM:SS' to total seconds."""
    parts = time_str.split(":")
    if len(parts) != 3:
        raise ValueError("Time must be in format HH:MM:SS")
    h, m, s = map(int, parts)
    return h * 3600 + m * 60 + s


def project_directories_setup(base_dir: str) -> Tuple[Path, Path, Path, Path]:
    """Creates the standard directory structure for a training run."""
    train_dir = Path(base_dir)
    weights_dir = train_dir / "weights"
    logs_dir = train_dir / "logs"
    samples_dir = train_dir / "samples"

    for d in [train_dir, weights_dir, logs_dir, samples_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return train_dir, weights_dir, logs_dir, samples_dir


def get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"


def get_dataset_info(data_dir: str | Path) -> dict:
    """Scans data directory for versioning metadata."""
    try:
        max_mtime = 0.0
        total_size = 0
        data_path = Path(data_dir)
        for file_path in data_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in [".arrow", ".json"]:
                stats = file_path.stat()
                if stats.st_mtime > max_mtime:
                    max_mtime = stats.st_mtime
                total_size += stats.st_size
        last_modified = (
            datetime.datetime.fromtimestamp(max_mtime).strftime("%Y-%m-%d %H:%M:%S")
            if max_mtime > 0
            else "Unknown"
        )
        return {
            "data_last_modified": last_modified,
            "data_total_size_mb": round(total_size / (1024 * 1024), 2),
        }
    except Exception as e:
        return {"data_last_modified": "Unknown", "data_total_size_mb": 0.0, "error": str(e)}


def save_config_json(config: DiffusionTrainingConfig, save_dir: Path) -> None:
    """Persists the full configuration to JSON for reproducibility."""
    config_dict = asdict(config)
    config_dict["git_hash"] = get_git_commit_hash()
    config_dict.update(get_dataset_info(config.data_dir))
    with open(save_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)


def get_learning_rate(it: int, config: DiffusionTrainingConfig) -> float:
    """Cosine decay schedule with linear warmup."""
    if it < config.lr_warmup_iters:
        return config.learning_rate * (it + 1) / (config.lr_warmup_iters + 1)
    if it > config.lr_decay_iters:
        return config.lr_min
    decay_ratio = (it - config.lr_warmup_iters) / (config.lr_decay_iters - config.lr_warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.lr_min + coeff * (config.learning_rate - config.lr_min)


# ---------------------------------------------------------------------------
# StandardScaler for Diffusion
# ---------------------------------------------------------------------------

def compute_spectra_statistics(
    dataset: "PairedEmbeddingSpectraDataset",
    num_samples: int = 5000,
    logger: Optional[logging.Logger] = None,
) -> Tuple[float, float]:
    """
    Estimate per-pixel mean and std of the (already asinh-normalised) spectra
    from a random subset of the training set.

    The resulting μ and σ are used to z-score the spectra before feeding them
    to the diffusion model, ensuring the data distribution is centred around 0
    with unit variance — the optimal regime for DDPM noise schedules.

    Args:
        dataset: A PairedEmbeddingSpectraDataset (already asinh-normalised).
        num_samples: Number of random samples to use for estimation.
        logger: Optional logger for progress messages.

    Returns:
        (mean, std): Scalar statistics computed across all pixels and samples.
    """
    n = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), n, replace=False)

    # Accumulate running statistics (Welford's algorithm for memory efficiency)
    running_sum = 0.0
    running_sq_sum = 0.0
    total_pixels = 0

    for i, idx in enumerate(indices):
        _, spectrum, _ = dataset[int(idx)]  # [1, L]
        flat = spectrum.reshape(-1).double()
        running_sum += flat.sum().item()
        running_sq_sum += (flat * flat).sum().item()
        total_pixels += flat.numel()

        if logger and (i + 1) % 1000 == 0:
            logger.info(f"  Scaler estimation: {i + 1}/{n} samples processed.")

    mean = running_sum / total_pixels
    std = math.sqrt(running_sq_sum / total_pixels - mean * mean)
    std = max(std, 1e-8)  # Safety clamp

    if logger:
        logger.info(f"  Scaler statistics: μ = {mean:.6f}, σ = {std:.6f}")

    return mean, std


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PairedEmbeddingSpectraDataset(Dataset):
    """
    Pairs pre-extracted image embeddings with spectra from the Arrow dataset
    via shared TARGETID keys.

    The embeddings are loaded from a NumPy .npz file (keys: 'images', 'targetid').
    Spectra are loaded on-the-fly from the Arrow dataset and normalised with
    asinh transformation followed by an optional StandardScaler (z-score).
    """

    def __init__(
        self,
        embeddings_path: str,
        embeddings_key: str,
        arrow_dataset: EuclidDESIDatasetArrow,
        spectrum_length: int = 7781,
        scaler_mean: float = 0.0,
        scaler_std: float = 1.0,
        use_standard_scaling: bool = True,
    ):
        super().__init__()
        self.spectrum_length = spectrum_length
        self.use_standard_scaling = use_standard_scaling
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std

        # Load embeddings
        npz = np.load(embeddings_path)
        
        # Auto-detect embeddings key if not present in the NPZ archive
        if embeddings_key not in npz:
            if "EuclidImage" in npz:
                embeddings_key = "EuclidImage"
            elif "images" in npz:
                embeddings_key = "images"
                
        # Auto-detect target ID key
        id_key = "ids" if "ids" in npz else "targetid"
        
        self.embeddings = npz[embeddings_key].astype(np.float32)  # [N, context_dim]
        emb_ids = npz[id_key].astype(np.int64)                    # [N]

        # Build target ID lookup from arrow dataset (optimized for millisecond loading)
        self.arrow_ds = arrow_dataset
        tids = self.arrow_ds.ds['targetid']
        arrow_id_to_idx = {int(tid): i for i, tid in enumerate(tids)}

        # Match embeddings to spectra via TARGETID
        self.valid_pairs = []  # List of (emb_idx, arrow_idx)
        for emb_i, tid in enumerate(emb_ids):
            if int(tid) in arrow_id_to_idx:
                self.valid_pairs.append((emb_i, arrow_id_to_idx[int(tid)]))

        if len(self.valid_pairs) == 0:
            raise RuntimeError(
                f"No matching TARGETIDs found between embeddings ({len(emb_ids)}) "
                f"and Arrow dataset ({len(arrow_id_to_idx)})."
            )

    def __len__(self) -> int:
        return len(self.valid_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            embedding: [context_dim] float32 tensor.
            spectrum:  [1, spectrum_length] float32 tensor (normalised flux).
            redshift:  [] float32 scalar tensor.
        """
        emb_idx, arrow_idx = self.valid_pairs[idx]

        # Load embedding
        embedding = torch.from_numpy(self.embeddings[emb_idx])

        # Load spectrum from Arrow dataset (already normalised by the dataset transform)
        item = self.arrow_ds[arrow_idx]
        spectra = item["spectra"]  # [num_patches, patch_size] or [L, 1]

        # Load redshift
        redshift = torch.tensor(float(item.get("redshift", 0.0)), dtype=torch.float32)

        # Flatten back to 1D if the dataset returned patched spectra
        if isinstance(spectra, torch.Tensor):
            spectrum_flat = spectra.reshape(-1).float()
        else:
            spectrum_flat = torch.from_numpy(np.array(spectra)).reshape(-1).float()

        # Trim or pad to exact spectrum_length
        current_len = spectrum_flat.shape[0]
        if current_len > self.spectrum_length:
            spectrum_flat = spectrum_flat[: self.spectrum_length]
        elif current_len < self.spectrum_length:
            spectrum_flat = torch.nn.functional.pad(
                spectrum_flat, (0, self.spectrum_length - current_len)
            )

        # Apply StandardScaler: (x - μ) / σ to centre data at 0 with unit variance.
        # This is critical for DDPM: the noise schedule assumes data ~ N(0, 1).
        if self.use_standard_scaling:
            spectrum_flat = (spectrum_flat - self.scaler_mean) / self.scaler_std

        # Shape: [1, spectrum_length]
        spectrum = spectrum_flat.unsqueeze(0)

        return embedding, spectrum, redshift


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(
    model: SpectrumDiffusionModel,
    val_loader: DataLoader,
    config: DiffusionTrainingConfig,
    device: torch.device,
    ctx: Any,
) -> float:
    """Computes average noise prediction loss over a fixed number of validation batches."""
    model.eval()
    losses = []
    val_iter = iter(val_loader)

    for _ in range(config.eval_batches):
        try:
            embedding, spectrum, redshift = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            embedding, spectrum, redshift = next(val_iter)

        embedding = embedding.to(device)
        spectrum = spectrum.to(device)
        redshift = redshift.to(device)

        with ctx:
            loss = model(spectrum, embedding, redshift)

        losses.append(loss.item())

    model.train()
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def main() -> None:
    # Parse CLI arguments
    parser = HfArgumentParser((DiffusionTrainingConfig,))
    config = parser.parse_args_into_dataclasses()[0]

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Directory structure
    train_dir, weights_dir, logs_dir, samples_dir = project_directories_setup(config.train_dir)

    # Logging
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logs_dir / "training.log", mode="a"),
        ],
    )
    logger = logging.getLogger("Diffusion")

    # Validate embeddings path
    if not config.embeddings_path or not Path(config.embeddings_path).exists():
        logger.error(f"Embeddings file not found: '{config.embeddings_path}'")
        sys.exit(1)

    # Auto-load config.json of pre-trained model if it exists
    emb_path = Path(config.embeddings_path)
    pretrained_config_path = None
    
    # Check parent directory structures
    if len(emb_path.parents) >= 3:
        candidate = emb_path.parents[2] / "weights" / "config.json"
        if candidate.exists():
            pretrained_config_path = candidate
            
    if pretrained_config_path:
        logger.info(f"Auto-detecting pre-trained configurations from {pretrained_config_path}")
        try:
            with open(pretrained_config_path, "r") as f:
                pt_cfg = json.load(f)
            
            # Dynamically override default config values based on pre-trained model config
            if "n_embd" in pt_cfg:
                config.context_dim = pt_cfg["n_embd"]
                logger.info(f"  Overriding context_dim = {config.context_dim} (from n_embd)")
            if "spectra_norm_type" in pt_cfg:
                config.spectra_norm_type = pt_cfg["spectra_norm_type"]
                logger.info(f"  Overriding spectra_norm_type = {config.spectra_norm_type}")
            if "spectra_norm_scaler" in pt_cfg:
                config.spectra_norm_scaler = pt_cfg["spectra_norm_scaler"]
                logger.info(f"  Overriding spectra_norm_scaler = {config.spectra_norm_scaler}")
            if "spectra_norm_const" in pt_cfg:
                config.spectra_norm_const = pt_cfg["spectra_norm_const"]
                logger.info(f"  Overriding spectra_norm_const = {config.spectra_norm_const}")
            if "spectra_size" in pt_cfg:
                config.spectrum_length = pt_cfg["spectra_size"]
                logger.info(f"  Overriding spectrum_length = {config.spectrum_length} (from spectra_size)")
            elif "DESISpectrum_size" in pt_cfg:
                config.spectrum_length = pt_cfg["DESISpectrum_size"]
                logger.info(f"  Overriding spectrum_length = {config.spectrum_length} (from DESISpectrum_size)")
        except Exception as e:
            logger.warning(f"Failed to parse pre-trained config: {e}")

    save_config_json(config, weights_dir)

    # Parse channel dimensions from comma-separated string
    channel_dims = tuple(int(x.strip()) for x in config.channel_dims.split(","))

    # Build DiffusionModelConfig
    model_config = DiffusionModelConfig(
        spectrum_length=config.spectrum_length,
        spectrum_channels=1,
        context_dim=config.context_dim,
        time_emb_dim=config.time_emb_dim,
        channel_dims=channel_dims,
        num_groups=config.num_groups,
        num_timesteps=config.num_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        use_mid_attention=config.use_mid_attention,
    )

    logger.info(f"Starting Spectrum Diffusion Training")
    logger.info(f"  Embeddings: {config.embeddings_path}")
    logger.info(f"  Data:       {config.data_dir}")
    logger.info(f"  Output:     {config.train_dir}")
    logger.info(f"  Context dim: {config.context_dim}")
    logger.info(f"  Channels:   {channel_dims}")
    logger.info(f"  Timesteps:  {config.num_timesteps}")
    logger.info(f"  Mid-attn:   {config.use_mid_attention}")

    # --- Data Loading ---
    # Configure Arrow dataset for spectra-only access
    registry = ModalityRegistry([
        ModalityConfig(
            name="spectra",
            input_size=1,
            patch_size=1,
            pos_input_size=1,
            embed_pos=False,
            encoder_type="discrete",
        )
    ])

    tf_kwargs = {
        "norm_type_spec": config.spectra_norm_type,
        "norm_scaler_spec": config.spectra_norm_scaler,
        "norm_const_spec": config.spectra_norm_const,
    }
    val_tf = EuclidDESIDatasetArrow.data_transforms(stage="val", **tf_kwargs)

    logger.info("Loading Arrow datasets...")
    
    # First attempt: load train split
    base_ds_train = EuclidDESIDatasetArrow(
        arrow_folder_root=config.data_dir,
        split="train",
        modality_registry=registry,
        spiral=False,
        transform=val_tf,  # No augmentation for spectra in diffusion training
    )
    
    # Check overlap with train split
    npz = np.load(config.embeddings_path)
    id_key = "ids" if "ids" in npz else "targetid"
    emb_ids = npz[id_key].astype(np.int64)
    
    arrow_train_ids = set(base_ds_train.ds['targetid'])
    overlap_count = len(set(emb_ids).intersection(arrow_train_ids))
    
    if overlap_count == 0:
        logger.warning("No matching IDs found in 'train' split. Sourcing from 'test' split instead...")
        base_ds_train = EuclidDESIDatasetArrow(
            arrow_folder_root=config.data_dir,
            split="test",
            modality_registry=registry,
            spiral=False,
            transform=val_tf,
        )
        base_ds_val = base_ds_train  # Symmetrical split
        logger.info("Successfully switched to 'test' split with matching IDs!")
    else:
        logger.info(f"Matched {overlap_count} targets in 'train' split!")
        base_ds_val = EuclidDESIDatasetArrow(
            arrow_folder_root=config.data_dir,
            split="test",
            modality_registry=registry,
            spiral=False,
            transform=val_tf,
        )

    logger.info("Building paired datasets (embedding ↔ spectrum)...")

    # --- StandardScaler: compute μ and σ from training spectra (post-asinh) ---
    scaler_mean, scaler_std = 0.0, 1.0

    if config.use_standard_scaling:
        # Build a temporary dataset WITHOUT scaling to compute raw statistics
        temp_dataset = PairedEmbeddingSpectraDataset(
            embeddings_path=config.embeddings_path,
            embeddings_key=config.embeddings_key,
            arrow_dataset=base_ds_train,
            spectrum_length=config.spectrum_length,
            use_standard_scaling=False,  # Raw asinh values
        )
        logger.info(
            f"Computing StandardScaler statistics from {config.scaler_num_samples} "
            f"training samples (post-asinh, pre-scaling)..."
        )
        scaler_mean, scaler_std = compute_spectra_statistics(
            temp_dataset,
            num_samples=config.scaler_num_samples,
            logger=logger,
        )
        del temp_dataset  # Free memory

    train_dataset = PairedEmbeddingSpectraDataset(
        embeddings_path=config.embeddings_path,
        embeddings_key=config.embeddings_key,
        arrow_dataset=base_ds_train,
        spectrum_length=config.spectrum_length,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        use_standard_scaling=config.use_standard_scaling,
    )

    val_dataset = PairedEmbeddingSpectraDataset(
        embeddings_path=config.embeddings_path,
        embeddings_key=config.embeddings_key,
        arrow_dataset=base_ds_val,
        spectrum_length=config.spectrum_length,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        use_standard_scaling=config.use_standard_scaling,
    )

    logger.info(f"Train pairs: {len(train_dataset)} | Val pairs: {len(val_dataset)}")
    if config.use_standard_scaling:
        logger.info(f"StandardScaler active: μ={scaler_mean:.6f}, σ={scaler_std:.6f}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    # --- Model & Optimizer ---
    model = SpectrumDiffusionModel(model_config).to(device)
    total_params = model.get_num_params()
    logger.info(f"Model parameters: {total_params / 1e6:.2f}M")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
    )

    # Mixed precision context
    if device.type == "cuda":
        ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
    else:
        ctx = contextlib.nullcontext()

    # --- Checkpoint Resume ---
    iter_num = 0
    best_val_loss = float("inf")
    patience_counter = 0
    accumulated_time = 0.0

    if config.init_from == "resume":
        ckpt_path = weights_dir / "ckpt_last.pt"
        if ckpt_path.exists():
            logger.info(f"Resuming from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            iter_num = checkpoint["iter_num"]
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            patience_counter = checkpoint.get("patience_counter", 0)
            accumulated_time = checkpoint.get("total_run_time", 0.0)

            # Restore scaler statistics from checkpoint (overrides fresh computation)
            if "scaler_mean" in checkpoint:
                scaler_mean = checkpoint["scaler_mean"]
                scaler_std = checkpoint["scaler_std"]
                # Update datasets with restored scaler
                train_dataset.scaler_mean = scaler_mean
                train_dataset.scaler_std = scaler_std
                val_dataset.scaler_mean = scaler_mean
                val_dataset.scaler_std = scaler_std
                logger.info(f"Restored scaler from checkpoint: μ={scaler_mean:.6f}, σ={scaler_std:.6f}")

            logger.info(
                f"Resumed at iter {iter_num} (Best val loss: {best_val_loss:.6f})"
            )
        else:
            logger.warning(f"No checkpoint at {ckpt_path}. Starting from scratch.")

    # --- CSV Logging ---
    csv_path = logs_dir / "training_metrics.csv"
    if config.init_from == "scratch" or not csv_path.exists():
        with open(csv_path, "w") as f:
            f.write("iter,progress,train_loss,val_loss,lr,dt_ms,mem_gb,eta_hms\n")

    # --- Time Limit ---
    max_run_seconds = None
    if config.max_run_hours is not None:
        try:
            max_run_seconds = parse_time_to_seconds(config.max_run_hours)
            logger.info(f"Time limit: {config.max_run_hours} ({max_run_seconds}s)")
        except Exception as e:
            logger.error(f"Error parsing max_run_hours: {e}")
            sys.exit(1)

    # --- Training Loop ---
    logger.info(
        f"Training SpectrumDiffusionModel for {config.max_iters} iterations..."
    )

    train_iter = iter(train_loader)
    run_start_time = time.time()
    last_log_time = time.time()
    model.train()
    stop_training = False

    while iter_num <= config.max_iters:
        # Dynamic LR
        lr = get_learning_rate(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get batch
        try:
            embedding, spectrum, redshift = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            embedding, spectrum, redshift = next(train_iter)

        embedding = embedding.to(device)
        spectrum = spectrum.to(device)
        redshift = redshift.to(device)

        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            loss = model(spectrum, embedding, redshift)

        loss.backward()

        # Gradient clipping
        if config.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()

        # --- Logging ---
        if iter_num % config.log_interval == 0:
            current_time = time.time()
            dt = current_time - last_log_time
            last_log_time = current_time
            avg_dt = dt / config.log_interval if iter_num > 0 else 0

            mem_usage = (
                torch.cuda.max_memory_allocated() / (1024**3)
                if device.type == "cuda"
                else 0.0
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            progress = iter_num / config.max_iters
            remaining_iters = config.max_iters - iter_num
            eta_seconds = remaining_iters * avg_dt
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

            logger.info(
                f"Iter {iter_num}/{config.max_iters} ({progress:.2%}) | "
                f"Loss {loss.item():.6f} | LR {lr:.4e} | "
                f"Mem {mem_usage:.2f}GB | dt {avg_dt*1000:.1f}ms | ETA {eta_str}"
            )

            with open(csv_path, "a") as f:
                f.write(
                    f"{iter_num},{progress:.4f},{loss.item():.6f},,{lr:.4e},"
                    f"{avg_dt*1000:.1f},{mem_usage:.2f},{eta_str}\n"
                )

        # --- Evaluation & Checkpoint ---
        if iter_num > 0 and iter_num % config.eval_interval == 0:
            val_loss = estimate_loss(model, val_loader, config, device, ctx)
            logger.info(f"Validation Loss: {val_loss:.6f}")

            with open(csv_path, "a") as f:
                f.write(
                    f"{iter_num},{iter_num / config.max_iters:.4f},,{val_loss:.6f},,,, \n"
                )

            ckpt_data = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": asdict(model_config),
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "total_run_time": accumulated_time + (time.time() - run_start_time),
                "scaler_mean": scaler_mean,
                "scaler_std": scaler_std,
            }

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(ckpt_data, weights_dir / "ckpt_best.pt")
                logger.info(f"New best model saved! (Loss: {best_val_loss:.6f})")
            else:
                if iter_num >= config.early_stopping_min_iters:
                    patience_counter += 1
                    logger.info(
                        f"No improvement. Patience: {patience_counter}/{config.early_stop_patience}"
                    )
                    if patience_counter >= config.early_stop_patience:
                        logger.warning(
                            f"Early stopping at iter {iter_num}."
                        )
                        stop_training = True
                else:
                    patience_counter = 0
                    logger.info(
                        f"No improvement, but current iter {iter_num} < early_stopping_min_iters {config.early_stopping_min_iters}. Skipping patience increment."
                    )

        # --- Routine Checkpoint ---
        if iter_num > 0 and iter_num % config.checkpoint_interval == 0:
            ckpt_data = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": asdict(model_config),
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "total_run_time": accumulated_time + (time.time() - run_start_time),
                "scaler_mean": scaler_mean,
                "scaler_std": scaler_std,
            }
            torch.save(ckpt_data, weights_dir / "ckpt_last.pt")
            logger.info(f"Routine checkpoint saved at iter {iter_num}.")

        # --- Time Limit ---
        if max_run_seconds is not None:
            elapsed = time.time() - run_start_time
            if elapsed > max_run_seconds:
                logger.warning(
                    f"Time limit reached ({config.max_run_hours}). Saving and exiting."
                )
                ckpt_data = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_config": asdict(model_config),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "patience_counter": patience_counter,
                    "total_run_time": accumulated_time + elapsed,
                    "scaler_mean": scaler_mean,
                    "scaler_std": scaler_std,
                }
                torch.save(ckpt_data, weights_dir / "ckpt_last.pt")
                stop_training = True

        if stop_training:
            break

        iter_num += 1

    # --- Post-Training Sampling ---
    logger.info("Training complete. Generating sample spectra...")

    best_ckpt = weights_dir / "ckpt_best.pt"
    if best_ckpt.exists():
        checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        logger.info("Loaded best model weights for sampling.")

    model.eval()

    # Take the first N embeddings from the validation set for sampling
    num_samples = min(config.num_samples_at_end, len(val_dataset))
    sample_embeddings = []
    sample_real_spectra = []
    sample_redshifts = []
    for i in range(num_samples):
        emb, spec, z = val_dataset[i]
        sample_embeddings.append(emb)
        sample_real_spectra.append(spec)
        sample_redshifts.append(z)

    sample_embeddings = torch.stack(sample_embeddings).to(device)
    sample_real_spectra = torch.stack(sample_real_spectra).cpu().numpy()
    sample_redshifts = torch.stack(sample_redshifts).to(device)

    logger.info(f"Generating {num_samples} spectra via {config.num_timesteps}-step reverse process...")
    with ctx:
        generated_spectra = model.sample(sample_embeddings, sample_redshifts, device)

    generated_spectra = generated_spectra.cpu().numpy()

    # Inverse StandardScaler: x_original = x_scaled * σ + μ
    # This converts generated spectra back to asinh-normalised space for
    # downstream scientific comparison (e.g. vs real asinh-normalised spectra).
    if config.use_standard_scaling:
        generated_spectra = generated_spectra * scaler_std + scaler_mean
        logger.info(f"Inverse StandardScaler applied: μ={scaler_mean:.6f}, σ={scaler_std:.6f}")

    # Save generated and real spectra for comparison
    save_path = samples_dir / "generated_spectra.npz"
    np.savez(
        save_path,
        generated=generated_spectra,
        real=sample_real_spectra,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
    )
    logger.info(f"Sample spectra saved to {save_path}")
    logger.info("Diffusion training pipeline complete.")


if __name__ == "__main__":
    main()
