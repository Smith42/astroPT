"""
AstroPT Multimodal Training Scripts.

This script implements a Distributed Data Parallel (DDP) training loop for the 
AstroPT model, utilising the Euclid-DESI multimodal dataset (Arrow format).

Author: Victor Alonso Rodriguez
Date: March 2026
"""

from __future__ import annotations

import argparse
import datetime
import inspect
import json
import logging
import math
import numpy as np
import os
import subprocess
import sys
import time
import re
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from pathlib import Path
from transformers import HfArgumentParser
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.distributed as dist
from torch.distributed import (
    init_process_group, 
    destroy_process_group
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    from codecarbon import EmissionsTracker
    _CODECARBON_AVAILABLE = True
except ImportError:
    _CODECARBON_AVAILABLE = False


from astropt.model import GPT, GPTConfig, ModalityConfig, ModalityRegistry
from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow


@dataclass
class TrainingConfig:
    """
    Hyperparameters and configuration settings for AstroPT training.
    
    This dataclass acts as the single source of truth for the experiment.
    Arguments can be overridden via command line (CLI).
    
    It is saved by save_config_json() into a .json file.
    """
    
    #--- Training Metadata ---#
    train_name: Optional[str] = None            # Name of the training
    train_date: Optional[str] = None            # Date of the training
    train_description: Optional[str] = None     # Description or comment abouth the training

    #--- I/O & Paths ---#
    train_dir: Optional[str] = None               # Training output directory (built dynamically at the end of the class)
    data_dir: str = "/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"   # Dataset directory
            
    #--- Data Loading ---#
    batch_size: int = 16            # Micro-batch size per GPU (what fits in VRAM)
    num_workers: int = 8            # Optimized for Teide HPC (CPU cores for loading)
    persistent_workers: bool = True # Keep workers active
    prefetch_factor: int = 2        # How many batches to preload per worker
    pin_memory: bool = True         # Faster transfer RAM -> VRAM
    spiral: bool = True             # Whether to apply spiral readout to galaxy images
    init_from: str = "scratch"      # Training from scratch or resume training
    
    #--- Model Architecture for the 100M Parameter Setup ---#
    n_layer: int = 8                # Depth of the Transformer
    n_head: int = 8                 # Number of attention heads
    n_embd: int = 512               # Embedding dimension (width of the network)
    n_chan: int = 4                 # Input channels: 1 VIS + 3 NISP (Y, J, H)
    block_size: int = 1024          # Context length (max tokens per sample)
    dropout: float = 0.0            # Regularization (0.0 for pretraining is standard)
    bias: bool = False              # Learnable bias in Linear layers (False is modern/faster)
    attn_type: str = "causal"       # Attention mechanism type
    backbone: str = "native"        # Model bakcbone: native or llm
    use_qlora: bool = False         # Use Quantized Low-Rank Adaptation
    loss_type: str = "mae"          # Options: l1 / mae, mse, huber
    loss_huber_delta: float = 1.0   # Delta value for controlling Huber Loss Behaviour (default 1.0)
    use_aug: bool = True            # Active data augmentation by using image rotation
    use_pretokenized: bool = True   # Use pre-computed tokens from Arrow files (bypasses AION on-the-fly)
    
    #--- Multimodality Mixing Parameters ---#
    use_token_mixing: bool = True               # Enable cross-modal interleaving
    token_mixing_block_size: int = 128          # Interleaving block size
    token_mixing_stochastic: bool = False       # Enable stochastic block sizes
    token_mixing_min_block_size: int = 64       # Minimum block size for stochastic mixing
    token_mixing_max_block_size: int = 128      # Maximum block size for stochastic mixing
    token_mixing_seed: int = 61                 # Seed for reproducible stochastic mixing
    shuffle_modality_train: bool = True         # Shuffle modality order during training
    shuffle_modality_val: bool = False          # Shuffle modality order during validation
    modality_dropout_prob: float = 0.1          # Probability to zero one modality in a micro-step
    modality_dropout_mode: str = "spectra"      # none, images, spectra, random
    cross_reconstruction_loss_use: bool = True  # Enable cross reconstruction explicitly via targets
    cross_reconstruction_weight: float = 1.2    # Weight multiplier for cross-reconstructed modal loss
    
    # Dual Modality Identity
    use_cls_token: bool = True                  # Use a [CLS] token that is prepended to the sequence
    use_modality_embeddings: bool = True        # Use a different vector to distinguish modalities

    # Images
    images_train: bool = True           # Images bool flag for enabling training
    images_size: int = 224              # Images side size in pixels
    images_patch_size: int = 8          # Side size in pixels of each patch in an image
    images_channels: int = 4            # Channels per image (VIS + NISP Y,J,H)
    images_loss_weight: float = 2.0     # Images importance for training
    images_embed_pos: bool = True       # Images embedding positions learning
    images_pos_input_size: int = 1      # Images position input size
    images_norm_type: str = "asinh"     # Normalization method: constant, z_score or asinh
    images_norm_scaler: float = 1.0     # Scaler factor if normalization requieres it (default 1.0)
    images_norm_const: float = 1.0      # Normalization global constant for images: P99=7.603847
    images_mask: bool = True            # Enable tactical masking for image patches
    images_mask_prob: float = 0.20      # Probability to mask each image patch
    
    ## Images Tokenization Method
    images_tokenizer_method: str = "discrete"  # "discrete" (AION) or "aim" or "affine" (both continuous)
    images_tokeniser_discrete_vocab_size: int = 64000 # AION discrete tokeniser vocabulary size
    images_unet_weights_path: str = "logs/unet_adapter_weights/adapters_final.pt"  # Path to U-Net adapter weights
    images_aion_image_size: int = 112            # Target size for AION ImageCodec (controls token sequence length)
    images_aion_image_transform: str = "resize"  # "crop" or "resize" transform before U-Net
    
    # Spectra
    spectra_train: bool = True              # Spectra bool flag for enabling training
    spectra_inverse: bool = False           # Reading spectra from red to blue 
    spectra_size: int = 7781                # Spectra total size
    spectra_patch_size: int = 10            # Patch size for each spectrum
    spectra_loss_weight: float = 1.0        # Spectra importance for training 
    spectra_embed_pos: bool = True          # Spectra embedding positions learning
    spectra_pos_input_size: int = 1         # Spectra position input size
    spectra_norm_type: str = "asinh"        # Normalization method: constant, z_score or asinh
    spectra_norm_scaler: float = 1.0        # Scaler factor if normalization requieres it (default 1.0)
    spectra_norm_const: float = 1.0         # Normalization global constant for spectra: P99=7.956048
    spectra_mask: bool = True               # Enable tactical masking for spectra patches
    spectra_mask_prob: float = 0.20         # Probability to mask each spectrum patch
    spectra_tokenizer_method: str = "discrete"  # "discrete" (AION) or "aim" or "affine" (both continuous)
    spectra_tokeniser_discrete_vocab_size: int = 1024 # LFQ codebook_size
    
    #--- Optimization of the Learning Process ---#
    max_iters: int = 75_000         # Total training iters (NOT epochs)
    weight_decay: float = 1e-1      # Regularization to prevent overfitting
    beta1: float = 0.9              # AdamW parameter
    beta2: float = 0.95             # AdamW parameter
    grad_clip: float = 1.0          # Stabilizes training if gradients explode
    
    # Gradient Accumulation: Simulates a larger batch size. 
    # Effective batch = 16 * 40 = 640
    gradient_accumulation_steps: int = 40 
    
    #--- Learning Rate Scheduler ---#
    learning_rate: float = 3e-4     # Learning rate per weight update
    lr_min: float = 3e-5            # Minimum LR (usually 10% of max)
    lr_mult_images: float = 1.0    # LR multiplier for image encoder/decoder modality
    lr_mult_spectra: float = 1.0    # LR multiplier for spectra encoder/decoder modality
    lr_mult_backbone: float = 1.0   # LR multiplier for transformer/shared modality
    lr_decay: bool = True           # Activates the variable learning rate decay
    lr_warmup_iters: int = 4_000    # Steps to ramp up LR from 0 to max
    lr_decay_iters: int = 65_000    # Steps to decay LR down to min

    #--- Logging & Checkpointing ---#
    eval_interval: int = 1_000              # How often to validate
    eval_batches: int = 100                 # How many batches to use for validation
    log_interval: int = 200                 # How often to print to console/WandB
    checkpoint_interval: int = 1_000        # How often to save .pt files
    checkpoint_save_type: str = "both"      # Checkpoint saving mode: best, last, both or all
    early_stopping_patience: int = 15       # Stop if no improvement after N evals
    early_stopping_min_iters: int = 35000   # Minimum iterations before starting to count patience

    #--- System & Backend ---#
    device: str = "cuda"                    # CPU/GPU device interface: cpu, cuda or mps
    dtype: str = "bfloat16"                 # 'bfloat16' is best for A100 GPUs
    compile: bool = True                    # PyTorch 2.0 compiler
    compile_mode: str = "default"           # Compilation mode
    backend: str = "nccl"                   # Communication backend for DDP
    max_run_hours: Optional[str] = None     # Force stop after "HH:MM:SS"

    #--- External Monitoring ---#
    log_via_wandb: bool = False             # Weight and bias (wandb) logging
    wandb_project: str = "AstroPT-Arrow"    # wandb project name
    wandb_run_name: Optional[str] = None    # Training name
    log_emissions: bool = False             # CodeCarbon logging
    profile: bool = False                   # Enable PyTorch Profiler (Trace analysis)

    #--- Optional Diagnostics & Ablation Flags ---#
    diagnostics_enabled: bool = True        # Enable modality diagnostics CSV/WandB logs
    diagnostics_interval: int = 200         # Iter interval for diagnostics writes/logs
    diagnostics_track_losses: bool = True   # Track per-modality losses
    diagnostics_track_grads: bool = True    # Track per-branch gradient norms
    diagnostics_file_name: str = "training_diagnostics.csv"
    
    # Dynamic output directory with date
    def __post_init__(self):
        # Date
        if self.train_date is None:
            self.train_date = datetime.datetime.now().strftime("%Y%m%d")
        # Output directory
        if self.train_dir is None:
            clean_name = self.train_name.lower() if self.train_name else "default_run"
            clean_name = re.sub(r'[^a-z0-9]', ' ', clean_name)
            tokens = clean_name.split()
            suffix_name = "_".join(tokens)
            self.train_dir = f"./logs/astropt_100M_250K_arrow_{self.train_date}_{suffix_name}"

def get_git_commit_hash() -> str:
    """Returns the current git commit hash or 'unknown' if not in a git repo."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"
    

def get_dataset_info(data_dir: str | Path) -> dict:
    """
    Scans the data directory to create an informative version based on 
    filesystem metadata: latest modification time and total size.
    """
    try:
        max_mtime = 0.0
        total_size = 0
        file_count = 0
        data_path = Path(data_dir)

        # Walk through the directory
        for file_path in data_path.rglob('*'):
            
            # Filter arrow and json files
            if file_path.is_file() and file_path.suffix in ['.arrow', '.json']:
                    
                stats = file_path.stat()
                
                # Update max modification time
                if stats.st_mtime > max_mtime:
                    max_mtime = stats.st_mtime
                
                # Accumulate size
                total_size += stats.st_size
                file_count += 1

        # Convert timestamp
        if max_mtime > 0:
            last_modified_str = datetime.datetime.fromtimestamp(max_mtime).strftime('%Y-%m-%d %H:%M:%S')
        else:
            last_modified_str = "Unknown"

        # Returning a dictionary for adding it to the configuration file
        return {
            "data_last_modified": last_modified_str,
            "data_total_size_mb": round(total_size / (1024 * 1024), 2), # Size in MB
            "data_file_count": file_count
        }
        
    except Exception as e:
        return {"data_version_error": str(e)}
    

def project_directories_setup(base_dir: str | Path) -> Tuple[Path, Path, Path, Path, Path]:

    train_dir = Path(base_dir)
    weights_dir = train_dir / "weights"
    embeddings_dir = train_dir / "embeddings"
    plots_dir = train_dir / "plots"
    logs_dir = train_dir / "logs"
    
    for directory in [train_dir, weights_dir, embeddings_dir, plots_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        f"[INFO]: Created {directory}"

    return train_dir, weights_dir, embeddings_dir, plots_dir, logs_dir

def parse_time_to_seconds(time_str: str) -> float:
    """Converts a HH:MM:SS string to total seconds."""
    try:
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except ValueError:
        raise ValueError(f"Invalid time format: {time_str}. Expected HH:MM:SS")


def validate_runtime_flags(config: TrainingConfig) -> None:
    """Validates optional runtime flags introduced for diagnostics/ablations."""
    allowed_dropout_modes = {"none", "images", "spectra", "random"}

    if not (0.0 <= config.modality_dropout_prob <= 1.0):
        raise ValueError("modality_dropout_prob must be in [0, 1].")
    if config.modality_dropout_mode.lower() not in allowed_dropout_modes:
        raise ValueError(
            f"modality_dropout_mode must be one of: {sorted(allowed_dropout_modes)}"
        )
    if config.diagnostics_interval < 1:
        raise ValueError("diagnostics_interval must be >= 1.")

    for field_name in ("lr_mult_images", "lr_mult_spectra", "lr_mult_backbone"):
        value = getattr(config, field_name)
        if value <= 0.0:
            raise ValueError(f"{field_name} must be > 0.")


def _normalize_param_name(param_name: str) -> str:
    """Normalizes wrapped parameter names from DDP/compile wrappers."""
    clean_name = param_name
    for prefix in ("module._orig_mod.", "module.", "_orig_mod."):
        clean_name = clean_name.replace(prefix, "")
    return clean_name


def _param_branch_name(param_name: str) -> str:
    """Maps a parameter name to an optimizer/diagnostics branch."""
    name = _normalize_param_name(param_name)
    if any(token in name for token in ("encoders.images", "decoders.images", "embedders.images",
                                        "encoders.aion_images", "decoders.aion_images", "embedders.aion_images")):
        return "images"
    if any(token in name for token in ("encoders.spectra", "decoders.spectra", "embedders.spectra",
                                        "encoders.aion_spectra", "decoders.aion_spectra", "embedders.aion_spectra")):
        return "spectra"
    return "backbone"


def _compute_modality_losses(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    config: TrainingConfig,
) -> Dict[str, float]:
    """Computes unweighted loss per modality for diagnostics only.
    
    Note: When token_mixing is active, the model internally shifts and
    re-indexes targets to align with its outputs, but torch.compile
    prevents in-place mutations from propagating back to the caller.
    We handle this by truncating targets to match output shapes.
    """
    result: Dict[str, float] = {}

    for mod_name, pred in outputs.items():
        if mod_name not in targets:
            continue
        target = targets[mod_name]

        # Auto-align shapes when token_mixing causes length differences
        if pred.shape[1] != target.shape[1]:
            min_len = min(pred.shape[1], target.shape[1])
            pred = pred[:, :min_len]
            target = target[:, :min_len]

        # Use cross-entropy for discrete (AION) modalities
        if "aion" in mod_name:
            mod_loss = torch.nn.functional.cross_entropy(
                pred.reshape(-1, pred.size(-1)),
                target.reshape(-1),
            )
        elif config.loss_type in ["l1", "mae"]:
            mod_loss = torch.nn.functional.l1_loss(pred, target)
        elif config.loss_type == "mse":
            mod_loss = torch.nn.functional.mse_loss(pred, target)
        else:
            mod_loss = torch.nn.functional.huber_loss(
                pred,
                target,
                delta=config.loss_huber_delta,
            )
        result[mod_name] = float(mod_loss.detach().item())

    return result


def maybe_apply_modality_dropout(
    batch: Dict[str, Dict[str, torch.Tensor]],
    config: TrainingConfig,
) -> str:
    """Optionally zeros a modality as a robustness stress-test; returns dropped modality or 'none'."""
    drop_prob = float(config.modality_dropout_prob)
    if drop_prob <= 0.0:
        return "none"
    if float(np.random.random()) >= drop_prob:
        return "none"

    available = [m for m in batch["X"] if m in batch["Y"] and not m.endswith("_positions")]
    # Filter to actual modalities (exclude position keys)
    available = [m for m in available if m in ("images", "spectra", "aion_images", "aion_spectra")]
    if len(available) < 1:
        return "none"

    mode = config.modality_dropout_mode.lower().strip()
    if mode == "none":
        return "none"
    if mode == "random":
        chosen = str(np.random.choice(available))
    else:
        chosen = mode if mode in available else "none"

    if chosen == "none":
        return "none"

    batch["X"][chosen] = torch.zeros_like(batch["X"][chosen])
    
    # If Cross-Reconstruction Loss is active, we explicitly WANT to keep the target targets 
    # intact to force the model to predict the missing modality entirely from the other one.
    if getattr(config, 'cross_reconstruction_loss_use', False) == False:
        batch["Y"][chosen] = torch.zeros_like(batch["Y"][chosen])
        
    return chosen


def summarize_modality_dropout(drop_counts: Dict[str, int]) -> str:
    """Compacts micro-step dropout events into one per-iteration label."""
    drops = [k for k, v in drop_counts.items() if v > 0 and k != "none"]
    if len(drops) > 1:
        return "mixed"
    if len(drops) == 1:
        return drops[0]
    return "none"


def compute_branch_grad_norms(model: torch.nn.Module) -> Dict[str, float]:
    """Computes gradient L2 norms for image/spectra/backbone branches."""
    grad_sq = {"images": 0.0, "spectra": 0.0, "backbone": 0.0}

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        branch = _param_branch_name(name)
        grad = param.grad.detach()
        grad_sq[branch] += float(torch.sum(grad * grad).item())

    norms = {k: math.sqrt(v) for k, v in grad_sq.items()}
    norms["total"] = math.sqrt(sum(grad_sq.values()))
    return norms
    
    
def save_config_json(config: TrainingConfig, rank: int, save_dir: Path):
    """Saves the configuration to a JSON file for reproducibility."""
    if rank == 0:
        config_dict = asdict(config)
        
        # Adding Git hash
        config_dict["git_hash"] = get_git_commit_hash()
        
        # Adding dataset information
        data_stats = get_dataset_info(config.data_dir)
        config_dict.update(data_stats)
        
        config_path = save_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
            
def smart_config_merge(current_config: Any, saved_config_dict: Dict[str, Any], logger: logging.Logger) -> Any:
    """
    Merges a saved configuration dictionary into the current configuration object,
    respecting command-line interface (CLI) overrides.

    Args:
        current_config (Any): The current TrainingConfig dataclass instance 
                              initialized by the script/CLI.
        saved_config_dict (Dict[str, Any]): The configuration dictionary loaded 
                                            from the checkpoint file.
        logger (logging.Logger): Logger instance for tracking changes.

    Returns:
        Any: The updated configuration object with merged parameters.
    """
    
    # Get raw CLI arguments
    raw_argv = sys.argv[1:]
    
    # Iterate through the saved configuration keys
    for key, old_value in saved_config_dict.items():
        
        # Never restore the following values 
        if key in ['train_dir', 'init_from', 'run_name', 'train_date', 'wandb_run_name']:
            continue
            
        # If the saved key no longer exists in the current code, skip it
        if not hasattr(current_config, key):
            continue
            
        # CLI Detection
        kebab_key = key.replace('_', '-')
        candidate_flags = [f"--{key}", f"--{kebab_key}"]
        
        # Searching for --no-{key} for Flase booleans
        current_value = getattr(current_config, key)
        if isinstance(current_value, bool) or isinstance(old_value, bool):
            candidate_flags.append(f"--no-{key}")
            candidate_flags.append(f"--no-{kebab_key}")
            
        # Check the match
        user_override = False
        for flag in candidate_flags:
            # Check for exact match OR 'flag=' match
            if any(arg == flag or arg.startswith(f"{flag}=") for arg in raw_argv):
                user_override = True
                break    
        
        # Taking arguments logic
        if user_override:
            # User explicitly provided the argument
            if current_value != old_value:
                logger.info(f" --> MANUAL OVERRIDE: '{key}' | Old: {old_value} -> New: {current_value} (Set by CLI)")
        else:
            # Not provided. Using the saved in the checkpoint
            if current_value != old_value:
                setattr(current_config, key, old_value)
                logger.info(f" --> RESTORED: '{key}' | Script: {current_value} --> Saved: {old_value}")
    
    return current_config

def ddp_setup() -> Tuple[bool, int, int, str]:
    """
    Detects and initializes the Distributed Data Parallel (DDP) environment.
    
    If the script is launched via 'torchrun' (or srun in Slurm), specific environment
    variables (RANK, WORLD_SIZE) will be present. This function configures the
    process group and GPU device accordingly.

    Returns:
        ddp (bool): True if running in distributed mode, False otherwise.
        ddp_rank (int): Global rank of the process (0 is the master).
        ddp_world_size (int): Total number of processes (GPUs) participating.
        device (str): The specific device identifier for this process (e.g., 'cuda:1').
    """
    # Check if 'RANK' is in environment variables to determine if we are in DDP mode
    ddp = int(os.environ.get("RANK", -1)) != -1

    if ddp:
        # DDP Mode (Multi-GPU)
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        
        # Pin this process to a specific GPU based on its local rank
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        
        # Log initialization only on the master process to avoid spam
        if ddp_rank == 0:
            print(f"[INFO]: DDP Initialized: Global Rank {ddp_rank}/{ddp_world_size} | "
                  f"Local Rank {ddp_local_rank} | Device {device}")
            
    else:
        # Single Device Mode
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        
        if torch.cuda.is_available():
            device = "cuda"
            print(f"[INFO]: Single GPU Mode: Using {device}")
        else:
            device = "cpu"
            print("[WARNING]: Running on CPU. This will be slow!")

    return ddp, ddp_rank, ddp_world_size, device


def logging_setup(
    config: TrainingConfig, 
    ddp_rank: int,
    save_dir: str | Path
) -> logging.Logger:
    """
    Configures the logging system.
    
    - Master Process (Rank 0): Logs to Console (stdout) AND a file (training.log).
    - Worker Processes (Rank > 0): Only log ERRORS to avoid cluttering the output.

    Args:
        config (TrainingConfig): To know where the 'train_dir' is.
        ddp_rank (int): To decide verbosity (Master vs Workers).

    Returns:
        logging.Logger: The configured logger object.
    """
    
    # Ensuring the output directory exists
    logs_dir = Path(save_dir)
    if ddp_rank == 0:
        logs_dir.mkdir(parents=True, exist_ok=True)

    # Defining the log format
    log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Creating the Logger
    logger = logging.getLogger("AstroPT")
    logger.setLevel(logging.INFO)
    
    # Avoiding duplicate logs in interactive modes
    if logger.hasHandlers():
        logger.handlers.clear()

    # Configuration for Master Process (Rank 0)
    if ddp_rank == 0:
        # Handler 1: Console (Stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        logger.addHandler(console_handler)
        
        # Handler 2: File (training.log)
        log_file = logs_dir / "training.log"
        file_handler = logging.FileHandler(log_file, mode="a") # 'a' for append
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        logger.addHandler(file_handler)
        
        logger.info(f"Logging initialized. Saving logs to: {log_file}")
        
    # Configuration for Workers (Rank > 0)
    else:
        # Workers stay silent unless something explodes (ERROR/CRITICAL)
        null_handler = logging.NullHandler()
        logger.addHandler(null_handler)
        logger.setLevel(logging.ERROR)

    return logger


def create_dataloaders(
    config: TrainingConfig, 
    ddp: bool
) -> Tuple[DataLoader, DataLoader, ModalityRegistry]:
    """
    Initializes Datasets and DataLoaders for the training pipeline.

    This function acts as a bridge, translating the flat 'TrainingConfig'
    hyperparameters into the structured 'ModalityRegistry' objects required by 
    both the Model and the Dataset.

    Args:
        config (TrainingConfig): The global configuration object containing paths and params.
        ddp (bool): Flag indicating if Distributed Data Parallelism is active.

    Returns:
        Tuple[DataLoader, DataLoader, ModalityRegistry]: A tuple containing:
            - train_loader: The DataLoader for the training set.
            - val_loader: The DataLoader for the validation/test set.
            - registry: The configured ModalityRegistry (needed to initialize the GPT model).
    """
    
    # Configure ModalityRegistry
    modalities = []
    tf_kwargs = {}
    
    # Define configuration for each modality
    if config.images_train:
        if config.images_tokenizer_method == "discrete":
            # Discrete (AION) tokenization: tokens are integer IDs
            image_modality_config = ModalityConfig(
                name="aion_images",
                input_size=1,                           # Not used (Embedder uses vocab_size)
                patch_size=1,                           # Each token is a single ID
                pos_input_size=config.images_pos_input_size,
                loss_weight=config.images_loss_weight,
                embed_pos=True,                         # Learnt position embeddings
                vocab_size=config.images_tokeniser_discrete_vocab_size, # FSQ levels [8,8,8,5,5,5]
                encoder_type="discrete",
            )
        else:
            # Continuous regression: patches of pixel values
            img_input_batch_size = config.images_patch_size * config.images_patch_size * config.images_channels
            image_modality_config = ModalityConfig(
                name="images",
                input_size=img_input_batch_size,
                patch_size=config.images_patch_size,
                pos_input_size=config.images_pos_input_size,  
                loss_weight=config.images_loss_weight,
                embed_pos=config.images_embed_pos,
                encoder_type=config.images_tokenizer_method, # "aim" or "affine"
            )
        modalities.append(image_modality_config)
        
        # Transforms only needed for continuous mode
        if config.images_tokenizer_method != "discrete":
            tf_kwargs.update({
                'norm_type_img': config.images_norm_type,
                'norm_scaler_img': config.images_norm_scaler,
                'norm_const_img': config.images_norm_const,
            })
        
    if config.spectra_train:
        if config.spectra_tokenizer_method == "discrete":
            # Discrete (AION) tokenization: tokens are integer IDs
            spectra_modality_config = ModalityConfig(
                name="aion_spectra",
                input_size=1,                           # Not used (Embedder uses vocab_size)
                patch_size=1,                           # Each token is a single ID
                pos_input_size=config.spectra_pos_input_size,
                loss_weight=config.spectra_loss_weight,
                embed_pos=True,                         # Learnt position embeddings
                vocab_size=config.spectra_tokeniser_discrete_vocab_size, # LFQ codebook_size
                encoder_type="discrete",
            )
        else:
            # Continuous regression: patches of spectral flux
            spectra_modality_config = ModalityConfig(
                name="spectra",
                input_size=config.spectra_patch_size,
                patch_size=config.spectra_patch_size,
                pos_input_size=config.spectra_pos_input_size, 
                loss_weight=config.spectra_loss_weight,
                embed_pos=config.spectra_embed_pos,
                encoder_type=config.spectra_tokenizer_method, # "aim" or "affine"
            )
        modalities.append(spectra_modality_config)
        
        # Transforms only needed for continuous mode
        if config.spectra_tokenizer_method != "discrete":
            tf_kwargs.update({
                'norm_type_spec': config.spectra_norm_type,
                'norm_scaler_spec': config.spectra_norm_scaler,
                'norm_const_spec': config.spectra_norm_const,
            })
    
    # Instantiate the Registry
    registry = ModalityRegistry(modalities)
    
    # Use data augmentation for training
    train_stage = 'train' if config.use_aug else 'val'
        
    # 4. Instantiate transforms dynamically unpacking the dictionary
    train_tf = EuclidDESIDatasetArrow.data_transforms(
        stage=train_stage, 
        **tf_kwargs
    )
    
    val_tf = EuclidDESIDatasetArrow.data_transforms(
        stage='val', 
        **tf_kwargs
    )
    
    # Activating the logger object
    logger = logging.getLogger("AstroPT")
    
    # Informational log (Only printed by the Master Process to avoid spam)
    if not ddp or (ddp and int(os.environ.get("RANK", 0)) == 0):
        logger.info(f"Loading data from: {config.data_dir}")

    # Instantiate Train Dataset 
    train_dataset = EuclidDESIDatasetArrow(
        arrow_folder_root=config.data_dir,
        split="train",
        modality_registry=registry, 
        spiral=config.spiral,
        stochastic=True,
        transform=train_tf,
        spectra_inverse=config.spectra_inverse,
        spectra_mask=config.spectra_mask,
        spectra_mask_prob=config.spectra_mask_prob,
        images_mask=config.images_mask,
        images_mask_prob=config.images_mask_prob,
        unet_weights_path=config.images_unet_weights_path,
        aion_image_size=config.images_aion_image_size,
        aion_image_transform=config.images_aion_image_transform,
        use_pretokenized=config.use_pretokenized,
    )
    
    # Instantiate Validation/Test Dataset 
    val_dataset = EuclidDESIDatasetArrow(
        arrow_folder_root=config.data_dir,
        split="test", 
        modality_registry=registry,
        spiral=config.spiral,
        stochastic=False,
        transform=val_tf,
        spectra_inverse=config.spectra_inverse,
        spectra_mask=config.spectra_mask,
        spectra_mask_prob=config.spectra_mask_prob,
        images_mask=config.images_mask,
        images_mask_prob=config.images_mask_prob,
        unet_weights_path=config.images_unet_weights_path,
        aion_image_size=config.images_aion_image_size,
        aion_image_transform=config.images_aion_image_transform,
        use_pretokenized=config.use_pretokenized,
    )

    # Configure DDP Samplers
    if ddp:
        # In DDP the sampler splits the data among GPUs
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    # Create Final DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
        drop_last=True
    )

    return train_loader, val_loader, registry

def create_model(
    config: TrainingConfig, 
    registry: ModalityRegistry, 
    device: torch.device, 
    ddp: bool
) -> torch.nn.Module:
    """
    Instantiates the GPT model, moves it to the target device, compiles it,
    and wraps it in DDP if distributed training is active.

    Args:
        config (TrainingConfig): Hyperparameters for the model architecture.
        registry (ModalityRegistry): Configuration of input modalities (img, spectra).
        device (torch.device): The GPU (or CPU) where the model will live.
        ddp (bool): Whether Distributed Data Parallelism is enabled.

    Returns:
        torch.nn.Module: The ready-to-train model.
    """
    
    # Activating the logger object
    logger = logging.getLogger("AstroPT")
    
    # Model configuration
    gpt_config = GPTConfig(
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        bias=config.bias,
        dropout=config.dropout,
        attn_type=config.attn_type,
        use_qlora=config.use_qlora,
        loss_type=config.loss_type,
        backbone=config.backbone,
        use_token_mixing=config.use_token_mixing,
        token_mixing_block_size=config.token_mixing_block_size,
        token_mixing_stochastic=config.token_mixing_stochastic,
        token_mixing_min_block_size=config.token_mixing_min_block_size,
        token_mixing_max_block_size=config.token_mixing_max_block_size,
        token_mixing_seed=config.token_mixing_seed,
        cross_reconstruction_loss_use=config.cross_reconstruction_loss_use,
        cross_reconstruction_weight=config.cross_reconstruction_weight,
        use_cls_token=config.use_cls_token,
        use_modality_embeddings=config.use_modality_embeddings,
    )
    
    # Instantiate the Model
    model = GPT(gpt_config, registry)
    logger.info(f"Initializing GPT Model with {config.n_layer} layers")

    # Move Model to Selected Device
    model.to(device)

    # Pythorch Compilation
    if config.compile:
        if ddp:
            # Work around a known Dynamo distributed partition bug where SymInt
            # outputs can be materialized as Python ints in AOTAutograd wrappers.
            torch._dynamo.config.optimize_ddp = False
            torch._dynamo.config.capture_scalar_outputs = True
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model, mode=config.compile_mode, dynamic=False)

    # DDP Wrapping
    if ddp:
        
        # Local RANK for each process
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Model configuration for DDP
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=False 
        )

    return model


def create_optimizer(
    model: torch.nn.Module, 
    config: TrainingConfig
) -> torch.optim.Optimizer:
    """
    Creates the AdamW optimizer with weight decay handling.

    It separates parameters into two groups:
    1. Decay Group: Weights of Linear and Embedding layers (2D tensors).
    2. No-Decay Group: Biases, LayerNorms, and 1D tensors.

    Args:
        model (torch.nn.Module): The loaded GPT model.
        config (TrainingConfig): Configuration containing lr, weight_decay, and betas.

    Returns:
        torch.optim.Optimizer: Configured AdamW optimizer.
    """
    
    # Activating the logger object
    logger = logging.getLogger("AstroPT")
    
    # Filter parameters that require gradients
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    # Keep legacy two-group optimizer layout unless branch multipliers are requested.
    use_branch_lr = any(
        abs(v - 1.0) > 1e-12
        for v in (config.lr_mult_images, config.lr_mult_spectra, config.lr_mult_backbone)
    )

    if not use_branch_lr:
        decay_params = []
        nodecay_params = []

        for _, p in param_dict.items():
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.info(
            f"Optimizer Config: {len(decay_params)} tensors ({num_decay_params:,} params) with decay."
        )
        logger.info(
            f"Optimizer Config: {len(nodecay_params)} tensors ({num_nodecay_params:,} params) without decay."
        )

        optim_groups = [
            {"params": decay_params, "weight_decay": config.weight_decay, "lr_scale": 1.0, "group_name": "decay"},
            {"params": nodecay_params, "weight_decay": 0.0, "lr_scale": 1.0, "group_name": "nodecay"},
        ]
    else:
        branch_lr = {
            "images": float(config.lr_mult_images),
            "spectra": float(config.lr_mult_spectra),
            "backbone": float(config.lr_mult_backbone),
        }

        grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for name, param in param_dict.items():
            decay_key = "decay" if param.dim() >= 2 else "nodecay"
            branch = _param_branch_name(name)
            key = (branch, decay_key)

            if key not in grouped:
                grouped[key] = {
                    "params": [],
                    "weight_decay": config.weight_decay if decay_key == "decay" else 0.0,
                    "lr_scale": branch_lr[branch],
                    "group_name": f"{branch}_{decay_key}",
                }
            grouped[key]["params"].append(param)

        optim_groups = list(grouped.values())
        logger.info("Optimizer Config: branch LR multipliers enabled.")
        for group in optim_groups:
            n_params = sum(p.numel() for p in group["params"])
            logger.info(
                f"  - {group['group_name']}: {len(group['params'])} tensors "
                f"({n_params:,} params), wd={group['weight_decay']}, lr_scale={group['lr_scale']}"
            )

    # Check for Fused AdamW (faster CUDA kernel)
    use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused and torch.cuda.is_available() else dict()
    
    if use_fused:
        logger.info("Using Fused AdamW implementation (faster).")

    # Instantiate AdamW
    optimizer = torch.optim.AdamW(
        optim_groups, 
        lr=config.learning_rate, 
        betas=(config.beta1,config.beta2), 
        **extra_args
    )

    return optimizer


def get_learning_rate(it: int, config: TrainingConfig) -> float:
    """
    Calculates the learning rate for the current iteration
    using a Cosine Decay schedule with Warmup.
    
    Args:
        it (int): The current training step number.
        config (TrainingConfig): Configuration containing lr, min_lr, warmup_iters, etc.
        
    Returns:
        float: The calculated learning rate for this specific step.
    """
    
    # Linear Warmup Phase
    if it < config.lr_warmup_iters:
        # Linear increase from 0 to learning_rate
        return config.learning_rate * (it + 1) / (config.lr_warmup_iters + 1)
    
    # Post-Decay Phase (Constant Min LR)
    if it > config.lr_decay_iters:
        return config.lr_min
    
    # Cosine Decay Phase
    decay_ratio = (it - config.lr_warmup_iters) / (config.lr_decay_iters - config.lr_warmup_iters)
    assert 0 <= decay_ratio <= 1
    
    # Calculate the cosine coefficient
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    # Apply the coefficient to the range [min_lr, learning_rate]
    return config.lr_min + coeff * (config.learning_rate - config.lr_min)

def main():
    """
    Main training entry point.
    
    Orchestrates the entire training pipeline:
    1. Sets up the DDP environment and logging.
    2. Loads the configuration and datasets.
    3. Instantiates the Model and Optimizer.
    4. Manages Checkpoint loading (resume vs scratch).
    5. Executes the training loop (implemented in Part 2).
    """

    #--- DDP & CONFIG SETUP ---#
    
    # Initialize Distributed Data Parallel (DDP) if applicable
    ddp, ddp_rank, ddp_world_size, device = ddp_setup()
    
    # Huggingface Argument Parser
    parser = HfArgumentParser((TrainingConfig,))
    
    # Load configuration from CLI arguments (overriding defaults)
    config, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    validate_runtime_flags(config)
    
    # Creating directories structure
    (train_dir, weights_dir, 
     embed_dir, plots_dir, logs_dir) = project_directories_setup(config.train_dir)
    
    # Setup Logger (only Master Process logs to file/console)
    logger = logging_setup(config, ddp_rank, logs_dir)
    
    # Saving configuration in a .json file
    save_config_json(config, ddp_rank, weights_dir)
    
    # Imporved file for workflow control
    improve_file_path = train_dir / ".improved"
    
    # Log basic training info
    if ddp_rank == 0:
        logger.info(f"Starting AstroPT training on {device} (DDP: {ddp})")
        
        # Logging training configuration
        config_dict = asdict(config)
        config_str = "\n".join([f"    {k}: {v}" for k, v in config_dict.items()])
        logger.info(f"Training config:\n{config_str}")

    # Changing configuration for DDP mode
    if ddp: 
        
        # Automatic gradient accumulation steps
        if config.gradient_accumulation_steps % ddp_world_size != 0:
            if ddp_rank == 0:
                logger.warning(f"Grad Accum {config.gradient_accumulation_steps} "
                               f"is not divisible by {ddp_world_size}. It will be rounded down.")
        
        # Original configuration        
        original_accum = config.gradient_accumulation_steps
        
        # New value
        config.gradient_accumulation_steps = config.gradient_accumulation_steps // ddp_world_size
        
        # Effective batch size
        eff_batch_size = config.batch_size * config.gradient_accumulation_steps * ddp_world_size
        
        
        # Automatic workers number
        try:
            total_available_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            total_available_cpus = os.cpu_count() or 1
        
        # Computing how many CPU can be assigned to each GPU
        cpus_per_gpu = total_available_cpus // ddp_world_size
        
        # Assigning the half of the total
        suggested_workers = int(cpus_per_gpu * 0.5)
        
        # Selecting the suggested worker number
        suggested_workers = max(2, min(4, suggested_workers))
        
        # Changing the configuration
        config.num_workers = suggested_workers
        
        # Only master shows
        if ddp_rank == 0:
            logger.info(f"DDP Detected: {ddp_world_size} GPUs. {total_available_cpus} CPUs")
            logger.info(f"  -> Adjusting Gradient Accumulation: {original_accum} "
                        f"-> {config.gradient_accumulation_steps} per GPU.")
            logger.info(f"  -> Effective Batch Size maintained at: {eff_batch_size}")
            logger.info(f"  -> Assigning {suggested_workers} workers per DataLoader.")
        
                
    # Set reproducibility seeds
    seed_offset = ddp_rank 
    torch.manual_seed(61 + seed_offset)
    np.random.seed(61 + seed_offset)
    
    # Optimization: Allow TF32 on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    try:
        #--- DATA PIPELINE ---#
        
        logger.info("Initializing DataLoaders...")
        train_loader, val_loader, registry = create_dataloaders(config, ddp)
        
        #--- MODEL & OPTIMIZER ---#
        
        # Determine precision type for Mixed Precision Training (AMP)
        ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # Create the Automatic Mixed Precision context manager
        ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
        
        # Scaler for small grandient values
        scaler = torch.amp.GradScaler("cuda", enabled=(ptdtype == torch.float16))
        
        # Initialize Model
        model = create_model(config, registry, torch.device(device), ddp)
        
        # Initialize Optimizer
        optimizer = create_optimizer(model, config)

        #--- CHECKPOINT MANAGEMENT ---#
        
        # Define where the checkpoint file should be
        ckpt_path_best = weights_dir / "ckpt_best.pt"
        ckpt_path_last = weights_dir / "ckpt_last.pt"
        
        # State variables
        iter_num = 0
        epoch_num = 0
        best_val_loss = 1e9
        accumulated_time = 0.0
        early_stop_counter = 0
        
        # Resume a training
        if config.init_from == 'resume':
            
            ckpt_path = None
            
            # 1. Try lo load the LAST checkpoint file
            if ckpt_path_last.is_file():
                ckpt_path = ckpt_path_last
                logger.info(f"Resuming from LAST checkpoint: {ckpt_path}")
                
            # 2. If 'last' is missing, look for the latest 'ckpt_iter_XXXXXX.pt' (Mode "all")
            else:
                # Get all files starting with 'ckpt_iter_' and ending in '.pt'
                iter_checkpoints = list(weights_dir.glob('ckpt_iter_*.pt'))
                
                if iter_checkpoints:
                    # Sort them to find the latest iteration
                    iter_checkpoints.sort()
                    ckpt_path = iter_checkpoints[-1]
                    logger.info(f"Resuming from LATEST HISTORY checkpoint: {ckpt_path}")
            
            # 3. Try BEST if neither LAST nor HISTORY exist
            if ckpt_path is None and ckpt_path_best.is_file():
                ckpt_path = ckpt_path_best
                logger.info(f"Resuming from BEST checkpoint: {ckpt_path}")
            
            # 4. If nothing is found -> Scratch
            if ckpt_path is None:
                logger.warning(f"Resume requested but no checkpoints found. Starting from SCRATCH.")
            
            if ckpt_path:
                # Load the file to the current device (CPU or GPU)
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                
                # Load Model Weights
                state_dict = checkpoint['model']
                
                new_state_dict = {}
                # Clean the state_dict keys
                for k, v in list(state_dict.items()):
                    clean_k = k.replace("module.", "").replace("_orig_mod.", "")
                    new_k = f"module._orig_mod.{clean_k}"
                    new_state_dict[new_k] = v
                        
                msg = model.load_state_dict(new_state_dict, strict=False)
                logger.info(f"Load state result: {msg}")
                
                # Load Optimizer State
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                except ValueError as e:
                    logger.warning(
                        "Could not restore optimizer state (likely due to optimizer group changes). "
                        f"Continuing with fresh optimizer. Details: {e}"
                    )
                
                # Restore Training Counters
                iter_num = checkpoint['iter_num']
                epoch_num = checkpoint.get('epoch_num', 0)
                
                # Loading the correct loss
                loaded_best_loss = checkpoint['best_val_loss']
                
                if ckpt_path_best.is_file():
                    try:
                        # Load metadata
                        best_ckpt_data = torch.load(ckpt_path_best, map_location=device, weights_only=False)
                        disk_best_loss = best_ckpt_data.get('best_val_loss', 1e9)
                        
                        # Select the best loss
                        if disk_best_loss < loaded_best_loss:
                            logger.info(f"Correction: Disk loss {disk_best_loss} better than {loaded_best_loss}. Updating.")
                            best_val_loss = disk_best_loss
                        else:
                            best_val_loss = loaded_best_loss
                    except Exception as e:
                        logger.warning(f"Could not verify ckpt_best.pt on disk: {e}. Trusting loaded checkpoint.")
                        best_val_loss = loaded_best_loss
                else:
                    best_val_loss = loaded_best_loss
                

                accumulated_time = checkpoint.get('total_run_time', 0.0)
                
                # Changing time to hours
                prev_hours = accumulated_time / 3600
                
                logger.info(f"Resumed successfully at iteration {iter_num} - "
                            f"Best Loss: {best_val_loss:.4f} - Previous run time: {prev_hours:.2f} hours.")
                
                
                # Only remove .improved file in resume mode
                if ddp_rank == 0:
                    improve_file_path.unlink(missing_ok=True)
                    logger.info("Removing .improved file.")

            else:
                logger.info("Starting training from scratch.")
            
            # If a training is resume, patience to zero again
            early_stop_counter = 0
            
                

        #--- TRAINING LOOP SETUP ---#
        
        # Create an infinite iterator for the DataLoader
        train_iter = iter(train_loader)
        
        # Initialize timers and counters
        t0 = time.time()
        run_start_time = time.time()
        last_log_time = time.time()
        initial_iter = iter_num
        
        # Wait for all GPUs synchronization
        if ddp:
            dist.barrier()
        
        # CSV Logging setup
        if ddp_rank == 0:
            csv_path = logs_dir / "training_metrics.csv"
            if config.init_from == 'scratch' or not csv_path.is_file():
                with open(csv_path, "w") as f:
                    # Headers for the CSV
                    headers = [
                        "iter", "epoch", "progress", "timestamp", "train_loss", "val_loss",
                        "val_loss_spec_from_img", "val_loss_img_from_spec", "loss_images", "loss_spectra",
                        "cross_loss_images", "cross_loss_spectra",
                        "grad_norm", "grad_images", "grad_spectra", "grad_backbone",
                        "clipped", "dropped_modality", "lr", "lr_images", "lr_spectra", "lr_backbone",
                        "mfu", "mem_gb", "dt_ms", "rt_hms", "eta_hms"
                    ]
                    f.write(",".join(headers) + "\n")

            diagnostics_path = logs_dir / config.diagnostics_file_name
            if config.diagnostics_enabled and (config.init_from == 'scratch' or not diagnostics_path.is_file()):
                with open(diagnostics_path, "w") as f:
                    diag_headers = [
                        "iter", "epoch", "timestamp", "dropout_mode", "dropout_applied",
                        "loss_images", "loss_spectra",
                        "grad_images", "grad_spectra", "grad_backbone", "grad_total",
                        "lr_base", "lr_images", "lr_spectra", "lr_backbone"
                    ]
                    f.write(",".join(diag_headers) + "\n")
        
        # WANDB Configuration
        if ddp_rank == 0:
            logger.info(f"Starting training loop from iteration {iter_num}...")
            if config.log_via_wandb and _WANDB_AVAILABLE:
                wandb.init(project=config.wandb_project, 
                        name=config.wandb_run_name, 
                        config=asdict(config))


        # Pytorch profiler setup
        prof = nullcontext()

        if config.profile:
            # Schedule: 
            # - wait=10: Avoid 10 first iters
            # - warmup=2: Starting the tracer
            # - active=4: Saving iters 12, 13, 14 and 15
            # - repeat=1: Just one time
            wait, warmup, active = 10, 2, 4
            
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
                on_trace_ready=tensorboard_trace_handler(str(logs_dir / "profiler_trace")),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            
            if ddp_rank == 0:
                logger.info(f"Profiler enabled. Recording iterations {wait+warmup} to {wait+warmup+active}...")
            
            # Starting the profiler
            prof.start()

        # Max time running before autosaving and finishing
        max_run_seconds = None
        if config.max_run_hours is not None:
            try:
                max_run_seconds = parse_time_to_seconds(config.max_run_hours)
                if ddp_rank == 0:
                    logger.info(f"Time limit set to: {config.max_run_hours} ({max_run_seconds} seconds)")
            except Exception as e:
                if ddp_rank == 0:
                    logger.error(f"Error parsing max_run_hours: {e}")
                sys.exit(1)

        # Update correct epoch
        if ddp and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch_num)

        #--- THE TRAINING LOOP ---#
        while iter_num <= config.max_iters:
            
            # SET LEARNING RATE for this iteration
            lr = get_learning_rate(iter_num, config)
            for param_group in optimizer.param_groups:
                lr_scale = float(param_group.get('lr_scale', 1.0))
                param_group['lr'] = lr * lr_scale
                
            
            # Variable to accumulate loss
            loss_accum_for_log = 0.0

            # Optional modality diagnostics accumulators
            modality_loss_sums = {"images": 0.0, "spectra": 0.0, "aion_images": 0.0, "aion_spectra": 0.0}
            modality_loss_counts = {"images": 0, "spectra": 0, "aion_images": 0, "aion_spectra": 0}
            cross_loss_sums = {"images": 0.0, "spectra": 0.0, "aion_images": 0.0, "aion_spectra": 0.0}
            cross_loss_counts = {"images": 0, "spectra": 0, "aion_images": 0, "aion_spectra": 0}
            modality_drop_counts = {"images": 0, "spectra": 0, "aion_images": 0, "aion_spectra": 0, "none": 0}
            branch_grad_norms = {
                "images": float("nan"),
                "spectra": float("nan"),
                "aion_images": float("nan"),
                "aion_spectra": float("nan"),
                "backbone": float("nan"),
                "total": float("nan"),
            }
            
            # Variable for avoid NaNs
            skip_step = False
                
            # GRADIENT ACCUMULATION LOOP
            for micro_step in range(config.gradient_accumulation_steps):
                
                # Fetch Raw Batch (CPU)
                try:
                    raw_batch = next(train_iter)
                except StopIteration:
                    epoch_num += 1
                    if ddp_rank == 0:
                        logger.info(f"--> Iter {iter_num}: End of Epoch {epoch_num}. Restarting DataLoader.")
                    
                    if ddp:
                        train_loader.sampler.set_epoch(epoch_num)
                    
                    train_iter = iter(train_loader)
                    raw_batch = next(train_iter)
                
                # Calculate dynamic stochastic seed 
                batch_seed = config.token_mixing_seed + iter_num * config.gradient_accumulation_steps + micro_step + ddp_rank

                # Process Batch
                B = EuclidDESIDatasetArrow.process_modes(
                    batch_data=raw_batch, 
                    modality_registry=registry, 
                    device=torch.device(device),
                    shuf=config.shuffle_modality_train,
                    use_token_mixing=config.use_token_mixing,
                    token_mixing_seed=batch_seed
                )

                dropped_modality = maybe_apply_modality_dropout(B, config)
                modality_drop_counts[dropped_modality] = modality_drop_counts.get(dropped_modality, 0) + 1
                        
                
                # DDP Context
                if ddp:
                    
                    is_last_micro_step = (micro_step == config.gradient_accumulation_steps - 1)
                    
                    # context manager: 'no_sync' disables communication, 'nullcontext' enables it
                    context = model.no_sync() if not is_last_micro_step else nullcontext()
                    
                else:
                    context = nullcontext()

                # Forward & Backward Pass
                with context:
                    
                    # Automatic Mixed Precision (AMP)
                    with ctx: 
                        
                        # Forward pass
                        outputs, loss = model(B["X"], targets=B["Y"], dropped_modality=dropped_modality)
                        
                        # --- INITIAL LOSS BALANCE CHECK (Step 0) ---
                        if iter_num == 0 and micro_step == 0 and ddp_rank == 0:
                            logger.info(" --> INITIAL LOSS MODALITY BALANCE CHECK (Step 0) <-- ")
                            initial_losses = {}
                            for mod_name in registry.names():
                                if mod_name in outputs and mod_name in B["Y"]:
                                    pred = outputs[mod_name]
                                    target = B["Y"][mod_name]
                                    mod_config = registry.get_config(mod_name)
                                    
                                    # Auto-align shapes (token_mixing causes length differences
                                    # that torch.compile prevents from propagating back)
                                    if pred.shape[1] != target.shape[1]:
                                        min_len = min(pred.shape[1], target.shape[1])
                                        pred = pred[:, :min_len]
                                        target = target[:, :min_len]
                                    
                                    if "aion" in mod_name:
                                        mod_loss = torch.nn.functional.cross_entropy(
                                            pred.reshape(-1, pred.size(-1)),
                                            target.reshape(-1),
                                        ).item()
                                    elif config.loss_type in ["l1", "mae"]:
                                        mod_loss = torch.nn.functional.l1_loss(pred, target).item()
                                    elif config.loss_type == "mse":
                                        mod_loss = torch.nn.functional.mse_loss(pred, target).item()
                                    else:
                                        mod_loss = torch.nn.functional.huber_loss(pred, target, delta=config.loss_huber_delta).item()
                                    
                                    weighted_loss = mod_loss * mod_config.loss_weight
                                    initial_losses[f"initial_loss/{mod_name}_unweighted"] = mod_loss
                                    initial_losses[f"initial_loss/{mod_name}_weighted"] = weighted_loss
                                    
                                    logger.info(f"Modality: {mod_name:<10} | Unweighted Loss: {mod_loss:.6f} | Weight: {mod_config.loss_weight} | Weighted Contrib: {weighted_loss:.6f}")
                            if config.log_via_wandb and _WANDB_AVAILABLE:
                                wandb.log(initial_losses, step=0)

                        # Skipping the batch if the loss is NaN
                        if torch.isnan(loss) or torch.isinf(loss):
                            skip_step = True
                            if ddp_rank == 0:
                                logger.warning(f"NaN/Inf detected in loss at iter {iter_num}, "
                                               f"micro-step {micro_step}. Skipping batch.")
                            break

                        if config.diagnostics_enabled and config.diagnostics_track_losses:
                            micro_mod_losses = _compute_modality_losses(outputs, B["Y"], config)
                            for mod_name, mod_loss in micro_mod_losses.items():
                                if mod_name in modality_loss_sums:
                                    modality_loss_sums[mod_name] += mod_loss
                                    modality_loss_counts[mod_name] += 1
                                if mod_name == dropped_modality:
                                    cross_loss_sums[mod_name] += mod_loss
                                    cross_loss_counts[mod_name] += 1
                        
                        # Scale loss
                        loss = loss / config.gradient_accumulation_steps
                        
                    
                    # If there is a NaN, backward is avoided
                    if not skip_step:
                        scaler.scale(loss).backward()
                        loss_accum_for_log += loss.item()
            
            #--- END OF MICRO-BATCHES ---#
            
            # Not NaNs found
            if not skip_step:
                
                # Unscaling the gradient
                scaler.unscale_(optimizer)
                
                # Control variables
                grad_norm = 0.0
                is_clipped = 0.0
                
                if config.grad_clip != 0.0:
                    
                    # This fonction (finished in _) computes the gradient norm, 
                    # returns it and compares with the gradient clipping value
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    
                    # Logging if gradient is clipped
                    if grad_norm > config.grad_clip:
                        is_clipped = 1.0

                diagnostics_due = (
                    config.diagnostics_enabled
                    and (iter_num % config.diagnostics_interval == 0)
                    and (iter_num > initial_iter or iter_num == 0)
                )

                if diagnostics_due and config.diagnostics_track_grads:
                    branch_grad_norms = compute_branch_grad_norms(model)
                    
                    
                # Updating the weights
                scaler.step(optimizer)
                scaler.update()
                
            else:
                # NaNs found
                grad_norm = 0.0
                is_clipped = 0.0
                optimizer.zero_grad(set_to_none=True)
                
                # Loss value for logging
                loss_accum_for_log = float('nan')
            
            # Always set gradient to none
            optimizer.zero_grad(set_to_none=True)
            
            # LOGGING Console & WandB (Master Only)
            if ddp_rank == 0:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                
                # Training progress
                train_prog = iter_num / config.max_iters
                    
                
                if iter_num % config.log_interval == 0 and (iter_num > initial_iter or iter_num == 0):
                    
                    # Time since last iter log
                    current_log_time = time.time()
                    time_since_last_log = current_log_time - last_log_time
                    last_log_time = current_log_time
                    
                    # Average time between logging iteractions
                    avg_dt = time_since_last_log / config.log_interval
                    
                    # Computing MFU
                    raw_model = model.module if ddp else model
                    if hasattr(raw_model, "_orig_mod"):
                        raw_model = raw_model._orig_mod
                        
                    mfu_display = 0.0

                    # Searching for estimate_mfu()
                    if hasattr(raw_model, "estimate_mfu"):
                        
                        if iter_num == 0:
                            mfu_display = 0.0
                        else:
                            # Computing the total seen samples per GPU on each iter
                            fwdbwd_per_gpu = config.batch_size * config.gradient_accumulation_steps
                            mfu_display = raw_model.estimate_mfu(fwdbwd_per_gpu, avg_dt)
                    else:
                        mfu_display = -1.0
                        
                    # VRAM Computation
                    mem_usage = torch.cuda.max_memory_allocated() / (1024 ** 3) 
                    torch.cuda.reset_peak_memory_stats()   
                    
                    
                    # Recover the real average loss of the batch
                    lossf = loss_accum_for_log
                    
                    # Compute Running Time
                    current_session_seconds = time.time() - run_start_time
                    total_seconds = current_session_seconds + accumulated_time
                    
                    running_str = str(datetime.timedelta(seconds=int(total_seconds)))
                    
                    # Compute the ETA for finishing training
                    remaining_iters = config.max_iters - iter_num
                    eta_seconds = remaining_iters * avg_dt
                    eta_str = str(datetime.timedelta(seconds=int(eta_seconds))).replace(',', '')
                    
                    logger.info(
                        f"Iter {iter_num}/{config.max_iters} ({train_prog:.2%}) | "
                        f"Loss {loss_accum_for_log:.4f} | "
                        f"LR {lr:.4e} (Img:{lr*config.lr_mult_images:.1e}/Spc:{lr*config.lr_mult_spectra:.1e}) | "
                        f"Norm {grad_norm:.2f} | "
                        f"MFU {mfu_display*100:.2f}% (avg) | "
                        f"Mem {mem_usage:.2f}GB | "
                        f"dt {avg_dt*1000:.2f}ms (avg) | "
                        f"RT {running_str} | ETA {eta_str} | "
                        f"Mix:{int(config.use_token_mixing)} "
                        f"MskI:{int(config.images_mask)} MskS:{int(config.spectra_mask)}"
                    )

                    diagnostics_due = (
                        config.diagnostics_enabled
                        and (iter_num % config.diagnostics_interval == 0)
                        and (iter_num > initial_iter or iter_num == 0)
                    )

                    loss_images_diag = (
                        modality_loss_sums.get("images", 0) / modality_loss_counts.get("images", 1)
                        if modality_loss_counts.get("images", 0) > 0
                        else (modality_loss_sums.get("aion_images", 0) / modality_loss_counts.get("aion_images", 1) if modality_loss_counts.get("aion_images", 0) > 0 else float("nan"))
                    )
                    loss_spectra_diag = (
                        modality_loss_sums.get("spectra", 0) / modality_loss_counts.get("spectra", 1)
                        if modality_loss_counts.get("spectra", 0) > 0
                        else (modality_loss_sums.get("aion_spectra", 0) / modality_loss_counts.get("aion_spectra", 1) if modality_loss_counts.get("aion_spectra", 0) > 0 else float("nan"))
                    )
                    cross_images_diag = (
                        cross_loss_sums.get("images", 0) / cross_loss_counts.get("images", 1)
                        if cross_loss_counts.get("images", 0) > 0
                        else (cross_loss_sums.get("aion_images", 0) / cross_loss_counts.get("aion_images", 1) if cross_loss_counts.get("aion_images", 0) > 0 else float("nan"))
                    )
                    cross_spectra_diag = (
                        cross_loss_sums.get("spectra", 0) / cross_loss_counts.get("spectra", 1)
                        if cross_loss_counts.get("spectra", 0) > 0
                        else (cross_loss_sums.get("aion_spectra", 0) / cross_loss_counts.get("aion_spectra", 1) if cross_loss_counts.get("aion_spectra", 0) > 0 else float("nan"))
                    )
                    dropout_summary = summarize_modality_dropout(modality_drop_counts)

                    if diagnostics_due:
                        logger.info(
                            f"Diag | Drop={dropout_summary} | "
                            f"Loss(img/spec)=({loss_images_diag:.4f}/{loss_spectra_diag:.4f}) | "
                            f"Cross(img/spec)=({cross_images_diag:.4f}/{cross_spectra_diag:.4f}) | "
                            f"Grad(img/spec/back)=({branch_grad_norms['images']:.2f}/"
                            f"{branch_grad_norms['spectra']:.2f}/{branch_grad_norms['backbone']:.2f})"
                        )
                    
                    if config.log_via_wandb and _WANDB_AVAILABLE:
                        wandb_payload = {
                            "iter": iter_num,
                            "train/loss": lossf,
                            "train/lr_backbone": lr,
                            "train/lr_images": lr * config.lr_mult_images,
                            "train/lr_spectra": lr * config.lr_mult_spectra,
                            "train/grad_norm": grad_norm,
                            "train/mfu": mfu_display * 100,
                            "train/mem_gb": mem_usage,
                            "train/time_ms": avg_dt * 1000,
                            "train/rt_hours": total_seconds / 3600,
                            "train/eta_hours": eta_seconds / 3600,
                            # Structural Tracking Flags
                            "status/token_mixing": 1 if config.use_token_mixing else 0,
                            "status/images_masking": 1 if config.images_mask else 0,
                            "status/spectra_masking": 1 if config.spectra_mask else 0
                        }
                        if diagnostics_due and config.diagnostics_enabled:
                            wandb_payload.update(
                                {
                                    "diag/loss_images": loss_images_diag,
                                    "diag/loss_spectra": loss_spectra_diag,
                                    "diag/cross_loss_images": cross_images_diag,
                                    "diag/cross_loss_spectra": cross_spectra_diag,
                                    "diag/grad_images": branch_grad_norms["images"],
                                    "diag/grad_spectra": branch_grad_norms["spectra"],
                                    "diag/grad_backbone": branch_grad_norms["backbone"],
                                    "diag/dropout_images_microsteps": modality_drop_counts.get("images", 0),
                                    "diag/dropout_spectra_microsteps": modality_drop_counts.get("spectra", 0),
                                }
                            )
                        wandb.log(wandb_payload)
                        
                    # CSV Writing
                    try:
                        with open(csv_path, "a") as f:
                            # Obtaining the current timestamp
                            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            # Writing into the CSV
                            f.write(f"{iter_num},{epoch_num},{train_prog:.4f},{timestamp_str},{loss_accum_for_log:.6f},,,"
                                    f",{loss_images_diag:.6f},{loss_spectra_diag:.6f},{cross_images_diag:.6f},{cross_spectra_diag:.6f},"
                                    f"{grad_norm:.4f},{branch_grad_norms['images']:.4f},"
                                    f"{branch_grad_norms['spectra']:.4f},{branch_grad_norms['backbone']:.4f},{is_clipped:.0f},{dropout_summary},"
                                    f"{lr:.4e},{lr * config.lr_mult_images:.4e},{lr * config.lr_mult_spectra:.4e},{lr * config.lr_mult_backbone:.4e},"
                                    f"{mfu_display*100:.2f},{mem_usage:.2f},{avg_dt*1000:.2f},{running_str},{eta_str}\n"
                            )
                            
                    except Exception as e:
                        logger.error(f"CSV Write Error: {e}")

                    if diagnostics_due and config.diagnostics_enabled:
                        try:
                            with open(diagnostics_path, "a") as f:
                                timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                f.write(
                                    f"{iter_num},{epoch_num},{timestamp_str},{config.modality_dropout_mode},"
                                    f"{dropout_summary},{loss_images_diag:.6f},{loss_spectra_diag:.6f},"
                                    f"{branch_grad_norms['images']:.6f},{branch_grad_norms['spectra']:.6f},"
                                    f"{branch_grad_norms['backbone']:.6f},{branch_grad_norms['total']:.6f},"
                                    f"{lr:.6e},{lr*config.lr_mult_images:.6e},"
                                    f"{lr*config.lr_mult_spectra:.6e},{lr*config.lr_mult_backbone:.6e}\n"
                                )
                        except Exception as e:
                            logger.error(f"Diagnostics CSV Write Error: {e}")

            # VALIDATION & CHECKPOINT
            if iter_num > 0 and iter_num % config.eval_interval == 0:
                
                # Stopping training variable
                stop_training = False
                
                # Master performs validation
                if ddp_rank == 0:
                    logger.info(f"Running validation at iter {iter_num}...")
                    
                    # Switch to evaluation mode
                    model.eval() 
                    
                    val_losses = []
                    val_iso_img2spec = []
                    val_iso_spec2img = []
                    max_val_batches = config.eval_batches
                    
                    with torch.no_grad():
                        val_iter = iter(val_loader)
                        for _ in range(max_val_batches):
                            try:
                                vbatch = next(val_iter)
                            except StopIteration:
                                val_iter = iter(val_loader)
                                vbatch = next(val_iter)
                                
                            # Deterministic seed for validation 
                            val_batch_seed = config.token_mixing_seed + iter_num

                            B_val = EuclidDESIDatasetArrow.process_modes(
                                batch_data=vbatch, 
                                modality_registry=registry, 
                                device=torch.device(device),
                                shuf=config.shuffle_modality_val,
                                use_token_mixing=config.use_token_mixing,
                                token_mixing_seed=val_batch_seed
                            )
                            
                            with ctx:
                                _, v_loss = model(B_val["X"], targets=B_val["Y"])
                            val_losses.append(v_loss.item())

                            # --- ISOLATED RECONSTRUCTION VALIDATION ---
                            img_key = "aion_images" if "aion_images" in B_val["X"] else "images"
                            spec_key = "aion_spectra" if "aion_spectra" in B_val["X"] else "spectra"

                            if img_key in B_val["X"] and spec_key in B_val["X"]:
                                # 1. Images -> Spectra (Mask out Spectra Input completely)
                                X_img2spec = {k: v for k, v in B_val["X"].items()}
                                X_img2spec[spec_key] = torch.zeros_like(X_img2spec[spec_key])
                                
                                with ctx:
                                    out_img2spec, _ = model(X_img2spec, targets=B_val["Y"])
                                    iso_loss_s = _compute_modality_losses(out_img2spec, B_val["Y"], config)
                                    if spec_key in iso_loss_s:
                                        val_iso_img2spec.append(iso_loss_s[spec_key])

                                # 2. Spectra -> Images (Mask out Images Input completely)
                                X_spec2img = {k: v for k, v in B_val["X"].items()}
                                X_spec2img[img_key] = torch.zeros_like(X_spec2img[img_key])

                                with ctx:
                                    out_spec2img, _ = model(X_spec2img, targets=B_val["Y"])
                                    iso_loss_i = _compute_modality_losses(out_spec2img, B_val["Y"], config)
                                    if img_key in iso_loss_i:
                                        val_iso_spec2img.append(iso_loss_i[img_key])
                    
                    val_loss = sum(val_losses) / len(val_losses)
                    
                    avg_iso_img2spec = sum(val_iso_img2spec) / len(val_iso_img2spec) if val_iso_img2spec else float('nan')
                    avg_iso_spec2img = sum(val_iso_spec2img) / len(val_iso_spec2img) if val_iso_spec2img else float('nan')

                    logger.info(
                        f"Val Loss: {val_loss:.4f} | "
                        f"Zero-Shot Cross-Loss -> SpectraFromImg: {avg_iso_img2spec:.4f} | "
                        f"ImgFromSpectra: {avg_iso_spec2img:.4f}"
                    )
                    
                    if config.log_via_wandb and _WANDB_AVAILABLE:
                        val_payload = {"val/loss": val_loss, "iter": iter_num}
                        if val_iso_img2spec:
                            val_payload["val/isolated_loss_spectra_from_img"] = avg_iso_img2spec
                        if val_iso_spec2img:
                            val_payload["val/isolated_loss_img_from_spectra"] = avg_iso_spec2img
                        wandb.log(val_payload)
                        
                    # CSV Write: Just Validation properties
                    try:
                        with open(csv_path, "a") as f:
                            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            # iter, epoch, prog, time, t_loss(empty), val_loss, val_loss_spec_from_img, val_loss_img_from_spec, and the rest empty...
                            # Remember headers:
                            # "iter", "epoch", "progress", "timestamp", "train_loss", "val_loss",
                            # "val_loss_spec_from_img", "val_loss_img_from_spec", "loss_images", "loss_spectra",
                            # "cross_loss_images", "cross_loss_spectra",
                            # "grad_norm", "grad_images", "grad_spectra", "grad_backbone",
                            # "clipped", "dropped_modality", "lr", "lr_images", "lr_spectra", "lr_backbone",
                            # "mfu", "mem_gb", "dt_ms", "rt_hms", "eta_hms"
                            f.write(f"{iter_num},{epoch_num},{train_prog:.4f},{timestamp_str},,{val_loss:.6f},{avg_iso_img2spec:.6f},"
                                    f"{avg_iso_spec2img:.6f},,,,,,,,,,,,,,,,,,,\n")
                    except Exception as e:
                        logger.error(f"Val CSV Write Error: {e}")
                    
                    
                    # EARLY STOPPING MECHANISM
                    
                    # If loss improve
                    if val_loss < best_val_loss:
                        
                        # Computing the improvement
                        improvement = best_val_loss - val_loss
                        
                        # Reset the stop counter
                        best_val_loss = val_loss
                        early_stop_counter = 0
                        is_best = True
                        
                        # Logging the validation loss as the BEST
                        logger.info(f"Validation Loss: {val_loss:.4f} (-{improvement:.4f}) | NEW BEST")
                        
                        # Bash control hidden file for checking if the val loss has improved
                        if ddp_rank == 0:
                            with open(improve_file_path, "w") as f:
                                f.write(f"Iter: {iter_num}, Loss: {val_loss}")
                        
                    else:
                        # Increasing the counter ONLY if we have reached the minimum iters
                        if iter_num >= getattr(config, 'early_stopping_min_iters', 0):
                            early_stop_counter += 1
                        
                        is_best = False
                        logger.info(
                            f"Validation Loss: {val_loss:.4f} (Best: {best_val_loss:.4f}) | "
                            f"Patience: {early_stop_counter}/{config.early_stopping_patience}"
                        )
                        
                        # If counter eaches the patience limit
                        if early_stop_counter >= config.early_stopping_patience:
                            logger.info(f"Early stopping triggered! No improvement for {early_stop_counter} checks.")
                            stop_training = True
                    
                    # CHECKPOINT SAVING
                    if iter_num > 0:
                        
                        # 1. Calculate current total time
                        current_total_time = (time.time() - run_start_time) + accumulated_time
                        
                        # 2. Prepare Base Checkpoint Dictionary
                        checkpoint = {
                            'model': model.module.state_dict() if ddp else model.state_dict(),
                            'modality_registry': registry,
                            'optimizer': optimizer.state_dict(),
                            'iter_num': iter_num,
                            'epoch_num': epoch_num,
                            'best_val_loss': best_val_loss, # Current record (may be updated below)
                            'config': asdict(config),
                            'total_run_time': current_total_time,
                            'early_stop_counter': early_stop_counter,
                        }

                        # 3. Save "LAST" - Periodic save
                        if config.checkpoint_save_type in ["last", "both"]:
                            checkpoint['best_val_loss'] = best_val_loss
                            torch.save(checkpoint, ckpt_path_last)

                        # 4. Save "BEST"
                        if is_best and config.checkpoint_save_type in ["best", "both", "all"]:
                            checkpoint['best_val_loss'] = best_val_loss 
                            logger.info(f"New best model found ({val_loss:.4f}). Saving to {ckpt_path_best}")
                            torch.save(checkpoint, ckpt_path_best)
                            
                        if config.checkpoint_save_type == "all":
                            ckpt_iter_name = f"ckpt_iter_{iter_num:06d}.pt"
                            ckpt_path_iter = weights_dir / ckpt_iter_name
                            
                            # Update dictionary with current val loss for this snapshot
                            checkpoint['best_val_loss'] = val_loss 
                            torch.save(checkpoint, ckpt_path_iter)
                            logger.info(f"Snapshot saved: {ckpt_path_iter}")
                
                if ddp:
                    
                    # 1=Stop, 0=Continue
                    stop_signal = torch.tensor(1 if stop_training else 0, device=device)
                    
                    # Comunicating the decission to al GPUs
                    dist.broadcast(stop_signal, src=0)
                    
                    # Update the stopping variable
                    stop_training = stop_signal.item() == 1
                    
                    # Waiting to finish
                    dist.barrier()
                
                # Switch back to training modes
                model.train() 
                
                # Early stopping
                if stop_training:
                    if ddp_rank == 0:
                        logger.info("Stopping training loop due to Early Stopping.")
                    break # Exit the while iter_num < max_iters loop
                
            # PROFILER STEP
            if config.profile:
                prof.step()
                
            
            # AUTOSAVING
            # In case the Slurm time comes to an end, finishing earlier
            if max_run_seconds is not None:
                elapsed_seconds = (time.time() - run_start_time)
                
                # If own limit time is reached
                if elapsed_seconds > max_run_seconds:
                    # Stop
                    stop_signal = torch.tensor(1, device=device)
                else:
                    # Continue
                    stop_signal = torch.tensor(0, device=device)

                # Syncronizing all process
                if ddp:
                    dist.all_reduce(stop_signal, op=dist.ReduceOp.MAX)
                
                if stop_signal.item() == 1:
                    if ddp_rank == 0:
                        elapsed_str = str(datetime.timedelta(seconds=int(elapsed_seconds)))
                        logger.info(f"Time limit reached ({elapsed_str} > {config.max_run_hours}). "
                                    f"Saving checkpoint and exiting...")
                        
                        # Saving checkpoint
                        current_total_time = elapsed_seconds + accumulated_time
                        checkpoint = {
                            'model': model.module.state_dict() if ddp else model.state_dict(),
                            'modality_registry': registry,
                            'optimizer': optimizer.state_dict(),
                            'iter_num': iter_num,
                            'epoch_num': epoch_num,
                            'best_val_loss': best_val_loss,
                            'config': asdict(config),
                            'total_run_time': current_total_time,
                            'early_stop_counter': early_stop_counter,
                        }
                        
                        # Overwriting the LAST checkpoint
                        torch.save(checkpoint, ckpt_path_last)
                        logger.info(f"Emergency checkpoint saved to {ckpt_path_last}")

                    break # Exit the training loop
            
            # Increment iter counter
            iter_num += 1


    finally:
        
        # Stop profiler if active
        if config.profile:
            prof.stop()
        
        #--- CLEANUP DDP ---#
        if ddp:
            destroy_process_group()
            logger.info("Cleanup complete.")
        
        if ddp_rank == 0:
            logger.info("Training finished!")


if __name__ == "__main__":
    main()