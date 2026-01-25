"""
AstroPT Multimodal Training Scripts.

This script implements a Distributed Data Parallel (DDP) training loop for the 
AstroPT model, utilising the Euclid-DESI multimodal dataset (Arrow format).

Author: Victor Alonso Rodriguez
Date: January 2026
"""

from __future__ import annotations

import argparse
import datetime
import inspect
import json
import logging
import math
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
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
    """

    #--- 1. I/O & Paths ---#
    out_dir: str = "logs/astropt_100M_arrow"
    data_dir: str = "/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"
    
    #--- 2. Data Loading ---#
    batch_size: int = 16            # Micro-batch size per GPU (what fits in VRAM)
    num_workers: int = 8            # Optimized for Teide HPC (CPU cores for loading)
    persistent_workers: bool = True # Keep workers active
    prefetch_factor: int = 2        # How many batches to preload per worker
    pin_memory: bool = True         # Faster transfer RAM -> VRAM
    spiral: bool = True             # Whether to apply spiral readout to galaxy images
    init_from: str = "scratch"      # Training from scratch or resume training
    
    #--- 3. Model Architecture for the 100M Parameter Setup ---#
    n_layer: int = 12           # Depth of the Transformer
    n_head: int = 12            # Number of attention heads
    n_embd: int = 768           # Embedding dimension (width of the network)
    n_chan: int = 4             # Input channels: 1 VIS + 3 NISP (Y, J, H)
    block_size: int = 1024      # Context length (max tokens per sample)
    dropout: float = 0.0        # Regularization (0.0 for pretraining is standard)
    bias: bool = False          # Learnable bias in Linear layers (False is modern/faster)
    attn_type: str = "causal"   # Attention mechanism type
    backbone: str = "native"    # Model bakcbone: native or llm
    tokeniser: str = "aim"      # Model tokeniser method
    use_qlora: bool = False     # Use Quantized Low-Rank Adaptation
    
    #--- Multimodality Specifics ---#
    # Images
    images_size: int = 224              # Images side size in pixels
    images_patch_size: int = 16         # Side size in pixels of each patch in an image
    images_channels: int = 4            # Channels per image (VIS + NISP Y,J,H)
    images_loss_weight: float = 1.0     # Images importance for training
    images_embed_pos: bool = True       # Images embedding positions learning
    image_pos_input_size: int = 1       # Images position input size
    img_norm_type: str = "constant"     # Normalization method for images (or z_score)
    img_norm_const: float = 1.0         # Normalization constant for images
    
    # Spectra
    spectra_size: int = 7781            # Spectra total size
    spectra_patch_size: int = 10        # Patch size for each spectrum
    spectra_loss_weight: float = 1.0    # Spectra importance for training 
    spectra_embed_pos: bool = True      # Spectra embedding positions learning
    spectra_pos_input_size: int = 1     # Spectra position input size
    spectra_norm_type: str = "constant" # Normalization method for spectra (or z_score)
    spectra_norm_const: float = 1.0     # Normalization constant for spectra
    
    #--- 4. Optimization of the Learning Process) ---#
    learning_rate: float = 6e-4     # Learning rate per weight update
    max_iters: int = 100_000        # Total training steps (NOT epochs)
    weight_decay: float = 1e-1      # Regularization to prevent overfitting
    beta1: float = 0.9              # AdamW parameter
    beta2: float = 0.95             # AdamW parameter
    grad_clip: float = 1.0          # Stabilizes training if gradients explode
    
    # Gradient Accumulation: Simulates a larger batch size. 
    # Effective batch = 16 * 40 = 640
    gradient_accumulation_steps: int = 40 
    
    #--- 5. Learning Rate Scheduler ---#
    lr_decay: bool = True           # Activates the variable learning rate decay
    lr_warmup_iters: int = 2_000    # Steps to ramp up LR from 0 to max
    lr_decay_iters: int = 80_000    # Steps to decay LR down to min
    lr_min: float = 6e-5            # Minimum LR (usually 10% of max)

    #--- 6. Logging & Checkpointing ---#
    eval_interval: int = 1_000              # How often to validate
    eval_batches: int = 100                 # How many batches to use for validation
    log_interval: int = 200                 # How often to print to console/WandB
    checkpoint_interval: int = 2_000        # How often to save .pt files
    checkpoint_save_type: str = "both"      # Checkpoint saving mode: best, last or both
    early_stopping_patience: int = 10       # Stop if no improvement after N evals

    #--- 7. System & Backend ---#
    device: str = "cuda"                    # CPU/GPU device interface: cpu, cuda or mps
    dtype: str = "bfloat16"                 # 'bfloat16' is best for A100 GPUs
    compile: bool = True                    # PyTorch 2.0 compiler
    compile_mode: str = "default"           # Compilation mode
    backend: str = "nccl"                   # Communication backend for DDP
    max_run_hours: Optional[str] = None     # Force stop after "HH:MM:SS"

    #--- 8. External Monitoring ---#
    log_via_wandb: bool = False             # Weight and bias (wandb) logging
    wandb_project: str = "AstroPT-Arrow"    # wandb project name
    wandb_run_name: Optional[str] = None    # Training name
    log_emissions: bool = False             # CodeCarbon logging
    profile: bool = False                   # Enable PyTorch Profiler (Trace analysis)


def get_config_from_args() -> TrainingConfig:
    """
    Parses command line arguments to override default TrainingConfig values.

    Instead of manually defining flags, this function inspects the TrainingConfig 
    dataclass and automatically generates command-line arguments for every field.

    Example:
        python train.py --batch_size 32 --no-compile --wandb_project "New-Test"

    Returns:
        TrainingConfig: A new configuration object populated with CLI overrides.
    """
    
    # Creating the parser argument object
    parser = argparse.ArgumentParser(description="AstroPT Training Script")
    
    # Obtaining the defeault configuration
    default_config = TrainingConfig()
    
    # Iterate over the configuration parameters
    for key, value in asdict(default_config).items():
        arg_type = type(value)
        
        # Booleans (Flags)
        if arg_type == bool:
            
            # Create paired flags: --feature (True) and --no-feature (False)
            parser.add_argument(f"--{key}", action="store_true", default=value, help=f"Enable {key}")
            parser.add_argument(f"--no-{key}", dest=key, action="store_false", help=f"Disable {key}")
        
        # None defaults values
        elif value is None:
            # If default is None it expects a string
            parser.add_argument(f"--{key}", type=str, default=None, help="Default: None")
            
        # Standard Case: Int, Float, Str
        else:
            parser.add_argument(f"--{key}", type=arg_type, default=value, help=f"Default: {value}")

    # Argument parsed from terminal
    args = parser.parse_args()
    
    # Overriding default configuration
    return TrainingConfig(**vars(args))

def get_git_commit_hash() -> str:
    """Returns the current git commit hash or 'unknown' if not in a git repo."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"
    

def get_dataset_info(data_dir: str) -> dict:
    """
    Scans the data directory to create an informative version based on 
    filesystem metadata: latest modification time and total size.
    """
    try:
        max_mtime = 0.0
        total_size = 0
        file_count = 0

        # Walk through the directory
        for root, _, files in os.walk(data_dir):
            for name in files:
                # Filter arrow files
                if not name.endswith('.arrow') and not name.endswith('.json'):
                    continue
                
                full_path = os.path.join(root, name)
                stats = os.stat(full_path)
                
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


def parse_time_to_seconds(time_str: str) -> float:
    """Converts a HH:MM:SS string to total seconds."""
    try:
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except ValueError:
        raise ValueError(f"Invalid time format: {time_str}. Expected HH:MM:SS")
    
    
def save_config_json(config: TrainingConfig, rank: int):
    """Saves the configuration to a JSON file for reproducibility."""
    if rank == 0:
        config_dict = asdict(config)
        
        # Adding Git hash
        config_dict["git_hash"] = get_git_commit_hash()
        
        # Adding dataset information
        data_stats = get_dataset_info(config.data_dir)
        config_dict.update(data_stats)
        
        config_path = os.path.join(config.out_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)

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
            print(f"[INFO]: DDP Initialized: Global Rank {ddp_rank}/{ddp_world_size} | Local Rank {ddp_local_rank} | Device {device}")
            
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
    ddp_rank: int
) -> logging.Logger:
    """
    Configures the logging system.
    
    - Master Process (Rank 0): Logs to Console (stdout) AND a file (training.log).
    - Worker Processes (Rank > 0): Only log ERRORS to avoid cluttering the output.

    Args:
        config (TrainingConfig): To know where the 'out_dir' is.
        ddp_rank (int): To decide verbosity (Master vs Workers).

    Returns:
        logging.Logger: The configured logger object.
    """
    
    # Ensuring the output directory exists
    if ddp_rank == 0:
        os.makedirs(config.out_dir, exist_ok=True)

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
        log_file = os.path.join(config.out_dir, "training.log")
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
    # Calculate the flattened input image size for the Linear Encoder
    img_input_batch_size = config.images_patch_size * config.images_patch_size * config.images_channels
    
    # Define configuration for each modality
    modalities = [
        ModalityConfig(
            name="images",
            input_size=img_input_batch_size,
            patch_size=config.images_patch_size,
            pos_input_size=config.image_pos_input_size,  
            loss_weight=config.images_loss_weight,
            embed_pos=config.images_embed_pos,
        ),
        ModalityConfig(
            name="spectra",
            input_size=config.spectra_patch_size,
            patch_size=config.spectra_patch_size,
            pos_input_size=config.spectra_pos_input_size, 
            loss_weight=config.spectra_loss_weight,
            embed_pos=config.spectra_embed_pos,
        ),
    ]
    
    # Instantiate the Registry
    registry = ModalityRegistry(modalities)
    
    # Prepare data transformations
    data_tf = EuclidDESIDatasetArrow.data_transforms(
        norm_type_img=config.img_norm_type,
        norm_const_img=config.img_norm_const,
        norm_type_spec=config.spectra_norm_type,
        norm_const_spec=config.spectra_norm_const
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
        transform=data_tf
    )
    
    # Instantiate Validation/Test Dataset 
    val_dataset = EuclidDESIDatasetArrow(
        arrow_folder_root=config.data_dir,
        split="test", 
        modality_registry=registry,
        spiral=config.spiral,
        stochastic=False,
        transform=data_tf
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
        tokeniser=config.tokeniser,
        use_qlora=config.use_qlora,
        backbone=config.backbone,
    )
    
    # Instantiate the Model
    model = GPT(gpt_config, registry)
    logger.info(f"Initializing GPT Model with {config.n_layer} layers")

    # Move Model to Selected Device
    model.to(device)

    # Pythorch Compilation
    if config.compile:
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model, mode=config.compile_mode) 

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
    
    # Separate parameters into Decay and No-Decay groups
    decay_params = []
    nodecay_params = []
    
    for n, p in param_dict.items():
        if p.dim() >= 2:
            # Tensors with 2 or more dimensions GET decay
            decay_params.append(p)
        else:
            # Tensors with 1 dimension DO NOT GET decay
            nodecay_params.append(p)
            
    # Calculate total parameters for logging
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    
    logger.info(f"Optimizer Config: {len(decay_params)} tensors ({num_decay_params:,} params) with decay.")
    logger.info(f"Optimizer Config: {len(nodecay_params)} tensors ({num_nodecay_params:,} params) without decay.")

    # Create the param_groups list for PyTorch
    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

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
    
    # Load configuration from CLI arguments (overriding defaults)
    config = get_config_from_args()
    
    # Setup Logger (only Master Process logs to file/console)
    logger = logging_setup(config, ddp_rank)
    
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
                logger.warning(f"Grad Accum {config.gradient_accumulation_steps} is not divisible by {ddp_world_size}. It will be rounded down.")
        
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
            logger.info(f"  -> Adjusting Gradient Accumulation: {original_accum} -> {config.gradient_accumulation_steps} per GPU.")
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
        ckpt_path_best = os.path.join(config.out_dir, 'ckpt_best.pt')
        ckpt_path_last = os.path.join(config.out_dir, 'ckpt_last.pt')
        
        # State variables
        iter_num = 0
        best_val_loss = 1e9
        accumulated_time = 0.0
        early_stop_counter = 0
        
        # Resume a training
        if config.init_from == 'resume':
            
            # 1. Try lo load the LAST checkpoint file
            if os.path.exists(ckpt_path_last):
                ckpt_path = ckpt_path_last
                logger.info(f"Resuming from LAST checkpoint: {ckpt_path}")
            
            # 2. Try BEST if LAST is not found
            elif os.path.exists(ckpt_path_best):
                ckpt_path = ckpt_path_best
                logger.info(f"Resuming from BEST checkpoint: {ckpt_path}")
            
            # 3. If neither exists: From Scratch
            else:
                ckpt_path = None
                logger.warning(f"Resume requested but no checkpoints found. Starting from SCRATCH.")
            
            if ckpt_path:
                # Load the file to the current device (CPU or GPU)
                checkpoint = torch.load(ckpt_path, map_location=device)
                
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
                optimizer.load_state_dict(checkpoint['optimizer'])
                
                # Restore Training Counters
                iter_num = checkpoint['iter_num']
                best_val_loss = checkpoint['best_val_loss']
                accumulated_time = checkpoint.get('total_run_time', 0.0)
                
                # Changing time to hours
                prev_hours = accumulated_time / 3600
                
                logger.info(f"Resumed successfully at iteration {iter_num} - Best Loss: {best_val_loss:.4f} - Previous run time: {prev_hours:.2f} hours.")

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
        epoch_num = 0
        
        # Wait for all GPUs synchronization
        if ddp:
            dist.barrier()
        
        # CSV Logging setup
        if ddp_rank == 0:
            csv_path = os.path.join(config.out_dir, "metrics.csv")
            if config.init_from == 'scratch' or not os.path.exists(csv_path):
                with open(csv_path, "w") as f:
                    # Headers for the CSV
                    f.write("iter,epoch,progress,timestamp,train_loss,val_loss,grad_norm,clipped,lr,mfu,mem_gb,dt_ms,rt_hms,eta_hms\n")
        
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
                on_trace_ready=tensorboard_trace_handler(os.path.join(config.out_dir, "profiler_trace")),
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


        #--- THE TRAINING LOOP ---#
        while iter_num < config.max_iters:
            
            # SET LEARNING RATE for this iteration
            lr = get_learning_rate(iter_num, config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            
            # Variable to accumulate loss
            loss_accum_for_log = 0.0
                
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
                
                # Process Batch
                B = EuclidDESIDatasetArrow.process_modes(
                    batch_data=raw_batch, 
                    modality_registry=registry, 
                    device=torch.device(device)
                )
                
                # Fitting positions dimensions
                for key in B['X'].keys():
                    if 'positions' in key:
                        pos_tensor = B['X'][key]
                        if pos_tensor.dim() == 3 and pos_tensor.shape[-1] > config.spectra_pos_input_size:
                            B['X'][key] = pos_tensor.mean(dim=-1)
                
                
                # X and Y alignment
                for mode in B['Y'].keys():
                    if mode in B['X']:
                        x_len = B['X'][mode].shape[1]
                        y_len = B['Y'][mode].shape[1]
                        
                        if y_len != x_len:
                            
                            min_len = min(x_len, y_len)
                            
                            B['X'][mode] = B['X'][mode][:, :min_len, ...]
                            B['Y'][mode] = B['Y'][mode][:, :min_len, ...]
                            
                            pos_key = f"{mode}_positions"
                            if pos_key in B['X']:
                                B['X'][pos_key] = B['X'][pos_key][:, :min_len, ...]
                        
                
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
                        _, loss = model(B["X"], targets=B["Y"])
                        
                        # Skipping the batch if the loss is NaN
                        if torch.isnan(loss) or torch.isinf(loss):
                            skip_step = True
                            if ddp_rank == 0:
                                logger.warning(f"NaN/Inf detected in loss at iter {iter_num}, micro-step {micro_step}. Skipping batch.")
                            break
                        
                        # Scale loss
                        loss = loss / config.gradient_accumulation_steps
                        
                    
                    # If there is a NaN, backward is avoided
                    if not skip_step:
                        scaler.scale(loss).backward()
                        loss_accum_for_log += loss.item()
                    
                    # Scaled Backward
                    scaler.scale(loss).backward()
                    
                    # Accumulate the raw loss value for logging
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
                    
                
                if iter_num % config.log_interval == 0:
                    
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
                    eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                    
                    logger.info(
                        f"Iter {iter_num}/{config.max_iters} ({train_prog:.2%}) | "
                        f"Loss {loss_accum_for_log:.4f} | "
                        f"LR {lr:.4e} | "
                        f"Norm {grad_norm:.2f} |"
                        f"MFU {mfu_display*100:.2f}% (avg) | "
                        f"Mem {mem_usage:.2f}GB | "
                        f"dt {avg_dt*1000:.2f}ms (avg) | "
                        f"RT {running_str} (H:M:S) | "
                        f"ETA {eta_str} (H:M:S)" 
                    )
                    
                    if config.log_via_wandb and _WANDB_AVAILABLE:
                        wandb.log({
                            "iter": iter_num,
                            "train/loss": lossf,
                            "train/lr": lr,
                            "train/grad_norm": grad_norm,
                            "train/mfu": mfu_display * 100,
                            "train/mem_gb": mem_usage,
                            "train/time_ms": avg_dt * 1000,
                            "train/rt_hours": total_seconds / 3600,
                            "train/eta_hours": eta_seconds / 3600
                        })
                        
                        # CSV Writing
                        try:
                            with open(csv_path, "a") as f:
                                # Obtaining the current timestamp
                                timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                
                                # Writing into the CSV
                                f.write(f"{iter_num},{epoch_num},{train_prog},{timestamp_str},{loss_accum_for_log:.6f},,"
                                        f"{grad_norm:.4f},{is_clipped:.0f},{lr:.4e},{mfu_display*100:.2f},{mem_usage:.2f},"
                                        f"{avg_dt*1000:.2f},{running_str},{eta_str}\n"
                                )
                                
                        except Exception as e:
                            logger.error(f"CSV Write Error: {e}")

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
                    max_val_batches = config.eval_batches
                    
                    with torch.no_grad():
                        val_iter = iter(val_loader)
                        for _ in range(max_val_batches):
                            try:
                                vbatch = next(val_iter)
                            except StopIteration:
                                val_iter = iter(val_loader)
                                vbatch = next(val_iter)
                                
                            B_val = EuclidDESIDatasetArrow.process_modes(
                                batch_data=vbatch, 
                                modality_registry=registry, 
                                device=torch.device(device)
                            )
                            
                            with ctx:
                                _, v_loss = model(B_val["X"], targets=B_val["Y"])
                            val_losses.append(v_loss.item())
                    
                    val_loss = sum(val_losses) / len(val_losses)
                    logger.info(f"Validation Loss: {val_loss:.4f}")
                    
                    if config.log_via_wandb and _WANDB_AVAILABLE:
                        wandb.log({"val/loss": val_loss, "iter": iter_num})
                        
                    # CSV Write: Just Validation properties
                    try:
                        with open(csv_path, "a") as f:
                            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            # iter, prog, time, t_loss(empty), val_loss, grad(empty)...
                            f.write(f"{iter_num},{epoch_num},{train_prog:.4f},{timestamp_str},,{val_loss:.6f},,,,,,,,\n")
                    except Exception as e:
                        logger.error(f"Val CSV Write Error: {e}")
                    
                    
                    # EARLY STOPPING MECHANISM
                    
                    # If loss improve
                    if val_loss < best_val_loss:
                        # Reset the stop counter
                        best_val_loss = val_loss
                        early_stop_counter = 0
                        is_best = True
                        
                    else:
                        # Increasing the counter
                        early_stop_counter += 1
                        is_best = False
                        logger.info(f"Early Stopping Counter: {early_stop_counter}/{config.early_stopping_patience}")
                        
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
                        if is_best and config.checkpoint_save_type in ["best", "both"]:
                            checkpoint['best_val_loss'] = best_val_loss 
                            logger.info(f"New best model found ({val_loss:.4f}). Saving to {ckpt_path_best}")
                            torch.save(checkpoint, ckpt_path_best)
                
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
                        logger.info(f"Time limit reached ({elapsed_str} > {config.max_run_hours}). Saving checkpoint and exiting...")
                        
                        # Saving checkpoint
                        current_total_time = elapsed_seconds + accumulated_time
                        checkpoint = {
                            'model': model.module.state_dict() if ddp else model.state_dict(),
                            'modality_registry': registry,
                            'optimizer': optimizer.state_dict(),
                            'iter_num': iter_num,
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