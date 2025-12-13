"""
AstroPT Multimodal Training Scripts.

This script implements a Distributed Data Parallel (DDP) training loop for the 
AstroPT model, utilising the Euclid-DESI multimodal dataset (Arrow format).

Author: Victor Alonso Rodriguez
Date: December 2025
"""

from __future__ import annotations

import argparse
import inspect
import math
import logging
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, default_collate
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
    out_dir: str = "logs/astropt_100M_arrow_v1"
    data_dir: str = "/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"
    
    #--- 2. Data Loading ---#
    batch_size: int = 16            # Micro-batch size per GPU (what fits in VRAM)
    num_workers: int = 16           # Optimized for Arrow (CPU cores for loading)
    persistent_workers: bool = True # Keep workers active
    prefetch_factor: int = 2        # How many batches to preload per worker
    pin_memory: bool = True         # Faster transfer RAM -> VRAM
    spiral: bool = True             # Whether to apply spiral readout to galaxy images
    
    #--- 3. Model Architecture for the 100M Parameter Setup ---#
    n_layer: int = 12           # Depth of the Transformer
    n_head: int = 12            # Number of attention heads
    n_embd: int = 768           # Embedding dimension (width of the network)
    n_chan: int = 4             # Input channels: 1 VIS + 3 NISP (Y, J, H)
    block_size: int = 1024      # Context length (max tokens per sample)
    dropout: float = 0.0        # Regularization (0.0 for pretraining is standard)
    bias: bool = False          # Learnable bias in Linear layers (False is modern/faster)
    attn_type: str = "causal"   # Attention mechanism type
    
    #--- Multimodality Specifics ---#
    # Images
    images_size: int = 224          # Images side size in pixels
    images_patch_size: int = 16     # Side size in pixels of each patch in an image
    images_channels: int = 4        # Channels per image (VIS + NISP Y,J,H)
    images_loss_weight: float = 1.0 # Image importante for training
    
    # Spectra
    spectra_size: int = 7781            # Spectra total size
    spectra_patch_size: int = 10        # Patch size for each spectrum
    spectra_loss_weight: float = 1.0    # Spectra importance for training 
    
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
    always_save_checkpoint: bool = False    # If True, save every interval regardless of improvement

    #--- 7. System & Backend ---#
    device: str = "cuda"            # CPU/GPU device interface: cpu, cuda or mps
    dtype: str = "bfloat16"         # 'bfloat16' is best for A100 GPUs
    compile: bool = True            # PyTorch 2.0 compiler
    compile_mode: str = "default"   # Compilation mode
    backend: str = "nccl"           # Communication backend for DDP

    #--- 8. External Monitoring ---#
    log_via_wandb: bool = False             # Weight and bias (wandb) logging
    wandb_project: str = "AstroPT-Arrow"    # wandb project name
    wandb_run_name: Optional[str] = None    # Training name
    log_emissions: bool = False             # CodeCarbon logging


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


def normalise(x: torch.Tensor) -> torch.Tensor:
    """
    Standardizes the input tensor (Mean 0, Std 1) while preserving the original dtype.
    
    Calculations are performed in float32 for numerical stability, 
    then cast back to the input dtype (e.g., bfloat16).

    Args:
        x (torch.Tensor): Input tensor of shape (N, ...).

    Returns:
        torch.Tensor: Normalized tensor with the same dtype as input.
    """
    # Cast to float32 for precise mean/std calculation (avoids bfloat16 underflow)
    x_32 = x.float()
    
    # Calculate standard deviation and mean across the feature dimension (dim=1)
    std, mean = torch.std_mean(x_32, dim=1, keepdim=True)
    
    # Apply Z-score normalization: (Value - Mean) / Std
    # Added epsilon (1e-8) to prevent division by zero
    x_norm = (x_32 - mean) / (std + 1e-8)
    
    # Cast back to the original data type (e.g., bfloat16) to save memory
    return x_norm.to(x.dtype)


def data_transforms() -> Dict[str, Any]:
    """
    Returns the dictionary of transformations expected by the Dataset.
    
    Defines specific preprocessing pipelines for 'images' and 'spectra'.

    Returns:
        Dict[str, Any]: Keys match the modality names, values are torchvision Transforms.
    """
    return {
        "images": transforms.Compose([
            transforms.Lambda(normalise)
        ]),
        "spectra": transforms.Compose([
            transforms.Lambda(normalise)
        ])
    }


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
            pos_input_size=1,  
            loss_weight=config.images_loss_weight,
            embed_pos=True,
        ),
        ModalityConfig(
            name="spectra",
            input_size=config.spectra_patch_size,
            patch_size=config.spectra_patch_size,
            pos_input_size=config.spectra_patch_size, 
            loss_weight=config.spectra_loss_weight,
            embed_pos=False,
        ),
    ]
    
    # Instantiate the Registry
    registry = ModalityRegistry(modalities)
    
    # Prepare data transformations
    data_tf = data_transforms()
    
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
    
    # Instantiate the Model (CPU): Configuration and registry modality
    model = GPT(config, registry)
    logger.info(f"Initializing GPT Model with {config.n_layer} layers")

    # Move to Selected Device
    model.to(device)

    # Pythorch Compilation
    if config.compile:
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model, mode=config.compile_mode) 

    # DDP Wrapping
    if ddp:
        
        # Loal RANK for each process
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
        betas=config.betas, 
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
    if it < config.warmup_iters:
        # Linear increase from 0 to learning_rate
        return config.learning_rate * (it + 1) / (config.warmup_iters + 1)
    
    # Post-Decay Phase (Constant Min LR)
    if it > config.lr_decay_iters:
        return config.min_lr
    
    # Cosine Decay Phase
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    
    # Calculate the cosine coefficient
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    # Apply the coefficient to the range [min_lr, learning_rate]
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)



if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # default config values designed to test run a 100M parameter model on galaxy imagery and spectra
    # look at `config/astropt*.py` for a prod run example
    out_dir = "logs/astropt0100M_multimodal_250K_T1_20251210"
    eval_interval = 1000
    log_interval = 100
    checkpoint_interval = 2000
    assert checkpoint_interval % eval_interval == 0
    eval_iters = 100
    eval_only = False  # if True, script exits right after the first eval
    always_save_checkpoint = (
        False  # if True, always save a checkpoint at each checkpoint_interval
    )
    init_from = "scratch"  # 'scratch' or 'resume'
    # data
    gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
    batch_size = 16  # if gradient_accumulation_steps > 1, this is the micro-batch size
    spiral = True  # do we want to process the galaxy patches in spiral order?
    block_size = 1024
    image_size = 224
    num_workers = 8 # 8 for TiedeHPC #32  # 64
    # astroPT model
    n_layer = 12
    n_head = 12
    n_embd = 768
    n_chan = 4  # 3 imagery bands: r, i, z for jpeg, 1 imagery band for FITS
    dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    # NB dropout is NOT implemented for flex attention
    bias = False  # do we use bias inside LayerNorm and Linear layers?
    # Define modalities configuration
    modalities = [
        ModalityConfig(
            name="images",
            input_size=16 * 16 * n_chan,
            patch_size=16,
            pos_input_size=1,
            loss_weight=1.0,
            embed_pos=True,
        ),
        ModalityConfig(
            name="spectra",
            input_size=10,
            patch_size=10, #256,
            pos_input_size=10,
            loss_weight=1.0,
            embed_pos=False,
        ),
    ]
    # Create modality registry
    modality_registry = ModalityRegistry(modalities)
    # Choose tokenisers from "affine" and "aim"
    tokeniser = "aim"
    # adamw optimizer
    # we follow the same schedule here as Chinchilla
    learning_rate = 6e-4  # max learning rate
    max_iters = (
        30000  # total number of training iterations for one pass over our dataset
    )
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True  # whether to decay the learning rate
    warmup_iters = 2000  # how many steps to warm up for
    lr_decay_iters = 27000 * 1.1  # should be ~= max_iters per Chinchilla
    min_lr = (
        learning_rate / 10
    )  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    attn_type = "causal"
    # DDP settings
    backend = "nccl"  # 'nccl', 'gloo', etc.
    # system
    device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = "bfloat16"  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = True  # use PyTorch 2.0 to compile the model to be faster
    log_via_wandb = False
    wandb_project = None
    # -----------------------------------------------------------------------------
    config_keys = [
        k
        for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str))
    ]
    exec(
        open("src/astropt/configurator.py").read()
    )  # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys}  # will be useful for logging
    # -----------------------------------------------------------------------------

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = (
            ddp_rank == 0
        )  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        assert gradient_accumulation_steps % torch.cuda.device_count() == 0
        gradient_accumulation_steps //= torch.cuda.device_count()
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = (
        gradient_accumulation_steps
        * ddp_world_size
        * batch_size
        * block_size
        * len(modalities)
    )
    if master_process:
        if log_via_wandb:
            print("wandb detected, gonna log to that")            
        if log_emissions:
            print("codecarbon detected, will log emissions")
        print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in device else "cpu"
    )  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    # dataset init
    transforms = {"images": data_transforms(),
                  "spectra": data_transforms()}
    
    
    # training dataset and dataloader
    ARROW_FOLDER_ROOT = "/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"
    
    print("Using EuclidDESIdataset (MSc Thesis Version)...")




    # Splitting datasets
    tds = EuclidDESIDatasetArrow(
        arrow_folder_root=ARROW_FOLDER_ROOT,
        split="train",                       
        modality_registry=modality_registry, 
        spiral=spiral,                  
        stochastic=True,
        transform=transforms
    )
    
    vds = EuclidDESIDatasetArrow(
        arrow_folder_root=ARROW_FOLDER_ROOT,
        split="test",                        
        modality_registry=modality_registry,
        spiral=spiral,
        stochastic=False,
        transform=transforms
    )

    tdl = iter(
        DataLoader(
            tds,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            shuffle=True,
            collate_fn=robust_collate_fn,
            drop_last=True
        )
    )
    vdl = iter(
        DataLoader(
            vds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=robust_collate_fn
        )
    )

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # model init
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        n_chan=n_chan,
        block_size=block_size,
        dropout=dropout,
        modalities=modalities,
        attn_type=attn_type,
    )

    if init_from == "scratch":
        # init a new model from scratch
        if master_process:
            print("initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf, modality_registry, master_process=master_process)
    if init_from == "resume":
        if master_process:
            print(f"resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        # NOTE had to remove 'bias' key here -- where does it go?!
        for k in ["n_layer", "n_head", "n_embd", "block_size"]:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf, modality_registry, master_process=master_process)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

    # logging via wandb if available
    # this is here so we can get the number of params from model()
    if log_via_wandb and master_process:
        if wandb_project is None:
            wandb_project = f"TFM: AstroPT-{model.get_num_params() / 1e6:06.1f}M"
        
        try:
            print("Trying to connect with wandb (Online mode)...")
            wandb.init(
                project=wandb_project,
                config=config,
                settings=wandb.Settings(init_timeout=300)
            )
            print('Susccesfully online conexion with wandb')
            
        except:
            print('Error while conecting with wandb\nChanging to offline mode')
            
            try:
            
                if wandb.run is not None:
                    wandb.finish()
                    wandb.init(
                        project=wandb_project,
                        config=config,
                        mode="offline"
                    )
                    
            except Exception as e_offline:
                print(f'Fatal error while initializating wandb: {e_offline}')
                print('wand unabled')
                log_via_wandb = False


    # write config and important information to log file
    with open(f"{out_dir}/hparams.txt", "w") as fi:
        fi.write(f"AstroPT-{model.get_num_params() / 1e6:06.1f}M\n")
        fi.write(f"time: {int(time.time())}\n")
        for k, v in config.items():
            fi.write(f"{k}: {v}\n")

    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args["block_size"] = (
            block_size  # so that the checkpoint will have the right value
        )
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))

    # optimizer
    optimizer = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device_type
    )
    if init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free up memory

    # compile the model
    if compile:
        if master_process:
            print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        if master_process:
            print("Wrapping in DDP")
        # Note to future people: we had to turn off optimize_ddp due to a
        # torch compiler error when running DDP. This _may_ be fixed in a
        # future torch version so check periodically. I tested this on:
        # 2.6.0.dev20241126+cu124
        torch._dynamo.config.optimize_ddp = False
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(tds, vds):
        out = {}
        model.eval()
        for ds, split in zip(
            [tds, vds],
            ["train", "val"],
        ):
            # Creating an internal dl to avoid StopIteration error
            dl = iter(
                DataLoader(
                    ds,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            )
            
            out[split] = {}
            
            # For small datasets (as the one with just N=100)
            n_iters = min(eval_iters, len(ds) // batch_size) 
            if n_iters == 0:
                n_iters = 1
            
            losses = torch.zeros(n_iters)
            for k in range(n_iters):
                try:
                    B = EuclidDESIDatasetArrow.process_modes(next(dl), modality_registry, device)
                    with ctx:
                        logits, loss = model(B["X"], targets=B["Y"])
                    losses[k] = loss.item()
                except StopIteration:
                    losses = losses[:k] 
                    break
                
            out[split]["dummy"] = losses.mean()
        model.train()
        return out

    @torch.no_grad()
    def validate(iter_num, out_dir, tds, vds):
        model.eval()

        t_dl_viz = iter(DataLoader(tds, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True))
        v_dl_viz = iter(DataLoader(vds, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True))

        for dl, split in zip([t_dl_viz, v_dl_viz], ["train", "val"]):
            f, axs = plt.subplots(8, 4, figsize=(6, 12), constrained_layout=True)
            
            try:
                batch = next(dl)
            except StopIteration:
                tdl = iter(torch.utils.data.DataLoader(
                    tds, 
                    batch_size=batch_size, 
                    num_workers=16, 
                    pin_memory=True, 
                    shuffle=True,
                    collate_fn=robust_collate_fn, 
                    drop_last=True
                ))
                batch = next(tdl)
            
            if batch is None:
                print("None Batch. Skipping.")
                continue
            

            B = EuclidDESIDatasetArrow.process_modes(batch, modality_registry, device)
            
            with ctx:
                P, loss = model(B["X"], B["Y"])
                if "images" in modality_registry.names():
                    Yim = B["Y"]["images"].to(device)
                    b, t, c = Yim.size()
                    zero_block = torch.zeros((b, 1, c)).to(device)
                    Yim = torch.cat((zero_block, Yim), dim=1)
                    
                    if spiral:
                        Yim = torch.stack([vds.antispiralise_image(yy) for yy in Yim])
                        
                    im_patch = modality_registry.get_config("images").patch_size
                    Yim = einops.rearrange(
                        Yim,
                        "b (h w) (p1 p2 c) -> b (h p1) (w p2) c",
                        p1=im_patch,
                        p2=im_patch,
                        h=image_size // im_patch,
                        w=image_size // im_patch,
                    )
                    Pim = torch.cat((zero_block, P["images"]), dim=1)
                    
                    if spiral:
                        Pim = torch.stack([vds.antispiralise_image(pp) for pp in Pim])
                        
                    Pim = einops.rearrange(
                        Pim,
                        "b (h w) (p1 p2 c) -> b (h p1) (w p2) c",
                        p1=im_patch,
                        p2=im_patch,
                        h=image_size // im_patch,
                        w=image_size // im_patch,
                    )

                    for ax, p, y in zip(
                        axs, Pim.to(float).cpu().numpy(), Yim.to(float).cpu().numpy()
                    ):
                        ax[0].imshow(np.clip(y, 0, 1))
                        ax[1].imshow(np.clip(p, 0, 1))
                        ax[0].axis("off")
                        ax[1].axis("off")

                    if log_via_wandb:
                        wandb.log(
                            {
                                "Y": wandb.Image(Yim.swapaxes(1, -1)),
                                "P": wandb.Image(Pim.swapaxes(1, -1)),
                            }
                        )

            if "spectra" in modality_registry.names():
                Ysp = B["Y"]["spectra"]
                Psp = P["spectra"]
                for ax, p, y in zip(
                    axs, Psp.to(float).cpu().numpy(), Ysp.to(float).cpu().numpy()
                ):

                    ax[2].plot(np.concatenate(y, axis=0))
                    ax[2].plot(np.concatenate(p, axis=0)) 
                    ax[3].plot(np.concatenate(p, axis=0))
            f.savefig(
                os.path.join(out_dir, f"{iter_num:06d}_{split}.jpg"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close(f)
        model.train()

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    # training loop
    if master_process:
        print("starting training...")
    B = EuclidDESIDatasetArrow.process_modes(
        next(tdl), modality_registry, device, True
    )  # fetch the very first batch
    t0 = time.time()
    dts = []
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # unwrap DDP container if needed
    running_mfu = -1.0
    if log_emissions and master_process:
        tracker = EmissionsTracker(
            output_dir=out_dir,
            log_level="error",
            save_to_file=True,
            on_csv_write="update",
        )
        tracker.start()
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            validate(iter_num, out_dir, tds, vds)
            losses = estimate_loss(tds, vds)
            val_loss = np.mean(list(losses["val"].values()))
            print(
                f"iter {iter_num}:\ntrain loss:\n{losses['train']}\nval loss:\n{losses['val']}"
            )
            with open(os.path.join(out_dir, "loss.txt"), "a") as fi:
                if fi.tell() == 0:  # check if a new file and write header if so
                    train_head_str = ",".join(
                        map(lambda x: str(x) + "_train", losses["train"].keys())
                    )
                    valid_head_str = ",".join(
                        map(lambda x: str(x) + "_valid", losses["val"].keys())
                    )
                    fi.write(f"iter_num,{train_head_str},{valid_head_str},lr,mfu\n")
                train_loss_str = ",".join(
                    map(lambda x: str(x.item()), losses["train"].values())
                )
                valid_loss_str = ",".join(
                    map(lambda x: str(x.item()), losses["val"].values())
                )
                fi.write(
                    f"{iter_num},{train_loss_str},{valid_loss_str},{lr},{running_mfu * 100}\n"
                )
            if log_via_wandb:
                wandb.log({"valloss": losses["val"]}, step=iter_num)
            if iter_num != 0:
                loss_df = pd.read_csv(os.path.join(out_dir, "loss.txt"))
                f, axs = plt.subplots(
                    1,
                    len(losses["train"]) + 1,
                    figsize=(12, 4),
                    constrained_layout=True,
                )
                axs.ravel()[0].set_title("mean")
                axs.ravel()[0].plot(
                    loss_df["iter_num"],
                    loss_df.filter(like="train").mean(axis=1),
                    label="train",
                )
                axs.ravel()[0].plot(
                    loss_df["iter_num"],
                    loss_df.filter(like="valid").mean(axis=1),
                    label="valid",
                )
                for ax, train_loss, valid_loss in zip(
                    axs.ravel()[1:],
                    loss_df.filter(like="train"),
                    loss_df.filter(like="valid"),
                ):
                    ax.set_title(train_loss)
                    ax.plot(loss_df["iter_num"], loss_df[train_loss], label="train")
                    ax.plot(loss_df["iter_num"], loss_df[valid_loss], label="valid")
                [ax.set_yscale("log") for ax in axs.ravel()]
                [ax.legend() for ax in axs.ravel()]
                f.savefig(os.path.join(out_dir, "loss.png"))
                plt.close(f)

            if val_loss < best_val_loss or always_save_checkpoint:
                best_val_loss = val_loss
                if iter_num > 0:
                    model_state = raw_model.state_dict()
                    checkpoint = {
                        "model": model_state,
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                        "modality_registry": modality_registry,
                    }
                    if master_process:
                        print(f"saving checkpoint to {out_dir}")
                    if always_save_checkpoint:
                        torch.save(
                            checkpoint, os.path.join(out_dir, f"{iter_num:06d}_ckpt.pt")
                        )
                    else:
                        torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
        if iter_num == 0 and eval_only:
            break

        try:
            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(gradient_accumulation_steps):
                if ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = (
                        micro_step == gradient_accumulation_steps - 1
                    )

                with ctx:
                    logits, loss = model(B["X"], targets=B["Y"])
                
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                B = EuclidDESIDatasetArrow.process_modes(
                next(tdl), modality_registry, device, True
                )  # fetch the very first batch
                
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()    
                
        except StopIteration:
            print("End of the epoch. Restarting training dataloader")
            # Creating the next tdl in case it is missing
            tdl = iter(
                DataLoader(
                    tds,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            )
            B = EuclidDESIDatasetArrow.process_modes(next(tdl), modality_registry, device, True)
            
                
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        dts.append(dt)
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(
                    batch_size * gradient_accumulation_steps, dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            if log_via_wandb:
                wandb.log({"loss": lossf, "time": dt}, step=iter_num)
            if log_emissions:
                emissions: float = tracker.flush()
                print(
                    f"iter {iter_num}: loss {lossf:.6f}, time {np.mean(dts) * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%, tot co2 {emissions:.1f}kg"
                )
            else:
                print(
                    f"iter {iter_num}: loss {lossf:.6f}, time {np.mean(dts) * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%"
                )
            dts = []

        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            if log_emissions:
                emissions: float = tracker.stop()
                if master_process:
                    print(emissions)
            break

    if ddp:
        destroy_process_group()
    if log_via_wandb:
        wandb.finish()
