"""
AstroPT Multimodal Training Scripts.

This script implements a Distributed Data Parallel (DDP) training loop for the 
AstroPT model, utilising the Euclid-DESI multimodal dataset (Arrow format).

Author: Victor Alonso Rodriguez
Date: March 2026
"""


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

from astropt.config import TrainingConfig
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
    

def project_directories_setup(base_dir: str | Path) -> Tuple[Path, Path, Path, Path, Path, Path]:

    train_dir = Path(base_dir)
    weights_dir = train_dir / "weights"
    embeddings_dir = train_dir / "embeddings"
    plots_dir = train_dir / "plots"
    logs_dir = train_dir / "logs"
    analysis_dir = train_dir / "analysis"
    
    for directory in [train_dir, weights_dir, embeddings_dir, plots_dir, logs_dir, analysis_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[INFO]: Created {directory}")

    return train_dir, weights_dir, embeddings_dir, plots_dir, logs_dir, analysis_dir

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

    # V4: Contrastive Alignment Validation
    if config.use_contrastive_alignment:
        if config.clip_loss_weight < 0.0:
            raise ValueError("clip_loss_weight must be >= 0.0.")
        if config.clip_temperature <= 0.0:
            raise ValueError("clip_temperature must be > 0.0.")
        if config.clip_fusion_layer >= config.n_layer:
            raise ValueError(
                f"clip_fusion_layer ({config.clip_fusion_layer}) must be < n_layer ({config.n_layer})."
            )


def _normalize_param_name(param_name: str) -> str:
    """Normalizes wrapped parameter names from DDP/compile wrappers."""
    clean_name = param_name
    for prefix in ("module._orig_mod.", "module.", "_orig_mod."):
        clean_name = clean_name.replace(prefix, "")
    return clean_name


def _param_branch_name(param_name: str) -> str:
    """Maps a parameter name to an optimizer/diagnostics branch dynamically."""
    name = _normalize_param_name(param_name)
    # Match any encoder/decoder/embedder.[ModalityName]
    for part in ("encoders.", "decoders.", "embedders."):
        if part in name:
            # Extract the modality name after the dot
            # e.g. encoders.EuclidImage.weight -> EuclidImage
            suffix = name.split(part)[1]
            mod_name = suffix.split(".")[0]
            return mod_name
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

    # If the model provided aligned targets (due to token mixing), use them
    aligned_targets = outputs.get("_aligned_targets", targets)
    
    prefix_modality = outputs.get("_prefix_modality")
    
    for mod_name, pred in outputs.items():
        if mod_name.startswith("_") or mod_name not in aligned_targets:
            continue
            
        # Skip the prefix modality because its loss is not calculated (bidirectional context)
        if mod_name == prefix_modality:
            continue
            
        target = aligned_targets[mod_name]

        # Auto-align shapes when token_mixing causes length differences
        if pred.shape[1] != target.shape[1]:
            min_len = min(pred.shape[1], target.shape[1])
            pred = pred[:, :min_len]
            target = target[:, :min_len]

        # Alignment is already handled by the model's routing logic
        pred = pred.contiguous()
        target = target.contiguous()

        # Compute reconstruction loss
        if config.loss_type in ["l1", "mae"]:
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

def compute_cross_reconstruction_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    target_key: str,
    config: "TrainingConfig",
) -> float:
    """Compute the cross-reconstruction loss for a single target modality.

    Applies the base cross_reconstruction_weight to all directions, plus an
    additional spectral boost (cross_reconstruction_weight_to_spectra) when
    the target modality is the spectral modality (spectral-first design).

    Args:
        outputs:    Model output dict (may contain '_aligned_targets').
        targets:    Original targets dict from the batch.
        target_key: The modality key we zeroed out in the input (i.e., the
                    modality we want to reconstruct from context).
        config:     TrainingConfig instance with weight parameters.

    Returns:
        Weighted scalar loss (float) for the target modality reconstruction.
        Returns 0.0 if target_key is not found in model outputs.
    """
    per_mod = _compute_modality_losses(outputs, targets, config)
    raw_loss = per_mod.get(target_key, None)
    if raw_loss is None:
        return 0.0

    # Base cross-reconstruction weight
    weight = getattr(config, 'cross_reconstruction_weight', 1.0)

    # Additional boost when reconstructing the spectral modality (spectral-first design)
    spectral_key = getattr(config, 'spectral_modality_key', 'spectra')
    if target_key == spectral_key:
        weight *= getattr(config, 'cross_reconstruction_weight_to_spectra', 1.0)

    return raw_loss * weight


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
    available = [m for m in batch["X"] if m in batch["Y"] and not m.endswith("_positions") and not m.startswith("_")]
    if len(available) < 1:
        return "none"

    mode = config.modality_dropout_mode.lower().strip()
    if mode == "none":
        return "none"
    
    if mode == "random":
        chosen = str(np.random.choice(available))
    else:
        # Robust matching: 
        # 1. Exact match
        # 2. Case-insensitive substring match (e.g. 'spectra' matches 'DESISpectrum')
        matching = [m for m in available if mode == m.lower() or mode in m.lower()]
        chosen = matching[0] if matching else "none"

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

def summarize_prefix_modality(prefix_counts: Dict[str, int]) -> str:
    """Returns a string summary of which modalities acted as Prefix."""
    prefixes = [k for k, v in prefix_counts.items() if v > 0]
    if len(prefixes) > 1:
        return "mixed"
    if len(prefixes) == 1:
        return prefixes[0]
    return "none"


def compute_branch_grad_norms(model: torch.nn.Module) -> Dict[str, float]:
    """Computes gradient L2 norms for all branches dynamically."""
    grad_sq = {}

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        branch = _param_branch_name(name)
        if branch not in grad_sq:
            grad_sq[branch] = 0.0
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
    
    # Modality Factory: maps modality strings to their Processor and specific Config logic.
    # This allows adding new survey types (e.g., HSCImage, SED) without changing create_dataloaders.
    from astropt.processors.image import EuclidImageProcessor
    from astropt.processors.spectrum import DESISpectrumProcessor
    from astropt.dataloader_multimodal import MultimodalDatasetArrow
    
    MODALITY_PROCESSOR_MAP = {
        "EuclidImage": EuclidImageProcessor,
        "DESISpectrum": DESISpectrumProcessor,
    }
    
    # Backward compatibility mapping for old generic names
    LEGACY_MAP = {
        "images": "EuclidImage",
        "spectra": "DESISpectrum"
    }

    # Configure ModalityRegistry
    modalities = []
    tf_kwargs = {}
    train_processors = {}
    val_processors = {}
    
    active_modalities = config.modalities
    
    # Process each active modality defined in the config
    for mod_name in active_modalities:
        # Resolve actual class name if legacy key is used
        actual_mod_name = LEGACY_MAP.get(mod_name, mod_name)
        
        if actual_mod_name not in MODALITY_PROCESSOR_MAP:
            logging.warning(f"Modality '{mod_name}' (resolved to '{actual_mod_name}') not found in registry. Skipping.")
            continue
            
        processor_cls = MODALITY_PROCESSOR_MAP[actual_mod_name]
        
        # Hyperparameter lookup logic
        # We try: 1. Specific surveyed name (e.g. euclid_images_patch_size)
        #         2. Surveyed name without underscore (e.g. euclidimages_patch_size)
        #         3. Generic family name (e.g. images_patch_size)
        
        family_prefix = "images" if "Image" in actual_mod_name else "spectra"
        # Standardize prefixes for lookup
        # 1. Exact class name (e.g. EuclidImage)
        survey_prefix_exact = actual_mod_name 
        # 2. Lowercase with survey type (e.g. euclid_images)
        survey_prefix_std = actual_mod_name.lower().replace("image", "_images").replace("spectrum", "_spectra").strip("_")
        # 3. Concatenated lowercase (e.g. euclidimages)
        survey_prefix_alt = survey_prefix_std.replace("_", "")
        
        def get_mod_param(attr_suffix, default_val):
            # Try 1: Exact class name (e.g. EuclidImage_patch_size)
            val = getattr(config, f"{survey_prefix_exact}_{attr_suffix}", None)
            if val is not None: return val
            # Try 2: Standardized prefix (e.g. euclid_images_patch_size)
            val = getattr(config, f"{survey_prefix_std}_{attr_suffix}", None)
            if val is not None: return val
            # Try 3: Alternative prefix (e.g. euclidimages_patch_size)
            val = getattr(config, f"{survey_prefix_alt}_{attr_suffix}", None)
            if val is not None: return val
            # Try 4: Fallback to generic family (e.g. images_patch_size)
            return getattr(config, f"{family_prefix}_{attr_suffix}", default_val)

        if "Image" in actual_mod_name:
            p_size = get_mod_param("patch_size", config.images_patch_size)
            input_size = p_size * p_size * get_mod_param("channels", config.images_channels)
            pos_in = get_mod_param("pos_input_size", config.images_pos_input_size)
            l_weight = get_mod_param("loss_weight", config.images_loss_weight)
            e_pos = get_mod_param("embed_pos", config.images_embed_pos)
            
            mod_config = ModalityConfig(
                name=mod_name,
                input_size=input_size,
                patch_size=p_size,
                pos_input_size=pos_in,
                loss_weight=l_weight,
                embed_pos=e_pos,
                encoder_type="aim"
            )
            modalities.append(mod_config)
            
            # TRAINING PROCESSOR (with augmentation)
            train_processors[mod_name] = processor_cls(
                filters=get_mod_param("filters", None),
                spiral=config.spiral,
                stochastic=get_mod_param("mask", config.images_mask),
                mask_prob=get_mod_param("mask_prob", config.images_mask_prob),
                stage="train"
            )
            # VALIDATION PROCESSOR (without augmentation)
            val_processors[mod_name] = processor_cls(
                filters=get_mod_param("filters", None),
                spiral=config.spiral,
                stochastic=get_mod_param("mask", config.images_mask),
                mask_prob=get_mod_param("mask_prob", config.images_mask_prob),
                stage="val"
            )
            
            # Transforms: used by data_transforms()
            tf_kwargs.update({
                f'norm_type_{mod_name}': get_mod_param("norm_type", config.images_norm_type),
                f'norm_scaler_{mod_name}': get_mod_param("norm_scaler", config.images_norm_scaler),
                f'norm_const_{mod_name}': get_mod_param("norm_const", config.images_norm_const),
            })

        elif "Spectrum" in actual_mod_name:
            p_size = get_mod_param("patch_size", config.spectra_patch_size)
            pos_in = get_mod_param("pos_input_size", config.spectra_pos_input_size)
            l_weight = get_mod_param("loss_weight", config.spectra_loss_weight)
            e_pos = get_mod_param("embed_pos", config.spectra_embed_pos)

            mod_config = ModalityConfig(
                name=mod_name,
                input_size=p_size,
                patch_size=p_size,
                pos_input_size=pos_in,
                loss_weight=l_weight,
                embed_pos=e_pos,
                encoder_type="aim"
            )
            modalities.append(mod_config)
            
            # Spectra processors are stateless regarding stage for now
            proc = processor_cls(
                stochastic=get_mod_param("mask", config.spectra_mask),
                mask_prob=get_mod_param("mask_prob", config.spectra_mask_prob),
                inverse=get_mod_param("inverse", config.spectra_inverse)
            )
            train_processors[mod_name] = proc
            val_processors[mod_name] = proc
            
            tf_kwargs.update({
                f'norm_type_{mod_name}': get_mod_param("norm_type", config.spectra_norm_type),
                f'norm_scaler_{mod_name}': get_mod_param("norm_scaler", config.spectra_norm_scaler),
                f'norm_const_{mod_name}': get_mod_param("norm_const", config.spectra_norm_const),
            })
    
    # Instantiate the Registry
    registry = ModalityRegistry(modalities)
    
    # Use data augmentation for training
    train_stage = 'train' if config.use_aug else 'val'
        
    # 4. Instantiate transforms dynamically unpacking the dictionary
    train_tf = MultimodalDatasetArrow.data_transforms(
        stage=train_stage, 
        **tf_kwargs
    )
    
    val_tf = MultimodalDatasetArrow.data_transforms(
        stage='val', 
        **tf_kwargs
    )
    
    # Activating the logger object
    logger = logging.getLogger("AstroPT")
    
    # Informational log (Only printed by the Master Process to avoid spam)
    if not ddp or (ddp and int(os.environ.get("RANK", 0)) == 0):
        logger.info(f"Loading data from: {config.data_dir}")

    # Instantiate Train Dataset 
    train_dataset = MultimodalDatasetArrow(
        arrow_folder_root=config.data_dir,
        split="train",
        processors=train_processors,
        modality_registry=registry, 
        transform=train_tf,
        applied_filters=config.applied_filters,
        metadata_path=config.metadata_path
    )
    
    # Instantiate Validation/Test Dataset 
    val_dataset = MultimodalDatasetArrow(
        arrow_folder_root=config.data_dir,
        split="test", 
        processors=val_processors,
        modality_registry=registry,
        transform=val_tf,
        applied_filters=config.applied_filters,
        metadata_path=config.metadata_path
    )

    # Logging filtering statistics
    if not ddp or (ddp and int(os.environ.get("RANK", 0)) == 0):
        logger.info("Dataset statistics after dynamic filtering:")
        if train_dataset.initial_len > 0:
            logger.info(f"  [Train] Loaded {train_dataset.initial_len} original samples -> {len(train_dataset)} active samples processed (retained {len(train_dataset)/train_dataset.initial_len:.1%})")
        else:
            logger.info(f"  [Train] Loaded 0 original samples")
        if val_dataset.initial_len > 0:
            logger.info(f"  [Validation] Loaded {val_dataset.initial_len} original samples -> {len(val_dataset)} active samples processed (retained {len(val_dataset)/val_dataset.initial_len:.1%})")
        else:
            logger.info(f"  [Validation] Loaded 0 original samples")

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

