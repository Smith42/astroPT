#!/usr/bin/env python3
"""
AstroPT Multimodal Training Script (Production Release)

This script acts as the clean entry point for training the AstroPT model.
It relies on the modularized `src/astropt/config.py` for hyperparameters
and `src/astropt/trainer.py` for the core training loop.
"""

import os
import sys
import logging
from pathlib import Path
from transformers import HfArgumentParser

from astropt.config import TrainingConfig
from astropt.training_utils import (
    ddp_setup, 
    logging_setup, 
    project_directories_setup, 
    validate_runtime_flags,
    save_config_json,
    extract_extra_cli_args
)
from astropt.trainer import Trainer

def main():
    # Get valid dataclass fields
    from dataclasses import fields as dc_fields
    valid_fields = {f.name for f in dc_fields(TrainingConfig)}
    
    # 1. Filter and parse Command Line Arguments
    cleaned_argv, extra_cli_args = extract_extra_cli_args(valid_fields)
    parser = HfArgumentParser((TrainingConfig,))
    
    # Support for YAML/JSON config file + CLI overrides
    if len(cleaned_argv) > 0 and (cleaned_argv[0].endswith(".json") or cleaned_argv[0].endswith(".yaml")):
        config_path = os.path.abspath(cleaned_argv[0])
        cleaned_argv.pop(0) # Remove the config file from args
        
        # Load the file into a dictionary first
        if config_path.endswith(".yaml"):
            import yaml
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            import json
            with open(config_path, "r") as f:
                config_dict = json.load(f)
        
        if config_dict is None:
            config_dict = {}

        # Remove extra keys before parsing to avoid HfArgumentParser failure/warnings
        dataclass_config_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        extra_config_dict = {k: v for k, v in config_dict.items() if k not in valid_fields}
        
        config, = parser.parse_dict(dataclass_config_dict, allow_extra_keys=True)
        
        # Apply extra config keys from file
        for k, v in extra_config_dict.items():
            setattr(config, k, v)
        
        # For the CLI overrides, we parse them into a separate object
        cli_config, _ = parser.parse_args_into_dataclasses(args=cleaned_argv, return_remaining_strings=True)
        
        for arg in cleaned_argv:
            if arg.startswith("--"):
                field_name = arg.split("=")[0].lstrip("-").replace("-", "_")
                if hasattr(cli_config, field_name):
                    setattr(config, field_name, getattr(cli_config, field_name))
    else:
        # Standard parsing
        config, _ = parser.parse_args_into_dataclasses(args=cleaned_argv, return_remaining_strings=True)

    # Set extra keys from CLI on the config object
    for k, v in extra_cli_args.items():
        setattr(config, k, v)
    
    # 2. Validate Custom Flags
    validate_runtime_flags(config)
    
    # 3. Setup Distributed Data Parallel (DDP)
    ddp, ddp_rank, ddp_world_size, device = ddp_setup()
    
    # 4. Initialize Dynamic Paths (sets up config.train_dir, train_date, etc.)
    config.__post_init__()
    
    # 5. Create Directory Structure
    train_dir, weights_dir, embeddings_dir, plots_dir, logs_dir, analysis_dir = project_directories_setup(config.train_dir)
    
    # 6. Setup Logging
    logger = logging_setup(config, ddp_rank, logs_dir)
    
    # Save the configuration JSON to the weights directory (Rank 0 only)
    save_config_json(config, ddp_rank, weights_dir)
    
    # 7. Instantiate and Run Trainer
    trainer = Trainer(
        config=config,
        ddp=ddp,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        device=device,
        weights_dir=weights_dir,
        logs_dir=logs_dir
    )
    
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed due to an exception: {e}", exc_info=True)
        if ddp:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        sys.exit(1)

    # 8. Cleanup
    if ddp:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
