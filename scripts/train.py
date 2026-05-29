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
    save_config_json
)
from astropt.trainer import Trainer

def main():
    # 1. Parse Command Line Arguments
    parser = HfArgumentParser((TrainingConfig,))
    
    # Support for YAML/JSON config file + CLI overrides
    if len(sys.argv) > 1 and (sys.argv[1].endswith(".json") or sys.argv[1].endswith(".yaml")):
        config_path = os.path.abspath(sys.argv[1])
        sys.argv.pop(1) # Remove the config file from args
        
        # Load the file into a dictionary first
        if config_path.endswith(".yaml"):
            import yaml
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            import json
            with open(config_path, "r") as f:
                config_dict = json.load(f)
        
        # Now parse the CLI arguments. 
        # To merge correctly, we use parse_dict with the file data, 
        # then we manually update with whatever comes from the CLI.
        config, = parser.parse_dict(config_dict, allow_extra_keys=True)
        
        # For the CLI overrides, we parse them into a separate object 
        # and only update fields that were EXPLICITLY passed (not defaults)
        # A simple way is to check sys.argv for the flags
        cli_config, _ = parser.parse_args_into_dataclasses(args=sys.argv[1:], return_remaining_strings=True)
        
        for arg in sys.argv[1:]:
            if arg.startswith("--"):
                field_name = arg.split("=")[0].lstrip("-")
                if hasattr(cli_config, field_name):
                    setattr(config, field_name, getattr(cli_config, field_name))
    else:
        # Standard parsing
        config, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
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
