import os
import json
import logging
import torch
from typing import Tuple, Dict, Any, Union, Optional
from dataclasses import fields
from huggingface_hub import hf_hub_download

from astropt.model import GPT, GPTConfig, ModalityRegistry

# Defining a logger
logger = logging.getLogger(__name__)

def load_astropt(
    repo_id="smith42/astropt_sparse",
    path="astropt/p16k10",
    weights_filename="ckpt.pt",
    use_llm_backbone=False,
    llm_model_name=None,
):
    """
    Load an AstroPT model.

    Args:
        repo_id: The Hugging Face repo ID (e.g., "smith42/astropt_sparse"). If this is None we assume that we are loading a local model.
        path: Subdirectory containing the model (e.g., "sparse" or "dense")
        weights_filename: Name of the weights file

    Returns:
        Loaded model
    """
    from astropt.model import GPT, GPTConfig

    if repo_id is not None:
        # Ping root config.json to count as a download at the repo level
        try:
            _ = hf_hub_download(
                repo_id=repo_id,
                filename="config.json",  # Root config.json
                local_files_only=False,
            )
        except Exception as e:
            print(e)
            pass

        if path:
            weights_filepath = f"{path}/{weights_filename}"
        else:
            weights_filepath = weights_filename

        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=weights_filepath,
            force_download=False,
        )
    else:
        weights_path = path + weights_filename

    checkpoint = torch.load(weights_path, weights_only=False, map_location="cpu")
    model_args = checkpoint["model_args"]
    modality_registry = checkpoint["modality_registry"]

    config = GPTConfig(**model_args)
    model = GPT(
        config,
        modality_registry,
    )

    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # torch.compile adds _orig_mod. prefix to parameter names so we fix below
    unwanted_prefix = "_orig_mod."
    keys_to_update = []
    for k in state_dict.keys():
        if unwanted_prefix in k:
            new_key = k.replace(unwanted_prefix, "")
            keys_to_update.append((k, new_key))
    for old_key, new_key in keys_to_update:
        state_dict[new_key] = state_dict.pop(old_key)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as err:
        print(err)
        print("Assuming we are loading a version of AstroPT < 3.0, so altering state dict key names to fit...")
        state_dict["encoders.images.c_fc.weight"] = state_dict.pop("transformer.wte.images.c_fc.weight")
        state_dict["encoders.images.c_proj.weight"] = state_dict.pop("transformer.wte.images.c_proj.weight")
        state_dict["decoders.images.c_fc.weight"] = state_dict.pop("lm_head.images.c_fc.weight")
        state_dict["decoders.images.c_proj.weight"] = state_dict.pop("lm_head.images.c_proj.weight")
        state_dict["embedders.images.wpe.weight"] = state_dict.pop("transformer.wpe.images.wpe.weight")
        model.load_state_dict(state_dict)


    dir_info = f"/{path}" if path else ""
    print(f"model loaded successfully from {repo_id}{dir_info}")
    print("args:", model_args)
    return model


def load_local_model(
    ckpt_path: str, 
    device: str | torch.device = "cpu", 
    config_path: Optional[str] = None
) -> Tuple[GPT, GPTConfig, ModalityRegistry, Dict[str, Any]]:
    """
    General loader for local training runs.
    
    Logic:
    1. Loads Checkpoint (Weights + Registry).
    2. Loads Config (Priority: config.json > checkpoint['config']).
    3. Filters Config args to prevent version mismatches.
    4. Returns Model, Config, and Registry.

    Args:
        ckpt_path (str): Path to the .pt file.
        device (str/torch.device): Device to load the model onto.

    Returns:
        tuple: (model, config, registry)
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    # Load Checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Resolve Configuration (JSON > Checkpoint)
    raw_args = {}
    
    # Configuration searching method
    candidates = []
    if config_path: candidates.append(config_path)
    
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    candidates.append(os.path.join(ckpt_dir, "config.json")) 
    candidates.append(os.path.join(os.path.dirname(ckpt_dir), "config.json")) 
    
    # Obtaining the .json path
    final_config_path = None
    for path in candidates:
        if os.path.exists(path):
            final_config_path = path
            break
            
    if final_config_path:
        logger.info(f"Found config.json at {final_config_path}. Using it for hyperparameters.")
        with open(final_config_path, "r") as f:
            raw_args = json.load(f)
    else:
        logger.warning("config.json not found in immediate directories. Falling back to checkpoint internal config.")
        if 'config' in checkpoint:
            raw_args = checkpoint['config']
        elif 'model_args' in checkpoint:
            raw_args = checkpoint['model_args']
        else:
            raise KeyError("Config not found in checkpoint dictionary.")

    # Filter Arguments for GPT
    try:
        valid_fields = {f.name for f in fields(GPTConfig)}
        clean_args = {k: v for k, v in raw_args.items() if k in valid_fields}
    except Exception as e:
        logger.warning(f"Argument filtering failed ({e}). Using raw args.")
        clean_args = raw_args

    config = GPTConfig(**clean_args)
    
    # Get Modality Registry
    if "modality_registry" not in checkpoint:
        raise KeyError("Modality Registry not found in checkpoint!")
    registry = checkpoint["modality_registry"]
    
    # Initialize Model
    model = GPT(config, registry)
    
    # Load State Dict
    state_dict = checkpoint["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("_orig_mod.", "").replace("module.", "")
        new_state_dict[new_k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model, config, registry, raw_args