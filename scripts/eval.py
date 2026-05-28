import os
import sys
import torch
import logging
import datetime
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from tqdm import tqdm

from transformers import HfArgumentParser
from astropt.config import TrainingConfig
from astropt.model import GPT, ModalityRegistry
from astropt.dataloader_multimodal import MultimodalDatasetArrow
from astropt.training_utils import _compute_modality_losses

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

@dataclass
class EvalArguments:
    ckpt_path: str = field(metadata={"help": "Path to the model checkpoint (.pt) file."})
    eval_split: str = field(default="test", metadata={"help": "Dataset split to evaluate on (e.g., 'test' or 'val')."})
    max_batches: int = field(default=500, metadata={"help": "Maximum number of batches to evaluate. 0 for all."})
    save_predictions: bool = field(default=False, metadata={"help": "Save raw model outputs and targets to disk."})
    batch_size: int = field(default=16, metadata={"help": "Batch size for evaluation."})
    device: str = field(default="cuda", metadata={"help": "Device to run on (cuda or cpu)."})
    data_dir: str = field(default=None, metadata={"help": "Override data directory if different from training config."})

def main():
    parser = HfArgumentParser((EvalArguments,))
    eval_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    ckpt_path = Path(eval_args.ckpt_path)
    if not ckpt_path.is_file():
        logger.error(f"Checkpoint file not found: {ckpt_path}")
        sys.exit(1)

    logger.info(f"Loading checkpoint from {ckpt_path}...")
    device = torch.device(eval_args.device if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Reconstruct TrainingConfig
    config_dict = checkpoint.get('config', {})
    if not config_dict:
        logger.warning("No config found in checkpoint. Using default TrainingConfig.")
    
    # Override data_dir if provided
    if eval_args.data_dir:
        config_dict['data_dir'] = eval_args.data_dir
        
    
    valid_keys = {f.name for f in fields(TrainingConfig)}
    clean_config_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
    config = TrainingConfig(**clean_config_dict)
    
    # Force evaluation specific config overrides just in case
    config.batch_size = eval_args.batch_size
    config.device = str(device)
    config.compile = False # No need to compile for standard evaluation script
    
    # 1. Instantiate the Modality Registry
    registry = ModalityRegistry()
    registry.setup_from_config(config)

    if getattr(config, 'use_contrastive_alignment', False):
        from astropt.model_v4 import GPT_V4
        model = GPT_V4(gpt_config, registry)
        logger.info("Initializing GPT_V4 Model (Contrastive Alignment ENABLED) for evaluation")
    else:
        model = GPT(gpt_config, registry)
    
    # 3. Load Weights
    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in list(state_dict.items()):
        # Clean DDP/compile prefixes if present
        clean_k = k.replace("module.", "").replace("_orig_mod.", "")
        new_state_dict[clean_k] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    logger.info(f"Loaded weights with result: {msg}")
    
    model.to(device)
    model.eval()

    # 4. DataLoader
    logger.info(f"Initializing {eval_args.eval_split} DataLoader...")
    dataset = MultimodalDatasetArrow(
        data_dir=config.data_dir,
        split=eval_args.eval_split,
        block_size=config.block_size,
        modality_registry=registry,
        spiral=config.spiral,
        use_aug=False # Never augment during evaluation
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False, # Sequential reading for evaluation
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )

    # 5. Evaluation Loop
    logger.info("Starting evaluation...")
    
    val_losses = []
    val_iso_img2spec = []
    val_iso_spec2img = []
    
    # For saving predictions
    all_predictions = {}
    all_targets = {}

    max_batches = eval_args.max_batches if eval_args.max_batches > 0 else len(loader)
    
    ctx = torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=torch.bfloat16)

    with torch.no_grad():
        for b_idx, batch in enumerate(tqdm(loader, total=min(max_batches, len(loader)))):
            if b_idx >= max_batches:
                break
                
            # Deterministic seed for token mixing in evaluation
            val_batch_seed = config.token_mixing_seed + b_idx

            B_val = MultimodalDatasetArrow.process_modes(
                batch_data=batch, 
                modality_registry=registry, 
                device=device,
                shuf=config.shuffle_modality_val,
                use_token_mixing=config.use_token_mixing,
                token_mixing_seed=val_batch_seed,
                use_cls_token=config.use_cls_token,
                cls_position=config.cls_position
            )

            with ctx:
                model._token_mixing_seed = B_val["X"].pop("token_mixing_seed", config.token_mixing_seed)
                outputs, v_loss = model(B_val["X"], targets=B_val["Y"])
            
            val_losses.append(v_loss.item())

            # --- ISOLATED RECONSTRUCTION VALIDATION ---
            img_key = "aion_images" if "aion_images" in B_val["X"] else "images"
            spec_key = "aion_spectra" if "aion_spectra" in B_val["X"] else "spectra"

            if img_key in B_val["X"] and spec_key in B_val["X"]:
                is_prefix_mode = config.attn_type == "prefix"

                # 1. Images -> Spectra
                with ctx:
                    if is_prefix_mode:
                        X_img2spec = {}
                        for k, v in B_val["X"].items():
                            if k == img_key or k == img_key + "_positions":
                                X_img2spec[k] = v
                        for k, v in B_val["X"].items():
                            if k not in X_img2spec:
                                X_img2spec[k] = v
                        X_img2spec[spec_key] = torch.zeros_like(X_img2spec[spec_key])
                    else:
                        X_img2spec = {k: v for k, v in B_val["X"].items()}
                        X_img2spec[spec_key] = torch.zeros_like(X_img2spec[spec_key])

                    out_img2spec, _ = model(X_img2spec, targets=B_val["Y"])
                    iso_loss_s = _compute_modality_losses(out_img2spec, B_val["Y"], config)
                    if spec_key in iso_loss_s:
                        val_iso_img2spec.append(iso_loss_s[spec_key])

                # 2. Spectra -> Images
                with ctx:
                    if is_prefix_mode:
                        X_spec2img = {}
                        for k, v in B_val["X"].items():
                            if k == spec_key or k == spec_key + "_positions":
                                X_spec2img[k] = v
                        for k, v in B_val["X"].items():
                            if k not in X_spec2img:
                                X_spec2img[k] = v
                        X_spec2img[img_key] = torch.zeros_like(X_spec2img[img_key])
                    else:
                        X_spec2img = {k: v for k, v in B_val["X"].items()}
                        X_spec2img[img_key] = torch.zeros_like(X_spec2img[img_key])

                    out_spec2img, _ = model(X_spec2img, targets=B_val["Y"])
                    iso_loss_i = _compute_modality_losses(out_spec2img, B_val["Y"], config)
                    if img_key in iso_loss_i:
                        val_iso_spec2img.append(iso_loss_i[img_key])
            
            # Save predictions if requested
            if eval_args.save_predictions:
                # Store detached CPU tensors to save memory
                for k, v in outputs.items():
                    if not k.startswith("_"): # Ignore internal properties
                        if k not in all_predictions:
                            all_predictions[k] = []
                        all_predictions[k].append(v.detach().cpu())
                
                for k, v in B_val["Y"].items():
                    if k not in all_targets:
                        all_targets[k] = []
                    all_targets[k].append(v.detach().cpu())

    # 6. Summary
    val_loss = sum(val_losses) / len(val_losses) if val_losses else float('nan')
    avg_iso_img2spec = sum(val_iso_img2spec) / len(val_iso_img2spec) if val_iso_img2spec else float('nan')
    avg_iso_spec2img = sum(val_iso_spec2img) / len(val_iso_spec2img) if val_iso_spec2img else float('nan')

    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Split         : {eval_args.eval_split}")
    logger.info(f"Batches       : {min(max_batches, len(loader))}")
    logger.info(f"Overall Loss  : {val_loss:.4f}")
    if not math.isnan(avg_iso_img2spec):
        logger.info(f"Zero-Shot Img -> Spec : {avg_iso_img2spec:.4f}")
        logger.info(f"Zero-Shot Spec -> Img : {avg_iso_spec2img:.4f}")
    logger.info("=" * 50)

    # 7. Dump predictions
    if eval_args.save_predictions:
        output_file = ckpt_path.parent / f"eval_predictions_{eval_args.eval_split}.pt"
        logger.info(f"Saving predictions to {output_file}...")
        
        # Concatenate lists
        for k in all_predictions:
            all_predictions[k] = torch.cat(all_predictions[k], dim=0)
        for k in all_targets:
            all_targets[k] = torch.cat(all_targets[k], dim=0)
            
        torch.save({
            "predictions": all_predictions,
            "targets": all_targets,
            "metrics": {
                "val_loss": val_loss,
                "zero_shot_img2spec": avg_iso_img2spec,
                "zero_shot_spec2img": avg_iso_spec2img
            },
            "config": config_dict
        }, output_file)
        logger.info("Predictions saved successfully.")

if __name__ == "__main__":
    main()
