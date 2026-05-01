"""
AstroPT Supervised Baseline (ResNet18).

This script trains a ResNet18 from scratch using the AstroPT dataset
to predict specific downstream properties directly from images.
It serves as a strong supervised baseline to compare against zero/few-shot probing.

Features:
- Modifies ResNet18 conv1 to accept 4 channels (VIS, Y, J, H).
- Modifies fc to output the required dimension.
- Uses EuclidDESIDatasetArrow (without spiral) and un-patchifies for 2D convolutions.
- AstroPT Architecture: Infinite iterations, Checkpoints, Logging, Cosine LR, config.json.
"""

import logging
import sys
import numpy as np
import pandas as pd
import einops
import math
import time
import os
import json
import subprocess
import datetime
import re
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow
from astropt.model import ModalityRegistry, ModalityConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from astropy.table import Table
import warnings
from astropy.utils.exceptions import AstropyWarning
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Any, Dict
from transformers import HfArgumentParser

warnings.simplefilter('ignore', category=AstropyWarning)

@dataclass
class TrainingConfig:
    """
    Configuration for ResNet18 Supervised Baseline.
    Aligned with AstroPT TrainingConfig where applicable.
    """
    #--- Metadata & Paths ---#
    target: str = "Z"                                      # Target column in catalog
    train_name: str = "resnet_baseline"                    # Name for the run
    train_date: Optional[str] = None                       # Date of the training
    train_dir: Optional[str] = None                        # Training output directory
    seed: int = 61                                         # Random seed
    metadata_path: str = "/home/valonso/iac18_aasensio_shared/euclid_dr1/test_catalog_cleaned.fits"
    data_dir: str = "/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"
    
    #--- Data Loading ---#
    batch_size: int = 128                                  # Batch size
    num_workers: int = 4                                   # DataLoader workers
    spiral: bool = False                                   # ResNet needs 2D raster order
    
    #--- Hyperparameters (AstroPT Aligned) ---#
    max_iters: int = 75000                                  # Total training iters
    init_from: str = "scratch"                             # "scratch" or "resume"
    learning_rate: float = 3e-4                            # Max Learning rate
    lr_min: float = 3e-5                                   # Min Learning rate
    lr_warmup_iters: int = 4000                             # Warmup steps
    lr_decay_iters: int = 6500                             # Decay steps
    weight_decay: float = 0.1                              # Weight decay
    beta1: float = 0.9                                     # AdamW parameter
    beta2: float = 0.95                                    # AdamW parameter
    
    #--- Image Processing ---#
    images_channels: int = 4                               # VIS + Y, J, H
    images_patch_size: int = 8                             # Patch size (AstroPT uses 8)
    images_norm_type: str = "asinh"                        # Normalization method
    images_norm_scaler: float = 1.0
    images_norm_const: float = 1.0
    use_aug: bool = True                                   # Use data augmentation
    
    #--- Logging & Checkpointing ---#
    eval_interval: int = 1000                               # How often to evaluate
    eval_batches: int = 100                                 # How many batches for evaluation
    log_interval: int = 200                                 # Console/CSV log interval
    checkpoint_interval: int = 1000                         # How often to save model weights
    early_stop_patience: int = 10                           # Limits iterations without validation loss improvement
    
    #--- System ---#
    max_run_hours: Optional[str] = None                    # Time limit ("HH:MM:SS")
    
    def __post_init__(self):
        if self.train_date is None:
            self.train_date = datetime.datetime.now().strftime("%Y%m%d")
        
        safe_target = str(self.target).replace('/', '_').replace(' ', '_')
        
        if self.train_dir is None:
            clean_name = self.train_name.lower()
            clean_name = re.sub(r'[^a-z0-9]', ' ', clean_name)
            suffix_name = "_".join(clean_name.split())
            self.train_dir = f"./logs/{self.train_date}_{suffix_name}_{safe_target}"
        else:
            base_path = Path(self.train_dir)
            if base_path.name != safe_target:
                self.train_dir = str(base_path / safe_target)

def parse_time_to_seconds(time_str: str) -> float:
    parts = time_str.split(':')
    if len(parts) != 3:
        raise ValueError("Time must be in format HH:MM:SS")
    hours, minutes, seconds = map(int, parts)
    return hours * 3600 + minutes * 60 + seconds

def project_directories_setup(base_dir: str) -> Tuple[Path, Path, Path, Path]:
    train_dir = Path(base_dir)
    weights_dir = train_dir / "weights"
    logs_dir = train_dir / "logs"
    predictions_dir = train_dir / "predictions"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    return train_dir, weights_dir, logs_dir, predictions_dir

def get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def get_dataset_info(data_dir: str | Path) -> dict:
    try:
        max_mtime = 0.0
        total_size = 0
        data_path = Path(data_dir)
        for file_path in data_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in ['.arrow', '.json']:
                stats = file_path.stat()
                if stats.st_mtime > max_mtime:
                    max_mtime = stats.st_mtime
                total_size += stats.st_size
        
        last_modified_str = datetime.datetime.fromtimestamp(max_mtime).strftime('%Y-%m-%d %H:%M:%S') if max_mtime > 0 else "Unknown"
        return {
            "data_last_modified": last_modified_str,
            "data_total_size_mb": round(total_size / (1024 * 1024), 2),
        }
    except Exception as e:
        return {"data_last_modified": "Unknown", "data_total_size_mb": 0.0, "error": str(e)}

def save_config_json(config: TrainingConfig, save_dir: Path):
    config_dict = asdict(config)
    config_dict["git_hash"] = get_git_commit_hash()
    data_stats = get_dataset_info(config.data_dir)
    config_dict.update(data_stats)
    config_path = save_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)

def get_learning_rate(it: int, config: TrainingConfig) -> float:
    if it < config.lr_warmup_iters:
        return config.learning_rate * (it + 1) / (config.lr_warmup_iters + 1)
    if it > config.lr_decay_iters:
        return config.lr_min
    decay_ratio = (it - config.lr_warmup_iters) / (config.lr_decay_iters - config.lr_warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.lr_min + coeff * (config.learning_rate - config.lr_min)

class ResNet18Astro(nn.Module):
    def __init__(self, in_channels=4, output_dim=1):
        super().__init__()
        self.resnet = resnet18(weights=None)
        
        # Modify conv1 for 4 channels
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels, 
            original_conv1.out_channels, 
            kernel_size=original_conv1.kernel_size, 
            stride=original_conv1.stride, 
            padding=original_conv1.padding, 
            bias=original_conv1.bias is not None
        )
        
        # Modify final fc layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)
        
    def forward(self, x):
        return self.resnet(x)

def unpatchify(patch_image, patch_size=8, c=4):
    num_patches = patch_image.shape[1]
    grid_size = int(np.sqrt(num_patches))
    return einops.rearrange(
        patch_image,
        'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
        h=grid_size, w=grid_size, p1=patch_size, p2=patch_size, c=c
    )

class FilteredAstroDataset(Dataset):
    def __init__(self, base_dataset, target_dict, valid_indices):
        self.base_dataset = base_dataset
        self.target_dict = target_dict
        self.valid_indices = valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        item = self.base_dataset[real_idx]
        targetid = item['targetid']
        while targetid not in self.target_dict:
            idx = (idx + 1) % len(self)
            real_idx = self.valid_indices[idx]
            item = self.base_dataset[real_idx]
            targetid = item['targetid']
            
        y = self.target_dict[targetid]
        return item['images'], y, targetid

def compute_metrics(y_true, y_pred, task_type, target_name):
    if task_type == 'regression':
        diff = y_pred - y_true
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        if target_name == 'Z':
            norm_diff = diff / (1 + y_true)
            bias = np.median(norm_diff) 
            nmad = 1.4826 * np.median(np.abs(norm_diff - np.median(norm_diff)))
            outliers = np.mean(np.abs(norm_diff) > 0.15) * 100
        elif target_name == 'LOGMSTAR':
            bias = np.median(diff)      
            nmad = 1.4826 * np.median(np.abs(diff - np.median(diff)))
            outliers = np.mean(np.abs(diff) > 0.25) * 100 
        else:
            bias = np.median(diff)      
            nmad = 1.4826 * np.median(np.abs(diff - np.median(diff)))
            relative_error = np.abs(diff) / (np.abs(y_true) + 1e-5)
            outliers = np.mean(relative_error > 0.30) * 100
        return {"R2": r2, "RMSE": rmse, "Bias": bias, "NMAD": nmad, "Outliers": outliers}
    else:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        FP = cm.sum(axis=0) - np.diag(cm)
        TN = cm.sum() - (FP + (cm.sum(axis=1) - np.diag(cm)) + np.diag(cm))
        with np.errstate(divide='ignore', invalid='ignore'):
            fpr_macro = np.mean(np.nan_to_num(FP / (FP + TN)))
        return {"Accuracy": acc, "Precision": precision, "Recall": recall, "F1_score": f1, "FPR": fpr_macro}

@torch.no_grad()
def estimate_loss(model, val_loader, config, device, task_type, ctx, criterion):
    model.eval()
    losses = []
    
    val_iter = iter(val_loader)
    for _ in range(config.eval_batches):
        try:
            images, y, _ = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            images, y, _ = next(val_iter)
            
        images = images.to(device)
        images_2d = unpatchify(images, patch_size=config.images_patch_size)
        
        if task_type == 'regression':
            y = y.to(device).float()
        else:
            y = y.to(device).long()
            
        with ctx:
            preds = model(images_2d)
            if task_type == 'regression':
                preds = preds.squeeze()
            loss = criterion(preds, y)
            
        losses.append(loss.item())
        
    model.train()
    return np.mean(losses)

def main():
    parser = HfArgumentParser((TrainingConfig,))
    config = parser.parse_args_into_dataclasses()[0]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # --- SETUP DIRECTORIES & LOGGING ---
    train_dir, weights_dir, logs_dir, predictions_dir = project_directories_setup(config.train_dir)
    
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logs_dir / "training.log", mode='a')
        ]
    )
    logger = logging.getLogger("ResNet-Baseline")
    
    save_config_json(config, weights_dir)
    
    logger.info(f"Starting ResNet18 Supervised Baseline for target: {config.target}")
    logger.info(f"Loading metadata from {config.metadata_path}")
    catalog = Table.read(config.metadata_path).to_pandas()
    
    # Process TARGETID
    target_id_col = 'TARGETID' if 'TARGETID' in catalog.columns else 'targetid'
    catalog[target_id_col] = catalog[target_id_col].astype('int64')
    
    # Task definition
    vals = catalog[config.target]
    classification_targets = ['SPECTYPE', 'data_set_release']
    if config.target not in classification_targets or pd.api.types.is_numeric_dtype(vals):
        task_type = 'regression'
        mask = vals.notna() & (vals > -90)
    else:
        task_type = 'classification'
        mask = vals.notna() & (vals != "") & (vals != "N/A")
        
    filtered_cat = catalog[mask].copy()
    
    # Setup target scaling / encoding
    if task_type == 'regression':
        y_vals = filtered_cat[config.target].values.astype(np.float32)
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y_vals.reshape(-1, 1)).flatten()
        output_dim = 1
    else:
        y_vals = filtered_cat[config.target].values
        encoder = LabelEncoder()
        y_scaled = encoder.fit_transform(y_vals)
        output_dim = len(encoder.classes_)
        scaler_y = None
        
    filtered_cat['y_processed'] = y_scaled
    target_dict = dict(zip(filtered_cat[target_id_col], filtered_cat['y_processed']))
    
    # Configure Arrow Dataset
    registry = ModalityRegistry([ModalityConfig(
        name="images", 
        input_size=config.images_patch_size**2 * config.images_channels,
        pos_input_size=2,
        patch_size=config.images_patch_size, 
        embed_pos=True
    )])
    
    tf_kwargs = {
        'norm_type_img': config.images_norm_type,
        'norm_scaler_img': config.images_norm_scaler,
        'norm_const_img': config.images_norm_const
    }
    
    train_stage = 'train' if config.use_aug else 'val'
    train_tf = EuclidDESIDatasetArrow.data_transforms(stage=train_stage, **tf_kwargs)
    val_tf = EuclidDESIDatasetArrow.data_transforms(stage='val', **tf_kwargs)
    
    base_ds_train = EuclidDESIDatasetArrow(
        arrow_folder_root=config.data_dir,
        split="train",
        modality_registry=registry,
        spiral=config.spiral,
        transform=train_tf
    )
    
    base_ds_val = EuclidDESIDatasetArrow(
        arrow_folder_root=config.data_dir,
        split="test",
        modality_registry=registry,
        spiral=config.spiral,
        transform=val_tf
    )
    
    logger.info("Aligning Arrow dataset with filtered catalog...")
    train_valid_indices = []
    for i in tqdm(range(len(base_ds_train)), desc="Scanning Train Target IDs"):
        try:
            item = base_ds_train.ds[i]
            tid = int(item['targetid'])
            if tid in target_dict:
                train_valid_indices.append(i)
        except Exception:
            continue

    val_valid_indices = []
    for i in tqdm(range(len(base_ds_val)), desc="Scanning Val Target IDs"):
        try:
            item = base_ds_val.ds[i]
            tid = int(item['targetid'])
            if tid in target_dict:
                val_valid_indices.append(i)
        except Exception:
            continue
            
    logger.info(f"Found {len(train_valid_indices)} Train & {len(val_valid_indices)} Val matched samples for {config.target}.")
    if len(train_valid_indices) == 0 or len(val_valid_indices) == 0:
        logger.error("No matches found for train or val. Exiting.")
        sys.exit(1)
        
    train_ds = FilteredAstroDataset(base_ds_train, target_dict, train_valid_indices)
    test_ds = FilteredAstroDataset(base_ds_val, target_dict, val_valid_indices)
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True)
    val_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True)
    
    # Initialize Model & Optimizer
    model = ResNet18Astro(in_channels=config.images_channels, output_dim=output_dim).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(config.beta1, config.beta2))
    criterion = nn.MSELoss() if task_type == 'regression' else nn.CrossEntropyLoss()
    
    import contextlib
    if DEVICE.type == "cuda":
        ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
    else:
        ctx = contextlib.nullcontext()
    
    # --- CHECKPOINT RESUME ---
    iter_num = 0
    best_val_loss = float('inf')
    patience_counter = 0
    accumulated_time = 0.0
    
    if config.init_from == "resume":
        ckpt_path = weights_dir / "ckpt_last.pt"
        if ckpt_path.exists():
            logger.info(f"Resuming from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            patience_counter = checkpoint.get('patience_counter', 0)
            accumulated_time = checkpoint.get('total_run_time', 0.0)
            logger.info(f"Resumed successfully at iter {iter_num} (Best Loss: {best_val_loss:.4f})")
        else:
            logger.warning(f"No checkpoint found at {ckpt_path}. Starting from scratch.")
    
    # --- CSV LOGGING ---
    csv_path = logs_dir / "training_metrics.csv"
    if config.init_from == "scratch" or not csv_path.exists():
        with open(csv_path, "w") as f:
            f.write("iter,progress,train_loss,val_loss,lr,dt_ms,mem_gb,eta_hms\n")
    
    # --- MAX RUN HOURS SETUP ---
    max_run_seconds = None
    if config.max_run_hours is not None:
        try:
            max_run_seconds = parse_time_to_seconds(config.max_run_hours)
            logger.info(f"Time limit set to: {config.max_run_hours} ({max_run_seconds} seconds)")
        except Exception as e:
            logger.error(f"Error parsing max_run_hours: {e}")
            sys.exit(1)
            
    # --- TRAINING LOOP ---
    logger.info(f"Training ResNet18 on {config.target} for {config.max_iters} iterations...")
    
    train_iter = iter(train_loader)
    t0 = time.time()
    run_start_time = time.time()
    last_log_time = time.time()
    
    model.train()
    stop_training = False
    
    while iter_num <= config.max_iters:
        # Dynamic LR
        lr = get_learning_rate(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Get Batch
        try:
            images, y, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, y, _ = next(train_iter)
            
        images = images.to(DEVICE)
        images_2d = unpatchify(images, patch_size=config.images_patch_size)
        
        if task_type == 'regression':
            y = y.to(DEVICE).float()
        else:
            y = y.to(DEVICE).long()
            
        # Forward pass
        optimizer.zero_grad()
        with ctx:
            preds = model(images_2d)
            if task_type == 'regression':
                preds = preds.squeeze()
            loss = criterion(preds, y)
            
        loss.backward()
        optimizer.step()
        
        # Logging
        if iter_num % config.log_interval == 0:
            current_time = time.time()
            dt = current_time - last_log_time
            last_log_time = current_time
            avg_dt = dt / config.log_interval if iter_num > 0 else 0
            
            mem_usage = torch.cuda.max_memory_allocated() / (1024 ** 3)
            torch.cuda.reset_peak_memory_stats()
            
            progress = iter_num / config.max_iters
            
            # ETA calculation
            remaining_iters = config.max_iters - iter_num
            eta_seconds = remaining_iters * avg_dt
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            
            logger.info(
                f"Iter {iter_num}/{config.max_iters} ({progress:.2%}) | "
                f"Loss {loss.item():.4f} | LR {lr:.4e} | "
                f"Mem {mem_usage:.2f}GB | dt {avg_dt*1000:.1f}ms | ETA {eta_str}"
            )
            
            # Save to CSV
            with open(csv_path, "a") as f:
                f.write(f"{iter_num},{progress:.4f},{loss.item():.4f},,{lr:.4e},{avg_dt*1000:.1f},{mem_usage:.2f},{eta_str}\n")
                
        # Evaluation & Checkpoint
        if iter_num > 0 and iter_num % config.eval_interval == 0:
            val_loss = estimate_loss(model, val_loader, config, DEVICE, task_type, ctx, criterion)
            logger.info(f"Validation Loss: {val_loss:.4f}")
            
            # Update CSV with val loss
            with open(csv_path, "a") as f:
                f.write(f"{iter_num},{iter_num/config.max_iters:.4f},,{val_loss:.4f},,,, \n")
            
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'total_run_time': accumulated_time + (time.time() - run_start_time)
            }
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(checkpoint, weights_dir / "ckpt_best.pt")
                logger.info(f"New best model saved! (Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                logger.info(f"No validation improvement. Patience: {patience_counter}/{config.early_stop_patience}")
                if patience_counter >= config.early_stop_patience:
                    logger.warning(f"Early stopping triggered after {iter_num} iterations without improvement.")
                    stop_training = True
                
        # Routine Checkpoint
        if iter_num > 0 and iter_num % config.checkpoint_interval == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'total_run_time': accumulated_time + (time.time() - run_start_time)
            }
            torch.save(checkpoint, weights_dir / "ckpt_last.pt")
            logger.info(f"Routine checkpoint saved at iter {iter_num}.")
            
        # Autosaving and Stop based on Time
        if max_run_seconds is not None:
            elapsed_seconds = time.time() - run_start_time
            if elapsed_seconds > max_run_seconds:
                logger.warning(f"Time limit reached ({config.max_run_hours}). Saving checkpoint and exiting gracefully.")
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                    'total_run_time': accumulated_time + elapsed_seconds
                }
                torch.save(checkpoint, weights_dir / "ckpt_last.pt")
                stop_training = True
                
        if stop_training:
            break
            
        iter_num += 1
        
    logger.info("Training complete. Starting final evaluation on full test set for predictions...")
    
    # Final Evaluation over full Test Set
    # Load best weights if available
    best_ckpt = weights_dir / "ckpt_best.pt"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE, weights_only=False)['model'])
        logger.info("Loaded best model weights for final evaluation.")
        
    model.eval()
    all_preds = []
    all_trues = []
    all_ids = []
    
    # We use full test_loader without drop_last
    final_test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    with torch.no_grad():
        for images, y, batch_ids in tqdm(final_test_loader, desc="Final Evaluation"):
            images = images.to(DEVICE)
            images_2d = unpatchify(images, patch_size=config.images_patch_size)
            
            with ctx:
                logits = model(images_2d)
            
            if task_type == 'regression':
                batch_preds = logits.cpu().float().numpy().flatten()
                batch_preds = scaler_y.inverse_transform(batch_preds.reshape(-1, 1)).flatten()
                batch_trues = scaler_y.inverse_transform(y.numpy().reshape(-1, 1)).flatten()
            else:
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                batch_trues = y.numpy()
                
            all_preds.append(batch_preds)
            all_trues.append(batch_trues)
            all_ids.append(batch_ids.numpy())
            
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    all_ids = np.concatenate(all_ids)
    
    metrics = compute_metrics(all_trues, all_preds, task_type, config.target)
    
    row = {
        "Target": config.target,
        "Task": task_type,
        "Modality": "images",
        "Probe": "RESNET18"
    }
    row.update(metrics)
    
    df = pd.DataFrame([row])
    results_csv = train_dir / "resnet_baseline_results.csv"
    header = not results_csv.exists()
    df.to_csv(results_csv, mode='a', header=header, index=False)
    logger.info(f"Final metrics saved to {results_csv}")
    
    # Save predictions
    safe_target = str(config.target).replace('/', '_').replace(' ', '_')
    pred_filename = predictions_dir / f"preds_{safe_target}_images_resnet18.npz"
    np.savez(pred_filename, preds=all_preds, true_vals=all_trues, targetid=all_ids)
    logger.info(f"Predictions saved to {pred_filename}")

if __name__ == "__main__":
    main()
