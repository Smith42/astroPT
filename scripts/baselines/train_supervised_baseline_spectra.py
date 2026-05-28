"""
Spectra Supervised Baseline Training Script.

This script implements a supervised training pipeline for the SpectraSupervisedBaseline
model, designed for 1D DESI spectra regression tasks. This model serves as a 
supervised benchmark against foundation models like AstroPT.

Author: Antigravity (Senior ML Engineer)
Date: May 2026
"""

import logging
import sys
import numpy as np
import pandas as pd
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

# --- Configuration ---

@dataclass
class TrainingConfig:
    """
    Configuration for Spectra Supervised Baseline.
    Aligned with AstroPT TrainingConfig and ResNet Baseline.
    """
    #--- Metadata & Paths ---#
    target: str = "Z"                                      # Target column in catalog
    train_name: str = "spectra_supervised_baseline"        # Name for the run
    train_date: Optional[str] = None                       # Date of the training
    train_dir: Optional[str] = None                        # Training output directory
    seed: int = 42                                         # Random seed
    metadata_path: str = "/home/valonso/iac18_aasensio_shared/euclid_dr1/test_catalog_cleaned.fits"
    data_dir: str = "/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"
    
    #--- Model Architecture ---#
    seq_len_in: int = 7781          # Input sequence length (DESI spectra)
    embed_dim: int = 256            # Embedding dimension for transformer
    num_layers: int = 4             # Transformer encoder layers
    num_heads: int = 8              # Transformer attention heads
    dim_feedforward: int = 1024     # Expansion factor (4x embed_dim)
    dropout: float = 0.1
    
    #--- Data Loading ---#
    batch_size: int = 64
    num_workers: int = 4                                   # DataLoader workers
    spiral: bool = False                                   # Not used for 1D spectra
    
    #--- Hyperparameters (AstroPT Aligned) ---#
    max_iters: int = 75000                                 # Total training iters
    init_from: str = "scratch"                             # "scratch" or "resume"
    learning_rate: float = 1e-4                            # Max Learning rate
    lr_min: float = 1e-5                                   # Min Learning rate
    lr_warmup_iters: int = 4000                            # Warmup steps
    lr_decay_iters: int = 65000                            # Decay steps
    weight_decay: float = 1e-2                             # Weight decay
    beta1: float = 0.9                                     # AdamW parameter
    beta2: float = 0.95                                    # AdamW parameter
    
    #--- Spectra Processing ---#
    spectra_norm_type: str = "asinh"
    spectra_norm_scaler: float = 1.0
    spectra_norm_const: float = 1.0
    
    #--- Logging & Checkpointing ---#
    eval_interval: int = 1000                               # How often to evaluate
    eval_batches: int = 100                                 # How many batches for evaluation
    log_interval: int = 200                                 # Console/CSV log interval
    checkpoint_interval: int = 1000                         # How often to save model weights
    early_stop_patience: int = 10                           # Limits iterations without validation loss improvement
    early_stopping_min_iters: int = 35000                   # Minimum iterations before starting to count patience
    
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

# --- Model Definition ---

class SpectraSupervisedBaseline(nn.Module):
    """
    Supervised Baseline Architecture for 1D DESI spectra.
    """
    def __init__(self, config: TrainingConfig, output_dim: int = 1):
        super().__init__()
        self.output_dim = output_dim
        self.embed_dim = config.embed_dim
        
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=21, stride=5),
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            nn.Conv1d(64, 128, kernel_size=17, stride=4),
            nn.BatchNorm1d(128),
            nn.GELU(),
            
            nn.Conv1d(128, self.embed_dim, kernel_size=9, stride=2),
            nn.BatchNorm1d(self.embed_dim),
            nn.GELU()
        )
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, config.seq_len_in)
            self.seq_len_out = self.stem(dummy_input).shape[-1]
            
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len_out, self.embed_dim))
        self.pos_drop = nn.Dropout(p=config.dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.embed_dim // 2, self.output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle [B, L, 1] from arrow dataset
        if x.dim() == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        # Ensure [B, 1, L] for conv1d stem
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = x.transpose(1, 2)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        out = self.head(x)
        return out

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
        return item['spectra'], y, targetid

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
            spectra, y, _ = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            spectra, y, _ = next(val_iter)
            
        spectra = spectra.to(device)
        
        if task_type == 'regression':
            y = y.to(device).float()
        else:
            y = y.to(device).long()
            
        with ctx:
            preds = model(spectra)
            if task_type == 'regression':
                preds = preds.squeeze(-1)
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
    logger = logging.getLogger("Spectra-Baseline")
    
    save_config_json(config, weights_dir)
    
    logger.info(f"Starting Spectra Supervised Baseline for target: {config.target}")
    logger.info(f"Loading metadata from {config.metadata_path}")
    catalog = Table.read(config.metadata_path).to_pandas()
    
    # Process TARGETID
    target_id_col = 'TARGETID' if 'TARGETID' in catalog.columns else 'targetid'
    catalog[target_id_col] = catalog[target_id_col].astype('int64')
    
    # Apply redshift-dependent global spectrograph SNR filter
    if 'SNR_SPEC_R' in catalog.columns and 'SNR_SPEC_Z' in catalog.columns and 'Z' in catalog.columns:
        snr_mask = ((catalog['Z'] < 0.15) & (catalog['SNR_SPEC_R'] > 3.0)) | ((catalog['Z'] >= 0.15) & (catalog['SNR_SPEC_Z'] > 3.0))
        catalog = catalog[snr_mask].copy()
        logger.info(f"Applied redshift-dependent global spectrograph SNR filter. Catalog shape: {catalog.shape}")
    
    # Task definition
    vals = catalog[config.target]
    classification_targets = ['SPECTYPE', 'data_set_release']
    if config.target not in classification_targets or pd.api.types.is_numeric_dtype(vals):
        task_type = 'regression'
        mask = vals.notna() & (vals > -90)
        
        # Apply task-specific IVAR quality filter
        ivar_col = f"{config.target}_IVAR"
        if ivar_col in catalog.columns:
            ivar_vals = catalog[ivar_col]
            err_vals = np.where(ivar_vals > 0, 1.0 / np.sqrt(ivar_vals.clip(lower=1e-10)), np.inf)
            mask &= (vals > 3.0 * err_vals)
            logger.info(f"Applied dynamic SNR > 3 quality filter for regression target '{config.target}' using '{ivar_col}'.")
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
        name="spectra", 
        input_size=1,
        patch_size=1, 
        pos_input_size=1,
        embed_pos=False,
        encoder_type="discrete"
    )])
    
    tf_kwargs = {
        'norm_type_spec': config.spectra_norm_type,
        'norm_scaler_spec': config.spectra_norm_scaler,
        'norm_const_spec': config.spectra_norm_const
    }
    
    train_tf = EuclidDESIDatasetArrow.data_transforms(stage='train', **tf_kwargs)
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
    model = SpectraSupervisedBaseline(config=config, output_dim=output_dim).to(DEVICE)
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
    logger.info(f"Training SpectraSupervisedBaseline on {config.target} for {config.max_iters} iterations...")
    
    train_iter = iter(train_loader)
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
            spectra, y, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            spectra, y, _ = next(train_iter)
            
        spectra = spectra.to(DEVICE)
        
        if task_type == 'regression':
            y = y.to(DEVICE).float()
        else:
            y = y.to(DEVICE).long()
            
        # Forward pass
        optimizer.zero_grad()
        with ctx:
            preds = model(spectra)
            if task_type == 'regression':
                preds = preds.squeeze(-1)
            loss = criterion(preds, y)
            
        loss.backward()
        optimizer.step()
        
        # Logging
        if iter_num % config.log_interval == 0:
            current_time = time.time()
            dt = current_time - last_log_time
            last_log_time = current_time
            avg_dt = dt / config.log_interval if iter_num > 0 else 0
            
            mem_usage = torch.cuda.max_memory_allocated() / (1024 ** 3) if DEVICE.type == "cuda" else 0.0
            if DEVICE.type == "cuda":
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
                if iter_num >= config.early_stopping_min_iters:
                    patience_counter += 1
                    logger.info(f"No validation improvement. Patience: {patience_counter}/{config.early_stop_patience}")
                    if patience_counter >= config.early_stop_patience:
                        logger.warning(f"Early stopping triggered after {iter_num} iterations without improvement.")
                        stop_training = True
                else:
                    logger.info(f"No validation improvement, but within min_iters ({iter_num}/{config.early_stopping_min_iters}). Skipping patience.")
                
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
    best_ckpt = weights_dir / "ckpt_best.pt"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE, weights_only=False)['model'])
        logger.info("Loaded best model weights for final evaluation.")
        
    model.eval()
    all_preds = []
    all_trues = []
    all_ids = []
    
    final_test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    with torch.no_grad():
        for spectra, y, batch_ids in tqdm(final_test_loader, desc="Final Evaluation"):
            spectra = spectra.to(DEVICE)
            
            with ctx:
                logits = model(spectra)
            
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
        "Modality": "spectra",
        "Probe": "TRANSFORMER_BASELINE"
    }
    row.update(metrics)
    
    df = pd.DataFrame([row])
    results_csv = train_dir / "spectra_baseline_results.csv"
    header = not results_csv.exists()
    df.to_csv(results_csv, mode='a', header=header, index=False)
    logger.info(f"Final metrics saved to {results_csv}")
    
    # Save predictions
    safe_target = str(config.target).replace('/', '_').replace(' ', '_')
    pred_filename = predictions_dir / f"preds_{safe_target}_spectra_transformer.npz"
    np.savez(pred_filename, preds=all_preds, true_vals=all_trues, targetid=all_ids)
    logger.info(f"Predictions saved to {pred_filename}")

if __name__ == "__main__":
    main()
