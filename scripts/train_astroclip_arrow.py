"""
AstroCLIP training on AstroPT Arrow data with AstroPT-like logging.

Goals:
- Reuse AstroPT Arrow dataloader.
- Keep training/validation outputs as comparable as possible to AstroPT logs.
- Run DDP with 4 GPUs by default.
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import einops
import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop
from transformers import HfArgumentParser

# Assuming this script is executed from astroPT/scripts
ASTROPT_DIR = Path(__file__).resolve().parent.parent
ASTROCLIP_DIR = ASTROPT_DIR.parent / "AstroCLIP"

if str(ASTROPT_DIR / "src") not in sys.path:
    sys.path.append(str(ASTROPT_DIR / "src"))
if str(ASTROCLIP_DIR) not in sys.path:
    sys.path.append(str(ASTROCLIP_DIR))

from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow
from astropt.model import ModalityConfig, ModalityRegistry

from astroclip.models.astroclip import AstroClipModel, ImageHead, SpectrumHead


@dataclass
class AstroCLIPArrowConfig:
    """
    Hyperparameters and configuration settings for AstroCLIP training.

    This dataclass keeps AstroPT-like block layout while containing only
    AstroCLIP-relevant training variables.
    """

    #--- Training Metadata ---#
    train_name: Optional[str] = "astroclip_arrow"
    train_date: Optional[str] = None
    train_description: Optional[str] = None

    #--- I/O & Paths ---#
    train_dir: Optional[str] = None
    data_dir: str = "/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated"

    #--- Data Loading ---#
    batch_size: int = 16
    num_workers: int = 8
    persistent_workers: bool = True
    prefetch_factor: int = 2
    pin_memory: bool = True
    init_from: str = "scratch"      # Training from scratch or resume training

    #--- Model Architecture ---#
    seed: int = 61

    # Images
    images_size: int = 224
    images_patch_size: int = 8
    images_channels: int = 4

    # Spectra
    spectra_patch_size: int = 10

    # AstroCLIP-specific image preprocessing
    center_crop: int = 144

    #--- Optimization of the Learning Process ---#
    max_iters: int = 75_000
    max_epochs: int = 100
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 40

    #--- Learning Rate Scheduler ---#
    learning_rate: float = 3e-4
    lr_min: float = 3e-5
    lr_warmup_steps: int = 4_000
    lr_decay_steps: int = 65_000
    max_steps: int = -1

    #--- Logging & Checkpointing ---#
    val_check_interval: int = 1000
    limit_val_batches: int = 100
    log_interval: int = 200
    checkpoint_interval: int = 1_000
    checkpoint_save_type: str = "both"      # best, last, both or all
    early_stopping_patience: int = 10

    #--- System & Backend ---#
    precision: str = "bf16-mixed"
    max_run_hours: Optional[str] = None     # Force stop after "HH:MM:SS"

    #--- External Monitoring ---#
    log_via_wandb: bool = False
    wandb_project: str = "astroclip-alignment"
    wandb_entity: Optional[str] = None

    #--- DDP ---#
    ddp_num_gpus: int = 4
    ddp_strategy: str = "ddp"
    ddp_find_unused_parameters: bool = True

    #--- AstroCLIP encoder tuning ---#
    freeze_image_backbone: bool = False
    freeze_spectrum_backbone: bool = True

    def __post_init__(self):
        if self.train_date is None:
            self.train_date = datetime.datetime.now().strftime("%Y%m%d")

        if self.train_dir is None:
            clean_name = self.train_name.lower() if self.train_name else "default_run"
            clean_name = re.sub(r"[^a-z0-9]", " ", clean_name)
            suffix_name = "_".join(clean_name.split())
            self.train_dir = f"./logs/astroclip_arrow_{self.train_date}_{suffix_name}"


def get_git_commit_hash() -> str:
    """Returns the current git commit hash or 'unknown' if unavailable."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ASTROPT_DIR))
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


def get_dataset_info(data_dir: str | Path) -> dict:
    """
    Scans the dataset directory and returns a lightweight data fingerprint
    based on last modification time, total size, and file count.
    """
    try:
        max_mtime = 0.0
        total_size = 0
        file_count = 0
        data_path = Path(data_dir)

        for file_path in data_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in [".arrow", ".json"]:
                stats = file_path.stat()
                if stats.st_mtime > max_mtime:
                    max_mtime = stats.st_mtime
                total_size += stats.st_size
                file_count += 1

        if max_mtime > 0:
            last_modified_str = datetime.datetime.fromtimestamp(max_mtime).strftime("%Y-%m-%d %H:%M:%S")
        else:
            last_modified_str = "Unknown"

        return {
            "data_last_modified": last_modified_str,
            "data_total_size_mb": round(total_size / (1024 * 1024), 2),
            "data_file_count": file_count,
        }
    except Exception as e:
        return {"data_version_error": str(e)}


def save_config_json(config: AstroCLIPArrowConfig, save_path: Path) -> None:
    """Saves run config with AstroPT-style reproducibility metadata."""
    config_dict = asdict(config)
    config_dict["created_at"] = datetime.datetime.now().isoformat()
    config_dict["git_hash"] = get_git_commit_hash()
    config_dict.update(get_dataset_info(config.data_dir))

    with open(save_path, "w") as f:
        json.dump(config_dict, f, indent=2)


def parse_time_to_seconds(time_str: str) -> float:
    """Converts a HH:MM:SS string to total seconds."""
    try:
        h, m, s = map(int, time_str.split(":"))
        if h < 0 or m < 0 or s < 0:
            raise ValueError
        return h * 3600 + m * 60 + s
    except ValueError:
        raise ValueError(f"Invalid time format: {time_str}. Expected HH:MM:SS")


def validate_runtime_flags(config: AstroCLIPArrowConfig) -> None:
    """Validates runtime flags before launching expensive GPU work."""
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if config.num_workers < 0:
        raise ValueError("num_workers must be >= 0.")
    if config.num_workers > 0 and config.prefetch_factor < 1:
        raise ValueError("prefetch_factor must be >= 1 when num_workers > 0.")

    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1.")
    if config.max_iters < 1:
        raise ValueError("max_iters must be >= 1.")
    if config.max_steps == 0:
        raise ValueError("max_steps cannot be 0. Use -1 or a positive integer.")

    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0.")
    if config.lr_min <= 0:
        raise ValueError("lr_min must be > 0.")
    if config.lr_min > config.learning_rate:
        raise ValueError("lr_min must be <= learning_rate.")
    if config.lr_warmup_steps < 1:
        raise ValueError("lr_warmup_steps must be >= 1.")
    if config.lr_decay_steps < config.lr_warmup_steps:
        raise ValueError("lr_decay_steps must be >= lr_warmup_steps.")

    if config.val_check_interval < 1:
        raise ValueError("val_check_interval must be >= 1.")
    if config.limit_val_batches < 1:
        raise ValueError("limit_val_batches must be >= 1.")
    if config.checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be >= 1.")
    if config.log_interval < 1:
        raise ValueError("log_interval must be >= 1.")
    if config.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be >= 1.")

    if config.ddp_num_gpus < 1:
        raise ValueError("ddp_num_gpus must be >= 1.")

    if config.max_run_hours is not None:
        parse_time_to_seconds(config.max_run_hours)


def configure_training_logger(logs_dir: Path) -> logging.Logger:
    """Configure AstroPT-style logger writing to stdout and logs/training.log."""
    logger = logging.getLogger("AstroCLIPTrain")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if logger.hasHandlers():
        logger.handlers.clear()

    if local_rank != 0:
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.ERROR)
        return logger

    log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(logs_dir / "training.log", mode="a")
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logger.addHandler(file_handler)

    return logger


class AstroCLIPArrowDataModule(L.LightningDataModule):
    def __init__(self, config: AstroCLIPArrowConfig):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.center_crop = CenterCrop(self.config.center_crop)

    def setup(self, stage: Optional[str] = None):
        img_input_size = (
            self.config.images_patch_size
            * self.config.images_patch_size
            * self.config.images_channels
        )
        modalities = [
            ModalityConfig(
                name="images",
                input_size=img_input_size,
                pos_input_size=1,
                patch_size=self.config.images_patch_size,
                embed_pos=True,
            ),
            ModalityConfig(
                name="spectra",
                input_size=self.config.spectra_patch_size,
                pos_input_size=1,
                patch_size=self.config.spectra_patch_size,
                embed_pos=True,
            ),
        ]
        registry = ModalityRegistry(modalities)

        train_tf = EuclidDESIDatasetArrow.data_transforms(
            norm_type_img="asinh",
            norm_scaler_img=1.0,
            norm_const_img=7.603847,
            norm_type_spec="asinh",
            norm_scaler_spec=1.0,
            norm_const_spec=7.956048,
            stage="train",
        )

        val_tf = EuclidDESIDatasetArrow.data_transforms(
            norm_type_img="asinh",
            norm_scaler_img=1.0,
            norm_const_img=7.603847,
            norm_type_spec="asinh",
            norm_scaler_spec=1.0,
            norm_const_spec=7.956048,
            stage="val",
        )

        self.train_dataset = EuclidDESIDatasetArrow(
            arrow_folder_root=self.config.data_dir,
            split="train",
            modality_registry=registry,
            spiral=False,
            stochastic=False,
            transform=train_tf,
        )

        self.val_dataset = EuclidDESIDatasetArrow(
            arrow_folder_root=self.config.data_dir,
            split="test",
            modality_registry=registry,
            spiral=False,
            stochastic=False,
            transform=val_tf,
        )

    def custom_collate(self, batch):
        images = []
        spectra = []
        targetids = []

        grid = self.config.images_size // self.config.images_patch_size
        p1 = self.config.images_patch_size
        p2 = self.config.images_patch_size

        valid_items = [item for item in batch if "images" in item and "spectra" in item]
        if len(valid_items) == 0:
            raise RuntimeError("Batch without paired image+spectrum samples")

        for item in valid_items:
            patch_img = item["images"]
            orig_img = einops.rearrange(
                patch_img,
                "(h w) (p1 p2 c) -> c (h p1) (w p2)",
                h=grid,
                w=grid,
                p1=p1,
                p2=p2,
                c=self.config.images_channels,
            )

            # Map 4 channels to 3 for AstroDINO compatibility.
            img_3ch = orig_img[:3, :, :]
            img_3ch = self.center_crop(img_3ch)
            images.append(img_3ch)

            patch_spec = item["spectra"]
            # SpectrumHead/SpecFormer expects (B, L, 1).
            orig_spec = patch_spec.flatten().unsqueeze(-1)
            spectra.append(orig_spec)
            targetids.append(int(item["targetid"]))

        return {
            "image": torch.stack(images).to(torch.float32),
            "spectrum": torch.stack(spectra).to(torch.float32),
            "targetid": torch.tensor(targetids, dtype=torch.int64),
        }

    def _loader_kwargs(self):
        kwargs = {
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
            "collate_fn": self.custom_collate,
            "drop_last": True,
        }
        if self.config.num_workers > 0:
            kwargs["persistent_workers"] = self.config.persistent_workers
            kwargs["prefetch_factor"] = self.config.prefetch_factor
        return kwargs

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            **self._loader_kwargs(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            **self._loader_kwargs(),
        )


class AstroClipTrainModule(AstroClipModel):
    def __init__(
        self,
        image_encoder,
        spectrum_encoder,
        temperature: float,
        lr: float,
        weight_decay: float,
        epochs: int,
        eta_min: float,
        beta1: float,
        beta2: float,
        lr_warmup_steps: int,
        lr_decay_steps: int,
    ):
        super().__init__(
            image_encoder=image_encoder,
            spectrum_encoder=spectrum_encoder,
            temperature=temperature,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            eta_min=eta_min,
        )
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self._last_grad_norms = {
            "images": float("nan"),
            "spectra": float("nan"),
            "backbone": float("nan"),
            "total": float("nan"),
        }

    @staticmethod
    def _clip_alignment_metrics(image_features: torch.Tensor, spectrum_features: torch.Tensor):
        image_features = F.normalize(image_features, dim=-1, eps=1e-3)
        spectrum_features = F.normalize(spectrum_features, dim=-1, eps=1e-3)

        logits = image_features @ spectrum_features.T
        labels = torch.arange(logits.shape[0], device=logits.device)

        i2s_acc = (logits.argmax(dim=1) == labels).float().mean()
        s2i_acc = (logits.argmax(dim=0) == labels).float().mean()

        pos_sim = torch.diag(logits).mean()
        neg_mask = ~torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        neg_sim = logits[neg_mask].mean() if neg_mask.any() else torch.tensor(0.0, device=logits.device)

        return i2s_acc, s2i_acc, pos_sim, neg_sim

    def training_step(self, batch, batch_idx):
        im, sp = batch["image"], batch["spectrum"]

        image_features = self.image_encoder(im)
        spectrum_features = self.spectrum_encoder(sp)

        loss_withlogit = self.criterion(
            image_features,
            spectrum_features,
            self.hparams.temperature,
        )
        loss_nologit = self.criterion(
            image_features,
            spectrum_features,
            self.hparams.logit_scale,
        )

        if not torch.isfinite(loss_withlogit):
            raise RuntimeError(
                f"NaN/Inf detected in train loss_withlogit at step={int(self.global_step)}, batch_idx={batch_idx}."
            )
        if not torch.isfinite(loss_nologit):
            raise RuntimeError(
                f"NaN/Inf detected in train loss_nologit at step={int(self.global_step)}, batch_idx={batch_idx}."
            )

        i2s_acc, s2i_acc, pos_sim, neg_sim = self._clip_alignment_metrics(
            image_features,
            spectrum_features,
        )

        self.log("train_loss", loss_withlogit, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_loss_withlogit", loss_withlogit, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_loss_nologit", loss_nologit, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_i2s_acc", i2s_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_s2i_acc", s2i_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_pos_sim", pos_sim, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_neg_sim", neg_sim, on_step=True, on_epoch=True, sync_dist=True)

        return loss_withlogit

    def validation_step(self, batch, batch_idx):
        im, sp = batch["image"], batch["spectrum"]

        image_features = self.image_encoder(im)
        spectrum_features = self.spectrum_encoder(sp)

        val_loss_nologit = self.criterion(
            image_features,
            spectrum_features,
            self.hparams.logit_scale,
        )
        val_loss_withlogit = self.criterion(
            image_features,
            spectrum_features,
            self.hparams.temperature,
        )

        i2s_acc, s2i_acc, pos_sim, neg_sim = self._clip_alignment_metrics(
            image_features,
            spectrum_features,
        )

        self.log("val_loss", val_loss_withlogit, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_loss_nologit", val_loss_nologit, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_loss_withlogit", val_loss_withlogit, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_i2s_acc", i2s_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_s2i_acc", s2i_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_pos_sim", pos_sim, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_neg_sim", neg_sim, on_step=False, on_epoch=True, sync_dist=True)

        return val_loss_withlogit

    def on_before_optimizer_step(self, optimizer):
        grad_sq = {"images": 0.0, "spectra": 0.0, "backbone": 0.0}

        for name, param in self.named_parameters():
            if param.grad is None:
                continue
            grad_val = float(torch.sum(param.grad.detach() * param.grad.detach()).item())
            if "image_encoder" in name:
                grad_sq["images"] += grad_val
            elif "spectrum_encoder" in name:
                grad_sq["spectra"] += grad_val
            else:
                grad_sq["backbone"] += grad_val

        self._last_grad_norms = {k: math.sqrt(v) for k, v in grad_sq.items()}
        self._last_grad_norms["total"] = math.sqrt(sum(grad_sq.values()))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(self.beta1, self.beta2),
            weight_decay=self.hparams.weight_decay,
        )

        base_lr = float(self.hparams.lr)
        min_lr = float(self.hparams.eta_min)
        warmup = max(1, int(self.lr_warmup_steps))
        decay = max(warmup + 1, int(self.lr_decay_steps))
        min_ratio = min_lr / base_lr if base_lr > 0 else 0.0

        def lr_lambda(step: int):
            if step < warmup:
                return float(step + 1) / float(warmup + 1)
            if step > decay:
                return min_ratio
            ratio = float(step - warmup) / float(decay - warmup)
            cosine = 0.5 * (1.0 + math.cos(math.pi * ratio))
            return min_ratio + cosine * (1.0 - min_ratio)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class AstroPTStyleCSVCallback(L.Callback):
    """Writes AstroPT-like CSV rows while training AstroCLIP."""

    headers = [
        "iter",
        "epoch",
        "progress",
        "timestamp",
        "train_loss",
        "val_loss",
        "val_loss_spec_from_img",
        "val_loss_img_from_spec",
        "loss_images",
        "loss_spectra",
        "cross_loss_images",
        "cross_loss_spectra",
        "grad_norm",
        "grad_images",
        "grad_spectra",
        "grad_backbone",
        "clipped",
        "dropped_modality",
        "lr",
        "lr_images",
        "lr_spectra",
        "lr_backbone",
        "mfu",
        "mem_gb",
        "dt_ms",
        "rt_hms",
        "eta_hms",
        "train_loss_nologit",
        "train_i2s_acc",
        "train_s2i_acc",
        "train_pos_sim",
        "train_neg_sim",
        "val_loss_nologit",
        "val_i2s_acc",
        "val_s2i_acc",
        "val_pos_sim",
        "val_neg_sim",
    ]

    def __init__(self, logs_dir: Path, log_interval: int, run_logger: Optional[logging.Logger] = None):
        super().__init__()
        self.logs_dir = logs_dir
        self.log_interval = max(1, int(log_interval))
        self.csv_path = self.logs_dir / "training_metrics.csv"
        self.run_logger = run_logger
        self.train_start = 0.0
        self.last_log_time = 0.0
        self.last_logged_step = -1

    @staticmethod
    def _as_float(value, default=float("nan")):
        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            return float(value.detach().float().cpu().item())
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _hms(seconds: float) -> str:
        seconds = max(0, int(seconds))
        return str(datetime.timedelta(seconds=seconds))

    def _ensure_header(self):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def on_train_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self._ensure_header()
        now = time.time()
        self.train_start = now
        self.last_log_time = now
        self.last_logged_step = -1
        if self.run_logger is not None:
            self.run_logger.info("Starting AstroCLIP training loop.")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return

        step = int(trainer.global_step)
        if step <= 0 or step == self.last_logged_step or (step % self.log_interval) != 0:
            return

        now = time.time()
        dt = now - self.last_log_time
        self.last_log_time = now

        metrics = trainer.callback_metrics

        max_steps = trainer.max_steps
        if max_steps is None or max_steps <= 0:
            max_steps = trainer.estimated_stepping_batches
        max_steps = max(1, int(max_steps))

        progress = float(step) / float(max_steps)
        elapsed = now - self.train_start
        eta = (max_steps - step) * (dt / float(self.log_interval))

        optimizer = trainer.optimizers[0] if trainer.optimizers else None
        lr = float("nan")
        if optimizer is not None and len(optimizer.param_groups) > 0:
            lr = float(optimizer.param_groups[0]["lr"])

        mem_gb = float("nan")
        if torch.cuda.is_available():
            mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            torch.cuda.reset_peak_memory_stats()

        grad_total = self._as_float(getattr(pl_module, "_last_grad_norms", {}).get("total"))
        grad_img = self._as_float(getattr(pl_module, "_last_grad_norms", {}).get("images"))
        grad_spec = self._as_float(getattr(pl_module, "_last_grad_norms", {}).get("spectra"))
        grad_back = self._as_float(getattr(pl_module, "_last_grad_norms", {}).get("backbone"))

        row = [
            step,
            int(trainer.current_epoch),
            f"{progress:.6f}",
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"{self._as_float(metrics.get('train_loss')):.6f}",
            "","","","nan","nan","nan","nan",
            f"{grad_total:.6f}",
            f"{grad_img:.6f}",
            f"{grad_spec:.6f}",
            f"{grad_back:.6f}",
            "nan","none",
            f"{lr:.6e}",
            f"{lr:.6e}",
            f"{lr:.6e}",
            f"{lr:.6e}",
            "nan",
            f"{mem_gb:.4f}",
            f"{(dt / float(self.log_interval)) * 1000.0:.3f}",
            self._hms(elapsed),
            self._hms(eta),
            f"{self._as_float(metrics.get('train_loss_nologit')):.6f}",
            f"{self._as_float(metrics.get('train_i2s_acc')):.6f}",
            f"{self._as_float(metrics.get('train_s2i_acc')):.6f}",
            f"{self._as_float(metrics.get('train_pos_sim')):.6f}",
            f"{self._as_float(metrics.get('train_neg_sim')):.6f}",
            "","","","","",
        ]

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        self.last_logged_step = step

        if self.run_logger is not None:
            self.run_logger.info(
                "Iter %d/%d (%.2f%%) | Loss %.4f | LR %.4e | Norm %.2f | Mem %.2fGB | dt %.2fms (avg) | RT %s | ETA %s",
                step,
                max_steps,
                progress * 100.0,
                self._as_float(metrics.get("train_loss")),
                lr,
                grad_total,
                mem_gb,
                (dt / float(self.log_interval)) * 1000.0,
                self._hms(elapsed),
                self._hms(eta),
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        step = int(trainer.global_step)
        metrics = trainer.callback_metrics

        val_i2s = self._as_float(metrics.get("val_i2s_acc"))
        val_s2i = self._as_float(metrics.get("val_s2i_acc"))

        # Analogue of directional cross-errors for comparability.
        val_spec_from_img = 1.0 - val_i2s if not math.isnan(val_i2s) else float("nan")
        val_img_from_spec = 1.0 - val_s2i if not math.isnan(val_s2i) else float("nan")

        row = [
            step,
            int(trainer.current_epoch),
            "",
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "",
            f"{self._as_float(metrics.get('val_loss')):.6f}",
            f"{val_spec_from_img:.6f}",
            f"{val_img_from_spec:.6f}",
            "nan",
            "nan",
            "nan",
            "nan",
            "","","","","",
            "none",
            "","","","",
            "nan",
            "","","","","","","","","",
            f"{self._as_float(metrics.get('val_loss_nologit')):.6f}",
            f"{val_i2s:.6f}",
            f"{val_s2i:.6f}",
            f"{self._as_float(metrics.get('val_pos_sim')):.6f}",
            f"{self._as_float(metrics.get('val_neg_sim')):.6f}",
        ]

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        if self.run_logger is not None:
            self.run_logger.info(
                "Val @ iter %d | Loss %.4f | SpectraFromImg %.4f | ImgFromSpectra %.4f",
                step,
                self._as_float(metrics.get("val_loss")),
                val_spec_from_img,
                val_img_from_spec,
            )


class MaxRunTimeCheckpointCallback(L.Callback):
    """Stops training after a wall-time threshold and saves an emergency checkpoint."""

    def __init__(self, max_run_hours: str, ckpt_dir: Path, run_logger: Optional[logging.Logger] = None):
        super().__init__()
        self.max_run_hours = max_run_hours
        self.max_run_seconds = parse_time_to_seconds(max_run_hours)
        self.ckpt_dir = ckpt_dir
        self.run_logger = run_logger
        self.train_start_time = 0.0
        self.triggered = False

    def on_train_start(self, trainer, pl_module):
        self.train_start_time = time.time()
        if trainer.is_global_zero and self.run_logger is not None:
            self.run_logger.info(
                "Time limit set to: %s (%d seconds)",
                self.max_run_hours,
                int(self.max_run_seconds),
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.triggered:
            return

        elapsed_seconds = time.time() - self.train_start_time
        if elapsed_seconds <= self.max_run_seconds:
            return

        self.triggered = True
        trainer.should_stop = True

        if trainer.is_global_zero:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            emergency_ckpt = self.ckpt_dir / "last.ckpt"
            elapsed_str = str(datetime.timedelta(seconds=int(elapsed_seconds)))

            if self.run_logger is not None:
                self.run_logger.info(
                    "Time limit reached (%s > %s). Saving emergency checkpoint and stopping...",
                    elapsed_str,
                    self.max_run_hours,
                )

            trainer.save_checkpoint(str(emergency_ckpt))

            if self.run_logger is not None:
                self.run_logger.info("Emergency checkpoint saved to %s", emergency_ckpt)


def build_checkpoint_callback(config: AstroCLIPArrowConfig, ckpt_dir: Path) -> ModelCheckpoint:
    save_type = config.checkpoint_save_type.lower().strip()
    if save_type not in {"best", "last", "both", "all"}:
        raise ValueError("checkpoint_save_type must be one of: best,last,both,all")

    save_last = save_type in {"last", "both"}
    if save_type == "best":
        save_top_k = 1
    elif save_type == "all":
        save_top_k = -1
    else:
        save_top_k = 1

    return ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="astroclip-step{step:07d}-valloss{val_loss_nologit:.4f}",
        monitor="val_loss_nologit",
        mode="min",
        save_last=save_last,
        save_top_k=save_top_k,
        every_n_train_steps=max(1, int(config.checkpoint_interval)),
        auto_insert_metric_name=False,
    )


def resolve_resume_checkpoint(config: AstroCLIPArrowConfig, ckpt_dir: Path) -> Optional[str]:
    if config.init_from.lower().strip() != "resume":
        return None

    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        return str(last_ckpt)

    candidates = sorted(ckpt_dir.glob("*.ckpt"))
    if len(candidates) > 0:
        return str(candidates[-1])

    return None


def main():
    parser = HfArgumentParser((AstroCLIPArrowConfig,))
    config, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    validate_runtime_flags(config)

    L.seed_everything(config.seed, workers=True)

    train_dir = Path(config.train_dir).resolve()
    logs_dir = train_dir / "logs"
    ckpt_dir = train_dir / "weights" / "astroclip_checkpoints"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_logger = configure_training_logger(logs_dir)
    run_logger.info("Logging initialized. Saving logs to: %s", logs_dir / "training.log")

    # Save run config for reproducibility (AstroPT-style).
    save_config_json(config, train_dir / "weights" / "config.json")

    datamodule = AstroCLIPArrowDataModule(config)

    import dotenv

    dotenv.load_dotenv(os.path.join(ASTROCLIP_DIR, "astroclip", ".env"))
    root = os.environ.get("ASTROCLIP_ROOT", str(ASTROCLIP_DIR))

    image_encoder = ImageHead(
        config=os.path.join(root, "astroclip", "astrodino", "config.yaml"),
        model_weights=os.path.join(root, "pretrained", "astrodino.ckpt"),
        save_directory=os.path.join(str(train_dir), "astrodino"),
        freeze_backbone=config.freeze_image_backbone,
    )

    spectrum_encoder = SpectrumHead(
        model_path=os.path.join(root, "pretrained", "specformer.ckpt"),
        freeze_backbone=config.freeze_spectrum_backbone,
    )

    model = AstroClipTrainModule(
        image_encoder=image_encoder,
        spectrum_encoder=spectrum_encoder,
        temperature=15.5,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        epochs=config.max_epochs,
        eta_min=config.lr_min,
        beta1=config.beta1,
        beta2=config.beta2,
        lr_warmup_steps=config.lr_warmup_steps,
        lr_decay_steps=config.lr_decay_steps,
    )

    if config.log_via_wandb:
        logger = WandbLogger(
            project=config.wandb_project,
            entity=(config.wandb_entity or os.environ.get("WANDB_ENTITY_NAME")),
            save_dir=str(train_dir),
            name=config.train_name,
            log_model=False,
        )
    else:
        logger = CSVLogger(save_dir=str(train_dir), name="lightning_csv")

    lr_monitor = LearningRateMonitor(logging_interval="step")
    ckpt_callback = build_checkpoint_callback(config, ckpt_dir)
    early_stopping = EarlyStopping(
        monitor="val_loss_nologit",
        mode="min",
        patience=max(1, int(config.early_stopping_patience)),
    )
    astropt_csv = AstroPTStyleCSVCallback(
        logs_dir=logs_dir,
        log_interval=config.log_interval,
        run_logger=run_logger,
    )

    # AstroPT-compatible stopping behavior:
    # - use max_steps if explicitly provided (>0)
    # - otherwise stop at max_iters
    effective_max_steps = int(config.max_iters)
    if int(config.max_steps) > 0:
        effective_max_steps = int(config.max_steps)

    run_logger.info(
        "Stopping config | max_iters=%d | max_steps=%d | effective_max_steps=%d | max_epochs=%d",
        int(config.max_iters),
        int(config.max_steps),
        int(effective_max_steps),
        int(config.max_epochs),
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. This training is configured for DDP on GPUs.")

    available_gpus = torch.cuda.device_count()
    requested_gpus = max(1, int(config.ddp_num_gpus))
    if available_gpus < requested_gpus:
        raise RuntimeError(
            f"Requested {requested_gpus} GPUs for DDP, but only {available_gpus} are visible."
        )
    devices = requested_gpus

    # Auto-tune workers in DDP to reduce CPU contention in shared HPC nodes.
    if devices > 1:
        try:
            total_available_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            total_available_cpus = os.cpu_count() or 1

        cpus_per_gpu = max(1, total_available_cpus // devices)
        suggested_workers = int(cpus_per_gpu * 0.5)
        suggested_workers = max(2, min(4, suggested_workers))

        original_workers = int(config.num_workers)
        config.num_workers = suggested_workers

        run_logger.info(
            "DDP CPU config | gpus=%d | visible_cpus=%d | num_workers: %d -> %d",
            int(devices),
            int(total_available_cpus),
            int(original_workers),
            int(suggested_workers),
        )

    # Match AstroPT behavior: gradient_accumulation_steps is specified globally,
    # then divided across DDP ranks to keep effective global batch consistent.
    requested_accum_steps = max(1, int(config.gradient_accumulation_steps))
    if devices > 1 and requested_accum_steps % devices == 0:
        accumulate_grad_batches = requested_accum_steps // devices
    else:
        accumulate_grad_batches = requested_accum_steps

    effective_global_batch = int(config.batch_size) * int(devices) * int(accumulate_grad_batches)
    # val_check_interval in the config is measured in optimizer steps (matching
    # train_multimodal_arrow.py's eval_interval semantics). Lightning's Trainer
    # counts val_check_interval in raw data batches, so we multiply by
    # accumulate_grad_batches to keep the same cadence. check_val_every_n_epoch=None
    # is required so Lightning does not enforce that the interval fits within one epoch.
    val_check_interval_batches = max(1, int(config.val_check_interval) * int(accumulate_grad_batches))

    # Overwrite config after runtime adjustments so saved json reflects effective settings.
    save_config_json(config, train_dir / "weights" / "config.json")

    run_logger.info(
        "Batch config | per_gpu_batch=%d | gpus=%d | accumulate_grad_batches=%d | effective_global_batch=%d",
        int(config.batch_size),
        int(devices),
        int(accumulate_grad_batches),
        int(effective_global_batch),
    )

    strategy = config.ddp_strategy
    if strategy == "ddp" and config.ddp_find_unused_parameters:
        strategy = "ddp_find_unused_parameters_true"

    runtime_guard = None
    if config.max_run_hours is not None:
        runtime_guard = MaxRunTimeCheckpointCallback(
            max_run_hours=config.max_run_hours,
            ckpt_dir=ckpt_dir,
            run_logger=run_logger,
        )

    callbacks = [ckpt_callback, lr_monitor, early_stopping, astropt_csv]
    if runtime_guard is not None:
        callbacks.append(runtime_guard)

    trainer = L.Trainer(
        default_root_dir=str(train_dir),
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        max_epochs=config.max_epochs,
        max_steps=effective_max_steps,
        accumulate_grad_batches=accumulate_grad_batches,
        precision=config.precision,
        gradient_clip_val=config.grad_clip,
        val_check_interval=val_check_interval_batches,
        check_val_every_n_epoch=None,
        limit_val_batches=max(1, int(config.limit_val_batches)),
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=max(1, int(config.log_interval)),
        num_sanity_val_steps=0,
    )

    ckpt_path = resolve_resume_checkpoint(config, ckpt_dir)
    if ckpt_path is not None:
        run_logger.info("Resuming from checkpoint: %s", ckpt_path)

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
