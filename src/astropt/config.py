import json
import os
from dataclasses import dataclass, asdict, field, fields
from typing import Optional, List, Dict, Any

@dataclass
class TrainingConfig:
    """
    Hyperparameters and configuration settings for AstroPT training.
    
    This dataclass acts as the single source of truth for the experiment.
    Arguments can be overridden via command line (CLI) using HfArgumentParser.
    """
    
    #--- Training Metadata ---#
    train_name: Optional[str] = None            # Name of the training
    train_date: Optional[str] = None            # Date of the training
    train_description: Optional[str] = None     # Description or comment about the training

    #--- I/O & Paths ---#
    train_dir: Optional[str] = None             # Training output directory
    data_dir: str = "/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated"   # Dataset directory
    metadata_path: str = "/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits" # FITS metadata catalog path
    
    #--- Model Architecture for the 100M Parameter Setup ---#
    n_layer: int = 12               # Depth of the Transformer
    n_head: int = 12                # Number of attention heads
    n_embd: int = 768               # Embedding dimension (width of the network)
    n_chan: int = 4                 # Input channels: 1 VIS + 3 NISP (Y, J, H)
    block_size: int = 1024          # Context length (max tokens per sample)
    dropout: float = 0.0            # Regularization (0.0 for pretraining is standard)
    bias: bool = False              # Learnable bias in Linear layers (False is modern/faster)
    attn_type: str = "causal"       # Attention mechanism type: casual or prefix 
    backbone: str = "native"        # Model backbone: native or llm
    use_qlora: bool = False         # Use Quantized Low-Rank Adaptation
    loss_type: str = "huber"        # Options: l1 / mae, mse, huber
    loss_huber_delta: float = 1.0   # Delta value for controlling Huber Loss Behaviour (default 1.0)
    
    #--- Data Loading ---#
    batch_size: int = 16            # Micro-batch size per GPU (what fits in VRAM)
    num_workers: int = 8            # Optimized for Teide HPC (CPU cores for loading)
    persistent_workers: bool = True # Keep workers active
    prefetch_factor: int = 2        # How many batches to preload per worker
    pin_memory: bool = True         # Faster transfer RAM -> VRAM
    spiral: bool = True             # Whether to apply spiral readout to galaxy images
    init_from: str = "scratch"      # Training from scratch or resume training
    use_aug: bool = False           # Active data augmentation by using image rotation
    use_pretokenized: bool = False  # Bypass tokenization for a discrete training
    applied_filters: List[str] = field(default_factory=lambda: []) # Dynamic dataset filters applied during loading
    
    #--- Modality Orchestration ---#
    # Defines which specific modalities to load and train on
    modalities: List[str] = field(default_factory=lambda: ["EuclidImage", "DESISpectrum"])
    
    #--- Multimodality Mixing Parameters ---#
    use_token_mixing: bool = False              # Enable cross-modal interleaving
    token_mixing_block_size: int = 16           # Interleaving block size
    token_mixing_stochastic: bool = False       # Enable stochastic block sizes
    token_mixing_min_block_size: int = 32       # Minimum block size for stochastic mixing
    token_mixing_max_block_size: int = 128      # Maximum block size for stochastic mixing
    token_mixing_seed: int = 61                 # Seed for reproducible stochastic mixing
    shuffle_modality_train: bool = False        # Shuffle modality order during training
    shuffle_modality_val: bool = False          # Shuffle modality order during validation
    modality_dropout_prob: float = 0.00         # Probability to zero one modality in a micro-step
    modality_dropout_mode: str = "none"         # none, images, spectra, random
    cross_reconstruction_loss_use: bool = False # Enable cross reconstruction explicitly via targets
    cross_reconstruction_weight: float = 0.0    # Default weight for all cross-modal directions
    cross_reconstruction_weight_to_spectra: float = 0.0  # Extra boost when spectra is the TARGET (spectral-first design)
    spectral_modality_key: str = "DESISpectrum" # Key that identifies the primary spectral modality
    
    # Dual Modality Identity
    use_cls_token: bool = False                 # Use a [CLS] token that is prepended to the sequence
    cls_position: str = "last"                  # Position of the [CLS] token: 'first' or 'last'
    use_modality_embeddings: bool = False       # Use a different vector to distinguish modalities
    use_aperture_embedding: bool = False        # Use aperture-aware embeddings to distinguish central fiber (core) from disk (outskirts)

    # CLIP: Contrastive Alignment Architecture
    use_contrastive_alignment: bool = False      # Master flag: activates the V4 architecture
    clip_fusion_layer: int = -1                  # Layer where fusion starts (-1 = n_layer // 2)
    clip_loss_weight: float = 0.5                # Weight of the CLIP loss relative to the reconstruction loss
    clip_projection_dim: int = 256               # CLIP projector dimension (bottleneck)
    clip_temperature: float = 0.07               # Temperature for the InfoNCE loss

    # Images General parameters
    images_train: bool = True           # Images bool flag for enabling training
    images_size: int = 224              # Images side size in pixels
    images_patch_size: int = 16         # Side size in pixels of each patch in an image
    images_channels: int = 4            # Channels per image (VIS + NISP Y,J,H)
    images_filters: List[str] = field(default_factory=lambda: ["VIS", "Y", "J", "H"]) # Euclid filters
    images_loss_weight: float = 1.0     # Images importance for training
    images_embed_pos: bool = True       # Images embedding positions learning
    images_pos_input_size: int = 1      # Images position input size
    images_norm_type: str = "z_score"   # Normalization method: constant, z_score or asinh
    images_norm_scaler: float = 1.0     # Scaler factor if normalization requieres it (default 1.0)
    images_norm_const: float = 1.0      # Normalization global constant for images: P99=7.603847
    images_mask: bool = False           # Enable tactical masking for image patches
    images_mask_prob: float = 0.0       # Probability to mask each image patch
    
    # Specific images modalities parameters
    # Euclidimages
    EuclidImage_channels: int = 4
    EuclidImage_filters: List[str] = field(default_factory=lambda: ["VIS", "Y", "J", "H"])
    
    # Spectra General parameters
    spectra_train: bool = True              # Spectra bool flag for enabling training
    spectra_inverse: bool = False           # Reading spectra from red to blue 
    spectra_size: int = 7781                # Spectra total size
    spectra_patch_size: int = 10            # Patch size for each spectrum
    spectra_loss_weight: float = 1.0        # Spectra importance for training 
    spectra_embed_pos: bool = True          # Spectra embedding positions learning
    spectra_pos_input_size: int = 1         # Spectra position input size
    spectra_norm_type: str = "z_score"      # Normalization method: constant, z_score or asinh
    spectra_norm_scaler: float = 1.0        # Scaler factor if normalization requieres it (default 1.0)
    spectra_norm_const: float = 1.0         # Normalization global constant for spectra: P99=7.956048
    spectra_mask: bool = False              # Enable tactical masking for spectra patches
    spectra_mask_prob: float = 0.0          # Probability to mask each spectrum patch
    
    # Specific spectra modalities parameters
    # DESISpectra
    DESISpectrum_size: int = 7781
    
    #--- Optimization of the Learning Process ---#
    max_iters: int = 75_000         # Total training iters (NOT epochs)
    weight_decay: float = 0.1       # Regularization to prevent overfitting
    beta1: float = 0.9              # AdamW parameter
    beta2: float = 0.95             # AdamW parameter
    grad_clip: float = 1.0          # Stabilizes training if gradients explode
    
    # Gradient Accumulation: Simulates a larger batch size. 
    # Effective batch = 16 * 40 = 640
    gradient_accumulation_steps: int = 40 
    
    #--- Learning Rate Scheduler ---#
    learning_rate: float = 0.0003   # Learning rate per weight update
    lr_min: float = 0.00003         # Minimum LR (usually 10% of max)
    lr_mult_images: float = 1.0     # LR multiplier for image encoder/decoder modality
    lr_mult_spectra: float = 1.0    # LR multiplier for spectra encoder/decoder modality
    lr_mult_backbone: float = 1.0   # LR multiplier for transformer/shared modality
    lr_decay: bool = True           # Activates the variable learning rate decay
    lr_warmup_iters: int = 4_000    # Steps to ramp up LR from 0 to max
    lr_decay_iters: int = 65_000    # Steps to decay LR down to min

    #--- Logging & Checkpointing ---#
    eval_interval: int = 1_000              # How often to validate
    eval_batches: int = 100                 # How many batches to use for validation
    log_interval: int = 200                 # How often to print to console/WandB
    checkpoint_interval: int = 1_000        # How often to save .pt files
    checkpoint_save_type: str = "both"      # Checkpoint saving mode: best, last, both or all
    early_stopping_patience: int = 15       # Stop if no improvement after N evals
    early_stopping_min_iters: int = 35000   # Minimum iterations before starting to count patience

    #--- System & Backend ---#
    device: str = "cuda"                    # CPU/GPU device interface: cpu, cuda or mps
    dtype: str = "bfloat16"                 # 'bfloat16' is best for A100 GPUs
    compile: bool = True                    # PyTorch 2.0 compiler
    compile_mode: str = "default"           # Compilation mode
    backend: str = "nccl"                   # Communication backend for DDP
    max_run_hours: Optional[str] = None     # Force stop after "HH:MM:SS"

    #--- External Monitoring ---#
    log_via_wandb: bool = False             # Weight and bias (wandb) logging
    wandb_project: str = "AstroPT-Arrow"    # wandb project name
    wandb_run_name: Optional[str] = None    # Training name
    log_emissions: bool = False             # CodeCarbon logging
    profile: bool = False                   # Enable PyTorch Profiler (Trace analysis)

    #--- Optional Diagnostics & Ablation Flags ---#
    diagnostics_enabled: bool = True        # Enable modality diagnostics CSV/WandB logs
    diagnostics_interval: int = 200         # Iter interval for diagnostics writes/logs
    diagnostics_track_losses: bool = True   # Track per-modality losses
    diagnostics_track_grads: bool = True    # Track per-branch gradient norms
    diagnostics_file_name: str = "training_diagnostics.csv"

    def __post_init__(self):
        import datetime
        import re
        # Date
        if self.train_date is None:
            self.train_date = datetime.datetime.now().strftime("%Y%m%d")
        # Output directory
        if self.train_dir is None:
            clean_name = self.train_name.lower() if self.train_name else "default_run"
            clean_name = re.sub(r'[^a-z0-9]', ' ', clean_name)
            tokens = clean_name.split()
            suffix_name = "_".join(tokens)
            self.train_dir = f"./logs/astropt_100M_250K_arrow_{self.train_date}_{suffix_name}"
            
        if self.attn_type == "prefix" and self.use_token_mixing:
            raise ValueError("Prefix attention is not compatible with token mixing. Set use_token_mixing = False.")

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in self.__dict__.items():
            if not k.startswith("_") and k not in d:
                d[k] = v
        return d

    def save_to_json(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def save_to_yaml(self, filepath: str):
        import yaml
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load_from_yaml(cls, filepath: str):
        import yaml
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        valid_fields = {f.name for f in fields(cls)}
        dataclass_data = {k: v for k, v in data.items() if k in valid_fields}
        extra_data = {k: v for k, v in data.items() if k not in valid_fields}
        
        instance = cls(**dataclass_data)
        for k, v in extra_data.items():
            setattr(instance, k, v)
        return instance
