import torch
import torch.distributed as dist
import logging
import os
import sys
import time
import math
import numpy as np
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
from contextlib import nullcontext
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
import datetime
import inspect
from dataclasses import asdict
from astropt.model import GPT, GPTConfig, ModalityRegistry, ModalityConfig
from astropt.model_v4 import GPT_V4
from astropt.config import TrainingConfig
from astropt.training_utils import (
    create_dataloaders,
    get_learning_rate,
    parse_time_to_seconds,
    maybe_apply_modality_dropout,
    summarize_modality_dropout,
    summarize_prefix_modality,
    compute_branch_grad_norms,
    compute_cross_reconstruction_loss,
    _compute_modality_losses,
    _param_branch_name,
)
from astropt.dataloader_multimodal import MultimodalDatasetArrow

class Trainer:
    def __init__(self, config: TrainingConfig, ddp: bool, ddp_rank: int, ddp_world_size: int, device: str, weights_dir: Path, logs_dir: Path):
        self.config = config
        self.ddp = ddp
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.device = device
        self.weights_dir = weights_dir
        self.logs_dir = logs_dir
        self.logger = logging.getLogger('AstroPT')
    def _create_model(self, registry: ModalityRegistry) -> torch.nn.Module:
        """
        Instantiates the GPT model, moves it to the target device, compiles it,
        and wraps it in DDP if distributed training is active.

        Args:
            registry (ModalityRegistry): Configuration of input modalities (img, spectra).

        Returns:
            torch.nn.Module: The ready-to-train model.
        """

        # Activating the logger object
        logger = logging.getLogger("AstroPT")
        config = self.config
        device = config.device
        ddp = self.ddp

        # Model configuration
        gpt_config = GPTConfig(
            block_size=config.block_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            bias=config.bias,
            dropout=config.dropout,
            attn_type=config.attn_type,
            use_qlora=config.use_qlora,
            loss_type=config.loss_type,
            backbone=config.backbone,
            use_token_mixing=config.use_token_mixing,
            token_mixing_block_size=config.token_mixing_block_size,
            token_mixing_stochastic=config.token_mixing_stochastic,
            token_mixing_min_block_size=config.token_mixing_min_block_size,
            token_mixing_max_block_size=config.token_mixing_max_block_size,
            token_mixing_seed=config.token_mixing_seed,
            cross_reconstruction_loss_use=config.cross_reconstruction_loss_use,
            cross_reconstruction_weight=config.cross_reconstruction_weight,
            use_cls_token=config.use_cls_token,
            cls_position=config.cls_position,
            use_modality_embeddings=config.use_modality_embeddings,
            use_aperture_embedding=config.use_aperture_embedding,
            use_contrastive_alignment=config.use_contrastive_alignment,
            clip_fusion_layer=config.clip_fusion_layer,
            clip_projection_dim=config.clip_projection_dim,
        )

        # Instantiate the Model
        if config.use_contrastive_alignment:
            model = GPT_V4(gpt_config, registry)
            logger.info(f"Initializing GPT_V4 Model with {config.n_layer} layers (Contrastive Alignment ENABLED)")
        else:
            model = GPT(gpt_config, registry)
            logger.info(f"Initializing GPT Model with {config.n_layer} layers")

        # Move Model to Selected Device
        model.to(device)

        # Pythorch Compilation
        if config.compile:
            if ddp:
                # Work around a known Dynamo distributed partition bug where SymInt
                # outputs can be materialized as Python ints in AOTAutograd wrappers.
                torch._dynamo.config.optimize_ddp = False
                torch._dynamo.config.capture_scalar_outputs = True
            logger.info("Compiling model with torch.compile...")
            model = torch.compile(model, mode=config.compile_mode, dynamic=False)

        # DDP Wrapping
        if ddp:

            # Local RANK for each process
            local_rank = int(os.environ.get("LOCAL_RANK", 0))

            # Model configuration for DDP
            model = DDP(
                model, 
                device_ids=[local_rank], 
                output_device=local_rank,
                find_unused_parameters=False 
            )

        return model


    def _create_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """
        Creates the AdamW optimizer with weight decay handling.

        It separates parameters into two groups:
        1. Decay Group: Weights of Linear and Embedding layers (2D tensors).
        2. No-Decay Group: Biases, LayerNorms, and 1D tensors.

        Args:
            model (torch.nn.Module): The loaded GPT model.

        Returns:
            torch.optim.Optimizer: Configured AdamW optimizer.
        """

        # Activating the logger object
        logger = logging.getLogger("AstroPT")
        # Local configuration alias
        config = self.config

        # Filter parameters that require gradients
        param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

        # Keep legacy two-group optimizer layout unless branch multipliers are requested.
        use_branch_lr = any(
            abs(v - 1.0) > 1e-12
            for v in (config.lr_mult_images, config.lr_mult_spectra, config.lr_mult_backbone)
        )

        if not use_branch_lr:
            decay_params = []
            nodecay_params = []

            for _, p in param_dict.items():
                if p.dim() >= 2:
                    decay_params.append(p)
                else:
                    nodecay_params.append(p)

            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            logger.info(
                f"Optimizer Config: {len(decay_params)} tensors ({num_decay_params:,} params) with decay."
            )
            logger.info(
                f"Optimizer Config: {len(nodecay_params)} tensors ({num_nodecay_params:,} params) without decay."
            )

            optim_groups = [
                {"params": decay_params, "weight_decay": config.weight_decay, "lr_scale": 1.0, "group_name": "decay"},
                {"params": nodecay_params, "weight_decay": 0.0, "lr_scale": 1.0, "group_name": "nodecay"},
            ]
        else:
            branch_lr = {
                "images": float(config.lr_mult_images),
                "spectra": float(config.lr_mult_spectra),
                "backbone": float(config.lr_mult_backbone),
            }

            grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for name, param in param_dict.items():
                decay_key = "decay" if param.dim() >= 2 else "nodecay"
                branch = _param_branch_name(name)
                key = (branch, decay_key)

                if key not in grouped:
                    grouped[key] = {
                        "params": [],
                        "weight_decay": config.weight_decay if decay_key == "decay" else 0.0,
                        "lr_scale": branch_lr[branch],
                        "group_name": f"{branch}_{decay_key}",
                    }
                grouped[key]["params"].append(param)

            optim_groups = list(grouped.values())
            logger.info("Optimizer Config: branch LR multipliers enabled.")
            for group in optim_groups:
                n_params = sum(p.numel() for p in group["params"])
                logger.info(
                    f"  - {group['group_name']}: {len(group['params'])} tensors "
                    f"({n_params:,} params), wd={group['weight_decay']}, lr_scale={group['lr_scale']}"
                )

        # Check for Fused AdamW (faster CUDA kernel)
        use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        extra_args = dict(fused=True) if use_fused and torch.cuda.is_available() else dict()

        if use_fused:
            logger.info("Using Fused AdamW implementation (faster).")

        # Instantiate AdamW
        optimizer = torch.optim.AdamW(
            optim_groups, 
            lr=config.learning_rate, 
            betas=(config.beta1,config.beta2), 
            **extra_args
        )

        return optimizer


    def train(self):
        config = self.config
        ddp = self.ddp
        ddp_rank = self.ddp_rank
        ddp_world_size = self.ddp_world_size
        device = self.device
        weights_dir = self.weights_dir
        logs_dir = self.logs_dir
        logger = self.logger
        train_dir = Path(config.train_dir)
        """
        Main training entry point.

        Orchestrates the entire training pipeline:
        1. Sets up the DDP environment and logging.
        2. Loads the configuration and datasets.
        3. Instantiates the Model and Optimizer.
        4. Manages Checkpoint loading (resume vs scratch).
        5. Executes the training loop (implemented in Part 2).
        """

        #--- TRAINING LOGIC ---#
        # Imporved file for workflow control
        improve_file_path = train_dir / ".improved"

        # Log basic training info
        if ddp_rank == 0:
            logger.info(f"Starting AstroPT training on {device} (DDP: {ddp})")

            # Logging training configuration
            config_dict = asdict(config)
            config_str = "\n".join([f"    {k}: {v}" for k, v in config_dict.items()])
            logger.info(f"Training config:\n{config_str}")

        # Changing configuration for DDP mode
        if ddp: 

            # Automatic gradient accumulation steps
            if config.gradient_accumulation_steps % ddp_world_size != 0:
                if ddp_rank == 0:
                    logger.warning(f"Grad Accum {config.gradient_accumulation_steps} "
                                   f"is not divisible by {ddp_world_size}. It will be rounded down.")

            # Original configuration        
            original_accum = config.gradient_accumulation_steps

            # New value
            config.gradient_accumulation_steps = config.gradient_accumulation_steps // ddp_world_size

            # Effective batch size
            eff_batch_size = config.batch_size * config.gradient_accumulation_steps * ddp_world_size


            # Automatic workers number
            try:
                total_available_cpus = len(os.sched_getaffinity(0))
            except AttributeError:
                total_available_cpus = os.cpu_count() or 1

            # Computing how many CPU can be assigned to each GPU
            cpus_per_gpu = total_available_cpus // ddp_world_size

            # Assigning the half of the total
            suggested_workers = int(cpus_per_gpu * 0.5)

            # Selecting the suggested worker number
            suggested_workers = max(2, min(4, suggested_workers))

            # Changing the configuration
            config.num_workers = suggested_workers

            # Only master shows
            if ddp_rank == 0:
                logger.info(f"DDP Detected: {ddp_world_size} GPUs. {total_available_cpus} CPUs")
                logger.info(f"  -> Adjusting Gradient Accumulation: {original_accum} "
                            f"-> {config.gradient_accumulation_steps} per GPU.")
                logger.info(f"  -> Effective Batch Size maintained at: {eff_batch_size}")
                logger.info(f"  -> Assigning {suggested_workers} workers per DataLoader.")


        # Set reproducibility seeds
        seed_offset = ddp_rank 
        torch.manual_seed(61 + seed_offset)
        np.random.seed(61 + seed_offset)

        # Optimization: Allow TF32 on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True 
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.autograd.set_detect_anomaly(True)

        try:
            #--- DATA PIPELINE ---#

            logger.info("Initializing DataLoaders...")
            train_loader, val_loader, registry = create_dataloaders(config, ddp)

            #--- MODEL & OPTIMIZER ---#

            # Determine precision type for Mixed Precision Training (AMP)
            ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

            # Create the Automatic Mixed Precision context manager
            ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)

            # Scaler for small grandient values
            scaler = torch.amp.GradScaler("cuda", enabled=(ptdtype == torch.float16))

            # Initialize Model
            model = self._create_model(registry)
            # Reference to the raw (unwrapped) model for MFU estimation and seed setting
            raw_model = model.module if ddp else model
            if hasattr(raw_model, "_orig_mod"):
                raw_model = raw_model._orig_mod

            # Initialize Optimizer
            optimizer = self._create_optimizer(model)
            
            #--- CHECKPOINT MANAGEMENT ---#

            # Define where the checkpoint file should be
            ckpt_path_best = weights_dir / "ckpt_best.pt"
            ckpt_path_last = weights_dir / "ckpt_last.pt"

            # State variables
            iter_num = 0
            epoch_num = 0
            best_val_loss = 1e9
            accumulated_time = 0.0
            early_stop_counter = 0

            # Resume a training
            if config.init_from == 'resume':

                ckpt_path = None

                # 1. Try lo load the LAST checkpoint file
                if ckpt_path_last.is_file():
                    ckpt_path = ckpt_path_last
                    logger.info(f"Resuming from LAST checkpoint: {ckpt_path}")

                # 2. If 'last' is missing, look for the latest 'ckpt_iter_XXXXXX.pt' (Mode "all")
                else:
                    # Get all files starting with 'ckpt_iter_' and ending in '.pt'
                    iter_checkpoints = list(weights_dir.glob('ckpt_iter_*.pt'))

                    if iter_checkpoints:
                        # Sort them to find the latest iteration
                        iter_checkpoints.sort()
                        ckpt_path = iter_checkpoints[-1]
                        logger.info(f"Resuming from LATEST HISTORY checkpoint: {ckpt_path}")

                # 3. Try BEST if neither LAST nor HISTORY exist
                if ckpt_path is None and ckpt_path_best.is_file():
                    ckpt_path = ckpt_path_best
                    logger.info(f"Resuming from BEST checkpoint: {ckpt_path}")

                # 4. If nothing is found -> Scratch
                if ckpt_path is None:
                    logger.warning(f"Resume requested but no checkpoints found. Starting from SCRATCH.")

                if ckpt_path:
                    # Load the file to the current device (CPU or GPU)
                    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

                    # Load Model Weights
                    state_dict = checkpoint['model']

                    new_state_dict = {}
                    # Clean the state_dict keys
                    for k, v in list(state_dict.items()):
                        clean_k = k.replace("module.", "").replace("_orig_mod.", "")
                        new_k = f"module._orig_mod.{clean_k}"
                        new_state_dict[new_k] = v

                    msg = model.load_state_dict(new_state_dict, strict=False)
                    logger.info(f"Load state result: {msg}")

                    # Load Optimizer State
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer'])
                    except ValueError as e:
                        logger.warning(
                            "Could not restore optimizer state (likely due to optimizer group changes). "
                            f"Continuing with fresh optimizer. Details: {e}"
                        )

                    # Restore Training Counters
                    iter_num = checkpoint['iter_num']
                    epoch_num = checkpoint.get('epoch_num', 0)

                    # Loading the correct loss
                    loaded_best_loss = checkpoint['best_val_loss']

                    if ckpt_path_best.is_file():
                        try:
                            # Load metadata
                            best_ckpt_data = torch.load(ckpt_path_best, map_location=device, weights_only=False)
                            disk_best_loss = best_ckpt_data.get('best_val_loss', 1e9)

                            # Select the best loss
                            if disk_best_loss < loaded_best_loss:
                                logger.info(f"Correction: Disk loss {disk_best_loss} better than {loaded_best_loss}. Updating.")
                                best_val_loss = disk_best_loss
                            else:
                                best_val_loss = loaded_best_loss
                        except Exception as e:
                            logger.warning(f"Could not verify ckpt_best.pt on disk: {e}. Trusting loaded checkpoint.")
                            best_val_loss = loaded_best_loss
                    else:
                        best_val_loss = loaded_best_loss


                    accumulated_time = checkpoint.get('total_run_time', 0.0)

                    # Changing time to hours
                    prev_hours = accumulated_time / 3600

                    logger.info(f"Resumed successfully at iteration {iter_num} - "
                                f"Best Loss: {best_val_loss:.4f} - Previous run time: {prev_hours:.2f} hours.")


                    # Only remove .improved file in resume mode
                    if ddp_rank == 0:
                        improve_file_path.unlink(missing_ok=True)
                        logger.info("Removing .improved file.")

                else:
                    logger.info("Starting training from scratch.")

                # If a training is resume, patience to zero again
                early_stop_counter = 0



            #--- TRAINING LOOP SETUP ---#

            # Create an infinite iterator for the DataLoader
            train_iter = iter(train_loader)

            # Initialize timers and counters
            t0 = time.time()
            run_start_time = time.time()
            last_log_time = time.time()
            initial_iter = iter_num

            # Wait for all GPUs synchronization
            if ddp:
                dist.barrier()

            # Define dynamic CSV headers for N modalities
            names = registry.names()
            base_cols = ["iter", "epoch", "progress", "timestamp", "train_loss", "val_loss", "clip_loss", "recon_loss", "val_clip_loss", "val_recon_loss"]
            val_cross_cols = [f"val_loss_reconstruct_{mod}" for mod in names]
            loss_cols = [f"loss_{mod}" for mod in names]
            cross_loss_cols = [f"cross_loss_{mod}" for mod in names]
            grad_cols = ["grad_norm"] + [f"grad_{mod}" for mod in names] + ["grad_backbone", "clipped", "dropped_modality"]
            lr_cols = ["lr"] + [f"lr_{mod}" for mod in names] + ["lr_backbone"]
            sys_cols = ["mfu", "mem_gb", "dt_ms", "rt_hms", "eta_hms"]
            
            csv_headers = base_cols + val_cross_cols + loss_cols + cross_loss_cols + grad_cols + lr_cols + sys_cols
            diag_headers = ["iter", "epoch", "timestamp", "dropout_mode", "dropout_applied"] + loss_cols + ["grad_total"] + [g for g in grad_cols if g != "grad_norm" and g != "clipped" and g != "dropped_modality"] + ["lr_base"] + [l for l in lr_cols if l != "lr"]

            # CSV Logging setup
            if ddp_rank == 0:
                csv_path = logs_dir / "training_metrics.csv"
                if config.init_from == 'scratch' or not csv_path.is_file():
                    with open(csv_path, "w") as f:
                        f.write(",".join(csv_headers) + "\n")

                diagnostics_path = logs_dir / config.diagnostics_file_name
                if config.diagnostics_enabled and (config.init_from == 'scratch' or not diagnostics_path.is_file()):
                    with open(diagnostics_path, "w") as f:
                        f.write(",".join(diag_headers) + "\n")

            # WANDB Configuration
            if ddp_rank == 0:
                logger.info(f"Starting training loop from iteration {iter_num}...")
                if config.log_via_wandb and _WANDB_AVAILABLE:
                    wandb.init(project=config.wandb_project, 
                            name=config.wandb_run_name, 
                            config=asdict(config))

            # Pytorch profiler setup
            prof = nullcontext()

            if config.profile:
                # Schedule: 
                # - wait=10: Avoid 10 first iters
                # - warmup=2: Starting the tracer
                # - active=4: Saving iters 12, 13, 14 and 15
                # - repeat=1: Just one time
                wait, warmup, active = 10, 2, 4

                prof = profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
                    on_trace_ready=tensorboard_trace_handler(str(logs_dir / "profiler_trace")),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
                )

                if ddp_rank == 0:
                    logger.info(f"Profiler enabled. Recording iterations {wait+warmup} to {wait+warmup+active}...")

                # Starting the profiler
                prof.start()

            # Max time running before autosaving and finishing
            max_run_seconds = None
            if config.max_run_hours is not None:
                try:
                    max_run_seconds = parse_time_to_seconds(config.max_run_hours)
                    if ddp_rank == 0:
                        logger.info(f"Time limit set to: {config.max_run_hours} ({max_run_seconds} seconds)")
                except Exception as e:
                    if ddp_rank == 0:
                        logger.error(f"Error parsing max_run_hours: {e}")
                    sys.exit(1)

            # Update correct epoch
            if ddp and train_loader.sampler is not None:
                train_loader.sampler.set_epoch(epoch_num)

            #--- THE TRAINING LOOP ---#
            while iter_num <= config.max_iters:

                # SET LEARNING RATE for this iteration
                lr = get_learning_rate(iter_num, config)
                for param_group in optimizer.param_groups:
                    lr_scale = float(param_group.get('lr_scale', 1.0))
                    param_group['lr'] = lr * lr_scale


                # Variable to accumulate loss
                loss_accum_for_log = 0.0
                clip_loss_accum_for_log = 0.0
                recon_loss_accum_for_log = 0.0

                # Optional modality diagnostics accumulators (dynamic registry-driven)
                modality_loss_sums = {m: 0.0 for m in registry.names()}
                modality_loss_counts = {m: 0 for m in registry.names()}
                cross_loss_sums = {m: 0.0 for m in registry.names()}
                cross_loss_counts = {m: 0 for m in registry.names()}
                modality_drop_counts = {m: 0 for m in registry.names()}
                modality_drop_counts["none"] = 0
                modality_prefix_counts = {m: 0 for m in registry.names()}
                
                branch_grad_norms = {
                    m: float("nan") for m in registry.names()
                }
                branch_grad_norms["backbone"] = float("nan")
                branch_grad_norms["total"] = float("nan")

                # Variable for avoid NaNs
                skip_step = False

                # GRADIENT ACCUMULATION LOOP
                for micro_step in range(config.gradient_accumulation_steps):

                    # Fetch Raw Batch (CPU)
                    try:
                        raw_batch = next(train_iter)
                    except StopIteration:
                        epoch_num += 1
                        if ddp_rank == 0:
                            logger.info(f"--> Iter {iter_num}: End of Epoch {epoch_num}. Restarting DataLoader.")
                        
                        if ddp and train_loader.sampler is not None:
                            train_loader.sampler.set_epoch(epoch_num)
                            
                        train_iter = iter(train_loader)
                        raw_batch = next(train_iter)

                    # Dynamic Modality Dropout (V4/Ablation)
                    # NOT DOING DROPOUT HERE, letting the original processing handle it

                    # Calculate dynamic stochastic seed 
                    batch_seed = config.token_mixing_seed + iter_num * config.gradient_accumulation_steps + micro_step + ddp_rank

                    # Process Batch
                    B = MultimodalDatasetArrow.process_modes(
                        batch_data=raw_batch, 
                        modality_registry=registry, 
                        device=torch.device(device),
                        shuf=config.shuffle_modality_train,
                        use_token_mixing=config.use_token_mixing,
                        token_mixing_seed=batch_seed,
                        use_cls_token=config.use_cls_token,
                        cls_position=config.cls_position
                    )

                    dropped_modality = maybe_apply_modality_dropout(B, config)
                    modality_drop_counts[dropped_modality] = modality_drop_counts.get(dropped_modality, 0) + 1


                    # DDP Context
                    if ddp:

                        is_last_micro_step = (micro_step == config.gradient_accumulation_steps - 1)

                        # context manager: 'no_sync' disables communication, 'nullcontext' enables it
                        context = model.no_sync() if not is_last_micro_step else nullcontext()

                    else:
                        context = nullcontext()

                    # Forward & Backward Pass
                    with context:

                        # Automatic Mixed Precision (AMP)
                        with ctx: 

                            # Extract token_mixing_seed from inputs BEFORE forward to
                            # prevent torch.compile from guarding on its changing value.
                            raw_model._token_mixing_seed = B["X"].pop("token_mixing_seed", config.token_mixing_seed)

                            # Forward pass
                            # We pass a copy of the targets dictionary to avoid in-place mutations
                            # (re-indexing) from affecting the diagnostics later in the loop.
                            outputs, loss = model(B["X"], targets=B["Y"].copy(), dropped_modality=dropped_modality)

                            # --- INITIAL LOSS BALANCE CHECK (Step 0) ---
                            if iter_num == 0 and micro_step == 0 and ddp_rank == 0:
                                logger.info(" --> INITIAL LOSS MODALITY BALANCE CHECK (Step 0) <-- ")
                                initial_losses = {}
                                aligned_targets = outputs.get("_aligned_targets", B["Y"])
                                prefix_modality = outputs.get("_prefix_modality")

                                for mod_name in registry.names():
                                    if mod_name in outputs and mod_name in aligned_targets:
                                        if mod_name == prefix_modality:
                                            logger.info(f"Modality: {mod_name:<10} | SKIPPED (Acts as Prefix)")
                                            continue

                                        pred = outputs[mod_name]
                                        target = aligned_targets[mod_name]
                                        mod_config = registry.get_config(mod_name)

                                        # Auto-align shapes (token_mixing causes length differences
                                        # that torch.compile prevents from propagating back)
                                        if pred.shape[1] != target.shape[1]:
                                            min_len = min(pred.shape[1], target.shape[1])
                                            pred = pred[:, :min_len]
                                            target = target[:, :min_len]

                                        if "aion" in mod_name:
                                            mod_loss = torch.nn.functional.cross_entropy(
                                                pred.reshape(-1, pred.size(-1)),
                                                target.reshape(-1),
                                            ).item()
                                        elif config.loss_type in ["l1", "mae"]:
                                            mod_loss = torch.nn.functional.l1_loss(pred, target).item()
                                        elif config.loss_type == "mse":
                                            mod_loss = torch.nn.functional.mse_loss(pred, target).item()
                                        else:
                                            mod_loss = torch.nn.functional.huber_loss(pred, target, delta=config.loss_huber_delta).item()

                                        weighted_loss = mod_loss * mod_config.loss_weight
                                        initial_losses[f"initial_loss/{mod_name}_unweighted"] = mod_loss
                                        initial_losses[f"initial_loss/{mod_name}_weighted"] = weighted_loss

                                        logger.info(f"Modality: {mod_name:<10} | Unweighted Loss: {mod_loss:.6f} | Weight: {mod_config.loss_weight} | Weighted Contrib: {weighted_loss:.6f}")
                                if config.log_via_wandb and _WANDB_AVAILABLE:
                                    wandb.log(initial_losses, step=0)

                            # Skipping the batch if the loss is NaN
                            if torch.isnan(loss) or torch.isinf(loss):
                                skip_step = True
                                if ddp_rank == 0:
                                    logger.warning(f"NaN/Inf detected in loss at iter {iter_num}, "
                                                   f"micro-step {micro_step}. Skipping batch.")
                                break

                            if config.diagnostics_enabled and config.diagnostics_track_losses:
                                micro_mod_losses = _compute_modality_losses(outputs, B["Y"], config)
                                for mod_name, mod_loss in micro_mod_losses.items():
                                    if mod_name in modality_loss_sums:
                                        modality_loss_sums[mod_name] += mod_loss
                                        modality_loss_counts[mod_name] += 1
                                    if mod_name == dropped_modality:
                                        cross_loss_sums[mod_name] += mod_loss
                                        cross_loss_counts[mod_name] += 1

                                # Track which modality was the prefix
                                prefix_mod = outputs.get("_prefix_modality")
                                if prefix_mod and prefix_mod in modality_prefix_counts:
                                    modality_prefix_counts[prefix_mod] += 1

                            # Scale loss
                            loss = loss / config.gradient_accumulation_steps


                        # If there is a NaN, backward is avoided
                        if not skip_step:
                            scaler.scale(loss).backward()
                            loss_accum_for_log += loss.item()
                            if "_clip_loss" in outputs:
                                clip_loss_accum_for_log += outputs["_clip_loss"].item() / config.gradient_accumulation_steps
                                recon_loss_accum_for_log += outputs["_reconstruction_loss"].item() / config.gradient_accumulation_steps


                #--- END OF MICRO-BATCHES ---#

                # Not NaNs found
                if not skip_step:

                    # Unscaling the gradient
                    scaler.unscale_(optimizer)

                    # Control variables
                    grad_norm = 0.0
                    is_clipped = 0.0

                    if config.grad_clip != 0.0:

                        # This fonction (finished in _) computes the gradient norm, 
                        # returns it and compares with the gradient clipping value
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                        # Logging if gradient is clipped
                        if grad_norm > config.grad_clip:
                            is_clipped = 1.0

                    diagnostics_due = (
                        config.diagnostics_enabled
                        and (iter_num % config.diagnostics_interval == 0)
                        and (iter_num > initial_iter or iter_num == 0)
                    )

                    if diagnostics_due and config.diagnostics_track_grads:
                        branch_grad_norms = compute_branch_grad_norms(model)


                    # Updating the weights
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    # NaNs found
                    grad_norm = 0.0
                    is_clipped = 0.0
                    optimizer.zero_grad(set_to_none=True)

                    # Loss value for logging
                    loss_accum_for_log = float('nan')
                    clip_loss_accum_for_log = float('nan')
                    recon_loss_accum_for_log = float('nan')

                # Always set gradient to none
                optimizer.zero_grad(set_to_none=True)

                # LOGGING Console & WandB (Master Only)
                if ddp_rank == 0:
                    t1 = time.time()
                    dt = t1 - t0
                    t0 = t1

                    # Training progress
                    train_prog = iter_num / config.max_iters


                    if iter_num % config.log_interval == 0 and (iter_num > initial_iter or iter_num == 0):

                        # Time since last iter log
                        current_log_time = time.time()
                        time_since_last_log = current_log_time - last_log_time
                        last_log_time = current_log_time

                        # Average time between logging iteractions
                        avg_dt = time_since_last_log / config.log_interval

                        # MFU Computation
                        mfu_display = 0.0

                        # Searching for estimate_mfu()
                        if hasattr(raw_model, "estimate_mfu"):

                            if iter_num == 0:
                                mfu_display = 0.0
                            else:
                                # Computing the total seen samples per GPU on each iter
                                fwdbwd_per_gpu = config.batch_size * config.gradient_accumulation_steps
                                mfu_display = raw_model.estimate_mfu(fwdbwd_per_gpu, avg_dt)
                        else:
                            mfu_display = -1.0

                        # VRAM Computation
                        mem_usage = torch.cuda.max_memory_allocated() / (1024 ** 3) 
                        torch.cuda.reset_peak_memory_stats()   


                        # Recover the real average loss of the batch
                        lossf = loss_accum_for_log

                        # Compute Running Time
                        current_session_seconds = time.time() - run_start_time
                        total_seconds = current_session_seconds + accumulated_time

                        running_str = str(datetime.timedelta(seconds=int(total_seconds)))

                        # Compute the ETA for finishing training
                        remaining_iters = config.max_iters - iter_num
                        eta_seconds = remaining_iters * avg_dt
                        eta_str = str(datetime.timedelta(seconds=int(eta_seconds))).replace(',', '')

                        # Dynamic LR summary for the log
                        lr_summary = "/".join([f"{m[:3]}:{lr * getattr(config, f'lr_mult_{m}', 1.0):.1e}" for m in registry.names()])

                        logger.info(
                            f" --> Iter {iter_num}/{config.max_iters} ({train_prog:.2%}) | "
                            f"Loss {loss_accum_for_log:.4f} | "
                            f"LR {lr:.4e} ({lr_summary}) | "
                            f"Norm {grad_norm:.2f} | "
                            f"MFU {mfu_display*100:.2f}% (avg) | "
                            f"Mem {mem_usage:.2f}GB | "
                            f"dt {avg_dt*1000:.2f}ms (avg) | "
                            f"RT {running_str} | ETA {eta_str}"
                        )

                        diagnostics_due = (
                            config.diagnostics_enabled
                            and (iter_num % config.diagnostics_interval == 0)
                            and (iter_num > initial_iter or iter_num == 0)
                        )

                        mod_diagnostics = {
                            mod: (modality_loss_sums.get(mod, 0) / modality_loss_counts.get(mod, 1) if modality_loss_counts.get(mod, 0) > 0 else float("nan"))
                            for mod in registry.names()
                        }
                        cross_diagnostics = {
                            mod: (cross_loss_sums.get(mod, 0) / cross_loss_counts.get(mod, 1) if cross_loss_counts.get(mod, 0) > 0 else float("nan"))
                            for mod in registry.names()
                        }

                        dropout_summary = summarize_modality_dropout(modality_drop_counts)
                        prefix_summary = summarize_prefix_modality(modality_prefix_counts)

                        if diagnostics_due:
                            diag_log = " | ".join([f"L_{m}: {v:.4f}" for m, v in mod_diagnostics.items()])
                            cross_log = " | ".join([f"C_{m}: {v:.4f}" for m, v in cross_diagnostics.items()])
                            
                            # Format branch grads dynamically for available branches
                            grad_parts = [f"G_{m}: {branch_grad_norms.get(m, float('nan')):.2f}" for m in registry.names()]
                            grad_parts.append(f"G_back: {branch_grad_norms.get('backbone', float('nan')):.2f}")
                            grad_log = " | ".join(grad_parts)
                            
                            logger.info(
                                f"Diag | Drop={dropout_summary} | Prefix={prefix_summary} | "
                                f"L_clip: {clip_loss_accum_for_log:.4f} | L_recon: {recon_loss_accum_for_log:.4f} | "
                                f"Loss({diag_log}) | "
                                f"Cross({cross_log}) | "
                                f"Grad({grad_log})"
                            )

                        if config.log_via_wandb and _WANDB_AVAILABLE:
                            wandb_payload = {
                                "iter": iter_num,
                                "train/loss": lossf,
                                "train/clip_loss": clip_loss_accum_for_log,
                                "train/recon_loss": recon_loss_accum_for_log,
                                "train/lr_backbone": lr,
                                "train/grad_norm": grad_norm,
                                "train/mfu": mfu_display * 100,
                                "train/mem_gb": mem_usage,
                                "train/time_ms": avg_dt * 1000,
                                "train/rt_hours": total_seconds / 3600,
                                "train/eta_hours": eta_seconds / 3600,
                                "status/token_mixing": 1 if config.use_token_mixing else 0,
                                "status/images_masking": 1 if config.images_mask else 0,
                                "status/spectra_masking": 1 if config.spectra_mask else 0
                            }
                            # Add dynamic learning rates
                            for m in registry.names():
                                mult = getattr(config, f"lr_mult_{m}", 1.0)
                                wandb_payload[f"train/lr_{m}"] = lr * mult

                            if diagnostics_due and config.diagnostics_enabled:
                                for m, v in mod_diagnostics.items():
                                    wandb_payload[f"diag/loss_{m}"] = v
                                for m, v in cross_diagnostics.items():
                                    wandb_payload[f"diag/cross_loss_{m}"] = v
                                for m in registry.names():
                                    wandb_payload[f"diag/grad_{m}"] = branch_grad_norms.get(m, float("nan"))
                                    wandb_payload[f"diag/dropout_{m}_microsteps"] = modality_drop_counts.get(m, 0)
                                wandb_payload["diag/grad_backbone"] = branch_grad_norms.get("backbone", float("nan"))
                                
                            wandb.log(wandb_payload)

                        # CSV Writing
                        try:
                            with open(csv_path, "a") as f:
                                timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                                row = {
                                    "iter": iter_num, "epoch": epoch_num, "progress": f"{train_prog:.4f}", 
                                    "timestamp": timestamp_str, "train_loss": f"{loss_accum_for_log:.6f}",
                                    "clip_loss": f"{clip_loss_accum_for_log:.6f}", "recon_loss": f"{recon_loss_accum_for_log:.6f}",
                                    "grad_norm": f"{grad_norm:.4f}", "grad_backbone": f"{branch_grad_norms.get('backbone', float('nan')):.4f}", 
                                    "clipped": f"{is_clipped:.0f}", "dropped_modality": dropout_summary, 
                                    "lr": f"{lr:.4e}", "lr_backbone": f"{lr * getattr(config, 'lr_mult_backbone', 1.0):.4e}",
                                    "mfu": f"{mfu_display*100:.2f}", "mem_gb": f"{mem_usage:.2f}", 
                                    "dt_ms": f"{avg_dt*1000:.2f}", "rt_hms": running_str, "eta_hms": eta_str
                                }
                                for mod in names:
                                    row[f"loss_{mod}"] = f"{mod_diagnostics.get(mod, float('nan')):.6f}"
                                    row[f"cross_loss_{mod}"] = f"{cross_diagnostics.get(mod, float('nan')):.6f}"
                                    row[f"grad_{mod}"] = f"{branch_grad_norms.get(mod, float('nan')):.4f}"
                                    row[f"lr_{mod}"] = f"{lr * getattr(config, f'lr_mult_{mod}', 1.0):.4e}"

                                f.write(",".join(str(row.get(h, "")) for h in csv_headers) + "\n")

                        except Exception as e:
                            logger.error(f"CSV Write Error: {e}")

                        if diagnostics_due and config.diagnostics_enabled:
                            try:
                                with open(diagnostics_path, "a") as f:
                                    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    row = {
                                        "iter": iter_num, "epoch": epoch_num, "timestamp": timestamp_str,
                                        "dropout_mode": config.modality_dropout_mode, "dropout_applied": dropout_summary,
                                        "grad_total": f"{branch_grad_norms.get('total', float('nan')):.4f}",
                                        "grad_backbone": f"{branch_grad_norms.get('backbone', float('nan')):.4f}",
                                        "lr_base": f"{lr:.4e}", "lr_backbone": f"{lr * getattr(config, 'lr_mult_backbone', 1.0):.4e}"
                                    }
                                    for mod in names:
                                        row[f"loss_{mod}"] = f"{mod_diagnostics.get(mod, float('nan')):.6f}"
                                        row[f"grad_{mod}"] = f"{branch_grad_norms.get(mod, float('nan')):.4f}"
                                        row[f"lr_{mod}"] = f"{lr * getattr(config, f'lr_mult_{mod}', 1.0):.4e}"
                                    
                                    f.write(",".join(str(row.get(h, "")) for h in diag_headers) + "\n")
                            except Exception as e:
                                logger.error(f"Diagnostics CSV Write Error: {e}")

                # VALIDATION & CHECKPOINT
                if iter_num > 0 and iter_num % config.eval_interval == 0:

                    # Stopping training variable
                    stop_training = False

                    # Master performs validation
                    if ddp_rank == 0:
                        logger.info(f"Running validation at iter {iter_num}...")

                        # Switch to evaluation mode
                        model.eval() 

                        val_losses = []
                        val_clip_losses = []
                        val_recon_losses = []
                        # General accumulator: one list per modality, keyed by modality name
                        from collections import defaultdict
                        cross_val_losses = defaultdict(list)
                        max_val_batches = config.eval_batches

                        with torch.no_grad():
                            val_iter = iter(val_loader)
                            for _ in range(max_val_batches):
                                try:
                                    vbatch = next(val_iter)
                                except StopIteration:
                                    val_iter = iter(val_loader)
                                    vbatch = next(val_iter)

                                # Deterministic seed for validation 
                                val_batch_seed = config.token_mixing_seed + iter_num

                                B_val = MultimodalDatasetArrow.process_modes(
                                    batch_data=vbatch, 
                                    modality_registry=registry, 
                                    device=torch.device(device),
                                    shuf=config.shuffle_modality_val,
                                    use_token_mixing=config.use_token_mixing,
                                    token_mixing_seed=val_batch_seed,
                                    use_cls_token=config.use_cls_token,
                                    cls_position=config.cls_position
                                )

                                with ctx:
                                    # Extract seed before forward (same pattern as training)
                                    raw_model._token_mixing_seed = B_val["X"].pop("token_mixing_seed", config.token_mixing_seed)
                                    v_outputs, v_loss = raw_model(B_val["X"], targets=B_val["Y"])
                                val_losses.append(v_loss.item())
                                if "_clip_loss" in v_outputs:
                                    val_clip_losses.append(v_outputs["_clip_loss"].item())
                                    val_recon_losses.append(v_outputs["_reconstruction_loss"].item())

                                # --- GENERALIZED N-MODALITY CROSS-RECONSTRUCTION VALIDATION ---
                                # For each modality present in the batch, zero it out and ask
                                # the model to reconstruct it from the remaining context.
                                # The spectral modality gets a boosted weight (spectral-first design).
                                active_mods = [
                                    m for m in registry.names()
                                    if m in B_val["X"]
                                ]

                                is_prefix_mode = config.attn_type == "prefix"

                                for target_mod in active_mods:
                                    with ctx:
                                        if is_prefix_mode:
                                            # In Prefix-LM: put all context modalities first,
                                            # then the zeroed target modality at the end.
                                            X_cross = {}
                                            for k, v in B_val["X"].items():
                                                if k != target_mod and k != target_mod + "_positions":
                                                    X_cross[k] = v
                                            for k, v in B_val["X"].items():
                                                if k not in X_cross:
                                                    X_cross[k] = v
                                            X_cross[target_mod] = torch.zeros_like(X_cross[target_mod])
                                        else:
                                            X_cross = {k: v for k, v in B_val["X"].items()}
                                            X_cross[target_mod] = torch.zeros_like(X_cross[target_mod])

                                        out_cross, _ = raw_model(X_cross, targets=B_val["Y"])
                                        weighted_loss = compute_cross_reconstruction_loss(
                                            out_cross, B_val["Y"],
                                            target_key=target_mod,
                                            config=config,
                                        )
                                        if weighted_loss > 0.0:
                                            cross_val_losses[target_mod].append(weighted_loss)

                        val_loss = sum(val_losses) / len(val_losses)
                        val_clip_loss = sum(val_clip_losses) / len(val_clip_losses) if val_clip_losses else float('nan')
                        val_recon_loss = sum(val_recon_losses) / len(val_recon_losses) if val_recon_losses else float('nan')

                        # Compute averages for each cross-modal direction
                        spectral_key = getattr(config, 'spectral_modality_key', 'spectra')
                        cross_avgs = {
                            mod: (sum(vals) / len(vals) if vals else float('nan'))
                            for mod, vals in cross_val_losses.items()
                        }

                        # Primary metric: spectral reconstruction from all other modalities
                        avg_cross_to_spec = cross_avgs.get(spectral_key, float('nan'))

                        # Build log string for all cross directions
                        cross_log = " | ".join(
                            f"{mod}: {v:.4f}" for mod, v in cross_avgs.items()
                        )
                        logger.info(
                            f"Val Loss: {val_loss:.4f} (CLIP: {val_clip_loss:.4f}, Recon: {val_recon_loss:.4f}) | "
                            f"Cross-Recon (weighted) [{cross_log}]"
                        )

                        if config.log_via_wandb and _WANDB_AVAILABLE:
                            val_payload = {
                                "val/loss": val_loss, 
                                "val/clip_loss": val_clip_loss,
                                "val/recon_loss": val_recon_loss,
                                "iter": iter_num
                            }
                            for mod, avg in cross_avgs.items():
                                val_payload[f"val/cross_loss_to_{mod}"] = avg
                            wandb.log(val_payload)

                        # CSV Write: Just Validation properties
                        try:
                            with open(csv_path, "a") as f:
                                timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                                row = {
                                    "iter": iter_num, "epoch": epoch_num, "progress": f"{train_prog:.4f}",
                                    "timestamp": timestamp_str, "val_loss": f"{val_loss:.6f}",
                                    "val_clip_loss": f"{val_clip_loss:.6f}", "val_recon_loss": f"{val_recon_loss:.6f}"
                                }
                                for mod in names:
                                    row[f"val_loss_reconstruct_{mod}"] = f"{cross_avgs.get(mod, float('nan')):.6f}"

                                f.write(",".join(str(row.get(h, "")) for h in csv_headers) + "\n")
                        except Exception as e:
                            logger.error(f"Val CSV Write Error: {e}")


                        # EARLY STOPPING MECHANISM

                        # If loss improve
                        if val_loss < best_val_loss:

                            # Computing the improvement
                            improvement = best_val_loss - val_loss

                            # Reset the stop counter
                            best_val_loss = val_loss
                            early_stop_counter = 0
                            is_best = True

                            # Logging the validation loss as the BEST
                            logger.info(f"Validation Loss: {val_loss:.4f} (-{improvement:.4f}) | NEW BEST")

                            # Bash control hidden file for checking if the val loss has improved
                            if ddp_rank == 0:
                                with open(improve_file_path, "w") as f:
                                    f.write(f"Iter: {iter_num}, Loss: {val_loss}")

                        else:
                            # Increasing the counter ONLY if we have reached the minimum iters
                            if iter_num >= getattr(config, 'early_stopping_min_iters', 0):
                                early_stop_counter += 1

                            is_best = False
                            logger.info(
                                f"Validation Loss: {val_loss:.4f} (Best: {best_val_loss:.4f}) | "
                                f"Patience: {early_stop_counter}/{config.early_stopping_patience}"
                            )

                            # If counter eaches the patience limit
                            if early_stop_counter >= config.early_stopping_patience:
                                logger.info(f"Early stopping triggered! No improvement for {early_stop_counter} checks.")
                                stop_training = True

                        # CHECKPOINT SAVING
                        if iter_num > 0:

                            # 1. Calculate current total time
                            current_total_time = (time.time() - run_start_time) + accumulated_time

                            # 2. Prepare Base Checkpoint Dictionary
                            checkpoint = {
                                'model': model.module.state_dict() if ddp else model.state_dict(),
                                'modality_registry': registry,
                                'optimizer': optimizer.state_dict(),
                                'iter_num': iter_num,
                                'epoch_num': epoch_num,
                                'best_val_loss': best_val_loss, # Current record (may be updated below)
                                'config': asdict(config),
                                'total_run_time': current_total_time,
                                'early_stop_counter': early_stop_counter,
                            }

                            # 3. Save "LAST" - Periodic save
                            if config.checkpoint_save_type in ["last", "both"]:
                                checkpoint['best_val_loss'] = best_val_loss
                                torch.save(checkpoint, ckpt_path_last)

                            # 4. Save "BEST"
                            if is_best and config.checkpoint_save_type in ["best", "both", "all"]:
                                checkpoint['best_val_loss'] = best_val_loss 
                                logger.info(f"New best model found ({val_loss:.4f}). Saving to {ckpt_path_best}")
                                torch.save(checkpoint, ckpt_path_best)

                            if config.checkpoint_save_type == "all":
                                ckpt_iter_name = f"ckpt_iter_{iter_num:06d}.pt"
                                ckpt_path_iter = weights_dir / ckpt_iter_name

                                # Update dictionary with current val loss for this snapshot
                                checkpoint['best_val_loss'] = val_loss 
                                torch.save(checkpoint, ckpt_path_iter)
                                logger.info(f"Snapshot saved: {ckpt_path_iter}")

                    if ddp:

                        # 1=Stop, 0=Continue
                        stop_signal = torch.tensor(1 if stop_training else 0, device=device)

                        # Comunicating the decission to al GPUs
                        dist.broadcast(stop_signal, src=0)

                        # Update the stopping variable
                        stop_training = stop_signal.item() == 1

                        # Waiting to finish
                        dist.barrier()

                    # Switch back to training modes
                    model.train() 

                    # Early stopping
                    if stop_training:
                        if ddp_rank == 0:
                            logger.info("Stopping training loop due to Early Stopping.")
                        break # Exit the while iter_num < max_iters loop

                # PROFILER STEP
                if config.profile:
                    prof.step()


                # AUTOSAVING
                # In case the Slurm time comes to an end, finishing earlier
                if max_run_seconds is not None:
                    elapsed_seconds = (time.time() - run_start_time)

                    # If own limit time is reached
                    if elapsed_seconds > max_run_seconds:
                        # Stop
                        stop_signal = torch.tensor(1, device=device)
                    else:
                        # Continue
                        stop_signal = torch.tensor(0, device=device)

                    # Syncronizing all process
                    if ddp:
                        dist.all_reduce(stop_signal, op=dist.ReduceOp.MAX)

                    if stop_signal.item() == 1:
                        if ddp_rank == 0:
                            elapsed_str = str(datetime.timedelta(seconds=int(elapsed_seconds)))
                            logger.info(f"Time limit reached ({elapsed_str} > {config.max_run_hours}). "
                                        f"Saving checkpoint and exiting...")

                            # Saving checkpoint
                            current_total_time = elapsed_seconds + accumulated_time
                            checkpoint = {
                                'model': model.module.state_dict() if ddp else model.state_dict(),
                                'modality_registry': registry,
                                'optimizer': optimizer.state_dict(),
                                'iter_num': iter_num,
                                'epoch_num': epoch_num,
                                'best_val_loss': best_val_loss,
                                'config': asdict(config),
                                'total_run_time': current_total_time,
                                'early_stop_counter': early_stop_counter,
                            }

                            # Overwriting the LAST checkpoint
                            torch.save(checkpoint, ckpt_path_last)
                            logger.info(f"Emergency checkpoint saved to {ckpt_path_last}")

                        break # Exit the training loop

                # Increment iter counter
                iter_num += 1


        finally:

            # Stop profiler if active
            if config.profile:
                prof.stop()

            #--- CLEANUP DDP ---#
            if ddp and dist.is_initialized():
                dist.destroy_process_group()
                logger.info("Cleanup complete.")

            if ddp_rank == 0:
                logger.info("Training finished!")


