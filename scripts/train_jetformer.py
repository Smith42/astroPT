"""
Training script for AstroPT with Jetformer tokenization.

This script extends the standard AstroPT training to support Jetformer's
continuous tokenization approach using normalizing flows and GMM outputs.

To run on a single GPU:
$ python train_jetformer.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node:
$ torchrun --standalone --nproc_per_node=4 scripts/train_jetformer.py
"""

import math
import os
import time
from contextlib import nullcontext
from functools import partial

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    import wandb
    log_via_wandb = True
except ImportError:
    log_via_wandb = False
try:
    from codecarbon import EmissionsTracker
    log_emissions = False
except ImportError:
    log_emissions = False

from astropt.local_datasets import GalaxyImageDataset
from astropt.model import GPT, GPTConfig, ModalityConfig, ModalityRegistry


def prepare_batch_for_jetformer(batch, tokeniser):
    """Prepare batch for Jetformer: use raw images instead of patches.
    
    For Jetformer, replaces 'images' (patches) with 'images_raw' (raw images [B,C,H,W])
    and adjusts positions accordingly.
    
    Args:
        batch: Dictionary with 'images' (patches) and optionally 'images_raw' (raw images)
        tokeniser: Tokeniser type ('jetformer' or other)
    
    Returns:
        Modified batch with 'images' replaced by raw images for Jetformer
    """
    if tokeniser != "jetformer":
        return batch
    
    if "images_raw" not in batch:
        raise ValueError(
            "Jetformer requires raw images. For HF datasets, this is handled automatically. "
            "For local datasets, you need to modify GalaxyImageDataset to return raw images."
        )
    
    # Replace patches with raw images
    raw_images = batch["images_raw"]  # [B, C, H, W] from dataset
    B, C, H, W = raw_images.shape
    patch_size = 16  # Should match modality config
    T = (H // patch_size) * (W // patch_size)
    
    batch["images"] = raw_images  # Replace patches with raw images
    batch["images_positions"] = torch.arange(T, dtype=torch.long).unsqueeze(0).expand(B, T)
    batch["images_is_raw"] = True
    
    return batch


def normalise(x, use_hf=False):
    """Normalize images to zero mean, unit variance."""
    if use_hf:
        x = torch.from_numpy(x).to(torch.float32)
    std, mean = torch.std_mean(x, dim=1, keepdim=True)
    x_norm = (x - mean) / (std + 1e-8)
    return x_norm.to(torch.float16)


def data_transforms(use_hf):
    """Data transformation pipeline."""
    norm = partial(normalise, use_hf=use_hf)
    transform = transforms.Compose(
        [
            # transforms.Lambda(lambda x: x/255.),
            transforms.Lambda(norm),
        ]
    )
    return transform


def process_galaxy_wrapper(galdict, func, return_raw=False):
    """Wrapper for processing galaxy images from HF dataset.
    
    Args:
        galdict: Dictionary with "image" key containing image data
        func: Function to process galaxy (process_galaxy)
        return_raw: If True, also return raw image [C, H, W] for Jetformer
    
    Returns:
        Dictionary with "images" (patches) and optionally "images_raw" (raw image)
    """
    raw_image = np.array(galdict["image"]).swapaxes(0, 2)  # [C, H, W]
    patch_galaxy = func(raw_image)
    result = {
        "images": patch_galaxy.to(torch.float),
        "images_positions": torch.arange(0, len(patch_galaxy), dtype=torch.long),
    }
    if return_raw:
        # Convert to [C, H, W] tensor and normalize to [0,1] if needed
        raw_tensor = torch.from_numpy(raw_image).to(torch.float)
        if raw_tensor.max() > 1.0:
            raw_tensor = raw_tensor / 255.0
        result["images_raw"] = raw_tensor
    return result


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # Configuration for Jetformer Training
    # -----------------------------------------------------------------------------
    tokeniser = "jetformer"
    out_dir = "logs/astropt_jetformer_5epochs_resume"
    eval_interval = 1000
    log_interval = 100
    checkpoint_interval = 5000
    assert checkpoint_interval % eval_interval == 0
    eval_iters = 100
    eval_only = False
    always_save_checkpoint = False
    init_from = "scratch"  # 'scratch' or 'resume'
    use_hf = True  # use the huggingface dataset version of our galz
    stream_hf_dataset = True  # stream the galaxies from huggingface
    # data
    gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
    batch_size = 32  # if gradient_accumulation_steps > 1, this is the micro-batch size
    spiral = False  # do we want to process the galaxy patches in spiral order?
    block_size = 1024
    image_size = 256
    num_workers = 32
    num_epochs = 5  # number of epochs to train (None = use max_iters instead)
    dataset_size = None  # dataset size for epoch calculation (None = try to get automatically, required for streaming)
    # astroPT model
    n_layer = 12
    n_head = 12
    n_embd = 768
    n_chan = 3  # 3 imagery bands: r, i, z for jpeg, 1 imagery band for FITS
    dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    # NB dropout is NOT implemented for flex attention
    bias = False  # do we use bias inside LayerNorm and Linear layers?
    # Define modalities configuration
    modalities = [
        ModalityConfig(
            name="images",
            input_size=16 * 16 * n_chan,  # Will be overridden by Jetformer
            patch_size=16,
            loss_weight=1.0,
            embed_pos=True,
            pos_input_size=1,
        ),
    ]
    # Create modality registry
    modality_registry = ModalityRegistry(modalities)
    # Jetformer-specific hyperparameters
    jetformer_flow_steps = 4  # Number of coupling layers in normalizing flow
    jetformer_gmm_K = 4  # Number of Gaussian components in mixture
    jetformer_noise_max = 0.1  # Maximum noise for curriculum
    jetformer_noise_min = 0.0  # Minimum noise for curriculum
    # Optimiser configuration
    learning_rate = 6e-4  # max learning rate
    max_iters = (
        1_000_000  # total number of training iterations (overridden if num_epochs is set)
    )
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True  # whether to decay the learning rate
    warmup_iters = 2000  # how many steps to warm up for
    lr_decay_iters = 27000 * 1.1  # should be ~= max_iters per Chinchilla
    min_lr = (
        learning_rate / 10
    )  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    attn_type = "causal"
    # DDP settings
    backend = "nccl"  # 'nccl', 'gloo', etc.
    # system
    device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = "bfloat16"  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = False  # use PyTorch 2.0 to compile the model to be faster
    log_via_wandb = False
    wandb_project = None
    # -----------------------------------------------------------------------------
    config_keys = [
        k for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str))
    ]
    exec(
        open("src/astropt/configurator.py").read()
    )  # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys}  # will be useful for logging
    # -----------------------------------------------------------------------------

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(ddp_local_rank)
        master_process = (
            ddp_rank == 0
        )  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        assert gradient_accumulation_steps % torch.cuda.device_count() == 0
        gradient_accumulation_steps //= torch.cuda.device_count()
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        ddp_rank = 0
        ddp_local_rank = 0
    tokens_per_iter = (
        gradient_accumulation_steps
        * ddp_world_size
        * batch_size
        * block_size
        * len(modalities)
    )
    if master_process:
        if log_via_wandb:
            print("Logging to wandb enabled")
        if log_emissions:
            print("codecarbon detected, will log emissions")
        print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in device else "cpu"
    )  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    transforms = {"images": data_transforms(use_hf)}
    
    # Training dataset
    tpaths = None if use_hf else "./data/train.txt"
    tds = GalaxyImageDataset(
        paths={"images": tpaths},
        spiral=spiral,
        transform=transforms,
        modality_registry=modality_registry,
    )
    # validation dataset and dataloader
    vpaths = None if use_hf else "./data/tests.txt"
    vds = GalaxyImageDataset(
        paths={"images": vpaths},
        spiral=spiral,
        transform=transforms,
        modality_registry=modality_registry,
    )

    if use_hf:
        from datasets import load_dataset

        tds_hf = load_dataset(
            "/scratch02/public/sao/msmith/data/galaxies/",
            revision="v2.0",
            split="train",
            streaming=(True if stream_hf_dataset else False),
        )
        # For Jetformer, we need raw images, not just patches
        return_raw = tokeniser == "jetformer"
        tds_hf = (
            tds_hf
            .select_columns("image_crop")
            .rename_column("image_crop", "image")
            .map(
                partial(process_galaxy_wrapper, func=tds.process_galaxy, return_raw=return_raw)
            )
        )
        tds_hf = tds_hf.remove_columns("image")

        vds_hf = load_dataset(
            "/scratch02/public/sao/msmith/data/galaxies/",
            revision="v2.0",
            split="test",
            streaming=(True if stream_hf_dataset else False),
        )
        # For Jetformer, we need raw images, not just patches
        return_raw = tokeniser == "jetformer"
        vds_hf = (
            vds_hf
            .select_columns("image_crop")
            .rename_column("image_crop", "image")
            .map(
                partial(process_galaxy_wrapper, func=tds.process_galaxy, return_raw=return_raw)
            )
        )
        vds_hf = vds_hf.remove_columns("image")

    # Create infinite dataloader wrapper for streaming datasets
    def infinite_dataloader(dataloader):
        """Wrap a DataLoader to cycle infinitely, enabling multiple epochs with streaming datasets."""
        while True:
            yield from dataloader

    # Create base dataloaders
    train_loader = DataLoader(
        tds_hf if use_hf else tds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 and stream_hf_dataset else False,
    )
    val_loader = DataLoader(
        vds_hf if use_hf else vds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 and stream_hf_dataset else False,
    )
    
    # Wrap with infinite iterator if streaming, otherwise use regular iterator
    if stream_hf_dataset and use_hf:
        tdl = infinite_dataloader(train_loader)
        vdl = infinite_dataloader(val_loader)
    else:
        tdl = iter(train_loader)
        vdl = iter(val_loader)

    # Calculate dataset size and max_iters from num_epochs if specified
    if num_epochs is not None:
        # Try to get dataset size automatically if not provided
        if dataset_size is None:
            if use_hf and not stream_hf_dataset:
                # Can get size from non-streaming HF dataset
                try:
                    dataset_size = len(tds_hf)
                    if master_process:
                        print(f"Dataset size (auto-detected): {dataset_size:,} samples")
                except (TypeError, AttributeError):
                    dataset_size = None
            elif use_hf and stream_hf_dataset:
                # For streaming datasets, get size from dataset info without loading data
                try:
                    from datasets import load_dataset_builder
                    if master_process:
                        print("Getting dataset size from info (not loading data)...")
                    builder = load_dataset_builder(
                        "/scratch02/public/sao/msmith/data/galaxies/",
                        revision="v2.0",
                    )
                    # Access the split info directly without loading data
                    if hasattr(builder.info, 'splits') and 'train' in builder.info.splits:
                        dataset_size = builder.info.splits['train'].num_examples
                        if master_process:
                            print(f"Dataset size (auto-detected from info): {dataset_size:,} samples")
                    else:
                        dataset_size = None
                        if master_process:
                            print("Warning: Could not get dataset size from info splits")
                except (TypeError, AttributeError, Exception) as e:
                    if master_process:
                        print(f"Warning: Could not get dataset size from info: {e}")
                        print("Falling back to loading dataset (this may take a moment)...")
                    # Fallback: load the dataset if info API doesn't work
                    try:
                        from datasets import load_dataset
                        tds_info = load_dataset(
                            "/scratch02/public/sao/msmith/data/galaxies/",
                            revision="v2.0",
                            split="train",
                            streaming=False,
                        )
                        dataset_size = len(tds_info)
                        if master_process:
                            print(f"Dataset size (auto-detected): {dataset_size:,} samples")
                        del tds_info  # Free memory
                    except Exception as e2:
                        if master_process:
                            print(f"Error: Could not auto-detect dataset size: {e2}")
                        dataset_size = None
            else:
                # Local dataset
                try:
                    dataset_size = len(tds)
                    if master_process:
                        print(f"Dataset size (auto-detected): {dataset_size:,} samples")
                except (TypeError, AttributeError):
                    dataset_size = None
        
        # If still None, require user to specify
        if dataset_size is None:
            raise ValueError(
                "num_epochs is set but dataset_size cannot be determined automatically. "
                "Please set dataset_size parameter."
            )
        
        # Calculate iterations per epoch
        effective_batch_size = batch_size * gradient_accumulation_steps * ddp_world_size
        iterations_per_epoch = (dataset_size + effective_batch_size - 1) // effective_batch_size  # ceiling division
        max_iters = num_epochs * iterations_per_epoch
        
        if master_process:
            print(f"Training for {num_epochs} epochs:")
            print(f"  Dataset size: {dataset_size:,} samples")
            print(f"  Effective batch size: {effective_batch_size:,}")
            print(f"  Iterations per epoch: {iterations_per_epoch:,}")
            print(f"  Total iterations: {max_iters:,}")
        
        # Set training start point when starting from scratch
        if init_from == "scratch":
            training_start_samples = 0

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9
    current_epoch = 0
    samples_seen = 0
    training_start_samples = 0  # Track where current training run started (for relative epoch calculation)

    # model init
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        n_chan=n_chan,
        block_size=block_size,
        dropout=dropout,
        modalities=modalities,
        attn_type=attn_type,
        tokeniser=tokeniser,
        jetformer_flow_steps=jetformer_flow_steps,
        jetformer_gmm_K=jetformer_gmm_K,
        jetformer_noise_max=jetformer_noise_max,
        jetformer_noise_min=jetformer_noise_min,
        img_size=image_size,  # Image size for Jetformer image-space flow
    )

    if init_from == "scratch":
        # init a new model from scratch
        if master_process:
            print("initializing a new model from scratch with Jetformer tokenization")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf, modality_registry, master_process=master_process)
    if init_from == "resume":
        if master_process:
            print(f"resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        # NOTE had to remove 'bias' key here -- where does it go?!
        for k in ["n_layer", "n_head", "n_embd", "block_size"]:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf, modality_registry, master_process=master_process)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
        # Restore epoch tracking if available
        current_epoch = checkpoint.get("current_epoch", 0)
        samples_seen = checkpoint.get("samples_seen", 0)
        # Restore training start point, or set to current samples_seen if old checkpoint
        training_start_samples = checkpoint.get("training_start_samples", samples_seen)
        if master_process:
            if "current_epoch" in checkpoint:
                # Calculate relative epoch for current training run
                if num_epochs is not None and dataset_size is not None:
                    relative_epoch = int((samples_seen - training_start_samples) // dataset_size)
                    print(f"Resuming from epoch {relative_epoch}/{num_epochs}, iteration {iter_num}")
                else:
                    print(f"Resuming from epoch {current_epoch}, iteration {iter_num}")
            else:
                print(f"Resuming from iteration {iter_num} (epoch tracking not available in checkpoint)")

    # logging via wandb if available
    # this is here so we can get the number of params from model()
    if log_via_wandb and master_process:
        if wandb_project is None:
            wandb.init(
                project=f"AstroPT-Jetformer-{model.get_num_params() / 1e6:06.1f}M",
                config=config,
            )
        else:
            wandb.init(
                project=wandb_project,
                config=config,
            )
    # write config and important information to log file
    with open(f"{out_dir}/hparams.txt", "w") as fi:
        fi.write(f"AstroPT-Jetformer-{model.get_num_params() / 1e6:06.1f}M\n")
        fi.write(f"time: {int(time.time())}\n")
        for k, v in config.items():
            fi.write(f"{k}: {v}\n")

    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args["block_size"] = (
            block_size  # so that the checkpoint will have the right value
        )
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))

    # optimizer
    optimizer = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device_type
    )
    if init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free up memory

    # compile the model
    if compile:
        if master_process:
            print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        if master_process:
            print("Wrapping in DDP")
        # Note to future people: we had to turn off optimize_ddp due to a
        # torch compiler error when running DDP. This _may_ be fixed in a
        # future torch version so check periodically. I tested this on:
        # 2.6.0.dev20241126+cu124
        torch._dynamo.config.optimize_ddp = False
        # if we have only one modality all params are used in a forward pass:
        # BUT: Jetformer decoder isn't used in loss, so need find_unused_parameters=True
        if len(modalities) == 1 and tokeniser != "jetformer":
            model = DDP(model, device_ids=[ddp_local_rank])
        else:
            model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for dl, split in zip(
            [tdl, vdl],
            ["train", "val"],
        ):
            out[split] = {}
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                batch_raw = next(dl)
                batch_raw = prepare_batch_for_jetformer(batch_raw, tokeniser)
                B = tds.process_modes(batch_raw, modality_registry, device)
                with ctx:
                    logits, loss = model(B["X"], targets=B["Y"])
                losses[k] = loss.item()
            out[split]["dummy"] = losses.mean()
        model.train()
        return out

    @torch.no_grad()
    def validate(iter_num, out_dir):
        model.eval()
        raw_model = model.module if ddp else model
        for dl, split in zip([tdl, vdl], ["train", "val"]):
            f, axs = plt.subplots(8, 2, figsize=(3, 12), constrained_layout=True)
            batch_raw = next(vdl)
            batch_raw = prepare_batch_for_jetformer(batch_raw, tokeniser)
            B = vds.process_modes(batch_raw, modality_registry, device)
            with ctx:
                P, loss = model(B["X"], B["Y"])
                if "images" in modality_registry.names():
                    # For Jetformer, B["Y"]["images"] is raw images [B,C,H,W]
                    # For non-Jetformer, B["Y"]["images"] is patches [B,T,D]
                    im_patch = modality_registry.get_config("images").patch_size
                    is_jetformer = (
                        isinstance(raw_model, GPT)
                        and raw_model.config.tokeniser == "jetformer"
                    )
                    
                    if is_jetformer:
                        # Raw images [B,C,H,W] - convert directly to image format for visualization
                        Yim_raw = B["Y"]["images"].to(device)  # [B,C,H,W]
                        # Permute to [B,H,W,C] for visualization
                        Yim = Yim_raw.permute(0, 2, 3, 1)  # [B,H,W,C]
                        
                        # Reconstruct images
                        x_recon = raw_model.jetformer_reconstruct_images(B["Y"]["images"])
                        # Permute to [B,H,W,C] for visualization
                        Pim = x_recon.permute(0, 2, 3, 1)  # [B,H,W,C]
                    else:
                        # Non-Jetformer: patches already [B,T,D]
                        Yim = B["Y"]["images"].to(device)
                        b, t, c = Yim.size()
                        zero_block = torch.zeros((b, 1, c)).to(device)
                        Yim = torch.cat((zero_block, Yim), dim=1)
                        if spiral:
                            Yim = torch.stack([vds.antispiralise(yy) for yy in Yim])
                        Yim = einops.rearrange(
                            Yim,
                            "b (h w) (p1 p2 c) -> b (h p1) (w p2) c",
                            p1=im_patch,
                            p2=im_patch,
                            h=image_size // im_patch,
                            w=image_size // im_patch,
                        )
                        
                        Pim = torch.cat((zero_block, P["images"]), dim=1)
                        if spiral:
                            Pim = torch.stack([vds.antispiralise(pp) for pp in Pim])
                        Pim = einops.rearrange(
                            Pim,
                            "b (h w) (p1 p2 c) -> b (h p1) (w p2) c",
                            p1=im_patch,
                            p2=im_patch,
                            h=image_size // im_patch,
                            w=image_size // im_patch,
                        )

                    for ax, p, y in zip(
                        axs, Pim.to(float).cpu().numpy(), Yim.to(float).cpu().numpy()
                    ):
                        ax[0].imshow(np.clip(y, 0, 1))
                        ax[1].imshow(np.clip(p, 0, 1))
                        ax[0].axis("off")
                        ax[1].axis("off")

                    if log_via_wandb:
                        wandb.log(
                            {
                                "Y": [wandb.Image(np.clip(yy.swapaxes(0, -1).cpu(), 0, 1)) for yy in Yim],
                                "P": [wandb.Image(np.clip(pp.swapaxes(0, -1).cpu(), 0, 1)) for pp in Pim],
                            }
                        )

            f.savefig(
                os.path.join(out_dir, f"{iter_num:06d}_{split}.jpg"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close(f)
        model.train()

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    # training loop
    if master_process:
        print("starting training...")
    batch_raw = next(tdl)
    batch_raw = prepare_batch_for_jetformer(batch_raw, tokeniser)
    B = tds.process_modes(batch_raw, modality_registry, device)  # fetch the very first batch
    t0 = time.time()
    dts = []
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # unwrap DDP container if needed
    running_mfu = -1.0
    if log_emissions and master_process:
        tracker = EmissionsTracker(
            output_dir=out_dir,
            log_level="error",
            save_to_file=True,
            on_csv_write="update",
        )
        tracker.start()
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # update Jetformer noise schedule if applicable
        raw_model = model.module if ddp else model
        if isinstance(raw_model, GPT) and raw_model.config.tokeniser == "jetformer":
            raw_model.set_jetformer_schedule(iter_num, max_iters)

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            validate(iter_num, out_dir)
            losses = estimate_loss()
            val_loss = np.mean(list(losses["val"].values()))
            print(
                f"iter {iter_num}:\ntrain loss:\n{losses['train']}\nval loss:\n{losses['val']}"
            )
            with open(os.path.join(out_dir, "loss.txt"), "a") as fi:
                if fi.tell() == 0:  # check if a new file and write header if so
                    train_head_str = ",".join(
                        map(lambda x: str(x) + "_train", losses["train"].keys())
                    )
                    valid_head_str = ",".join(
                        map(lambda x: str(x) + "_valid", losses["val"].keys())
                    )
                    fi.write(f"iter_num,{train_head_str},{valid_head_str},lr,mfu\n")
                train_loss_str = ",".join(
                    map(lambda x: str(x.item()), losses["train"].values())
                )
                valid_loss_str = ",".join(
                    map(lambda x: str(x.item()), losses["val"].values())
                )
                fi.write(
                    f"{iter_num},{train_loss_str},{valid_loss_str},{lr},{running_mfu * 100}\n"
                )
            if log_via_wandb:
                wandb.log({"valloss": losses["val"]}, step=iter_num)
            if iter_num != 0:
                loss_df = pd.read_csv(os.path.join(out_dir, "loss.txt"))
                f, axs = plt.subplots(
                    1,
                    len(losses["train"]) + 1,
                    figsize=(12, 4),
                    constrained_layout=True,
                )
                axs.ravel()[0].set_title("mean")
                axs.ravel()[0].plot(
                    loss_df["iter_num"],
                    loss_df.filter(like="train").mean(axis=1),
                    label="train",
                )
                axs.ravel()[0].plot(
                    loss_df["iter_num"],
                    loss_df.filter(like="valid").mean(axis=1),
                    label="valid",
                )
                for ax, train_loss, valid_loss in zip(
                    axs.ravel()[1:],
                    loss_df.filter(like="train"),
                    loss_df.filter(like="valid"),
                ):
                    ax.set_title(train_loss)
                    ax.plot(loss_df["iter_num"], loss_df[train_loss], label="train")
                    ax.plot(loss_df["iter_num"], loss_df[valid_loss], label="valid")
                # [ax.set_yscale("log") for ax in axs.ravel()]
                [ax.legend() for ax in axs.ravel()]
                f.savefig(os.path.join(out_dir, "loss.png"))
                plt.close(f)

            # Save checkpoint if validation improved or always_save_checkpoint is True
            save_checkpoint_now = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint_now = True
                if master_process:
                    print(f"Validation loss improved, saving checkpoint...")
            elif always_save_checkpoint:
                save_checkpoint_now = True
                if master_process:
                    print(f"Saving checkpoint (always_save_checkpoint=True)...")
            
            # Also save periodic checkpoints regardless of validation loss
            # This ensures we can resume even if validation didn't improve
            if iter_num > 0 and iter_num % checkpoint_interval == 0:
                save_checkpoint_now = True
                if master_process:
                    print(f"Periodic checkpoint at iteration {iter_num}...")
            
            if save_checkpoint_now and iter_num > 0:
                model_state = raw_model.state_dict()
                checkpoint = {
                    "model": model_state,
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                    "modality_registry": modality_registry,
                    "current_epoch": current_epoch,
                    "samples_seen": samples_seen,
                    "training_start_samples": training_start_samples,
                }
                if master_process:
                    print(f"saving checkpoint to {out_dir}")
                # Always save the latest checkpoint (for resume)
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
                # Also save numbered checkpoint if always_save_checkpoint or at checkpoint_interval
                if always_save_checkpoint or (iter_num % checkpoint_interval == 0):
                    torch.save(
                        checkpoint, os.path.join(out_dir, f"{iter_num:06d}_ckpt.pt")
                    )
        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (
                    micro_step == gradient_accumulation_steps - 1
                )

            with ctx:
                logits, loss = model(B["X"], targets=B["Y"])
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            batch_raw = next(tdl)
            batch_raw = prepare_batch_for_jetformer(batch_raw, tokeniser)
            B = tds.process_modes(batch_raw, modality_registry, device)  # fetch the very first batch
            
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

        # Track samples seen for epoch calculation (only once per iteration, after all micro steps)
        if num_epochs is not None and dataset_size is not None:
            batch_size_actual = B["X"]["images"].shape[0] if "images" in B["X"] else batch_size
            # Count samples once per iteration: batch_size * gradient_accumulation_steps
            samples_seen += batch_size_actual * gradient_accumulation_steps
            # Calculate epoch relative to current training run start
            new_epoch = int((samples_seen - training_start_samples) // dataset_size)
            if new_epoch > current_epoch:
                current_epoch = new_epoch
                if master_process:
                    print(f"Completed epoch {current_epoch}/{num_epochs} (samples seen: {samples_seen:,}/{dataset_size * num_epochs:,})")

        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        dts.append(dt)
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(
                    batch_size * gradient_accumulation_steps, dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            if log_via_wandb:
                log_dict = {"loss": lossf, "time": dt}
                if num_epochs is not None:
                    log_dict["epoch"] = current_epoch
                wandb.log(log_dict, step=iter_num)
            epoch_str = f", epoch {current_epoch}/{num_epochs}" if num_epochs is not None else ""
            if log_emissions:
                emissions: float = tracker.flush()
                print(
                    f"iter {iter_num}{epoch_str}: loss {lossf:.6f}, time {np.mean(dts) * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%, tot co2 {emissions:.1f}kg"
                )
            else:
                print(
                    f"iter {iter_num}{epoch_str}: loss {lossf:.6f}, time {np.mean(dts) * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%"
                )
            dts = []

        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            if log_emissions:
                emissions: float = tracker.stop()
                if master_process:
                    print(emissions)
            break

    # Cleanup: destroy process group before exiting
    if ddp:
        try:
            destroy_process_group()
        except Exception as e:
            # HeartbeatMonitor may already be shutting down - this is harmless
            if master_process:
                print(f"Note: Process group cleanup warning (harmless): {e}")
    if log_via_wandb:
        wandb.finish()
