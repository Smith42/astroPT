"""
BERT-style masked-autoencoder (MAE) pretraining for AstroPT.

Sibling of train.py: instead of the autoregressive next-patch objective it
trains a BERT-style masked autoencoder. A fraction of image patches is replaced
by a learnable mask token, the full sequence is processed by the bidirectional
encoder, and the masked patches are reconstructed. The objective is config-gated
in the model (objective="mae", attn_type="full"), so the rest of the loop
mirrors train.py.

To run on a single GPU, example:
$ python scripts/train_mae.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 scripts/train_mae.py
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


def normalise(x, use_hf=False):
    # HF is in numpy format. Need to change that here if so:
    if use_hf:
        x = torch.from_numpy(x).to(torch.float32)
    std, mean = torch.std_mean(x, dim=1, keepdim=True)
    x_norm = (x - mean) / (std + 1e-8)
    return x_norm.to(torch.float16)


def data_transforms(use_hf):
    norm = partial(normalise, use_hf=use_hf)
    transform = transforms.Compose([transforms.Lambda(norm)])
    return transform


def process_galaxy_wrapper(galdict, func):
    patch_galaxy = func(np.array(galdict["image"]).swapaxes(0, 2))
    return {
        "images": patch_galaxy.to(torch.float),
        "images_positions": torch.arange(0, len(patch_galaxy), dtype=torch.long),
    }


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # default config values to test run a ~100M parameter MAE on DESI galaxy imagery
    out_dir = "logs/astropt_mae0100M"
    eval_interval = 1000
    log_interval = 100
    checkpoint_interval = 5000
    assert checkpoint_interval % eval_interval == 0
    eval_iters = 100
    eval_only = False
    always_save_checkpoint = False
    init_from = "scratch"  # 'scratch' or 'resume'
    use_hf = True
    stream_hf_dataset = True
    # data
    gradient_accumulation_steps = 5 * 8
    batch_size = 16  # micro-batch size if gradient_accumulation_steps > 1
    spiral = True
    block_size = 1024
    image_size = 256
    num_workers = 32
    # astroPT model
    n_layer = 12
    n_head = 12
    n_embd = 768
    n_chan = 3  # 3 imagery bands: r, i, z for jpeg
    dropout = 0.0
    bias = False
    # MAE objective. MAE needs bidirectional attention, so attn_type must be
    # "full" (the model raises if it is left as "causal").
    objective = "mae"
    attn_type = "full"
    mae_mask_ratio = 0.15  # fraction of patches replaced by the mask token
    norm_pix_loss = False  # data pipeline already per-patch normalises
    # Define modalities configuration. embed_pos=True gives BERT-style learned
    # absolute positional embeddings.
    modalities = [
        ModalityConfig(
            name="images",
            input_size=16 * 16 * n_chan,
            patch_size=16,
            loss_weight=1.0,
            embed_pos=True,
            pos_input_size=1,
        ),
    ]
    modality_registry = ModalityRegistry(modalities)
    tokeniser = "aim"
    # adamw optimizer (Chinchilla schedule)
    learning_rate = 6e-4
    max_iters = 30000
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    decay_lr = True
    warmup_iters = 2000
    lr_decay_iters = 27000 * 1.1
    min_lr = learning_rate / 10
    # DDP / system
    backend = "nccl"
    device = "cuda"
    dtype = "bfloat16"
    compile = True
    log_via_wandb = False
    wandb_project = None
    # -----------------------------------------------------------------------------
    config_keys = [
        k
        for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str))
    ]
    exec(open("src/astropt/configurator.py").read())
    config = {k: globals()[k] for k in config_keys}
    # -----------------------------------------------------------------------------

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert gradient_accumulation_steps % torch.cuda.device_count() == 0
        gradient_accumulation_steps //= torch.cuda.device_count()
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = (
        gradient_accumulation_steps
        * ddp_world_size
        * batch_size
        * block_size
        * len(modalities)
    )
    if master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in device else "cpu"
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

    # dataset init
    transforms = {"images": data_transforms(use_hf)}
    tpaths = None if use_hf else "./data/train.txt"
    tds = GalaxyImageDataset(
        paths={"images": tpaths},
        spiral=spiral,
        transform=transforms,
        modality_registry=modality_registry,
    )
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
            "Smith42/galaxies",
            revision="v2.0",
            split="train",
            streaming=(True if stream_hf_dataset else False),
        )
        tds_hf = tds_hf.select_columns("image").map(
            partial(process_galaxy_wrapper, func=tds.process_galaxy)
        )
        tds_hf = tds_hf.remove_columns("image")

        vds_hf = load_dataset(
            "Smith42/galaxies",
            revision="v2.0",
            split="test",
            streaming=(True if stream_hf_dataset else False),
        )
        vds_hf = vds_hf.select_columns("image").map(
            partial(process_galaxy_wrapper, func=tds.process_galaxy)
        )
        vds_hf = vds_hf.remove_columns("image")

    tdl = iter(
        DataLoader(
            tds_hf if use_hf else tds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
    )
    vdl = iter(
        DataLoader(
            vds_hf if use_hf else vds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
    )

    iter_num = 0
    best_val_loss = 1e9

    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        n_chan=n_chan,
        block_size=block_size,
        dropout=dropout,
        modalities=modalities,
        attn_type=attn_type,
        objective=objective,
        mae_mask_ratio=mae_mask_ratio,
        norm_pix_loss=norm_pix_loss,
    )

    if init_from == "scratch":
        if master_process:
            print("initializing a new MAE model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf, modality_registry, master_process=master_process)
    if init_from == "resume":
        if master_process:
            print(f"resuming training from {out_dir}")
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        checkpoint_model_args = checkpoint["model_args"]
        for k in ["n_layer", "n_head", "n_embd", "block_size"]:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf, modality_registry, master_process=master_process)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

    if log_via_wandb and master_process:
        if wandb_project is None:
            wandb.init(
                project=f"AstroPT-MAE-{model.get_num_params() / 1e6:06.1f}M", config=config
            )
        else:
            wandb.init(project=wandb_project, config=config)
    with open(f"{out_dir}/hparams.txt", "w") as fi:
        fi.write(f"AstroPT-MAE-{model.get_num_params() / 1e6:06.1f}M\n")
        fi.write(f"time: {int(time.time())}\n")
        for k, v in config.items():
            fi.write(f"{k}: {v}\n")

    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args["block_size"] = block_size
    model.to(device)

    scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))

    optimizer = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device_type
    )
    if init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None

    if compile:
        if master_process:
            print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)

    if ddp:
        if master_process:
            print("Wrapping in DDP")
        torch._dynamo.config.optimize_ddp = False
        # single-modality MAE uses every parameter each step, so we do not need
        # find_unused_parameters
        model = DDP(model, device_ids=[ddp_local_rank])

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for dl, split in zip([tdl, vdl], ["train", "val"]):
            out[split] = {}
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                B = tds.process_modes(
                    next(dl), modality_registry, device, objective=objective
                )
                with ctx:
                    logits, loss = model(B["X"], targets=B["Y"])
                losses[k] = loss.item()
            out[split]["dummy"] = losses.mean()
        model.train()
        return out

    @torch.no_grad()
    def validate(iter_num, out_dir):
        """Plot masked-input | reconstruction | target for a batch of galaxies.

        The reconstruction composites the model's predicted patches at the
        masked positions with the original patches everywhere else.
        """
        model.eval()
        im_patch = modality_registry.get_config("images").patch_size
        grid = image_size // im_patch

        def to_image(seq):
            if spiral:
                seq = torch.stack([vds.antispiralise(s) for s in seq])
            return einops.rearrange(
                seq,
                "b (h w) (p1 p2 c) -> b (h p1) (w p2) c",
                p1=im_patch,
                p2=im_patch,
                h=grid,
                w=grid,
            )

        for dl, split in zip([tdl, vdl], ["train", "val"]):
            f, axs = plt.subplots(8, 3, figsize=(4.5, 12), constrained_layout=True)
            axs[0, 0].set_title("masked")
            axs[0, 1].set_title("recon")
            axs[0, 2].set_title("target")
            B = vds.process_modes(
                next(dl), modality_registry, device, objective=objective
            )
            with ctx:
                P, loss = model(B["X"], B["Y"])
            Yim = B["Y"]["images"].to(device).float()
            Pim = P["images"].float()
            mask = P["images_mask"].unsqueeze(-1).float()
            masked_input = Yim * (1 - mask)
            recon = Yim * (1 - mask) + Pim * mask

            masked_img = to_image(masked_input).cpu().numpy()
            recon_img = to_image(recon).cpu().numpy()
            target_img = to_image(Yim).cpu().numpy()

            for ax, mi, ri, ti in zip(axs, masked_img, recon_img, target_img):
                ax[0].imshow(np.clip(mi, 0, 1))
                ax[1].imshow(np.clip(ri, 0, 1))
                ax[2].imshow(np.clip(ti, 0, 1))
                for a in ax:
                    a.axis("off")

            if log_via_wandb:
                wandb.log(
                    {
                        f"{split}_masked": [wandb.Image(np.clip(mi, 0, 1)) for mi in masked_img],
                        f"{split}_recon": [wandb.Image(np.clip(ri, 0, 1)) for ri in recon_img],
                        f"{split}_target": [wandb.Image(np.clip(ti, 0, 1)) for ti in target_img],
                    }
                )

            f.savefig(
                os.path.join(out_dir, f"{iter_num:06d}_{split}.jpg"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close(f)
        model.train()

    def get_lr(it):
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    if master_process:
        print("starting training...")
    B = tds.process_modes(next(tdl), modality_registry, device, objective=objective)
    t0 = time.time()
    dts = []
    local_iter_num = 0
    raw_model = model.module if ddp else model
    running_mfu = -1.0
    if log_emissions and master_process:
        tracker = EmissionsTracker(
            output_dir=out_dir, log_level="error", save_to_file=True, on_csv_write="update"
        )
        tracker.start()
    while True:
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % eval_interval == 0 and master_process:
            validate(iter_num, out_dir)
            losses = estimate_loss()
            val_loss = np.mean(list(losses["val"].values()))
            print(
                f"iter {iter_num}:\ntrain loss:\n{losses['train']}\nval loss:\n{losses['val']}"
            )
            with open(os.path.join(out_dir, "loss.txt"), "a") as fi:
                if fi.tell() == 0:
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

            if val_loss < best_val_loss or always_save_checkpoint:
                best_val_loss = val_loss
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                        "modality_registry": modality_registry,
                    }
                    if master_process:
                        print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
        if iter_num == 0 and eval_only:
            break

        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == gradient_accumulation_steps - 1
                )
            with ctx:
                logits, loss = model(B["X"], targets=B["Y"])
            B = tds.process_modes(
                next(tdl), modality_registry, device, objective=objective
            )
            scaler.scale(loss).backward()

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        dts.append(dt)
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            if log_via_wandb:
                wandb.log({"loss": lossf, "time": dt}, step=iter_num)
            print(
                f"iter {iter_num}: loss {lossf:.6f}, time {np.mean(dts) * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%"
            )
            dts = []

        iter_num += 1
        local_iter_num += 1

        if iter_num > max_iters:
            if log_emissions:
                emissions = tracker.stop()
                if master_process:
                    print(emissions)
            break

    if ddp:
        destroy_process_group()
    if log_via_wandb:
        wandb.finish()
