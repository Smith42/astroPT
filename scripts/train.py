"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
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
    transform = transforms.Compose(
        [
            # transforms.Lambda(lambda x: x/255.),
            transforms.Lambda(norm),
        ]
    )
    return transform


def process_galaxy_wrapper(galdict, func):
    patch_galaxy = func(np.array(galdict["image_crop"]).swapaxes(0, 2))
    return {
        "images": patch_galaxy.to(torch.float),
        "images_positions": torch.arange(
            0, len(patch_galaxy), dtype=torch.long
        ),
    }

if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # default config values designed to test run a 100M parameter model on DESI galaxy imagery
    # look at `config/astropt*.py` for a prod run example
    out_dir = "logs/astropt0100M"
    eval_interval = 1000
    log_interval = 100
    checkpoint_interval = 5000
    assert checkpoint_interval % eval_interval == 0
    eval_iters = 100
    eval_only = False  # if True, script exits right after the first eval
    always_save_checkpoint = (
        False  # if True, always save a checkpoint at each checkpoint_interval
    )
    init_from = "scratch"  # 'scratch' or 'resume'
    use_hf = True  # use the huggingface dataset version of our galz
    stream_hf_dataset = True  # stream the galaxies from huggingface
    # data
    gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
    batch_size = 16  # if gradient_accumulation_steps > 1, this is the micro-batch size
    spiral = True  # do we want to process the galaxy patches in spiral order?
    block_size = 1024
    image_size = 256
    num_workers = 32  # 64
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
            input_size=16 * 16 * n_chan,
            patch_size=16,
            loss_weight=1.0,
            embed_pos=True,
            pos_input_size=1,
        ),
    ]
    # Create modality registry
    modality_registry = ModalityRegistry(modalities)
    # adamw optimizer
    # we follow the same schedule here as Chinchilla
    learning_rate = 6e-4  # max learning rate
    max_iters = (
        30000  # total number of training iterations for one pass over our dataset
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
    # DDP settings
    backend = "nccl"  # 'nccl', 'gloo', etc.
    # system
    device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = "bfloat16"  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    attn_type = "causal"
    compile = True  # use PyTorch 2.0 to compile the model to be faster
    log_via_wandb = False
    # -----------------------------------------------------------------------------
    config_keys = [
        k
        for k, v in globals().items()
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
        torch.cuda.set_device(device)
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
    tokens_per_iter = (
        gradient_accumulation_steps
        * ddp_world_size
        * batch_size
        * block_size
        * len(modalities)
    )
    if master_process:
        if log_via_wandb:
            print("wandb detected, gonna log to that")
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

    # dataset init
    transforms = {"images": data_transforms(use_hf)}
    # training dataset and dataloader
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
            "Smith42/galaxies",
            split="train",
            streaming=(True if stream_hf_dataset else False),
        )
        tds_hf = tds_hf.select_columns("image_crop").map(
            partial(process_galaxy_wrapper, func=tds.process_galaxy)
        )
        tds_hf = tds_hf.remove_columns("image_crop")

        vds_hf = load_dataset(
            "Smith42/galaxies",
            split="test",
            streaming=(True if stream_hf_dataset else False),
        )
        vds_hf = vds_hf.select_columns("image_crop").map(
            partial(process_galaxy_wrapper, func=tds.process_galaxy)
        )
        vds_hf = vds_hf.remove_columns("image_crop")

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

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

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
    )

    if init_from == "scratch":
        # init a new model from scratch
        if master_process:
            print("initializing a new model from scratch")
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

    # logging via wandb if available
    # this is here so we can get the number of params from model()
    if log_via_wandb and master_process:
        wandb.init(
            project=f"AstroPT-{model.get_num_params() / 1e6:06.1f}M",
            config=config,
        )
    # write config and important information to log file
    with open(f"{out_dir}/hparams.txt", "w") as fi:
        fi.write(f"AstroPT-{model.get_num_params() / 1e6:06.1f}M\n")
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
    try:  # initting gradscaler changed in Pytorch 2.5.1
        scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))
    except Exception:  # fallback to old scaler if we hit an error
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

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
        if len(modalities) == 1:
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
                B = tds.process_modes(next(dl), modality_registry, device)
                with ctx:
                    logits, loss = model(B["X"], targets=B["Y"])
                losses[k] = loss.item()
            out[split]["dummy"] = losses.mean()
        model.train()
        return out

    @torch.no_grad()
    def validate(iter_num, out_dir):
        model.eval()
        for dl, split in zip([tdl, vdl], ["train", "val"]):
            f, axs = plt.subplots(8, 2, figsize=(3, 12), constrained_layout=True)
            B = vds.process_modes(next(vdl), modality_registry, device)
            with ctx:
                P, loss = model(B["X"], B["Y"])
                if "images" in modality_registry.names():
                    Yim = B["Y"]["images"].to(device)
                    b, t, c = Yim.size()
                    zero_block = torch.zeros((b, 1, c)).to(device)
                    Yim = torch.cat((zero_block, Yim), dim=1)
                    if spiral:
                        Yim = torch.stack([vds.antispiralise(yy) for yy in Yim])
                    im_patch = modality_registry.get_config("images").patch_size
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
                                "Y": wandb.Image(Yim.swapaxes(1, -1)),
                                "P": wandb.Image(Pim.swapaxes(1, -1)),
                            }
                        )

            f.savefig(
                os.path.join(out_dir, f"{iter_num:06d}_{split}.jpg"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()
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
    B = tds.process_modes(
        next(tdl), modality_registry, device
    )  # fetch the very first batch
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
                wandb.log({"valloss": losses["val"]})
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
                [ax.set_yscale("log") for ax in axs.ravel()]
                [ax.legend() for ax in axs.ravel()]
                f.savefig(os.path.join(out_dir, "loss.png"))
                plt.close()

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
                    if always_save_checkpoint:
                        torch.save(
                            checkpoint, os.path.join(out_dir, f"{iter_num:06d}_ckpt.pt")
                        )
                    else:
                        torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
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
            B = tds.process_modes(
                next(tdl), modality_registry, device
            )  # fetch the very first batch
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
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
                wandb.log({"loss": lossf, "time": dt})
            if log_emissions:
                emissions: float = tracker.flush()
                print(
                    f"iter {iter_num}: loss {lossf:.6f}, time {np.mean(dts) * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%, tot co2 {emissions:.1f}kg"
                )
            else:
                print(
                    f"iter {iter_num}: loss {lossf:.6f}, time {np.mean(dts) * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%"
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

    if ddp:
        destroy_process_group()
    if log_via_wandb:
        wandb.finish()
