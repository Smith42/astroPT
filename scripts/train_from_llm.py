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

import itertools
import math
import os
import time
from contextlib import nullcontext

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from datasets import load_dataset
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

from astropt.local_datasets import LLMModalityDataset, llm_collate_fn
from astropt.model import GPT, GPTConfig, ModalityConfig, ModalityRegistry


def normalise(x):
    std, mean = torch.std_mean(x, dim=1, keepdim=True)
    x_norm = (x - mean) / (std + 1e-8)
    return x_norm.to(torch.float16)


def data_transforms():
    transform = transforms.Compose(
        [
            # transforms.Lambda(lambda x: x/255.),
            transforms.Lambda(normalise),
        ]
    )
    return transform


def to_device(x, device):
    if hasattr(x, "to"):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [to_device(i, device) for i in x]
    return x


def stringify(input_text, modality_infos):
    unwrapped_tokenizer = (
        model.module if hasattr(model, "module") else model
    ).tokenizer
    final_text = unwrapped_tokenizer.convert_ids_to_tokens(input_text)
    for ii, mod_name in enumerate(modality_infos["names"]):
        start_pos = modality_infos["starts"][ii]
        length = modality_infos["lengths"][ii]
        if mod_name != "images":
            mod_data = modality_infos["data"][ii]
            end_pos = start_pos + length
            final_text[start_pos] = f"{mod_data.squeeze().item():04f}"

    for ii, mod_name in enumerate(modality_infos["names"]):
        start_pos = modality_infos["starts"][ii]
        length = modality_infos["lengths"][ii]
        if mod_name == "images":
            end_pos = start_pos + length
            final_text = final_text[: start_pos + 1] + final_text[end_pos:]

    return " ".join(final_text)


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # default config values designed to test run a 4B LLM backbone AstroPT model on DESI galaxy imagery
    out_dir = "logs/smollm3B"
    eval_interval = 500
    log_interval = 50
    checkpoint_interval = 1000
    assert checkpoint_interval % eval_interval == 0
    eval_iters = 100
    eval_only = False  # if True, script exits right after the first eval
    always_save_checkpoint = (
        False  # if True, always save a checkpoint at each checkpoint_interval
    )
    init_from = "scratch"  # 'scratch' or 'resume'
    stream_hf_dataset = True  # stream the galaxies from huggingface
    leak_check = (
        True  # check for RAM leaks and reset dataloader if we reach > 80% RAM used
    )
    # data
    gradient_accumulation_steps = 5  # * 8  # used to simulate larger batch sizes
    batch_size = 32  # if gradient_accumulation_steps > 1, this is the micro-batch size
    spiral = True  # do we want to process the galaxy patches in spiral order?
    block_size = 1024
    image_size = 256
    num_workers = 64
    n_chan = 3  # 3 imagery bands: r, i, z for jpeg, 1 imagery band for FITS
    # Define modalities configuration
    # fmt: off
    galaxy_params = [
        "smooth-or-featured_smooth_fraction", "disk-edge-on_yes_fraction", "has-spiral-arms_yes_fraction", "bar_strong_fraction", "bulge-size_dominant_fraction", "how-rounded_cigar-shaped_fraction", "edge-on-bulge_boxy_fraction", "spiral-winding_tight_fraction", "merging_merger_fraction", "mag_u", "mag_g", "mag_r", "mag_i", "mag_z", "u_minus_r", "elpetro_absmag_r", "est_petro_th50_kpc", "petro_ba50", "petro_ba90", "elpetro_ba", "elpetro_phi", "sersic_n", "sersic_ba", "sersic_phi", "elpetro_mass_log", "redshift", "fibre_sfr_median", "fibre_ssfr_median", "total_sfr_median", "total_ssfr_median",
    ]
    # fmt: on
    modalities = [
        ModalityConfig(
            name="images",
            input_size=16 * 16 * n_chan,
            patch_size=16,
            loss_weight=1.0,
            embed_pos=True,
            pos_input_size=1,
        ),
        *[
            ModalityConfig(
                name=param,
                input_size=1,
                patch_size=1,
                loss_weight=0.005,  # less than the stable image training
                embed_pos=True,
                pos_input_size=1,
            )
            for param in galaxy_params
        ],
    ]
    # Create modality registry
    modality_registry = ModalityRegistry(modalities)
    # Which backbone and LoRA rank do we use?
    llm_model_name = "HuggingFaceTB/SmolLM3-3B"  # or "HuggingFaceTB/SmolLM3-3B-Base"
    lora_r = 32
    lora_alpha = 32
    use_qlora = False
    # Choose tokenisers from "affine" and "aim"
    tokeniser = "affine"
    # adamw optimizer
    # we follow the same schedule here as Chinchilla
    learning_rate = 6e-4  # max learning rate
    max_iters = (
        12000  # total number of training iterations for one pass over our dataset
    )
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True  # whether to decay the learning rate
    warmup_iters = 1000  # how many steps to warm up for
    lr_decay_iters = 10000 * 1.1  # should be ~= max_iters per Chinchilla
    min_lr = (
        learning_rate / 10
    )  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # DDP settings
    backend = "nccl"  # 'nccl', 'gloo', etc.
    # system
    device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = "bfloat16"  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = True  # use PyTorch 2.0 to compile the model to be faster
    wandb_project = None
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

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # model init
    model_args = dict(
        backbone="llm",
        llm_model_name=llm_model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        use_qlora=use_qlora,
        n_chan=n_chan,
        modalities=modalities,
        tokeniser=tokeniser,
    )

    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, modality_registry, master_process=master_process)

    # Setup special tokens for the model
    special_tokens = []
    for mod_name in modality_registry.names():
        special_tokens.extend(
            [f"<|begin_{mod_name}|>", f"<|{mod_name}|>", f"<|end_{mod_name}|>"]
        )

    model.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.llm.resize_token_embeddings(len(model.tokenizer))
    model.special_token_ids = {
        token: model.tokenizer.convert_tokens_to_ids(token) for token in special_tokens
    }

    if init_from == "scratch":
        # init a new model from scratch
        if master_process:
            print("initializing a new model from scratch")
    if init_from == "resume":
        if master_process:
            print(f"resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
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
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

    galaxies_train = load_dataset(
            "Smith42/galaxies",
            revision="v2.0",
            streaming=stream_hf_dataset,
            split="train",
    )
    if not stream_hf_dataset:
        galaxies_train = galaxies_train.to_iterable_dataset(num_shards=num_workers)
    galaxies_train = (galaxies_train
        .select_columns(["image"] + galaxy_params)
        .shuffle(seed=None, buffer_size=1000)
    )
    galaxies_train = itertools.cycle(galaxies_train)
    galaxies_test = load_dataset(
            "Smith42/galaxies",
            revision="v2.0",
            streaming=stream_hf_dataset,
            split="test",
    )
    if not stream_hf_dataset:
        galaxies_test = galaxies_test.to_iterable_dataset(num_shards=num_workers)
    galaxies_test = (galaxies_test
        .select_columns(["image"] + galaxy_params)
        .shuffle(seed=None, buffer_size=1000)
    )
    galaxies_test = itertools.cycle(galaxies_test)

    transforms = {"images": data_transforms()}

    def create_dataloaders():
        unwrapped_model = model.module if hasattr(model, "module") else model
        tds = LLMModalityDataset(
            hf_dataset=galaxies_train,
            modality_registry=modality_registry,
            tokenizer=unwrapped_model.tokenizer,
            special_token_ids=unwrapped_model.special_token_ids,
            transforms=transforms,
            random_order=True,
        )
        vds = LLMModalityDataset(
            hf_dataset=galaxies_test,
            modality_registry=modality_registry,
            tokenizer=unwrapped_model.tokenizer,
            special_token_ids=unwrapped_model.special_token_ids,
            transforms=transforms,
            random_order=True,
        )
        tdl = iter(
            DataLoader(
                tds,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=llm_collate_fn,
                pin_memory=True,
            )
        )
        vdl = iter(
            DataLoader(
                vds,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=llm_collate_fn,
                pin_memory=True,
            )
        )
        return tds, vds, tdl, vdl

    tds, vds, tdl, vdl = create_dataloaders()

    # logging via wandb if available
    # this is here so we can get the number of params from model()
    if log_via_wandb and master_process:
        if wandb_project is None:
            wandb.init(
                project=f"AstroPT-{model.get_num_params() / 1e6:06.1f}M",
                config=config,
            )
        else:
            wandb.init(
                project=wandb_project,
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

    print(f"Allocated model memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved model memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # For DDP with LLM backbone, ensure all components are on the correct device
    if ddp and hasattr(model, "llm") and model.llm is not None:
        # The .to(device) call above should handle this, but let's be explicit
        model.llm.to(device)
        for module in [model.encoders, model.decoders, model.embedders]:
            module.to(device)

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
        # here we only compile the llm, the most expensive part
        # TODO rewrite encoders and decoders to allow performant compilation of
        # those (atm they have too many dynamic ops to be worth it!)
        model.llm = torch.compile(model.llm)  # requires PyTorch 2.0

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

        P_data = {name: [] for name in modality_registry.names()}
        Y_data = {name: [] for name in modality_registry.names()}

        for dl, split in zip(
            [tdl, vdl],
            ["train", "val"],
        ):
            out[split] = {}
            losses = torch.zeros(eval_iters)
            mode_losses = {name: [] for name in modality_registry.names()}
            for k in range(eval_iters):
                B = to_device(next(dl), device=device)
                target_infos = B["Y"]["modality_infos"]
                with ctx:
                    pred_infos, loss = model(B["X"], targets=B["Y"])
                for batch_idx in range(len(pred_infos)):
                    P_info = pred_infos[batch_idx]
                    Y_info = target_infos[batch_idx]
                    for ii, pred_name in enumerate(P_info["names"]):
                        mode_losses[pred_name].append(P_info["losses"][ii])
                        if pred_name != "images":
                            P_data[pred_name].append(
                                P_info["data"][ii].squeeze().detach().cpu().item()
                            )
                            Y_data[pred_name].append(
                                Y_info["data"][ii].squeeze().detach().cpu().item()
                            )
                losses[k] = loss.item()
            out[split]["all"] = losses.mean()
            for name in modality_registry.names():
                out[split][name] = np.array(mode_losses[name]).mean()

        if log_via_wandb:
            for name in modality_registry.names():
                wandb.log(
                    {
                        name: wandb.Table(
                            columns=["Y", "P"],
                            data=list(zip(Y_data[name], P_data[name])),
                        )
                    },
                    step=iter_num,
                )

        model.train()
        return out

    @torch.no_grad()
    def validate(iter_num, out_dir):
        model.eval()
        for dl, split in zip([tdl, vdl], ["train", "val"]):
            f, axs = plt.subplots(8, 2, figsize=(3, 12), constrained_layout=True)
            B = to_device(next(vdl), device=device)
            with ctx:
                pred_modality_infos, loss = model(B["X"], B["Y"])

                print(
                    stringify(B["Y"]["token_sequences"][0], B["Y"]["modality_infos"][0])
                )
                print(stringify(B["Y"]["token_sequences"][0], pred_modality_infos[0]))

                if "images" in modality_registry.names():
                    Yim = []
                    for mod_info in B["Y"]["modality_infos"]:
                        for ii, name in enumerate(mod_info["names"]):
                            if name == "images":
                                Yim.append(mod_info["data"][ii].to(device))
                    Yim = torch.stack(Yim)
                    b, t, c = Yim.size()
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
                    Pim = []
                    for mod_info in pred_modality_infos:
                        for ii, name in enumerate(mod_info["names"]):
                            if name == "images":
                                Pim.append(mod_info["data"][ii].to(device))
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
                        mmn = lambda x: 255 * (x - x.min()) / (x.max() - x.min())
                        wandb.log(
                            {
                                "Y": [
                                    wandb.Image(mmn(im.swapaxes(0, -1)))
                                    for im in Yim[:32]
                                ],
                                "P": [
                                    wandb.Image(mmn(im.swapaxes(0, -1)))
                                    for im in Pim[:32]
                                ],
                            },
                            step=iter_num,
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
    B = to_device(next(tdl), device=device)
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
            val_loss = losses["val"]["all"]
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
                    figsize=((len(losses["train"]) + 1) * 2, 2.5),
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
                plt.close(f)

            if val_loss < best_val_loss or always_save_checkpoint:
                best_val_loss = val_loss
                if iter_num > 0:
                    model_state = raw_model.state_dict()
                    checkpoint = {
                        "model": model_state,
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
                _, loss = model(B["X"], targets=B["Y"])
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            B = to_device(next(tdl), device=device)
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
            lossf = loss.item()
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size, dt)
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            if log_via_wandb:
                wandb.log({"loss": lossf, "time": dt}, step=iter_num)
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

        if leak_check and psutil.virtual_memory().percent > 80 and iter_num > 0:
            # We reset the dataloaders as Hugging Face has an annoying data leak for its streaming dataloaders!
            if master_process:
                print("Resetting DataLoader workers as RAM is >80% used")
            tds, vds, tdl, vdl = create_dataloaders()

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
