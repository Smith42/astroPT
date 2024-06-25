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

import os
import time
import math
from functools import partial
import pickle
from pathlib import Path
from contextlib import nullcontext
import einops
import PIL

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torchvision import transforms, io
from tqdm import trange
try:
    import wandb
    log_via_wandb = True
except:
    log_via_wandb = False
try:
    from codecarbon import EmissionsTracker
    log_emissions = True
except:
    log_emissions = False

from model import GPTConfig, GPT
from local_datasets import GalaxyImageDataset

if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # default config values designed to test run a model on DESI
    # look at config/astropt300m.py for a prod run example
    out_dir = 'logs/test'
    eval_interval = 1000
    log_interval = 100
    checkpoint_interval = 10000
    assert checkpoint_interval % eval_interval == 0
    eval_iters = 10
    eval_only = False # if True, script exits right after the first eval
    always_save_checkpoint = True # if True, always save a checkpoint at each checkpoint_interval
    init_from = 'scratch' # 'scratch' or 'resume'
    use_hf = True # use the huggingface dataset version of our galz
    stream_hf_dataset = False # stream the galaxies from huggingface
    # data
    gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
    batch_size = 8 # if gradient_accumulation_steps > 1, this is the micro-batch size
    spiral = True # do we want to process the galaxy patches in spiral order?
    block_size = 1024
    num_workers = 64 
    # astroPT model
    n_layer = 26
    n_head = 16
    n_embd = 768
    n_chan = 3 # 3 imagery bands: r, i, z
    dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False # do we use bias inside LayerNorm and Linear layers?
    patch_size = 16 # size of image patches for ViT tokenisation
    # adamw optimizer
    # we follow the same schedule here as Chinchilla
    learning_rate = 2e-4 # max learning rate
    max_iters = 30000 # total number of training iterations for one pass over our dataset
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 2000 # how many steps to warm up for
    lr_decay_iters = 27000 * 1.1 # should be ~= max_iters per Chinchilla
    min_lr = learning_rate/10 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # DDP settings
    backend = 'nccl' # 'nccl', 'gloo', etc.
    # system
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = True # use PyTorch 2.0 to compile the model to be faster
    log_via_wandb = False
    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    exec(open('src/configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------
    
    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        assert gradient_accumulation_steps % torch.cuda.device_count() == 0
        gradient_accumulation_steps //= torch.cuda.device_count()
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    if master_process:
        if log_via_wandb: print("wandb detected, gonna log to that")
        if log_emissions: print("codecarbon detected, will log emissions")
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
    
    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # dataset init
    def normalise(x):
        std, mean = torch.std_mean(x, dim=1, keepdim=True)
        return (x - mean)/(std + 1e-8)
    def data_transforms():
        transform = transforms.Compose([
            #transforms.Lambda(lambda x: x/255.),
            transforms.Lambda(normalise),
        ])
        return transform
    # training dataset and dataloader
    tpaths = None if use_hf else "./train.txt"
    tds = GalaxyImageDataset(tpaths, spiral=spiral, transform=data_transforms())
    # validation dataset and dataloader
    vpaths = None if use_hf else "./test.txt"
    vds = GalaxyImageDataset(vpaths, spiral=spiral, transform=data_transforms())

    if use_hf:
        from datasets import load_dataset, Image
        import io

        def filter_bumf(galdict):
            """ Lazily remove galaxies that are borked """
            try:
                gal = PIL.Image.open(io.BytesIO(galdict["image"]["bytes"]))
                return True
            except:
                return False
        def process_galaxy_wrapper(galdict, func):
            gal = PIL.Image.open(io.BytesIO(galdict["image"]["bytes"]))
            patch_galaxy = func(np.array(gal).swapaxes(0, 2))
            return { "X": patch_galaxy[:-1], "Y": patch_galaxy[1:], }

        tds_hf = load_dataset(
            "Smith42/galaxies",
            split="train",
            streaming=(True if stream_hf_dataset else False),
            cache_dir="/raid/data/cache",
        )
        tds_hf = tds_hf.cast_column("image", Image(decode=False))
        tds_hf = tds_hf.filter(filter_bumf).map(partial(process_galaxy_wrapper, func=tds.process_galaxy))
        tds_hf = tds_hf.remove_columns(["image", "dr8_id"])

        vds_hf = load_dataset(
            "Smith42/galaxies",
            split="test",
            streaming=(True if stream_hf_dataset else False),
            cache_dir="/raid/data/cache",
        )
        vds_hf = vds_hf.cast_column("image", Image(decode=False))
        vds_hf = vds_hf.filter(filter_bumf).map(partial(process_galaxy_wrapper, func=vds.process_galaxy))
        vds_hf = vds_hf.remove_columns(["image", "dr8_id"])

    sampler = None #DistributedSampler(dataset) if ddp else None
    tdl = iter(DataLoader(
        tds_hf if use_hf else tds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    ))
    vdl = iter(DataLoader(
        vds_hf if use_hf else vds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    ))
    
    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9
    
    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, n_chan=n_chan, block_size=block_size, dropout=dropout, patch_size=patch_size)
    
    if init_from == 'scratch':
        # init a new model from scratch
        if master_process: print("initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf, master_process=master_process)
    if init_from == 'resume':
        if master_process: print(f"resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        # NOTE had to remove 'bias' key here -- where does it go?!
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf, master_process=master_process)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

    # logging via wandb if available
    # this is here so we can get the number of params from model()
    if log_via_wandb and master_process:
        wandb.init(
            project = f"AstroPT-{model.get_num_params()/1e6:06.1f}M",
            config = config,
        )
    # write config and important information to log file
    with open(f"{out_dir}/hparams.txt", "w") as fi:
        fi.write(f"AstroPT-{model.get_num_params()/1e6:06.1f}M\n")
        fi.write(f"time: {int(time.time())}\n")
        for k, v in config.items():
            fi.write(f"{k}: {v}\n")

    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value
    model.to(device)
    
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    
    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory
    
    # compile the model
    if compile:
        if master_process: print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0
    
    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for dl, split in zip([tdl, vdl], ["train", "val"]):
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                B = next(dl)
                X = B["X"].to(device)
                Y = B["Y"].to(device)
                T = B["T"].to(device)
                with ctx:
                    logits, loss = model(X, "galaxy", Y, pos=T)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    
    @torch.no_grad()
    def validate(iter_num, out_dir):
        model.eval()
        for dl, split in zip([tdl, vdl], ["train", "val"]):
            f, axs = plt.subplots(2, 4, figsize=(6, 3), constrained_layout=True)
            B = next(dl)
            X = B["X"].to(device)
            Y = B["Y"].to(device)
            T = B["T"].to(device)
            with ctx:
                P, _ = model(X, Y)
            b, t, c = Y.size()
            zero_block = torch.zeros((b, 1, c)).to(device)
            Y = torch.cat((zero_block, Y), dim=1)
            if spiral: Y = torch.stack([vds.antispiralise(yy) for yy in Y])
            Y = einops.rearrange(Y, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c', p1=patch_size, p2=patch_size, h=32, w=32)
            P = torch.cat((zero_block, P), dim=1)
            if spiral: P = torch.stack([vds.antispiralise(pp) for pp in P])
            P = einops.rearrange(P, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c', p1=patch_size, p2=patch_size, h=32, w=32)
            if log_via_wandb:
                wandb.log({"Y": wandb.Image(Y.swapaxes(1, -1)), "P": wandb.Image(P.swapaxes(1, -1))})

            for ax, p, y in zip(axs.T, P.to(float).cpu().numpy(), Y.cpu().numpy()):
                ax[0].imshow(np.clip(y, 0, 1))
                ax[1].imshow(np.clip(p, 0, 1))
                ax[0].axis("off")
                ax[1].axis("off")
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
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)
    
    # training loop
    if master_process: print("starting training...")
    B = next(tdl) # fetch the very first batch
    X = B["X"].to(device)
    Y = B["Y"].to(device)
    T = B["T"].to(device)
    t0 = time.time()
    dts = []
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    running_mfu = -1.0
    if log_emissions:
        tracker = EmissionsTracker(output_dir=out_dir, log_level="error", save_to_file=True)
        tracker.start()
    while True:
    
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"iter {iter_num}: train loss {losses['train']:.6f}, val loss {losses['val']:.6f}")
            validate(iter_num, out_dir)
            with open(os.path.join(out_dir, "loss.txt"), "a") as fi:
                fi.write(f"{iter_num},{losses['train']},{losses['val']},{lr},{running_mfu*100}\n")
            if log_via_wandb:
                wandb.log({"valloss": losses['val']})
            if iter_num != 0:
                loss_ar = np.genfromtxt(os.path.join(out_dir, "loss.txt"), delimiter=",")
                f, ax = plt.subplots(1, 1, figsize=(8, 3))
                ax.plot(loss_ar[:, 0], loss_ar[:, 1], label="train")
                ax.plot(loss_ar[:, 0], loss_ar[:, 2], label="val")
                ax.set_yscale("log")
                ax.legend()
                f.savefig(os.path.join(out_dir, "loss.png"))
                plt.close()
    
        if ((iter_num % checkpoint_interval == 0 or iter_num == max_iters) and master_process):
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    if master_process: print(f"saving checkpoint to {out_dir}")
                    if always_save_checkpoint:
                        torch.save(checkpoint, os.path.join(out_dir, f'{iter_num:06d}_ckpt.pt'))
                    else:
                        torch.save(checkpoint, os.path.join(out_dir, f'ckpt.pt'))
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
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, "galaxy", Y, pos=T)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            B = next(tdl)
            X = B["X"].to(device)
            Y = B["Y"].to(device)
            T = B["T"].to(device)
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
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            if log_via_wandb:
                wandb.log({"loss": lossf, "time": dt})
            if log_emissions:
                emissions: float = tracker.flush()
                print(f"iter {iter_num}: loss {lossf:.6f}, time {np.mean(dts)*1000:.2f}ms, mfu {running_mfu*100:.2f}%, co2 {emissions:.1f}kg")
            else:
                print(f"iter {iter_num}: loss {lossf:.6f}, time {np.mean(dts)*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            dts = []

        iter_num += 1
        local_iter_num += 1
    
        # termination conditions
        if iter_num > max_iters:
            if log_emissions:
                emissions: float = tracker.stop()
                if master_process: print(emissions)
            break
    
    if ddp:
        destroy_process_group()
    if log_via_wandb:
        wandb.finish()
