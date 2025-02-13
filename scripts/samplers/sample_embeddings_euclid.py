"""
Sample from a trained astropt model for Euclid data
"""
import os
import pickle
from contextlib import nullcontext
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
from astropt.model import GPTConfig, GPT
from datasets import load_dataset, concatenate_datasets 
from astropt.local_datasets import GalaxyImageDataset
from torchvision import transforms
from torchvision.transforms import ToTensor
import functools
from einops import rearrange
import pandas as pd

# -----------------------------------------------------------------------------
init_from = 'resume'
out_dir = 'logs/astropt090M_euclid/' # ignored if init_from is not 'resume'
refresh_cache = False # resample the embeddings
batch_size = 256
seed = 1337
spiral = True # do we want to process the galaxy patches in spiral order?
patch_size = 16 # size of image patches for ViT tokenisation
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('src/configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, '002500_ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    # TODO remove this for latest models
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# set up Euclid galaxies in test set to be processed
def normalise(x):
    std, mean = torch.std_mean(x, dim=1, keepdim=True)
    return (x - mean)/(std + 1e-8)
def data_transforms():
    transform = transforms.Compose([
        transforms.Lambda(normalise),
    ])
    return transform
ds = GalaxyImageDataset(
    paths='test.txt',
    spiral=spiral, 
    transform=data_transforms(), 
    patch_size=patch_size,
    stochastic=False,
)
dl = iter(DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=8,
    pin_memory=True,
))

n_tokens = 64
norm = "mean"
if (not (
        os.path.isfile(os.path.join(out_dir, f"zss_{n_tokens}t_{norm}.npy")) and 
        os.path.isfile(os.path.join(out_dir, f"idxs_{n_tokens}t_{norm}.npy")))
   ) or refresh_cache:
    # run generation
    xss = []
    zss = []
    idxs = []
    with torch.no_grad():
        with ctx:
            tt = tqdm(unit="galz", total=len(ds), unit_scale=True)
            for B in dl:
                if n_tokens == "all":
                    prefix_len = B["X"].shape[1]
                else:
                    prefix_len = n_tokens
                # flex attention does not yet work with variable inputs so error on compiled flex attention
                # see https://github.com/pytorch/pytorch/issues/139064
                xs = B["X"][:, :prefix_len]
                idx = B["idx"]
                if model.config.attn_type == "prefix":
                    # forward and backward attention over whole image if pretrained with prefix attention
                    zs = model.generate_embeddings(xs.to(device), prefix_len=prefix_len)
                else:
                    zs = model.generate_embeddings(xs.to(device))
                if not os.path.isfile(os.path.join(out_dir, f"xss_{n_tokens}t.npy")):
                    xss.append(rearrange(xs, "b t c -> b (t c)").detach().to(torch.float16).cpu().numpy())
                zss.append(zs.detach().cpu().numpy())
                idxs.append(idx)
                tt.update(batch_size)
            tt.close()

    if not os.path.isfile(os.path.join(out_dir, f"xss_{n_tokens}t.npy")):
        xss = np.concatenate(xss, axis=0)
        np.save(os.path.join(out_dir, f"xss_{n_tokens}t.npy"), xss)
    zss = np.concatenate(zss, axis=0)
    idxs = np.concatenate(idxs, axis=0)
    np.save(os.path.join(out_dir, f"zss_{n_tokens}t_{norm}.npy"), zss)
    np.save(os.path.join(out_dir, f"idxs_{n_tokens}t_{norm}.npy"), idxs)
else:
    print("loading from cache")
    zss = np.load(os.path.join(out_dir, f"zss_{n_tokens}t_{norm}.npy"))
    #xss = np.load(os.path.join(out_dir, f"xss_{n_tokens}t.npy"))
    idxs = np.load(os.path.join(out_dir, f"idxs_{n_tokens}t_{norm}.npy"))
