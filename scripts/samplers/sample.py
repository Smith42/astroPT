"""
Sample from a trained astropt model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
from astropt.model import GPTConfig, GPT
from datasets import load_dataset
from astropt.local_datasets import GalaxyImageDataset
from torchvision import transforms
import functools
import einops

# -----------------------------------------------------------------------------
init_from = 'resume'
out_dir = 'logs/spiralized_astropt_300M' # ignored if init_from is not 'resume'
prompt = '' # promptfile (numpy)
num_samples = 4 # number of samples to draw
max_new_tokens = 1024 # number of tokens generated in each sample
#temperature = 0.002 # 0.0 = no change, > = more random in predictions
spread = False
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('src/astropt/configurator.py').read()) # overrides from command line or config file
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
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
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

# set up HF galaxies in test set to be processed
def data_transforms():
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x/255.),
    ])
    return transform
def _process_galaxy_wrapper(gal, func):
    patch_galaxy = func(np.array(gal["image"]).swapaxes(0, 2))
    return {"image": patch_galaxy}
galproc = GalaxyImageDataset(None, spiral=True, transform=data_transforms())
ds = load_dataset("Smith42/galaxies", split="test", streaming=True)
ds = iter(ds.map(
    functools.partial(_process_galaxy_wrapper, func=galproc.process_galaxy)
).with_format("torch"))

# get samples for processing
xs = torch.stack([next(ds)["image"] for _ in range(num_samples)])[:, 0:16]

def plot_galaxies(xs, dumpto=os.path.join(out_dir, "test.png")):
    f, axs = plt.subplots(8, 4, figsize=(8, 16), constrained_layout=True)

    for ax, x in zip(axs.ravel(), xs):
        ax.axis("off")
        ax.imshow(x)

    for i, axlabel in enumerate(np.logspace(-4, 0, 8)):
        axs[i, 0].text(0.15, 0.8, f"T={axlabel:0.5f}", color='black', 
                       bbox=dict(facecolor='white', edgecolor='white', pad=2.0))

    f.savefig(dumpto)

# run generation
pss = []
with torch.no_grad():
    with ctx:
        for temperature in np.logspace(-4, 0, 8):
            print("inferring", temperature)
            ps = model.generate(xs.to(device), max_new_tokens - xs.shape[1], temperature=temperature)
            ps = (ps - ps.min())/(ps.max() - ps.min())
            # Rearrange galaxy images so that they are nice and in png format:
            ps = torch.stack([galproc.antispiralise(pp) for pp in ps])
            ps = einops.rearrange(ps, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c', p1=16, p2=16, h=32, w=32)
            pss.append(ps.detach().cpu().numpy())

        print("Plotting...")
        plot_galaxies(np.concatenate(pss), dumpto=os.path.join(out_dir, f"ps.png"))
