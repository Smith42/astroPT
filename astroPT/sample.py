"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
from model import GPTConfig, GPT
import functools

# -----------------------------------------------------------------------------
init_from = 'resume'
out_dir = 'logs/astropt' # ignored if init_from is not 'resume'
prompt = '' # promptfile (numpy)
num_samples = 1 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.00 # 0.0 = no change, < = less random, > = more random, in predictions
spread = False
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
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

# start file (numpy)
if start != '':
    x = torch.tensor(np.load(start)).to(device=device)
else:
    # This is an initial random input
    train_data = np.load('data/big_eo/TL64_train.npy', mmap_mode='r+')
    test_data = np.load('data/big_eo/TL64_test.npy', mmap_mode='r+')
    ii = torch.randint(len(train_data), (2,))
    print(ii)
    #ii = torch.tensor([397*1024 + 442, 442*1024 + 397])
    train_xs = torch.stack([torch.from_numpy((train_data[i]).astype(float)) for i in ii])
    test_xs = torch.stack([torch.from_numpy((test_data[i]).astype(float)) for i in ii])
    xs = torch.cat((train_xs, test_xs), dim=1)
    t_embs = xs[0, :, 10:12].to(device).float() # time embeddings for year
    ts = xs[0, 500:, 10:14].to(device).float()
    gts = xs[:, :, :10].to(device).float()
    xs = xs[:, :500, :14].to(device).float()

def plot_prediction(y, gt, spread=None, dumpto=os.path.join(out_dir, "test.png")):
    f, axs = plt.subplots(5, 2, figsize=(30, 16))
    names = [ "Blue", "Green", "NIR", "Red", "Red Edge 1", "Red Edge 2", "Red Edge 3", "Red Edge 4", "SWIR 1", "SWIR 2" ]

    for ax, ch, name in zip(axs.ravel(), range(10), names):
        ax.plot(y[:, ch], color="blue", label="Prediction")
        if spread is not None:
            ax.fill_between(
                range(len(y)),
                y[:, ch] - spread[:, ch], 
                y[:, ch] + spread[:, ch], 
                alpha=0.4,
                color="blue",
            )
        ax.plot(gt[:, ch], color="orange", label="Ground Truth")
        ax.set_title(name)
        ax.legend()
        f.savefig(dumpto)

# run generation
with torch.no_grad():
    with ctx:
        ys = model.generate(xs, ts, max_new_tokens, temperature=temperature).detach().cpu().numpy()
        gts = gts.detach().cpu().numpy()
        t_embs = t_embs.detach().cpu().numpy()
        unnorm = lambda ar: ar*255.
        ys = unnorm(ys)
        gts = unnorm(gts)

        print("Plotting...")
        for i, y, gt in tqdm(zip(range(len(ys)), ys, gts), total=len(ys)):
            plot_prediction(y[:], gt[:], dumpto=os.path.join(out_dir, f"p_{i:03d}.png"))
