"""
Sample from a trained astropt model
"""
import os
import pickle
from contextlib import nullcontext
import torch
from torch.utils.data import DataLoader
import tiktoken
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
from model import GPTConfig, GPT
from datasets import load_dataset
from train import GalaxyImageDataset
from torchvision import transforms
import functools
from einops import rearrange
import pandas as pd

from sklearn.decomposition import PCA

# -----------------------------------------------------------------------------
init_from = 'resume'
out_dir = 'logs/spiralized_astropt_300M' # ignored if init_from is not 'resume'
refresh_cache = False # resample the embeddings
batch_size = 256
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('astroPT/configurator.py').read()) # overrides from command line or config file
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
ds = ds.map(
    functools.partial(_process_galaxy_wrapper, func=galproc.process_galaxy)
).with_format("torch")
dl = iter(DataLoader(
    ds, batch_size=batch_size, num_workers=2,
))

def run_pca(zs):
    pca = PCA(n_components=2)
    zs_pca = pca.fit_transform(zs)
    return zs_pca

#metadata = pd.read_parquet("metadata.parquet")
#print([key for key in metadata.keys()])

def plot_embeddings(xs, ids, dumpto=os.path.join(out_dir, "test.png")):
    f, axs = plt.subplots(2, 4, figsize=(16, 4*2), constrained_layout=True)
    print("Reading metadata")
    metadata = pd.read_parquet("metadata.parquet")
    metadata = metadata.set_index(["dr8_id"])
    metadata = pd.concat([metadata.loc[id_] for id_ in tqdm(ids)])

    pcs = run_pca(xs)
    for ax, metadatum in zip(axs.ravel(), [
                "redshift", "mag_g", "sersic_n", "has-spiral-arms_yes_fraction",
                "u_minus_r", "elpetro_mass_log", "total_sfr_avg", "total_ssfr_avg",
            ]):
        md_to_plot = metadata[metadatum]
        im = ax.scatter(
            pcs[:, 0], pcs[:, 1], 
            c=md_to_plot.clip(md_to_plot.quantile(0.01), md_to_plot.quantile(0.99)),
            s=1, cmap="magma", alpha=0.4
        )
        ax.set_xlabel(metadatum)
        cbar = plt.colorbar(im)
    f.savefig(dumpto, dpi=300)

if (not (
        os.path.isfile(os.path.join(out_dir, "zss.npy")) and 
        os.path.isfile(os.path.join(out_dir, "idss.npy"))
   )) or refresh_cache:
    # run generation
    xss = []
    zss = []
    idss = []
    with torch.no_grad():
        with ctx:
            tt = tqdm(unit="galz", unit_scale=True)
            for B in dl:
                xs = B["image"][:, :64]
                ids = B["dr8_id"]
                zs = model.generate_embeddings(xs.to(device))
                xss.append(rearrange(xs, "b t c -> b (t c)").detach().cpu().numpy())
                zss.append(zs.detach().cpu().numpy())
                idss.append(ids)
                tt.update(batch_size)
            tt.close()

    zss = np.concatenate(zss, axis=0)
    xss = np.concatenate(xss, axis=0)
    idss = np.concatenate(idss, axis=0)
    np.save(os.path.join(out_dir, "xss_64t.npy"), xss)
    np.save(os.path.join(out_dir, "zss_64t.npy"), zss)
    np.save(os.path.join(out_dir, "idss_64t.npy"), idss)
else:
    print("loading from cache")
    xss = np.load(os.path.join(out_dir, "xss_64t.npy"))
    zss = np.load(os.path.join(out_dir, "zss_64t.npy"))
    idss = np.load(os.path.join(out_dir, "idss_64t.npy"))

print("plotting...")
plot_embeddings(zss, idss, dumpto=os.path.join(out_dir, f"embeddings_z_64t.png"))
plot_embeddings(xss, idss, dumpto=os.path.join(out_dir, f"embeddings_x_64t.png"))
