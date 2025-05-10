"""
Sample from a trained astropt model and finetune embeddings on linear probes 
"""
import os
from contextlib import nullcontext
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from model import GPTConfig, GPT
from sklearn.linear_model import LinearRegression
from datasets import load_dataset, concatenate_datasets
from train import GalaxyImageDataset
from torchvision import transforms
from torchvision.transforms import ToTensor
import functools
import pandas as pd

# -----------------------------------------------------------------------------
init_from = 'resume'
out_dir = './logs/9B_tokens/astropt001M/' # ignored if init_from is not 'resume'
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
    print("loading from checkpoint")
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, '030000_ckpt.pt')
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
def normalise(x):
    std, mean = torch.std_mean(x, dim=1, keepdim=True)
    return (x - mean)/(std + 1e-8)
def data_transforms():
    transform = transforms.Compose([
        transforms.Lambda(normalise),
    ])
    return transform
def _process_galaxy_wrapper(gal, func):
    gal = ToTensor()(gal["image"]).to(torch.bfloat16)
    patch_galaxy = func(gal)
    return {"image": patch_galaxy}
galproc = GalaxyImageDataset(None, spiral=True, transform=data_transforms())
ds = concatenate_datasets(( 
    load_dataset("Smith42/galaxies", split="test", streaming=True),
    load_dataset("Smith42/galaxies", split="validation", streaming=True),
))
ds = ds.map(
    functools.partial(_process_galaxy_wrapper, func=galproc.process_galaxy)
).with_format("torch")
dl = iter(DataLoader(
    ds, batch_size=batch_size, num_workers=2,
))

def train_probe(zs, ys):
    probe = LinearRegression()
    probe.fit(zs, ys)
    return probe

def run_probes(xs, ids, metadata):
    mdata_fields = metadata.keys()
    def reject_outliers(data, m=3.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else np.zeros(len(d))
        return (s < m)

    losses = []
    fields = []
    for metadatum in tqdm(mdata_fields):
        if metadatum == "g_minus_r":
            ys = pd.Series(metadata["mag_g_desi"].to_numpy() - metadata["mag_r_desi"].to_numpy())
        if metadatum == "r_minus_z":
            ys = pd.Series(metadata["mag_r_desi"].to_numpy() - metadata["mag_z_desi"].to_numpy())
        if ys.dtype != "object":

            nonnans = (np.isfinite(ys) & np.all(np.isfinite(xs), axis=1))
            xs_ = xs[nonnans]
            ys_ = ys[nonnans]

            # remove outliers above 3 sigma
            inliers = reject_outliers(ys_)
            xs_ = xs_[inliers]
            ys_ = ys_[inliers]

            # robust normalisation
            ys_ = (ys_ - np.median(ys_))/(np.quantile(ys_, 0.75) - np.quantile(ys_, 0.25))
            if np.all(np.isfinite(ys_)):
                halfway = len(ys_)//2
                probe = train_probe(xs_[:halfway], ys_[:halfway])
                losses.append(
                  (np.abs(probe.predict(xs_[halfway:]) - ys_[halfway:])).median()
                )
            else:
                losses.append(
                    np.nan
                )
            fields.append(metadatum)

    return fields, losses

n_tokens = 64
norm = "mean"
if (not (
        os.path.isfile(os.path.join(out_dir, f"zss_{n_tokens}t_{norm}.npy")) and 
        os.path.isfile(os.path.join(out_dir, f"idss_{n_tokens}t_{norm}.npy")) and
        os.path.isfile(os.path.join(out_dir, "metadata_processed.parquet"))
   )) or refresh_cache:
    # run generation
    zss = []
    idss = []
    with torch.no_grad():
        with ctx:
            tt = tqdm(unit="galz", unit_scale=True, total=87000)
            for B in dl:
                xs = B["image"][:, :n_tokens]
                ids = B["dr8_id"]
                zs = model.generate_embeddings(xs.to(device), average_type=norm)
                zss.append(zs.detach().cpu().numpy())
                idss.append(ids)
                tt.update(batch_size)
            tt.close()

    zss = np.concatenate(zss, axis=0)
    idss = np.concatenate(idss, axis=0)
    np.save(os.path.join(out_dir, f"zss_{n_tokens}t_{norm}.npy"), zss)
    np.save(os.path.join(out_dir, f"idss_{n_tokens}t_{norm}.npy"), idss)

    print("processing metadata file")
    metadata = pd.read_parquet("/raid/data/metadata.parquet")
    metadata = metadata.set_index(["dr8_id"])
    metadata = metadata.loc[list(idss)]
    metadata.to_parquet(os.path.join(out_dir, "metadata_processed.parquet"))
else:
    print("loading from cache")
    metadata = pd.read_parquet(os.path.join(out_dir, "metadata_processed.parquet"))
    zss = np.load(os.path.join(out_dir, f"zss_{n_tokens}t_{norm}.npy"))
    idss = np.load(os.path.join(out_dir, f"idss_{n_tokens}t_{norm}.npy"))

print("probing...")
labels, loss_zs = run_probes(zss, idss, metadata)
print(loss_zs)
file_path = "probe_losses.txt"
if (not os.path.exists(file_path)) or os.path.getsize(file_path) == 0:
    with open(file_path, 'w') as f:
        f.write(','.join(labels) + '\n')
with open(file_path, "a") as f:
    np.savetxt(f, np.array(loss_zs)[np.newaxis], delimiter=",")
