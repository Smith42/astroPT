"""
Sample embeddings from a trained AstroPT model
"""

import os
import functools
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor

import numpy as np
import pandas as pd
from tqdm import tqdm
from einops import rearrange

from datasets import load_dataset, concatenate_datasets

from astropt.model import GPT, GPTConfig
from astropt.local_datasets import GalaxyImageDataset

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
init_from = "resume"
out_dir = "logs/AIM"
batch_size = 256
seed = 1337
spiral = True
prefix_len = 64
device = "cuda"
dtype = "bfloat16"
compile = False

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]

ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(
    device_type=device_type, dtype=ptdtype
)

# -----------------------------------------------------------------------------
# Load checkpoint
# -----------------------------------------------------------------------------
ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

modality_registry = checkpoint["modality_registry"]

gptconf = GPTConfig(**checkpoint["model_args"])
model = GPT(gptconf, modality_registry)
state_dict = checkpoint["model"]

# clean DDP prefix if present
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval().to(device)

if compile:
    model = torch.compile(model)

# -----------------------------------------------------------------------------
# Dataset + transforms (MATCH TRAINING)
# -----------------------------------------------------------------------------
def normalise(x):
    std, mean = torch.std_mean(x, dim=1, keepdim=True)
    return (x - mean) / (std + 1e-8)


def data_transforms():
    return transforms.Compose([
        transforms.Lambda(normalise),
    ])


from PIL import Image
import numpy as np

def process_galaxy_wrapper(gal, func):
    img = gal["image"]

    # Case 1: PIL image (JPEG/PNG from HF)
    if isinstance(img, Image.Image):
        gal_tensor = ToTensor()(img).to(torch.float16)

    # Case 2: numpy array
    elif isinstance(img, np.ndarray):
        gal_tensor = torch.from_numpy(img).permute(2, 0, 1).to(torch.float16)

    # Case 3: torch tensor (local HF parquet)
    elif torch.is_tensor(img):
        gal_tensor = img.to(torch.float16)

    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    patches = func(gal_tensor)

    return {
        "images": patches,
        "images_positions": torch.arange(
            patches.shape[0], dtype=torch.long
        ),
        "dr8_id": gal.get("dr8_id", -1),
    }


galproc = GalaxyImageDataset(
    paths=None,
    spiral=spiral,
    transform={"images": data_transforms()},
    modality_registry=modality_registry,
)

# -----------------------------------------------------------------------------
# HF dataset (test + validation)
# -----------------------------------------------------------------------------
ds = concatenate_datasets((
    load_dataset("/scratch02/public/sao/msmith/data/galaxies/", revision="v2.0", split="test", streaming=True),
    load_dataset("/scratch02/public/sao/msmith/data/galaxies/", revision="v2.0", split="validation", streaming=True),
))

ds = ds.map(
    functools.partial(process_galaxy_wrapper, func=galproc.process_galaxy)
).with_format("torch")
ds = (
    ds
    .select_columns("image_crop")
    .rename_column("image_crop", "image")
    .map(
        functools.partial(process_galaxy_wrapper, func=galproc.process_galaxy)
    )
    ).with_format("torch")
ds = ds.remove_columns("image")

dl = DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=2,
    pin_memory=True,
)

# -----------------------------------------------------------------------------
# Embedding extraction
# -----------------------------------------------------------------------------
n_tokens = prefix_len
norm = "mean"

zss = []
idxs = []

with torch.no_grad():
    with ctx:
        tt = tqdm(unit="galaxies", unit_scale=True)

        for B in dl:
            xs = B["images"][:, :prefix_len].to(device)
            pos = B["images_positions"][:, :prefix_len].to(device)

            inputs = {
                "images": xs,
                "images_positions": pos,
            }

            zs = model.generate_embeddings(inputs)

            zss.append(zs["images"].detach().cpu().numpy())
            idxs.append(np.array(B["dr8_id"]))

            tt.update(xs.size(0))

        tt.close()

# -----------------------------------------------------------------------------
# Save outputs
# -----------------------------------------------------------------------------
zss = np.concatenate(zss, axis=0)
idxs = np.concatenate(idxs, axis=0)

np.save(os.path.join(out_dir, f"zss_{n_tokens}t_{norm}.npy"), zss)
np.save(os.path.join(out_dir, f"idxs_{n_tokens}t_{norm}.npy"), idxs)

print("Saved embeddings:", zss.shape)

# -----------------------------------------------------------------------------
# Optional: metadata join
# -----------------------------------------------------------------------------
# metadata = pd.read_parquet("/scratch02/public/sao/msmith/data/metadata.parquet").set_index("dr8_id")
# metadata = metadata.loc[idxs]
# metadata.to_parquet(os.path.join(out_dir, "metadata_processed.parquet"))

print("Done.")
