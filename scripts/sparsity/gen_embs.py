from tqdm import tqdm
import PIL
import io
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Image
from torchvision import transforms

from astropt.model_utils import load_astropt
from astropt.local_datasets import GalaxyImageDataset


def filter_bumf(galdict):
    """ Lazily remove galaxies that are borked """
    try:
        gal = PIL.Image.open(io.BytesIO(galdict["image_crop"]["bytes"]))
        # Force full image load to catch truncation errors
        gal.load()
        if galdict["galaxy_size"] < 25000:
            return True
        else:
            return False
    except Exception as e:
        print(f"Filtering out corrupted image: {e}")
        return False

def process_galaxy_wrapper(galdict, func):
    gal = np.array(
        PIL.Image.open(io.BytesIO(galdict["image_crop"]["bytes"]))
    ).swapaxes(0, 2)
    patch_galaxy = func(gal)
    return { "X": patch_galaxy[:-1], "Y": patch_galaxy[1:], "raw_image":gal,}

def normalise(x):
    x = torch.from_numpy(x).to(torch.float32)
    std, mean = torch.std_mean(x, dim=1, keepdim=True)
    x_norm = (x - mean)/(std + 1e-8)
    return x_norm.to(torch.float16)

if __name__ == "__main__":
    for model_name in ["p16k00", "p16k01", "p16k10", "p32k00", "p32k01", "p32k10"]:
        print(f"Processing {model_name}")
        model, model_args = load_astropt(path=f"astropt/{model_name}")
        model = model.to("cuda")

        ds = GalaxyImageDataset(
                None, spiral=True, 
                transform=transforms.Compose([transforms.Lambda(normalise)]), 
                patch_size=model_args["patch_size"]
        )
        tds = load_dataset(
            "Smith42/galaxies",
            split="validation",
            streaming=True,
            cache_dir="/raid/huggingface_cache/galaxies",
        ).remove_columns("image")
        columns_to_remove = tds.column_names
        tds = tds.cast_column("image_crop", Image(decode=False))
        tds = tds.filter(filter_bumf).map(partial(process_galaxy_wrapper, func=ds.process_galaxy))
        tds = tds.remove_columns(columns_to_remove)
        tdl = DataLoader(tds, batch_size=64, num_workers=32, pin_memory=True)

        # register hook on bottleneck layer for us to access later
        class FeatureExtractor:
            def __init__(self, model, layer_path):
                self.features = None
                layer = model
                for attr in layer_path.split('.'):
                    if attr.isdigit():
                        layer = layer[int(attr)]
                    else:
                        layer = getattr(layer, attr)
                self.handle = layer.register_forward_hook(self.hook)
                
            def hook(self, module, input, output):
                self.features = output
                
            def remove(self):
                self.handle.remove()

        if model_args["k_ratio"] == 0:
            extractor = FeatureExtractor(model, "transformer.pre_blocks.5.mlp.dropout")
        else:
            extractor = FeatureExtractor(model, "sparse.to_overcomplete")

        all_embeddings = []
        with torch.no_grad():
            for B in tqdm(tdl, total=int(86500/64)):
                gal = B["X"].to("cuda", dtype=torch.float)
                _ = model(gal)
                all_embeddings.append(torch.mean(extractor.features, dim=1).cpu().numpy())

        all_embeddings = np.vstack(all_embeddings)
        np.save(f"p{model_args['patch_size']}k{int(100*model_args['k_ratio']):02d}.npy", all_embeddings)
