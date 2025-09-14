"""
Example script that samples from a trained astropt model and finetunes
embeddings on a linear probe
"""

from functools import partial
from os.path import isfile

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from astropt.local_datasets import LLMModalityDataset, llm_collate_fn
from astropt.model_utils import load_astropt

if __name__ == "__main__":
    # set up HF galaxies in test set to be processed
    def normalise(x):
        std, mean = torch.std_mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + 1e-8)

    def data_transforms():
        transform = transforms.Compose(
            [transforms.Lambda(normalise),]
        )
        return transform

    model = load_astropt(repo_id="Smith42/astroPT_v3.0", path="astropt/smollm")

    # load dataset: we select mag_g as it is easy for this example but there are many values in Smith42/galaxies v2.0, you can choose from any column in hf.co/datasets/Smith42/galaxies_metadata
    galaxies = (
        load_dataset(
            "Smith42/galaxies",
            revision="v2.0",
            streaming=True,
            split="test",
        )
        .select_columns(("image", "mag_g"))
        .filter(lambda idx: idx["mag_g"] is not None)
        .shuffle(seed=None, buffer_size=1000)
        .take(1000)
    )

    transforms = {"images": data_transforms()}

    unwrapped_model = model.module if hasattr(model, "module") else model
    ds = LLMModalityDataset(
        hf_dataset=galaxies,
        modality_registry=unwrapped_model.modality_registry,
        tokenizer=unwrapped_model.tokenizer,
        special_token_ids=unwrapped_model.special_token_ids,
        transforms=transforms,
        random_order=False,
    )
    dl = iter(
        DataLoader(
            ds,
            batch_size=16,
            num_workers=0,
            collate_fn=llm_collate_fn,
            pin_memory=True,
        )
    )

    def train_probe(zs, ys):
        probe = LinearRegression()
        probe.fit(zs, ys)
        return probe

    if not isfile("zss.npy"):
        # here we run a loop over the dataset to generate our embeddings
        zss = []
        yss = []
        for B in tqdm(dl):
            zs = model.generate_embeddings({
                "images": B["images"], "images_positions": B["images_positions"]
            })["images"].detach().numpy()
            zss.append(zs)
            yss.append(B["mag_g"].detach().numpy())
        zss = np.concatenate(zss, axis=0)
        yss = np.concatenate(yss, axis=0)
        np.save("zss.npy", zss)
        np.save("yss.npy", yss)
    else:
        zss = np.load("zss.npy")
        yss = np.load("yss.npy")
        print(
            "Embeddings file (zss.npy) detected so moving straight to linear probe and viz"
        )

    print("Training probe...")
    # Now we train a linear probe on half the data and test on the other half
    # In a "real" setting you may want to use a more powerful model than a linear regressor
    # (and possibly a more difficult problem then magnitude prediction ;) !)
    halfway = len(zss) // 2
    probe = train_probe(zss[:halfway], yss[:halfway])
    pss = probe.predict(zss[halfway:])
    print(
        f"MSE: {mean_squared_error(pss, yss[halfway:])} R2: {r2_score(pss, yss[halfway:])}"
    )

    # Now let's visualise the embedding space by performing PCA and plotting
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(zss)

    vmax = np.percentile(yss, 95)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=yss, vmax=vmax, cmap="viridis")
    plt.colorbar(label="mag_g")
    plt.show()

    plt.plot(yss[halfway:], pss, ".")
    plt.ylabel("Ground truth magnitude")
    plt.ylabel("Predicted magnitude")
    plt.show()
