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

from astropt.local_datasets import GalaxyImageDataset
from astropt.model_utils import load_astropt

if __name__ == "__main__":
    # set up HF galaxies in test set to be processed
    def normalise(x):
        std, mean = torch.std_mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + 1e-8)

    def data_transforms():
        transform = transforms.Compose(
            [
                transforms.Lambda(normalise),
            ]
        )
        return transform

    def _process_galaxy_wrapper(idx, func):
        """This function ensures that the image is tokenised in the same way as
        the pre-trained model is expecting"""
        galaxy = func(
            torch.from_numpy(np.array(idx["image"]).swapaxes(0, 2)).to(float)
        ).to(torch.float)
        galaxy_positions = torch.arange(0, len(galaxy), dtype=torch.long)
        mag_g = idx["mag_g"]
        return {
            "images": galaxy,
            "images_positions": galaxy_positions,
            "mag_g": mag_g,
        }

    model = load_astropt("Smith42/astroPT_v2.0", path="astropt/095M")
    galproc = GalaxyImageDataset(
        None,
        spiral=True,
        transform={"images": data_transforms()},
        modality_registry=model.modality_registry,
    )
    # load dataset: we select mag_g as it is easy for this example but there are many values in Smith42/galaxies v2.0, you can choose from any column in hf.co/datasets/Smith42/galaxies_metadata
    ds = (
        load_dataset("Smith42/galaxies", split="test", revision="v2.0", streaming=True)
        .select_columns(("image", "mag_g"))
        .filter(lambda idx: idx["mag_g"] is not None)
        .map(partial(_process_galaxy_wrapper, func=galproc.process_galaxy))
        .with_format("torch")
        .take(
            1000
        )  # use the first 1k examples of our dataset to shorten total inference time
    )
    dl = iter(
        DataLoader(
            ds,
            batch_size=32,
            num_workers=0,
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
            zs = model.generate_embeddings(B)["images"].detach().numpy()
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
