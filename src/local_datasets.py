"""
A place to store our pytorch datasets.
"""
import einops
import numpy as np
import torch
from torch.utils.data import Dataset
# reading from parquet with the following or via a HF dataset method does not
# work. I therefore read in for now with pandas. TODO change this when
# AstroPile is integrated.
#from torchdata.datapipes.iter import FileLister
#import torcharrow.dtypes as dt
#import pyarrow_hotfix; pyarrow_hotfix.uninstall()
import pandas as pd

class GalaxyImageDataset(Dataset):

    def __init__(self, paths=None, transform=None, stochastic=True, spiral=False, patch_size=16):
        """
        Arguments:
            paths: file with all the galaxy paths. Paths can be None if streaming from HF.
            transform (callable, optional): Optional transform to be applied on a sample.
            spiral: spiral form instead of raster form
            patch_size: size of ViT patch
        """
        if paths is not None:
            self.paths = np.genfromtxt(paths, dtype=str)
        self.transform = transform
        self.patch_size = patch_size
        self.stochastic = stochastic
        self.spiral = spiral

    def __len__(self):
        return int(1e10) # len(self.paths)

    @staticmethod
    def _spiral(n):
        """ 
        generate a spiral index array of side length 'n'
        there must be a better way to do this: any suggestions? 
        """
        a = np.arange(n*n)
        b = a.reshape((n,n))
        m = None
        for i in range(n, 0, -2):
            m = np.r_[m, b[0, :], b[1:, -1], b[-1, :-1][::-1], b[1:-1, 0][::-1]]
            b = b[1:-1, 1:-1]
        a[list(m[1:])] = list(a)
        a = abs(a - n*n + 1)
        return a.reshape((n,n))

    def spiralise(self, galaxy):
        """ 
        Change ViT patch ordering to a 'spiral order'. See Fig 8 in
        https://arxiv.org/pdf/2401.08541.pdf for an illustration.

        Alternate function available here:
        https://www.procook.co.uk/product/procook-spiralizer-black-and-stainless-steel
        """
        # Generate a spiralised matrix and then flatten it to the same shape as 'galaxy'
        indices = einops.rearrange(
            self._spiral(int(np.sqrt(len(galaxy)))),
            'h w -> (h w)',
        )
        assert len(indices) == len(galaxy), "tokenised galaxy must have a square rootable length!"
        spiraled = [ii for _, ii in sorted(zip(indices, galaxy))]
        return torch.stack(spiraled) if isinstance(spiraled[0], torch.Tensor) else np.stack(spiraled)

    def antispiralise(self, galaxy):
        """ 
        Change ViT patch ordering from spiral to raster order. See 'spiralise'.
        """
        # Generate a spiralised matrix and then flatten it to the same shape as 'galaxy'
        indices = einops.rearrange(
            self._spiral(int(np.sqrt(len(galaxy)))),
            'h w -> (h w)',
        )
        assert len(indices) == len(galaxy), "tokenised galaxy must have a square rootable length!"
        antispiraled = [galaxy[ii] for ii in indices]
        return torch.stack(antispiraled) if isinstance(antispiraled[0], torch.Tensor) else np.stack(antispiraled)

    def process_galaxy(self, raw_galaxy):
        patch_galaxy = einops.rearrange(
            raw_galaxy,
            'c (h p1) (w p2) -> (h w) (p1 p2 c)', 
            p1=self.patch_size, p2=self.patch_size
        )

        if self.transform:
            patch_galaxy = self.transform(patch_galaxy)
        if self.spiral:
            patch_galaxy = self.spiralise(patch_galaxy)

        return patch_galaxy

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.stochastic == True:
            idx = np.random.randint(len(self.paths))

        while True:
            try:
                raw_galaxy = io.read_image(str(self.paths[idx]))
                break
            except Exception as err:
                idx = np.random.randint(len(self.paths))

        patch_galaxy = self.process_galaxy(raw_galaxy)
        return {"X": patch_galaxy[:-1], "Y": patch_galaxy[1:]}


class AstroCLIPDataset(Dataset):
    def __init__(self, paths=None, transform=None, norm_per_patch=True, stochastic=True, spiral=False, patch_size=16):
        """
        Arguments:
            paths: file with all the parquet paths.
            transform (callable, optional): Optional transform to be applied on a sample.
            patch_size: size of ViTesque patch.
        """
        self.df = pd.read_parquet("/raid/data/astroclip/data/train-00000-of-00138-de54d6200ce4d5fa.parquet")
        self.transform = transform
        self.patch_size = patch_size
        self.stochastic = stochastic
        self.norm_per_patch = norm_per_patch
        self.spiral = spiral

    def __len__(self):
        return int(1e10) # len(self.paths)

    @staticmethod
    def _spiral(n):
        """ 
        generate a spiral index array of side length 'n'
        there must be a better way to do this: any suggestions? 
        """
        a = np.arange(n*n)
        b = a.reshape((n,n))
        m = None
        for i in range(n, 0, -2):
            m = np.r_[m, b[0, :], b[1:, -1], b[-1, :-1][::-1], b[1:-1, 0][::-1]]
            b = b[1:-1, 1:-1]
        a[list(m[1:])] = list(a)
        a = abs(a - n*n + 1)
        return a.reshape((n,n))

    def spiralise(self, galaxy):
        """ 
        Change ViT patch ordering to a 'spiral order'. See Fig 8 in
        https://arxiv.org/pdf/2401.08541.pdf for an illustration.

        Alternate function available here:
        https://www.procook.co.uk/product/procook-spiralizer-black-and-stainless-steel
        """
        # Generate a spiralised matrix and then flatten it to the same shape as 'galaxy'
        indices = einops.rearrange(
            self._spiral(int(np.sqrt(len(galaxy)))),
            'h w -> (h w)',
        )
        assert len(indices) == len(galaxy), "tokenised galaxy must have a square rootable length!"
        spiraled = [ii for _, ii in sorted(zip(indices, galaxy))]
        return torch.stack(spiraled) if isinstance(spiraled[0], torch.Tensor) else np.stack(spiraled)

    def antispiralise(self, galaxy):
        """ 
        Change ViT patch ordering from spiral to raster order. See 'spiralise'.
        """
        # Generate a spiralised matrix and then flatten it to the same shape as 'galaxy'
        indices = einops.rearrange(
            self._spiral(int(np.sqrt(len(galaxy)))),
            'h w -> (h w)',
        )
        assert len(indices) == len(galaxy), "tokenised galaxy must have a square rootable length!"
        antispiraled = [galaxy[ii] for ii in indices]
        return torch.stack(antispiraled) if isinstance(antispiraled[0], torch.Tensor) else np.stack(antispiraled)

    def process_galaxy(self, raw_galaxy):
        patch_galaxy = einops.rearrange(
            raw_galaxy,
            '(h p1) (w p2) c -> (h w) (p1 p2 c)', 
            p1=self.patch_size, p2=self.patch_size
        )

        if self.transform:
            patch_galaxy = self.transform(patch_galaxy)
        if self.spiral:
            patch_galaxy = self.spiralise(patch_galaxy)

        return patch_galaxy

    def process_spectrum(self, raw_spectrum):
        sp = 3*self.patch_size**2 # spectrum patch size
        t = raw_spectrum.shape[0] # spectrum length
        padded_spectrum = np.zeros(sp*(t//sp + 1))
        padded_spectrum[:len(raw_spectrum)] = raw_spectrum
        patch_spectrum = einops.rearrange(
            padded_spectrum,
            '(t p) -> t p', 
            p=3*self.patch_size**2
        )

        if self.transform:
            patch_spectrum = self.transform(patch_spectrum)

        return patch_spectrum

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.stochastic == True:
            idx = np.random.randint(len(self.df))

        # NOTE this will not be needed for the AstroPile data input.
        # this is a dirty hack so that we can just get things started.
        spectrum = np.array(self.df["spectrum"][idx], dtype=np.float32)
        prelim_image = np.stack(self.df["image"][idx])
        h, w = prelim_image.shape
        # make 152 pixel galaxy 144 pixels so we can patch at 16 pix level
        galaxy = np.stack(prelim_image.ravel()).reshape(h, w, 3)[4:-4, 4:-4, :]
        # /end dirty hack

        patch_galaxy = self.process_galaxy(galaxy)
        patch_spectrum = self.process_spectrum(spectrum)

        patch_series = np.concatenate((patch_galaxy, patch_spectrum), axis=0)
        if self.norm_per_patch: 
            token_sums = patch_series.sum(axis=1, keepdims=True)
            patch_series = patch_series/token_sums
        return {"X": patch_series[:-1], "Y": patch_series[1:]}

if __name__ == "__main__":
    astroclip = iter(AstroCLIPDataset())
    gal = next(astroclip)
