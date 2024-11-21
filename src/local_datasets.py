"""
A place to store our pytorch datasets.
"""
import einops
import numpy as np
import torch
import os
import sys
from torch.utils.data import Dataset
from torchvision import io
from astropy.io import fits

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
        # set to a big number if stochastic as a fudge for epochism in pytorch
        return int(1e10) if self.stochastic else len(self.paths)

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
            # be careful in this loop -- it fails quietly if there is an error!!!
            try:
                # get extension to filename and read via FITS or JPG
                _, ext = os.path.splitext(self.paths[idx])
                if ext == ".jpg" or ext == ".jpeg":
                    raw_galaxy = io.read_image(str(self.paths[idx])).to(torch.bfloat16)
                    break
                elif ext == ".fits" or ext == ".FITS":
                    with fits.open(self.paths[idx]) as hdul:
                        raw_galaxy = hdul[0].data.astype(np.float32)  # Assuming the image data is in the first HDU
                        # we need to convert to float32 as FITS has bigendian issues (https://stackoverflow.com/questions/59247385/why-does-torch-from-numpy-require-a-different-byte-ordering-while-matplotlib-doe)
                    # @gosia I have got this working with a single channel, ofc we can can extend to multichannel too by stacking more FITS files along the zeroth axis.
                    raw_galaxy = torch.tensor(raw_galaxy[np.newaxis]).to(torch.bfloat16)
                    break
                else:
                    raise NotImplementedError(f"File must be FITS or JPEG, it is instead {ext}.")
            except Exception as err:
                print(err)
                if self.stochastic == True:
                    idx = np.random.randint(len(self.paths))
                else:
                    sys.exit(1)

        patch_galaxy = self.process_galaxy(raw_galaxy)
        return {"X": patch_galaxy[:-1], "Y": patch_galaxy[1:], "idx": idx}
