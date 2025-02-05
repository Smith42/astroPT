"""
A place to store our pytorch datasets.
"""
import einops
import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
from torch.utils.data import Dataset
from torchvision import io
from astropy.io import fits
from astropy.table import Table

class GalaxyImageDataset(Dataset):

    def __init__(self, paths=None, paths_spect=None, transform=None, stochastic=True, spiral=False, patch_size=16):
        """
        Arguments:
            paths: file with all the galaxy paths. Paths can be None if streaming from HF.
            paths_spect: file with all the spectra paths in same order as paths.
            transform (callable, optional): Optional transform to be applied on a sample.
            spiral: spiral form instead of raster form
            patch_size: size of ViT patch
        """
        if paths is not None:
            self.paths = np.genfromtxt(paths, delimiter=",", dtype=str)
        if paths_spect is not None:
            self.paths_spect = np.genfromtxt(paths_spect, delimiter=",", dtype=str)
        else:
            self.paths_spect = None
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
            p1=self.patch_size["images"], p2=self.patch_size["images"]
        )

        if "galaxy" in self.transform:
            patch_galaxy = self.transform["galaxy"](patch_galaxy)
        if self.spiral:
            patch_galaxy = self.spiralise(patch_galaxy)

        return patch_galaxy

    def process_spectra(self, raw_spectra):
        # Apply padding to the spectrum
        w = raw_spectra.shape[0]
        window = self.patch_size["spectra"]
        pad_w = (window - w % window) % window
        padded_spectra = F.pad(raw_spectra, (0, pad_w))

        # Now rearrange into patches
        patch_spectra = einops.rearrange(
            padded_spectra,
            '(w p) -> (w) (p)',
            p=window,
        )

        if "spectra" in self.transform:
            patch_spectra = self.transform["spectra"](patch_spectra)

        return patch_spectra

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.stochastic == True:
            idx = np.random.randint(len(self.paths))

        while True:
            # be careful in this loop -- it fails quietly if there is an error!!!
            try:
                # get extension to filename and read via FITS or JPG
                _, ext = os.path.splitext(self.paths[idx]) if len(self.paths.shape) == 1 else os.path.splitext(self.paths[idx][0])
                if ext == ".jpg" or ext == ".jpeg":
                    raw_galaxy = io.read_image(str(self.paths[idx])).to(torch.bfloat16)
                    break
                elif ext == ".fits" or ext == ".FITS":
                    if len(self.paths.shape) == 1:
                        # case with 1 FITS file
                        with fits.open(self.paths[idx]) as hdul:
                            raw_galaxy = hdul[0].data.astype(np.float32)  # Assuming the image data is in the first HDU
                            # we need to convert to float32 as FITS has bigendian issues (https://stackoverflow.com/questions/59247385/why-does-torch-from-numpy-require-a-different-byte-ordering-while-matplotlib-doe)
                        raw_galaxy = torch.tensor(raw_galaxy[np.newaxis]).to(torch.bfloat16)
                        if self.paths_spect is None:
                            break
                    else:
                        # case with N FITS files
                        raw_galaxy = []
                        for path in self.paths[idx]:
                            with fits.open(path) as hdul:
                                raw_galaxy.append(hdul[0].data.astype(np.float32))  # Assuming the image data is in the first HDU
                                # we need to convert to float32 as FITS has bigendian issues
                        raw_galaxy = torch.tensor(np.stack(raw_galaxy)).to(torch.bfloat16)
                        if self.paths_spect is None:
                            break
                    # Fetch spectrum if we have one
                    with fits.open(self.paths_spect[idx]) as hdul:
                        raw_spectra = hdul[1].data["Flux"].astype(np.float32)
                    raw_spectra = torch.tensor(raw_spectra).to(torch.bfloat16)
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
        patch_spectra = self.process_spectra(raw_spectra)
        return {
            "X": {"images": patch_galaxy, "spectra": patch_spectra[:-1]},
            "Y": {"images": patch_galaxy[1:], "spectra": patch_spectra},
            "idx": idx,
        }
