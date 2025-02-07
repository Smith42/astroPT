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
import matplotlib.pyplot as plt

class GalaxyImageDataset(Dataset):

    def __init__(
            self, paths=None, transform=None, stochastic=True, spiral=False, 
            patch_size=16, surprisal_patching=False
        ):
        """
        Arguments:
            paths: file with all the galaxy paths. Paths can be None if streaming from HF.
            transform (callable, optional): Optional transform to be applied on a sample.
            spiral: spiral form instead of raster form
            patch_size: size of ViT patch
        """
        if paths is not None:
            self.paths = np.genfromtxt(paths, delimiter=",", dtype=str)
        self.transform = transform
        self.patch_size = patch_size
        self.stochastic = stochastic
        self.spiral = spiral
        self.surprisal_patching = surprisal_patching
        self.image_size = 512

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

    def surprisal_patch(self, galaxy):
        """
        Implement a 'surprisal patching' on a stream of data.
        """
        def _rolling_z(values, new_value):
            mean = np.mean(values)
            std = np.std(values)
            return (new_value - mean) / std if std > 0 else 0

        galaxy = einops.rearrange(
            galaxy, 
            '(h w) (p1 p2 c) -> (h w p1) (p2 c)', 
            p1=self.patch_size, p2=self.patch_size, 
            h=self.image_size//self.patch_size, 
            w=self.image_size//self.patch_size
        )

        # Settings
        window = self.patch_size * 4
        rel_threshold = 1.0
        patch_starts = [0]  # Always start first patch at 0
        prev_z = 0
    
        z_scores = []
        rel_scores = []
        # For each value, is it surprising compared to previous window?
        for i in range(2, len(galaxy)):
            window_size = min(i, window)
            recent = galaxy[i-window_size:i]
            z_score = np.mean((galaxy[i] - np.mean(recent)) / (np.std(recent) + 1e-5))
            if np.abs(z_score - prev_z) > rel_threshold:
                patch_starts.append(i)
            
            z_scores.append(z_score)
            rel_scores.append(z_score - prev_z)
            prev_z = z_score
    
        #return patch_starts
        f, axs = plt.subplots(2, 1, sharex=True, figsize=(32, 4))
        axs[0].imshow(galaxy[:1600].T, aspect='auto')
        for start in patch_starts:
            axs[0].axvline(x=start, color='r', alpha=0.3)
            axs[1].axvline(x=start, color='r', alpha=0.3)
        axs[0].set_xlim(0, 1600)
        axs[1].set_ylim(-5, 5)
        axs[1].plot(z_scores)
        axs[1].plot(rel_scores)
        plt.savefig("/beegfs/general/mjsmith/tmp/surprisal_patching.png", dpi=300)
        print(patch_starts)
        exit(0)

    def process_galaxy(self, raw_galaxy):
        patch_galaxy = einops.rearrange(
            raw_galaxy,
            'c (h p1) (w p2) -> (h w) (p1 p2 c)', 
            p1=self.patch_size, p2=self.patch_size
        )

        if self.spiral:
            patch_galaxy = self.spiralise(patch_galaxy)
        if self.surprisal_patching:
            patch_galaxy = self.surprisal_patch(patch_galaxy)
        if self.transform:
            patch_galaxy = self.transform(patch_galaxy)

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
                        break
                    else:
                        # case with N FITS files
                        raw_galaxy = []
                        for path in self.paths[idx]:
                            with fits.open(path) as hdul:
                                raw_galaxy.append(hdul[0].data.astype(np.float32))  # Assuming the image data is in the first HDU
                                # we need to convert to float32 as FITS has bigendian issues
                        raw_galaxy = torch.tensor(np.stack(raw_galaxy)).to(torch.bfloat16)
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
