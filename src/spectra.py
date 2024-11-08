import astropy.io.fits as fits
from astropy.table import Table
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import io
import einops
import torch.nn.functional as F

class GalaxySpectraImageDataset(Dataset):
    def __init__(self, fits_dir="/home/msmith/iac18_shared_folder/desi_edr_low_z/spectra/", jpeg_dir="/home/msmith/iac18_shared_folder/desi_edr_low_z/jpeg/", transform=None, stochastic=True, spiral=False, patch_size=16, target_ids=None):
        """
        Arguments:
            fits_dir: Directory where FITS files (spectra) are stored.
            jpeg_dir: Directory where JPEG images (galaxies) are stored.
            transform: Optional transform to be applied on a sample.
            stochastic: If True, will pick random indices in __getitem__.
            spiral: Whether to rearrange patches in a spiral order.
            patch_size: Size of ViT patch for images.
        """
        self.fits_dir = fits_dir
        self.jpeg_dir = jpeg_dir
        self.transform = transform
        self.patch_size = patch_size
        self.stochastic = stochastic
        self.spiral = spiral
        if target_ids is None:
            self.target_ids = self._get_target_ids()  # Extract TARGETID from filenames
        else:
            self.target_ids = target_ids

    def _spiral(self, n):
        """Generate a spiral order for an `n x n` matrix."""
        matrix = np.arange(n * n).reshape(n, n)
        spiral_order = []
        while matrix.size:
            spiral_order += matrix[0].tolist()
            matrix = matrix[1:].T[::-1]
        return np.array(spiral_order).reshape(n, n)

    def spiralise(self, galaxy):
        """Rearrange patches in a spiral order."""
        indices = einops.rearrange(
            self._spiral(int(np.sqrt(len(galaxy)))),
            'h w -> (h w)',
        )
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


    def _get_target_ids(self):
        # Extract TARGETID (filename without extension) from both FITS and JPEG directories
        fits_files = [os.path.splitext(f)[0] for f in os.listdir(self.fits_dir) if f.endswith('.fits')]
        jpeg_files = [os.path.splitext(f)[0] for f in os.listdir(self.jpeg_dir) if f.endswith('.jpg')]
        # Ensure both FITS and JPEG files exist for the same TARGETID
        return list(set(fits_files) & set(jpeg_files))

    def __len__(self):
        return int(1e9)#len(self.target_ids)

    def _load_fits(self, target_id):
        """Load the FITS file corresponding to the TARGETID."""
        fits_path = os.path.join(self.fits_dir, f"{target_id}.fits")
        
        try:
               # Read the FITS file using Astropy's Table
                data = Table.read(fits_path, format='fits')
        
                # Extract the 'wave' and 'flux' columns
                #wave = data['wave']
                flux = data['flux']
        
                # Stack wave and flux to form a single tensor
                #spectrum_data = np.vstack([wave, flux]).astype(np.float32)  # Shape: (2, N)
                spectrum_data = flux.astype(np.float32)
        
                return torch.tensor(spectrum_data) 
    
        except Exception as e:
            print(f"Error loading FITS file {fits_path}: {e}")
            return None

    def _load_jpeg(self, target_id):
        """Load the JPEG image corresponding to the TARGETID."""
        jpeg_path = os.path.join(self.jpeg_dir, f"{target_id}.jpg")
        image = io.read_image(jpeg_path).to(torch.float32)  # Read as float32 for compatibility
        return image

    def process_image(self, raw_image, pad=False):
        # Pad the image to make it divisible by the patch size
        c, h, w = raw_image.shape

        # Apply padding to the image or crop image to patch size int divisor
        if pad:
            pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
            pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
            padded_image = F.pad(raw_image, (0, pad_w, 0, pad_h))
        else:
            antipad_h = (h // self.patch_size) * self.patch_size
            antipad_w = (w // self.patch_size) * self.patch_size
            padded_image = raw_image[:, :antipad_h, :antipad_w]

        # Now rearrange into patches
        patch_image = einops.rearrange(
            padded_image,
            'c (h p1) (w p2) -> (h w) (p1 p2 c)',
            p1=self.patch_size, p2=self.patch_size
        )

        if self.transform:
            patch_image = self.transform(patch_image)

        if self.spiral:
            patch_image = self.spiralise(patch_image)

        return patch_image

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.stochastic:
            idx = np.random.randint(len(self.target_ids))

        target_id = self.target_ids[idx]

        # Load spectrum
        raw_spectrum = self._load_fits(target_id)
        if raw_spectrum is None:
            print(f"Skipping {target_id}: Failed to load spectrum.")
            return self.__getitem__(np.random.randint(len(self.target_ids)))  # Retry with another index

        # Process spectrum
        # Apply padding to the spectrum
        w = raw_spectrum.shape[0]
        pad_w = (768 - w % 768) % 768
        padded_spectrum = F.pad(raw_spectrum, (0, pad_w))

        # Now rearrange into patches
        patch_spectrum = einops.rearrange(
            padded_spectrum,
            '(w p) -> (w) (p)',
            p=768,
        )

        # Load image
        try:
            raw_image = self._load_jpeg(target_id)
        except FileNotFoundError:
            print(f"Skipping {target_id}: JPEG image not found.")
            return self.__getitem__(np.random.randint(len(self.target_ids)))  # Retry with another index

        # Process image (e.g., split into patches)
        patch_image = self.process_image(raw_image)

        # Print for debugging (prints every 100th sample)
        #if idx % 100 == 0:
        #    print(f"Loaded target_id: {target_id} | Spectrum shape: {spectrum.shape} | Image shape: {raw_image.shape}")

        patch_both = torch.cat((patch_image, patch_spectrum), dim=0)
        # TODO this is a bit of a mess for now. refactor so that it is easy to add more modalities
        return {
            "spectra": patch_spectrum,   # Spectral data
            "images": patch_image,    # Image patches
        }

        #patch_both = torch.cat((patch_image, patch_spectrum), dim=0)
        #return {
        #    "X": patch_both[:-1],
        #    "Y": patch_both[1:],
        #} 
