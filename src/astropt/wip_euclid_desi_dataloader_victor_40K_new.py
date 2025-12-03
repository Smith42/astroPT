"""
This file contains the Euclid images and DESI spectra Dataloader
for traingin AstroPT trnasformer model
"""

import logging
import numpy as np
from numpy.typing import NDArray
import os
from typing import Optional, Dict, Any, List, Tuple

from astropy.io import fits
from astropy.table import Table

import einops
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class EuclidDESIDataset(Dataset): 
    
    def __init__(
        self, 
        metadata_path: str,
        vis_folder: str,
        nisp_folder: Dict[str, str],     
        spectra_folder: str,
        modality_registry: Any,
        spiral: bool = False,           
        stochastic = True,      
        transform: Dict = {},
    ):
        
        """
        Dataset to loading Euclid Images and DESI spectra
        
        Args:
            metadata_path: Path to crossmatch database .fits file
            vis_folder: Path to Euclid VIS images
            nisp_folders: Path dictionary to NISP filters images {'H': path, 'J': path, 'Y': path}.
            spectra_folder: Path to DESI spectra folder
            modality_registry: ModalityRegistry object to load the modality configuration
            spiral: For applying the spiral tokenization
            stochastic: For applyting an stochastic training
            transform: Transform method for each modality
        """
        
        # 1. Loading metadata
        self.meta = Table.read(metadata_path)
        
        # 2. .fits files paths
        self.vis_folder = vis_folder
        self.nisp_folder = nisp_folder
        self.spectra_folder = spectra_folder
        
        # 3. Configuration
        self.modality_registry = modality_registry
        self.spiral = spiral
        self.stochastic = stochastic
        self.transform = transform
        
        
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.meta)
    
    
    @staticmethod
    def _find_matching_indices(
        targets: List[int] | NDArray, 
        reference_ids: List[int] | NDArray
    ) -> NDArray[np.int64]:
        
        """Returns indices of `targets` in `reference_ids`."""
        id_to_index = {tid: i for i, tid in enumerate(reference_ids)}
        
        return np.array([id_to_index[tid] for tid in targets])

        
    @staticmethod
    def _spiral_sorting(n: int) -> NDArray[np.int64]:
        """
        Generate a spiral index array of side length 'n' (center=0).
        
        Args:
            n: The side length (width/height) of the patch grid (e.g., 14 for 224px/16).
            
        Returns:
            A 2D NumPy array of shape (n, n) containing the spiral indices.
        """
        # Creating a n x n matriz with secuential index. Example: n=3
        # (0, 1, 2)
        # (3, 4, 5)
        # (6, 7 ,8)
        layout = np.arange(n * n).reshape(n, n)
        
        # List to store extracted index
        spiral_indices = []
        
        while layout.size > 0:
            # 1. Extracting Upper row (0, 1, 2)
            spiral_indices.append(layout[0])
            layout = layout[1:] 
            if layout.size == 0: break
            
            # 2. Extracting Right column (5, 8)
            spiral_indices.append(layout[:, -1])
            layout = layout[:, :-1]
            if layout.size == 0: break
            
            # 3. Extracting Lower row (reversed) (7, 6)
            spiral_indices.append(layout[-1][::-1])
            layout = layout[:-1]
            if layout.size == 0: break
            
            # 4. Extracting Left column (reversed) (3)
            spiral_indices.append(layout[:, 0][::-1])
            layout = layout[:, 1:]

            # 5. Continuing with the inner matrix

        # Cocatenate index [(0, 1, 2),(5, 8),(7, 6),(3),(4)]
        # to have [0,1,2,5,8,7,6,3,4] 1D array
        spiral_order = np.concatenate(spiral_indices)
        
        # Creating and 'empty' matrix. 
        result = np.empty(n * n, dtype=int)
        
        # Changes from [0,1,2,5,8,7,6,3,4] 1D array
        # to [0,1,2,7,8,3,6,5,4] 
        result[spiral_order] = np.arange(n * n)
        
        # Substracting to have the 0 in the center
        # and the corrected 1D array [8,7,6,1,0,5,2,3,4]
        # with the zero in the middle
        result = (n * n - 1) - result
        
        # Returns the vector order
        return result
    
    
    def spiralise_image(self, 
        image: torch.Tensor | NDArray
    ) -> torch.Tensor | NDArray:
        """
        Reorder a sequence of image patches from 'raster order' (row-by-row)
        to 'spiral order' (center-to-outskirts). See Fig 8 in
        https://arxiv.org/pdf/2401.08541.pdf for an illustration.

        This allows the Transformer to focus on the central galaxy first.
        Requires the input length to be a perfect square (e.g., 196 patches for 14x14).

        Args:
            galaxy: Input tensor of shape (Num_Patches, Channels).
                    Can be a PyTorch Tensor or NumPy Array.

        Returns:
            The reordered tensor with the same shape, starting from the center patch.
        """
        
        n_patches = len(image)
        side_len = int(np.sqrt(n_patches))
        
        # Verify it's a perfect square
        assert side_len**2 == n_patches, (
            f"Galaxy patch count ({n_patches}) must be a perfect square!"
        )
        
        # Generate a spiralised vector positions
        spiral_indices = self._spiral_sorting(side_len)

        # Matches spiral index with their corresponding patch
        # Then they are sorted by the spiral index
        sorted_pairs = sorted(zip(spiral_indices, image), key=lambda pair: pair[0])
        
        # Extracting the patches in the correct order
        spiraled_patches = [patch for _, patch in sorted_pairs]
        
        # Returning the correct format
        if isinstance(image, torch.Tensor):
            return torch.stack(spiraled_patches)
        else:
            return np.stack(spiraled_patches)


    def antispiralise_image(self, 
        image: torch.Tensor | NDArray
    ) -> torch.Tensor | NDArray:
        """
        Reorder a sequence of image patches from 'spiral order' (center-to-outskirts)
        back to 'raster order' (row-by-row). It undoes what the spiralise function does.

        Used to reconstruct the 2D image from the model's sequential output.

        Args:
            galaxy: Input tensor/array of shape (Num_Patches, Channels) 
                    ordered in spiral sequence.

        Returns:
            The reordered tensor/array in raster order (ready to be reshaped to HxW).
        """
        
        n_patches = len(image)
        side_len = int(np.sqrt(n_patches))
        
        # Verify it's a perfect square
        assert side_len**2 == n_patches, (
            f"Galaxy patch count ({n_patches}) must be a perfect square!"
        )
        
        # Generate a spiralised vector positions
        spiral_indices = self._spiral_sorting(side_len)
        
        # Gather indexing operation
        return image[spiral_indices]
        
    def process_image(self, 
        raw_image: torch.Tensor | NDArray
    ) -> torch.Tensor | NDArray:
        """
        Convert a raw 2D image into a sequence of flattened patches.
        
        Performs three steps:
        1. 'Patchify': Cuts the image into a grid of small squares and flattens them.
           (e.g., 224x224 image -> 196 patches of size 16x16).
        2. Transform: Applies normalization (if configured).
        3. Spiralise: Reorders the patches from center to outskirts (if configured).

        Args:
            raw_image: Input image tensor of shape (Channels, Height, Width).

        Returns:
            Sequence of patches of shape (Num_Patches, Flattened_Patch_Size).
        """
        # 1. Obtaining patch size from configuration (ej. 16)
        cfg = self.modality_registry.get_config("images")
        patch_size = cfg.patch_size
        
        # 2. EINOP tensor transformation
        # Input: (Chanels, Height, Width) -> Output: (N_Patchs, Vector_Patch)
        # h,w = total image pixel per side / patch size
        # p1 y p2 are patch size (16, 16)
        patch_image = einops.rearrange(
            raw_image,
            "c (h p1) (w p2) -> (h w) (p1 p2 c)", # (h p1) = h * p1
            p1=patch_size,
            p2=patch_size,
        )

        # 3. Applied transformations to images
        if "images" in self.transform:
            patch_image = self.transform["images"](patch_image)
            
        # 4. Spiral order
        if self.spiral:
            patch_image = self.spiralise(patch_image)

        return patch_image
    
    def process_spectra(self, 
        raw_spectra: torch.Tensor, 
        wavelength: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process raw 1D spectra into a sequence of flattened patches (tokens).

        The function performs three steps:
        1. Padding: Extends the spectrum length so it is perfectly divisible by 'patch_size'.
        2. Patching: Reshapes the 1D array into (Num_Patches, Patch_Size).
        3. Transform: Applies normalization (if configured).

        Args:
            raw_spectra: 1D Tensor containing flux values.
            wavelength: 1D Tensor containing wavelength values.

        Returns:
            A tuple containing:
            - patch_spectra: Tensor of shape (Num_Patches, Patch_Size).
            - patch_wl: Tensor of shape (Num_Patches, Patch_Size) with corresponding wavelengths.
        """
        # 1. Get configuration
        cfg = self.modality_registry.get_config("spectra")
        patch_size = cfg.patch_size
        
        # 2. Calculate necessary padding
        # We need the total length to be a multiple of patch_size
        seq_len = raw_spectra.shape[0]
        remainder = seq_len % patch_size
        pad_len = (patch_size - remainder) % patch_size
        
        # 3. Apply padding (only if needed)
        if pad_len > 0:
            # F.pad tuple format for 1D: (padding_left, padding_right)
            raw_spectra = F.pad(raw_spectra, (0, pad_len))
            wavelength = F.pad(wavelength, (0, pad_len))

        # 4. Reshape into patches (Tokenization)
        # .view(-1, patch_size) automatically calculates the number of patches
        # Shape: (Total_Len) -> (Num_Patches, Patch_Size)
        patch_spectra = raw_spectra.view(-1, patch_size)
        patch_wl = wavelength.view(-1, patch_size)

        # 5. Apply transformations (e.g., Normalization)
        if "spectra" in self.transform:
            patch_spectra = self.transform["spectra"](patch_spectra)

        return patch_spectra, patch_wl

    @staticmethod
    def process_modes(
        batch_data: Dict[str, Any], 
        modality_registry: Any, 
        device: torch.device, 
        shuf: bool = False
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Prepare the batch for training: move to GPU and split into Inputs (X) and Targets (Y).

        This function applies the 'Autoregressive Shift':
        - X (Input): Represents sequence [0, 1, ..., N-1]
        - Y (Target): Represents sequence [1, 2, ..., N]
        
        For multimodal sequences (e.g., Image -> Spectrum), it handles the chaining:
        - Y of the FIRST modality is shifted forward (skips 1st token).
        - X of the LAST modality is shifted backward (drops last token).
        - Middle modalities connect perfectly (End of Mod A predicts Start of Mod B).

        Args:
            batch_data: Dictionary with raw tensors from the Dataloader.
            modality_registry: Registry to determine the order of modalities.
            device: The target device (CPU/CUDA/MPS).
            shuf: Whether to shuffle the order of modalities (Data Augmentation).

        Returns:
            A dictionary containing:
            - "X": Inputs dictionary mapping {modality: tensor}.
            - "Y": Targets dictionary mapping {modality: tensor}.
        """
        # 1. Determining modality order (e.g. ['images', 'spectra'])
        modes = modality_registry.generate_sequence(shuf=shuf)

        # 2. Moving all tensors tu GPU (Device) at same time
        # Using dict comprehension
        data_on_device = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
            for k, v in batch_data.items()
        }

        X = {}
        Y = {}
        
        # 3. Creating the X and Y sequences with the correct shift
        num_modes = len(modes)
        
        for i, mode in enumerate(modes):
            
            # Cloning originald ata to avoid modifications
            data = data_on_device[mode]
            pos = data_on_device[f"{mode}_positions"]
            
            # X logic (input) 
            # Removing last item if it is the last mode, no predicts nothing
            if i == num_modes - 1:
                X[mode] = data[:, :-1]
                X[f"{mode}_positions"] = pos[:, :-1]
            else:
                # If not, we using it full
                X[mode] = data
                X[f"{mode}_positions"] = pos

            # Y logic (target)
            # Removing first item if it is the first mode, nothing predict it
            if i == 0:
                Y[mode] = data[:, 1:]
            else:
                # If not, using it full
                Y[mode] = data

        return {"X": X, "Y": Y}


    def _load_image(self, 
        path: str
    ) -> Optional[NDArray]:
        """
        Safely load data from a FITS file.

        This method handles file existence checks and reading errors gracefully,
        returning None instead of raising an exception to avoid interrupting
        the data loading process.

        Args:
            path: Absolute path to the .fits file.

        Returns:
            A NumPy array containing the data from the primary HDU, 
            or None if the file is missing or corrupt.
        """
        # 1. Check existence to fail fast
        if not os.path.exists(path):
            logging.warning(f"Missing file: {path}")
            return None

        try:
            # 2. Open with context manager for safety
            # memmap=False ensures data is read into RAM and file handle is closed immediately
            with fits.open(path, memmap=False) as hdul:
                data = hdul[0].data
                
            # 3. Ensure we actually got data (some FITS can have empty primary HDUs)
            if data is None:
                logging.warning(f"Empty primary HDU in file: {path}")
                return None
                
            return data

        except Exception as e:
            logging.error(f"Failed to read {path}: {e}")
            return None


    def _load_spectrum(self, 
        path: str
    ) -> Optional[Dict[str, NDArray[np.float32]]]:
        """
        Load a processed 1D spectrum from a FITS file.

        Expects a FITS binary table in HDU 1 with columns 'WAVELENGTH' and 'FLUX'.

        Args:
            path: Absolute path to the .fits file.

        Returns:
            A dictionary with 'wavelength' and 'flux' arrays (float32),
            or None if loading fails.
        """
        if not os.path.exists(path):
            logging.warning(f"Spectrum file missing: {path}")
            return None

        try:
            with fits.open(path, memmap=False) as hdul:
                # In your processed files, data is in HDU 1 (Table)
                data = hdul[1].data 
                wave = data['WAVELENGTH'].astype(np.float32)
                flux = data['FLUX'].astype(np.float32)

            return {
                "wavelength": wave,
                "flux": flux
            }
        except Exception as e:
            logging.error(f"Failed to load spectrum {path}: {e}")
            return None
        
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single sample from the dataset by index.

        Loads the galaxy images (VIS + NISP) (if available), the spectrum (if available),
        and associated metadata. Handles missing files gracefully by returning None.

        Args:
            idx (int): Index of the sample in the metadata table.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing:
                - 'images': Tensor (4, H, W) [VIS, H, J, Y]
                - 'images_positions': Tensor (Spiral order indices)
                - 'spectra': Tensor (Tokens) (Optional)
                - 'spectra_positions': Tensor (Wavelengths) (Optional)
                - 'targetid': int (Unique Object ID)
                - 'idx': int (Original index)
            
            Returns None if there is neither Images nor Spectra to train with.
        """
        # Ensure index is a standard Python type for Astropy compatibility
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Retrieve metadata row and object target ID
        entry = self.meta[idx]
        targetid = entry['TARGETID']
        
        # Base returning dictionary
        sample = {
            "idx": idx,
            "targetid": int(targetid)
        }
        

        #--- 1. LOAD VIS & NISP IMAGES (OPTIONAL) ---#
        # Just loading images if there is a VIS folder configured
        if self.vis_folder is not None:
            
            # Determining possible VIS file names based on catalog version
            if 'VIS_cutout' in entry.colnames: 
                vis_filename = os.path.basename(str(entry['VIS_cutout']))
            elif 'name' in entry.colnames: 
                vis_filename = os.path.basename(str(entry['name']))
            else: 
                vis_filename = None
                logging.error(f"Skipping index {idx}: Catalog missing 'VIS_cutout' or 'name' column.")
    
            if vis_filename:
                # Extracting the name
                vis_image_path = os.path.join(self.vis_folder, vis_filename)
                
                # Loading the image
                vis_image = self._load_image(vis_image_path)

                # Managing possible errors
                if vis_image is not None:
                    
                    # --- NISP IMAGES LOADING ---
                    # Store reference shape from VIS image to ensure consistency
                    ref_shape = vis_image.shape
                    nisp_images_list = []
                    
                    # Iterate over the expected Near-Infrared bands
                    for band in ['H', 'J', 'Y']:
                        
                        # 1. Resolve NISP filename
                        col_name = f"NIR-{band}_cutout"
                        
                        if col_name in entry.colnames:
                            # 250K catalog: explicit column
                            nisp_filename = os.path.basename(str(entry[col_name]))
                        else:
                            # 40K catalog: infer from VIS name
                            nisp_filename = vis_filename.replace("VIS", f"NIR-{band}")
                        
                        # 2. Resolve Path
                        if isinstance(self.nisp_folder, dict):
                            # Dictionary config: {'H': path_h, ...}
                            # Fallback to 'default' key or empty string if band missing
                            band_folder = self.nisp_folder.get(band, self.nisp_folder.get(f'{band}-band', ''))
                            nisp_path = os.path.join(band_folder, nisp_filename)
                        else:
                            # String config: single folder for all
                            nisp_path = os.path.join(self.nisp_folder, nisp_filename)

                        # 3. Load image using the optimized internal method
                        nisp_img = self._load_image(nisp_path)
                        
                        # 4. Validation: Check if image exists and matches VIS dimensions
                        if nisp_img is None or nisp_img.shape != ref_shape:
                            # If missing/bad, append a black image (zeros) to maintain channel consistency
                            nisp_images_list.append(np.zeros_like(vis_image)) 
                            logging.warning(f"Missing/Bad NISP-{band} for target {targetid}. Filling with zeros.")
                        else:
                            nisp_images_list.append(nisp_img)
                    
                    # --- STACKING IMAGES (OUTSIDE THE LOOP) ---
                    # Combine VIS + 3 NISP bands into a single 4-channel array
                    # Result shape: (4, Height, Width)
                    raw_galaxy = np.stack([vis_image] + nisp_images_list, axis=0)
                    
                    # Convert to Tensor (bfloat16 for memory efficiency on A100)
                    raw_galaxy = torch.from_numpy(raw_galaxy).to(torch.bfloat16)

                    # Process image (Patching + Spiralisation)
                    patch_galaxy = self.process_image(raw_galaxy)
                    
                    # Update return dictionary
                    sample["images"] = patch_galaxy
                    sample["images_positions"] = torch.arange(0, len(patch_galaxy), dtype=torch.long)
                    
                else:
                    logging.warning(f"VIS image not found or corrupt: {vis_image_path} (TargetID: {targetid}). Skipping.")

        #--- 2. LOAD SPECTRA (OPTIONAL) ---#
        # Using spectra as optional for training the model
        if self.spectra_folder is not None:
            
            # 1. Define spectrum filename
            if 'SPEC_PATH' in entry.colnames:
                # Old catalog (40K): Use explicit path from column
                spec_filename = os.path.basename(str(entry['SPEC_PATH']))
            else:
                # New catalog (250K): Construct filename using TARGETID
                spec_filename = f"TARGETID_{targetid}.fits"
            
            spectrum_path = os.path.join(self.spectra_folder, spec_filename)

            # 2. Load spectrum using optimized internal method
            spec_data = self._load_spectrum(spectrum_path)
            
            # 3. Process and add to sample if loaded successfully
            if spec_data is not None:
                raw_flux = torch.from_numpy(spec_data['flux']).to(torch.bfloat16)
                
                # Normalize wavelength (Typical DESI Optical Range scaled to 0-1)
                raw_wave = torch.from_numpy(spec_data['wavelength']).to(torch.bfloat16)
                raw_wave = (raw_wave - 3000.0) / (10000.0 - 3000.0)

                # Apply padding and patching
                patch_spectra, patch_wl = self.process_spectra(raw_flux, raw_wave)
                
                # Safety checks (NaNs)
                if torch.isnan(patch_spectra).any():
                    # Raising error here to debug training data issues early
                    raise ValueError(f"NaNs found in spectrum {targetid}")

                # Add to output dictionary
                sample["spectra"] = patch_spectra
                sample["spectra_positions"] = patch_wl
            
            else:
                # If spectrum loading fails, warn and raise error (as requested)
                logging.warning(f"Spectrum missing: {spectrum_path}")
                raise FileNotFoundError(f"Spectrum missing: {spectrum_path}")


        #--- 3. FINAL VALIDATION ---#
        # If neither images nor spectra were loaded successfully, return None to skip sample
        if "images" not in sample and "spectra" not in sample:
            return None

        return sample


def adjust_dynamic_range(flux, q=100, clip=99.85):
    """Adjust dynamic range using asinh scaling and optional percentile clipping."""
    im = np.arcsinh(flux * q)
    if clip < 100:
        im = np.clip(im, 0, np.percentile(im, clip))
    return im


# new helper to test whether an image is usable
def is_valid_image(im, min_range=1e-6):
    """Return False if im is None, has only NaNs, or has negligible dynamic range."""
    if im is None:
        return False
    if not np.isfinite(im).any():
        return False
    try:
        im_min = np.nanmin(im)
        im_max = np.nanmax(im)
    except Exception:
        return False
    if not np.isfinite(im_min) or not np.isfinite(im_max):
        return False
    return (im_max - im_min) > min_range


def to_uint8(im, clip_below_zero=True):
    """Normalize to [0,255] uint8. If image is constant/invalid, return mid-gray."""
    if im is None:
        return None
    if clip_below_zero:
        im = np.clip(im, 0, None)
    # handle all-NaN or constant arrays gracefully
    if not np.isfinite(im).any():
        # return neutral gray image
        shape = im.shape
        return np.full(shape, 128, dtype=np.uint8)
    im_min = np.nanmin(im)
    im_max = np.nanmax(im)
    if not np.isfinite(im_min) or not np.isfinite(im_max) or im_max <= im_min:
        # constant image -> neutral gray
        shape = im.shape
        return np.full(shape, 128, dtype=np.uint8)
    im = (im - im_min) / (im_max - im_min)
    return (255 * im).astype(np.uint8)


def make_composite_cutout(vis_cutout, nisp_cutouts, vis_q=100, vis_clip=99.85, nisp_q=1, nisp_clip=99.85):
    """
    Create a composite RGB cutout from VIS and NISP images.

    - vis_cutout: 2D array or None
    - nisp_cutouts: dict of band->2D array (e.g. {'Y': arr, 'J': arr, 'H': arr})
    Returns an HxWx3 uint8 image or None if not enough data / shape mismatch.
    """
    logging.debug('Creating composite cutout')

    # require a valid VIS image
    if not is_valid_image(vis_cutout):
        logging.debug('VIS cutout missing or invalid, cannot create composite')
        return None

    # collect available and valid NISP images (filter out empty/constant)
    raw_available = [img for img in (nisp_cutouts or {}).values() if img is not None]
    available = [img for img in raw_available if is_valid_image(img)]
    if len(available) == 0:
        logging.debug('No valid NISP bands available, cannot create composite')
        return None

    # ensure shapes match
    for arr in available:
        if arr.shape != vis_cutout.shape:
            logging.debug(f'vis shape {vis_cutout.shape}, nisp shape {arr.shape} - cannot make composite')
            return None

    # mean of available NISP bands
    nisp_mean = np.mean(np.stack(available, axis=0), axis=0)

    # adjust dynamic ranges
    vis_adj = adjust_dynamic_range(vis_cutout, q=vis_q, clip=vis_clip)
    nisp_adj = adjust_dynamic_range(nisp_mean, q=nisp_q, clip=nisp_clip)
    mean_flux = np.mean([vis_adj, nisp_adj], axis=0)

    # convert to uint8 (to_uint8 now returns neutral gray for constant/invalid arrays)
    vis_uint8 = to_uint8(vis_adj)
    nisp_uint8 = to_uint8(nisp_adj)
    mean_uint8 = to_uint8(mean_flux)

    im = np.stack([nisp_uint8, mean_uint8, vis_uint8], axis=2)
    return im

# new helper: build RGB from specified NISP bands (use neutral gray for missing bands)
def make_nisp_rgb_composite(nisp_cutouts, band_order=('H', 'J', 'Y'), q=1, clip=99.85):
    """
    Build RGB image from NISP bands in given order.
    band_order: tuple/list where first element maps to R, second->G, third->B.
    Missing bands are substituted with neutral gray. Returns HxWx3 uint8 or None.
    """
    logging.debug('Creating NISP RGB composite with bands: %s', band_order)
    if not nisp_cutouts:
        logging.debug('No NISP inputs')
        return None

    # collect band arrays in requested order
    bands = [nisp_cutouts.get(b) for b in band_order]
    # require at least one valid band
    valid_any = any(is_valid_image(b) for b in bands if b is not None)
    if not valid_any:
        logging.debug('No valid NISP bands present for RGB composite')
        return None

    # determine reference shape from first valid band
    ref = None
    for b in bands:
        if b is not None and is_valid_image(b):
            ref = b.shape
            break
    if ref is None:
        return None

    # ensure shapes of valid bands match ref
    for b in bands:
        if b is not None and is_valid_image(b) and b.shape != ref:
            logging.debug('Shape mismatch among NISP bands; abort RGB composite')
            return None

    # prepare channels: if band valid -> adjust & convert, else neutral gray
    channels = []
    for b in bands:
        if b is not None and is_valid_image(b):
            b_adj = adjust_dynamic_range(b, q=q, clip=clip)
            ch = to_uint8(b_adj)
        else:
            ch = np.full(ref, 128, dtype=np.uint8)
        channels.append(ch)

    # stack as R,G,B where band_order maps to R,G,B
    rgb = np.stack(channels, axis=2)
    return rgb




