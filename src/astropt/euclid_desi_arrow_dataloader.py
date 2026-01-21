"""
This file contains the Euclid images and DESI spectra Dataloader
for traingin AstroPT transformer model from an Arrow format database.
"""
import einops
from datasets import load_from_disk, concatenate_datasets
import glob
import logging
import numpy as np
from numpy.typing import NDArray
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Optional, Dict, Any, List, Tuple


class EuclidDESIDatasetArrow(Dataset): 
    
    def __init__(
        self, 
        arrow_folder_root: str,
        split: str,
        modality_registry: Any,
        spiral: bool = False,           
        stochastic = True,      
        transform: Dict = {},
    ):
        """
        Dataset to loading Euclid Images and DESI spectra
        
        Args:
            arrow_folder_root: Path to the root folder of the arrow dataset
            split: Indicates the train and test spliting
            modality_registry: ModalityRegistry object to load the modality configuration
            spiral: For applying the spiral tokenization
            stochastic: For applyting an stochastic training
            transform: Transform method for each modality
        """
        
        # 1. Loading Arrow Dataset
        arrow_pattern = os.path.join(arrow_folder_root, f"{split}_*")
        arrow_folders = sorted(glob.glob(arrow_pattern))
        
        # Raising an error in case of data not found
        if not arrow_folders:
            raise ValueError(f"No Arrow data found at {arrow_pattern}")
        
        print(f"[{split}] Data found in: {len(arrow_folders)} directories")
        print(f"[{split}] Directories list: {[os.path.basename(f) for f in arrow_folders]}")
        
        # Keeping user informed
        logging.info(f"Loading {len(arrow_folders)} Arrow parts for split '{split}'...")
        
        # Creating the dataset with the corresponding split
        self.ds = concatenate_datasets([load_from_disk(p) for p in arrow_folders])
        self.ds = self.ds.with_format("numpy")
        
        logging.info(f"Dataset loaded. Total samples: {len(self.ds)}")
        print(f"[{split}] Dataset loaded. Total samples: {len(self.ds)}\n")
        
        # 2. Configuration
        self.modality_registry = modality_registry
        self.spiral = spiral
        self.stochastic = stochastic
        self.transform = transform
        
        
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.ds)
    
    
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
    
    @staticmethod
    def normalise(x: torch.Tensor) -> torch.Tensor:
        """
        Standardizes the input tensor (Mean 0, Std 1) while preserving the original dtype.
        
        Calculations are performed in float32 for numerical stability, 
        then cast back to the input dtype (e.g., bfloat16).

        Args:
            x (torch.Tensor): Input tensor of shape (N, ...).

        Returns:
            torch.Tensor: Normalized tensor with the same dtype as input.
        """
        # float32 for precise
        x_32 = x.float()
        
        # Standard deviation and mean 
        std, mean = torch.std_mean(x_32, dim=1, keepdim=True)
        
        # Apply Z-score normalization: (Value - Mean) / Std
        # Added epsilon (1e-8) to prevent division by zero
        x_norm = (x_32 - mean) / (std + 1e-8)
        
        return x_norm.to(x.dtype)

    @staticmethod
    def data_transforms() -> Dict[str, Any]:
        """
        Returns the dictionary of transformations expected by the Dataset.
        
        Defines specific preprocessing pipelines for 'images' and 'spectra'.

        Returns:
            Dict[str, Any]: Keys match the modality names, values are torchvision Transforms.
        """
        return {
            "images": transforms.Compose([
                transforms.Lambda(EuclidDESIDatasetArrow.normalise)
            ]),
            "spectra": transforms.Compose([
                transforms.Lambda(EuclidDESIDatasetArrow.normalise)
            ])
        }
    
    
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
            patch_image = self.spiralise_image(patch_image)

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
        Prepares the batch for training by moving tensors to GPU and creating Input (X) 
        and Target (Y) sequences with a Hybrid Autoregressive Shift.

        This method solves the dimension mismatch issue by applying different slicing 
        strategies based on the modality name, derived from empirical model behavior:

        1. Target (Y): Always represents the 'next token' (t+1).
           - Logic: Shift forward by 1 (drops the first token).
           
        2. Input (X): Depends on how the model processes each modality internally.
           - 'images': The model internally consumes 1 token (likely during embedding).
             Strategy: Feed the FULL sequence (N). Output will be N-1, matching Y.
           - 'spectra': The model preserves the sequence length.
             Strategy: Feed a TRIMMED sequence (N-1). Output will be N-1, matching Y.

        Args:
            batch_data (Dict[str, Any]): Dictionary containing raw tensors from the Dataloader.
            modality_registry (Any): Object containing the configuration and order of modalities.
            device (torch.device): The target device (CPU/CUDA) for tensor allocation.
            shuf (bool, optional): Whether to shuffle the order of modalities (Data Augmentation). 
                                   Defaults to False.

        Returns:
            Dict[str, Dict[str, torch.Tensor]]: A dictionary with two keys:
                - "X": Inputs dictionary mapping {modality_name: tensor}.
                - "Y": Targets dictionary mapping {modality_name: tensor}.
        """
        # 1. Determine the order of modalities (e.g., ['images', 'spectra'])
        modes = modality_registry.generate_sequence(shuf=shuf)

        # 2. Move all data to the target device (GPU) efficiently
        data_on_device = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
            for k, v in batch_data.items()
        }

        X = {}
        Y = {}
        
        for mode in modes:
            # Extract raw data and position tensors for the current modality
            data = data_on_device[mode]
            pos = data_on_device[f"{mode}_positions"]
            
            #--- TARGET (Y) GENERATION ---#
            # The goal is always to predict the next step.
            # We slice from index 1 to the end (removing the first token).
            # Shape: (Batch, N-1, Dim)
            Y[mode] = data[:, 1:]
            
            #--- INPUT (X) GENERATION (HYBRID STRATEGY) ---#
            if mode == 'images':
                # CASE 1: IMAGES
                # Empirical observation: Model output is 1 token shorter than input.
                # Action: Provide FULL sequence (N) so output becomes (N-1).
                # Matches Target Y (N-1).
                X[mode] = data
                X[f"{mode}_positions"] = pos
                
            elif mode == 'spectra':
                # CASE 2: SPECTRA
                # Empirical observation: Model output length equals input length.
                # Action: Manually TRIM the last token (N-1) so output stays (N-1).
                # Matches Target Y (N-1).
                X[mode] = data[:, :-1]
                X[f"{mode}_positions"] = pos[:, :-1]
                
            else:
                # Fallback for unknown modalities
                # Standard autoregressive behavior (Input is t, Target is t+1)
                X[mode] = data[:, :-1]
                X[f"{mode}_positions"] = pos[:, :-1]

        return {"X": X, "Y": Y}
        
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve a single sample from the arrow dataset by index.

        Loads the galaxy images (VIS + NISP), the spectrum,
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
        """
        # Ensure index is a standard Python type for Astropy compatibility
        if torch.is_tensor(idx):
            idx = idx.tolist()
        try:
            #--- 1. ARROW ACCESS ---#
            try:
                item = self.ds[idx]
                targetid = int(item['targetid']) 
            except Exception as e:
                logging.error(f"Error reading index {idx} from Arrow: {e}")
                new_idx = np.random.randint(0, len(self))
                return self.__getitem__(new_idx)
            
            # Base returning dictionary
            sample = {
                "idx": idx,
                "targetid": int(targetid)
            }
            

            #--- 2. LOAD VIS & NISP IMAGES ---#
            
            # Check for existence of VIS images (mandatory)
            if item['image_vis'] is not None:
                
                # Convert values to Tensor objects
                raw_vis = item['image_vis']
                if isinstance(raw_vis, list):
                    raw_vis = np.array(raw_vis) 
                vis = torch.from_numpy(raw_vis).to(torch.bfloat16)
                
                # Check for Numerical Stability (NaNs/Infs)
                if torch.isnan(vis).any() or torch.isinf(vis).any():
                    raise ValueError("NaNs or Infs detected in VIS image.")
                
                # Loading NISP images
                ref_shape = vis.shape
                nisp_tensors = []
                nisp_keys = ['image_nisp_h', 'image_nisp_j', 'image_nisp_y']
                
                for key in nisp_keys:
                    raw_data = item[key]
                    if isinstance(raw_data, list):
                        raw_data = np.array(raw_data) 
                    
                    if raw_data is None:
                        # Padding with zeros until fulfill the VIS shape
                        nisp_tensors.append(torch.zeros(ref_shape, dtype=torch.bfloat16))
                        continue
                    
                    try:
                        
                        # Converting to tensor objects
                        tensor = torch.from_numpy(raw_data).to(torch.bfloat16)
                        
                        # Checking possible NaNs or Infs values
                        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                            logging.warning(f"Target {targetid}: NaNs/Infs detected in {key}. Filling with zeros.")
                            nisp_tensors.append(torch.zeros(ref_shape, dtype=torch.bfloat16))
                        
                        # Checking the shape
                        elif tensor.shape != ref_shape:
                            logging.warning(f"Target {targetid}: Shape mismatch in {key} {tensor.shape} vs VIS {ref_shape}. Filling zeros.")
                            nisp_tensors.append(torch.zeros(ref_shape, dtype=torch.bfloat16))
                        
                        # Everything ok
                        else:
                            nisp_tensors.append(tensor)
                        
                    except Exception as e:
                        logging.error(f"Target {targetid}: Crash loading {key}: {e}. Filling zeros.")
                        nisp_tensors.append(torch.zeros(ref_shape, dtype=torch.bfloat16)) 
                

                # Stack and Process
                raw_galaxy = torch.stack([vis] + nisp_tensors, dim=0)
                patch_galaxy = self.process_image(raw_galaxy)
                
                # adding values to sample dictionary
                sample["images"] = patch_galaxy
                sample["images_positions"] = torch.arange(0, len(patch_galaxy), dtype=torch.long)

            else:
                logging.warning(f"Target {targetid}: VIS image is None in Arrow dataset. Skipping.")
                pass

            

            #--- 3. LOAD SPECTRA ---#
            
            # Only proceed if we have valid flux data
            if item['spectrum_flux'] is not None and len(item['spectrum_flux']) > 0:
                
                # Flux and Wavelength MUST have the same length
                if len(item['spectrum_flux']) != len(item['spectrum_wave']):
                    
                    logging.warning(f"Target {targetid}: Spectrum mismatch (Flux len {len(item['spectrum_flux'])} != Wave len {len(item['spectrum_wave'])}). Skipping spectrum.")
                    
                else:
                    
                    # Cheking the format
                    raw_flux = item['spectrum_flux']
                    if isinstance(raw_flux, list):
                        raw_flux = np.array(raw_flux) 
                    raw_flux = torch.from_numpy(raw_flux).to(torch.bfloat16)
                    
                    raw_wave = item['spectrum_wave']
                    if isinstance(raw_wave, list):
                        raw_wave = np.array(raw_wave) 
                    raw_wave = torch.from_numpy(raw_wave).to(torch.bfloat16)                    
                    
                    
                    # Check for NaNs in Spectrum
                    if torch.isnan(raw_flux).any():
                        logging.warning(f"Target {targetid}: NaNs in spectrum flux. Skipping spectrum.")
                        
                    else:
                        # Normalize wavelength
                        raw_wave = (raw_wave - 3000.0) / (10000.0 - 3000.0)
                        
                        # Apply padding and patching
                        patch_spectra, patch_wl = self.process_spectra(raw_flux, raw_wave)
                        
                        # Adding values to sample dictionary
                        sample["spectra"] = patch_spectra
                        sample["spectra_positions"] = patch_wl
                

            #--- 4. FINAL VALIDATION ---#
            # Ensure we are not returning a sample with only metadata
            if "images" not in sample and "spectra" not in sample:
                logging.debug(f"Target {targetid}: Sample ended up empty.")
            
                new_idx = np.random.randint(0, len(self))
                return self.__getitem__(new_idx)
            
            return sample
    
        except Exception as e:
            logging.error(f"Critical error loading idx {idx}: {e}. Retrying with random sample...")
            
            new_idx = np.random.randint(0, len(self))
            return self.__getitem__(new_idx)