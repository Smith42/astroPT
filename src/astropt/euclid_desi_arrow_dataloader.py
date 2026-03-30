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
import random
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
        spectra_inverse: bool = False,
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
        
        # Single modality safe mechanism
        cols_to_keep = ['targetid','redshift']
        
        try:
            if modality_registry.get_config("images") is not None:
                cols_to_keep.extend(['image_vis', 'image_nisp_h', 'image_nisp_j', 'image_nisp_y'])
        except Exception:
            pass
            
        try:
            if modality_registry.get_config("spectra") is not None:
                cols_to_keep.extend(['spectrum_flux', 'spectrum_wave'])
        except Exception:
            pass
        
        
        # Ensure we only ask for columns that actually exist in the Arrow schema
        available_cols = self.ds.column_names
        cols_to_keep = [c for c in cols_to_keep if c in available_cols]
        
        # Drop the rest
        self.ds = self.ds.select_columns(cols_to_keep)
        
        self.ds = self.ds.with_format("numpy")
        
        logging.info(f"Dataset loaded. Total samples: {len(self.ds)}")
        print(f"[{split}] Dataset loaded. Total samples: {len(self.ds)}\n")
        
        # 2. Configuration
        self.modality_registry = modality_registry
        self.spiral = spiral
        self.stochastic = stochastic
        self.transform = transform
        self.spectra_inverse = spectra_inverse
        
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
    def normalise_by_const(x: torch.Tensor, const: float) -> torch.Tensor:
        """
        Normalizes input by dividing by a fixed constant (e.g., global P99).
        """
        x_32 = x.float()
        
        # Avoid division by zero
        div_val = const if const > 1e-8 else 1.0
        
        # Normalization
        x_norm = x_32 / div_val
        
        return x_norm.to(x.dtype)
    
    @staticmethod
    def normalise_zscore(x: torch.Tensor) -> torch.Tensor:
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
    def normalise_asinh(x: torch.Tensor, a: float = 1.0, c: float = 1.0) -> torch.Tensor:
        """
        Applies Inverse Hyperbolic Sine (asinh) transformation.
        Linear near 0, Logarithmic for high values. Handles negatives gracefully.
        
        Args:
            x: Input tensor
            a: Softening parameter (scale). Default 1.0.
            c: Global Constant to normalize. Deafult 1.0.
        """
        
        return (torch.asinh(x.float() / a) / c).to(x.dtype)
    
    @staticmethod
    def _get_augmentation_pipeline() -> Any:
        """
        Returns the composition of data augmentations for training images.
        Includes:
        1. Discrete Rotation (0, 90, 180, 270 degrees) - Lossless.
        2. Horizontal Flip - 50% probability.
        """

        # Rotate images
        def rotate_image(x):
            k = random.randint(0, 3)
            return torch.rot90(x, k, dims=[1, 2])

        # Horizontal flip images
        def hflip_image(x):
            # 50% probability of flip
            if random.random() < 0.5:
                return torch.flip(x, dims=[2])
            return x

        # Transformations list
        aug_list = [
            transforms.Lambda(rotate_image),
            transforms.Lambda(hflip_image)
        ]

        return transforms.Compose(aug_list)


    @staticmethod
    def data_transforms(
        norm_type_img: str = "z_score",
        norm_scaler_img: float = 1.0,
        norm_const_img: float = 1.0,
        norm_type_spec: str = "z_score",
        norm_scaler_spec: float = 1.0,
        norm_const_spec: float = 1.0,
        stage: str = "val"
    ) -> Dict[str, Any]:
        """
        Generates the dictionary of data transformations dynamically based on the configuration.

        This method configures the preprocessing pipelines for 'images' and 'spectra'
        independently, allowing for Z-score normalization, normalization by a fixed constant
        (e.g., a pre-calculated percentile), or Identity (no normalization).

        Args:
            norm_type_img (str): Normalization strategy for images. Options: "z_score", "constant", "none".
            norm_scaler_img (float): Scaler parameter for the normalization image function. 
            norm_const_img (float): Global constant to divide images by. 
            norm_type_spec (str): Normalization strategy for spectra. Options: "z_score", "constant", "none".
            norm_scaler_spec (float): Scaler parameter for the normalization spectra function. 
            norm_const_spec (float): Global constant to divide spectra by. 
            stage (str): 'train' applies augmentations. 'val'/'test' do not.

        Returns:
            Dict[str, Any]: Dictionary where keys are modality names ('images', 'spectra')
                            and values are the composed torchvision Transforms.
        """
        
        # Internal helper function to select the correct normalization method
        def get_norm_transform(n_type: str, n_scaler: float, n_const: float):
            if n_type == "constant":
                # Using lambda for parsing two arguments to Compose module
                return transforms.Lambda(lambda x: EuclidDESIDatasetArrow.normalise_by_const(x, n_const))
            elif n_type == "z_score":
                return transforms.Lambda(EuclidDESIDatasetArrow.normalise_zscore)
            elif n_type == "asinh":
                return transforms.Lambda(lambda x: EuclidDESIDatasetArrow.normalise_asinh(x, a=n_scaler, c=n_const))
            else:
                # Identity transform (no change)
                return transforms.Lambda(lambda x: x)

        # Transformation dictionary
        transform_dict = {}

        # Image Rotation just in training
        if stage == 'train':
            transform_dict["images_aug"] = EuclidDESIDatasetArrow._get_augmentation_pipeline()

        # Image Normalization
        transform_dict["images_norm"] = get_norm_transform(norm_type_img, norm_scaler_img, norm_const_img)

        # Spectra transformations
        transform_dict["spectra"] = transforms.Compose([
            get_norm_transform(norm_type_spec, norm_scaler_spec, norm_const_spec)
        ])

        return transform_dict
    
    
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
        
        # Global image rotation
        if "images_aug" in self.transform:
             raw_image = self.transform["images_aug"](raw_image)
        
        # Patchify
        cfg = self.modality_registry.get_config("images")
        patch_size = cfg.patch_size
        
        # EINOP tensor transformation
        # Input: (Chanels, Height, Width) -> Output: (N_Patchs, Vector_Patch)
        # h,w = total image pixel per side / patch size
        # p1 y p2 are patch size (16, 16)
        patch_image = einops.rearrange(
            raw_image,
            "c (h p1) (w p2) -> (h w) (p1 p2 c)", # (h p1) = h * p1
            p1=patch_size,
            p2=patch_size,
        )

        # Images normalization
        if "images_norm" in self.transform:
            patch_image = self.transform["images_norm"](patch_image)
            
        # Spiral order
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
        # Get configuration
        cfg = self.modality_registry.get_config("spectra")
        patch_size = cfg.patch_size
        
        # Invert spectra for training if required
        if self.spectra_inverse:
            raw_spectra = torch.flip(raw_spectra, dims=[0])
            wavelength = torch.flip(wavelength, dims=[0])
        
        # Galculate necessary padding
        # We need the total length to be a multiple of patch_size
        seq_len = raw_spectra.shape[0]
        remainder = seq_len % patch_size
        pad_len = (patch_size - remainder) % patch_size
        
        # Apply padding (only if needed)
        if pad_len > 0:
            # F.pad tuple format for 1D: (padding_left, padding_right)
            raw_spectra = F.pad(raw_spectra, (0, pad_len))
            wavelength = F.pad(wavelength, (0, pad_len))

        # Reshape into patches (Tokenization)
        # .view(-1, patch_size) automatically calculates the number of patches
        # Shape: (Total_Len) -> (Num_Patches, Patch_Size)
        patch_spectra = raw_spectra.view(-1, patch_size)
        patch_wl = wavelength.view(-1, patch_size)

        # Apply transformations (Normalization)
        if "spectra" in self.transform:
            patch_spectra = self.transform["spectra"](patch_spectra)

        return patch_spectra, patch_wl

    @staticmethod
    def process_modes(
        batch_data: Dict[str, Any], 
        modality_registry: Any, 
        device: torch.device, 
        shuf: bool = False # Ignorado para mantener sincronía alfabética
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Prepares the batch for training by moving tensors to GPU and creating Input (X) 
        and Target (Y) sequences with a Hybrid Autoregressive Shift.
        Automatically handles Position Types (Float vs Long indices) based on config.
        """
        
        modes = modality_registry.generate_sequence(shuf=shuf)

        data_on_device = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
            for k, v in batch_data.items()
        }

        X = {}
        Y = {}

        num_modes = len(modes)

        for i, mode in enumerate(modes):
            data = data_on_device[mode]
            pos = data_on_device[f"{mode}_positions"]
            
            # Check configuration for this modality
            mod_config = modality_registry.get_config(mode)
            
            # If embed_pos=True
            if getattr(mod_config, 'embed_pos', False):
                b, s = pos.shape[:2]
                # Overwrite 'pos' with indices
                pos = torch.arange(s, device=device, dtype=torch.long).unsqueeze(0).expand(b, -1)

            # Create Targets (Shifted by 1)
            Y[mode] = data[:, 1:]

            # Create Inputs (X) and their Positions
            if i == 0 and num_modes > 1:
                # First modality in sequence (context) -> Full sequence
                X[mode] = data 
                X[f"{mode}_positions"] = pos
            else:
                # Subsequent modalities -> Autoregressive input (Shifted)
                X[mode] = data[:, :-1] 
                X[f"{mode}_positions"] = pos[:, :-1]

        return {"X": X, "Y": Y}
    

    @staticmethod
    def prepare_batch(
        batch_data: Dict[str, Any], 
        modality_registry: Any, 
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Prepares a batch for INFERENCE (Full Sequence, No Shifting).
        Handles device movement and type corrections (Float -> Long for Positions).
        """
        X = {}
        
        # Move to device
        data_on_device = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
            for k, v in batch_data.items()
        }

        # Process keys
        for key, val in data_on_device.items():
            
            # Spectra Position Fix
            if key == "spectra_positions":
                spec_config = modality_registry.get_config("spectra")
                if getattr(spec_config, 'embed_pos', False):
                    # Convert Float positions to Long Indices
                    b, s = val.shape[:2]
                    val = torch.arange(s, device=device, dtype=torch.long).unsqueeze(0).expand(b, -1)
            
            # Add to output dict
            X[key] = val

        return X
        
    
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
            if item.get('image_vis') is not None:
                
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
                    raw_data = item.get(key)
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
            if item.get('spectrum_flux') is not None and len(item.get('spectrum_flux', [])) > 0:
                
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