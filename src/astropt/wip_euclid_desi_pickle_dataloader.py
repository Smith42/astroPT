"""
Pickle/Arrow Dataloader for AstroPT.
Handles chunked datasets (lists of dicts) with LRU Caching for high-performance training.
"""

from collections import OrderedDict
import logging
import numpy as np
from numpy.typing import NDArray
import os
import glob
import pickle
import re
from typing import Optional, Dict, Any, List, Tuple, OrderedDict


import einops
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class EuclidDESIDatasetPickle(Dataset): 
    
    def __init__(
        self, 
        data_dir: str,
        split: str = 'train', # 'train', 'test', or 'val'
        modality_registry: Any = None,
        spiral: bool = False,           
        transform: Dict = {},
        cache_size: int = 4, # Number of .pkl files to keep in RAM (approx 2000 galaxies)
        samples_per_file: int = 500 # Standard chunk size of your dataset
    ):
        
        """
        Dataloader for Chunked Pickle Files.
        
        Args:
            data_dir: Root directory containing the .pkl files.
            split: Dataset split ('train', 'val', 'test').
            modality_registry: Configuration for modalities.
            spiral: Whether to apply spiral tokenization to images.
            cache_size: How many batch files to keep in memory to reduce Disk I/O.
            samples_per_file: Number of samples per .pkl file (usually 500).
        """
        self.data_dir = data_dir
        self.split = split
        self.modality_registry = modality_registry
        self.spiral = spiral
        self.transform = transform
        self.cache_size = cache_size
        self.samples_per_file = samples_per_file
        
        # 1. Index Files
        # Looking for files like "batch_train_1.pkl", "batch_train_2.pkl"
        pattern = os.path.join(data_dir, f"batch_{split}_*.pkl")
        self.file_paths = glob.glob(pattern)
        
        # Filter out metadata files ("_info.pkl")
        self.file_paths = [f for f in self.file_paths if not f.endswith("_info.pkl")]
        
        # Sort numerically (Crucial for deterministic indexing)
        # Regex extracts '10' from 'batch_train_10.pkl'
        def extract_number(path):
            match = re.search(r'batch_\w+_(\d+).pkl', path)
            return int(match.group(1)) if match else 0
            
        self.file_paths.sort(key=extract_number)
        
        if not self.file_paths:
            raise FileNotFoundError(f"No .pkl files found in {data_dir} for split '{split}'")
            
        logging.info(f"[{split.upper()}] Found {len(self.file_paths)} batch files.")
        
        # 2. Calculate Total Length
        # Assumption: All files have fixed size except maybe the last one.
        # This is faster than opening all files to count.
        self.total_len = len(self.file_paths) * samples_per_file
        
        # 3. Initialize LRU Cache
        # Stores loaded file content: {file_index: list_of_dicts}
        self.cache: OrderedDict[int, List[Dict]] = OrderedDict()

    def __len__(self) -> int:
        return self.total_len
    
    def _get_from_cache(self, file_idx: int) -> Optional[List[Dict]]:
        """
        Retrieves a data batch from RAM cache or loads it from disk if missing.
        Implements Least Recently Used (LRU) eviction policy.
        """
        # Hit: Return immediately and move to end (mark as recently used)
        if file_idx in self.cache:
            self.cache.move_to_end(file_idx)
            return self.cache[file_idx]
        
        # Miss: Load from disk
        path = self.file_paths[file_idx]
        try:
            with open(path, 'rb') as f:
                data_list = pickle.load(f)
        except Exception as e:
            logging.error(f"Failed to load batch {path}: {e}")
            return None

        # Insert into cache
        self.cache[file_idx] = data_list
        
        # Evict oldest if full
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False) # pop first item (oldest)
            
        return data_list

    # --------------------------------------------------------------
    # PROCESSING METHODS (Copied from previous robust version)
    # --------------------------------------------------------------
    
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

    # --------------------------------------------------------------
    # CORE LOGIC: __getitem__
    # --------------------------------------------------------------
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        
        # 1. Resolve File and Local Index
        if torch.is_tensor(idx): idx = idx.tolist()
        
        file_idx = idx // self.samples_per_file
        local_idx = idx % self.samples_per_file
        
        # Safety Check: Index out of bounds
        if file_idx >= len(self.file_paths):
            return None
            
        # 2. Load Data from Cache (or Disk)
        batch_data = self._get_from_cache(file_idx)
        
        # Safety Check: Empty file or local index out of bounds (last file case)
        if batch_data is None or local_idx >= len(batch_data):
            return None 
            
        # Get the specific galaxy dictionary
        entry = batch_data[local_idx]
        targetid = entry.get('targetid', -1)
        
        # --- 3. LOAD IMAGES ---
        # Keys: 'VIS_image', 'NISP_Y_image', 'NISP_J_image', 'NISP_H_image'
        vis = entry.get('VIS_image')
        
        if vis is None: 
            # Skip if primary image is missing
            return None 
        
        # Helper to validate NISP shapes against VIS
        ref_shape = vis.shape
        def validate_nisp(img):
            if img is None or img.shape != ref_shape:
                return np.zeros_like(vis)
            return img
            
        nisp_stack = [
            validate_nisp(entry.get('NISP_H_image')),
            validate_nisp(entry.get('NISP_J_image')),
            validate_nisp(entry.get('NISP_Y_image'))
        ]
        
        # Stack: (4, H, W) -> [VIS, H, J, Y]
        raw_image = np.stack([vis] + nisp_stack, axis=0).astype(np.float32)
        raw_image_tensor = torch.from_numpy(raw_image).to(torch.bfloat16)
        
        # --- FIX: ROBUST SIZE ENFORCEMENT ---
        # Aseguramos que la imagen sea 224x224 (o lo que espere el modelo)
        # Si es más pequeña, rellenamos (Pad). Si es más grande, recortamos (Crop).
        TARGET_H, TARGET_W = 224, 224 
        C, H, W = raw_image_tensor.shape
        
        if H != TARGET_H or W != TARGET_W:
            # Calculamos padding necesario
            pad_h = max(0, TARGET_H - H)
            pad_w = max(0, TARGET_W - W)
            
            # Si necesita relleno, aplicamos padding a derecha y abajo
            if pad_h > 0 or pad_w > 0:
                raw_image_tensor = F.pad(raw_image_tensor, (0, pad_w, 0, pad_h))
            
            # Si sobra (es más grande), recortamos el centro (o la esquina)
            # Aquí recortamos la esquina superior izquierda para simplificar y mantener alineación
            raw_image_tensor = raw_image_tensor[:, :TARGET_H, :TARGET_W]

        # ------------------------------------
        
        # Tokenize Image
        patch_image = self.process_image(raw_image_tensor)
        
        # --- 4. LOAD SPECTRA ---
        # Key: 'spectrum' (Dict with 'flux' and 'wavelength')
        spec_dict = entry.get('spectrum')
        patch_spectra = None
        patch_wl = None
        
        if spec_dict is not None and isinstance(spec_dict, dict):
            raw_flux = spec_dict.get('flux')
            raw_wave = spec_dict.get('wavelength')
            
            if raw_flux is not None and raw_wave is not None:
                # Convert to Tensor
                raw_flux = torch.from_numpy(raw_flux.astype(np.float32)).to(torch.bfloat16)
                raw_wave = torch.from_numpy(raw_wave.astype(np.float32)).to(torch.bfloat16)
                
                # Normalize Wavelength (Standard Min-Max 3000-10000A)
                # This keeps consistency with the previous dataloader scaling
                raw_wave = (raw_wave - 3000.0) / (10000.0 - 3000.0)
                
                # Tokenize Spectra
                patch_spectra, patch_wl = self.process_spectra(raw_flux, raw_wave)
        
        # --- 5. ASSEMBLE SAMPLE ---
        sample = {
            "images": patch_image,
            "images_positions": torch.arange(0, len(patch_image), dtype=torch.long),
            "idx": idx,
            "targetid": int(targetid)
        }
        
        # Add spectra if available (multimodal)
        if patch_spectra is not None:
            sample["spectra"] = patch_spectra
            sample["spectra_positions"] = patch_wl
            
        # Final validation (ensure at least images exist)
        if "images" not in sample:
            return None
            
        return sample