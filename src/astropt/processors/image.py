import torch
import numpy as np
import random
import einops
from typing import Dict, Any, List, Optional
from numpy.typing import NDArray
from torchvision import transforms
from .base import ModalityProcessor
from astropt.modalities.image import EuclidImage

class ImageProcessor(ModalityProcessor):
    """Base class for image processors that shares common logic."""
    
    @staticmethod
    def _spiral_sorting(n: int) -> NDArray[np.int64]:
        layout = np.arange(n * n).reshape(n, n)
        spiral_indices = []
        while layout.size > 0:
            spiral_indices.append(layout[0])
            layout = layout[1:] 
            if layout.size == 0: break
            spiral_indices.append(layout[:, -1])
            layout = layout[:, :-1]
            if layout.size == 0: break
            spiral_indices.append(layout[-1][::-1])
            layout = layout[:-1]
            if layout.size == 0: break
            spiral_indices.append(layout[:, 0][::-1])
            layout = layout[:, 1:]
        spiral_order = np.concatenate(spiral_indices)
        result = np.empty(n * n, dtype=int)
        result[spiral_order] = np.arange(n * n)
        result = (n * n - 1) - result
        return result

    def _spiralise(self, image: torch.Tensor) -> torch.Tensor:
        n_patches = len(image)
        side_len = int(np.sqrt(n_patches))
        assert side_len**2 == n_patches, f"Patch count ({n_patches}) must be a perfect square!"
        spiral_indices = self._spiral_sorting(side_len)
        sorted_pairs = sorted(zip(spiral_indices, image), key=lambda pair: pair[0])
        spiraled_patches = [patch for _, patch in sorted_pairs]
        if isinstance(image, torch.Tensor):
            return torch.stack(spiraled_patches)
        return np.stack(spiraled_patches)

    def _antispiralise(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Restores the original patch order from the spiral order.
        Used for reconstruction/visualization.
        """
        n_patches = len(patches)
        side_len = int(np.sqrt(n_patches))
        spiral_indices = self._spiral_sorting(side_len)
        inv_spiral_indices = np.argsort(spiral_indices)
        
        if isinstance(patches, torch.Tensor):
            return patches[inv_spiral_indices]
        return patches[inv_spiral_indices]

    @staticmethod
    def _get_augmentation_pipeline() -> transforms.Compose:
        """
        Configures augmentations: Discrete rotations and Flips.
        """
        def rotate_image(x):
            k = random.randint(0, 3)
            return torch.rot90(x, k, dims=[1, 2])

        def hflip_image(x):
            if random.random() < 0.5:
                return torch.flip(x, dims=[2])
            return x

        return transforms.Compose([
            transforms.Lambda(rotate_image),
            transforms.Lambda(hflip_image)
        ])

class EuclidImageProcessor(ImageProcessor):
    FILTER_MAP = {
        'VIS': 'image_vis',
        'Y': 'image_nisp_y',
        'J': 'image_nisp_j',
        'H': 'image_nisp_h'
    }

    def __init__(self, filters: Optional[List[str]] = None, spiral: bool = True, stochastic: bool = True, mask_prob: float = 0.0, stage: str = "train"):
        self.filters = filters or ['VIS', 'Y', 'J', 'H']
        self.spiral = spiral
        self.stochastic = stochastic
        self.mask_prob = mask_prob
        self.stage = stage
        self.aug_pipeline = self._get_augmentation_pipeline() if stage == "train" else None

    def required_columns(self) -> List[str]:
        return [self.FILTER_MAP[f] for f in self.filters if f in self.FILTER_MAP]

    def is_available(self, item: Dict[str, Any]) -> bool:
        # Require at least the first requested filter to be present
        req_cols = self.required_columns()
        return len(req_cols) > 0 and item.get(req_cols[0]) is not None

    def process(self, item: Dict[str, Any], transform: Dict[str, Any], config: Any = None) -> Optional[EuclidImage]:
        selected_tensors = []
        ref_shape = None
        
        for f in self.filters:
            key = self.FILTER_MAP.get(f)
            if not key: continue
            
            raw_data = item.get(key)
            if raw_data is None:
                if ref_shape:
                    selected_tensors.append(torch.zeros(ref_shape, dtype=torch.bfloat16))
                continue
                
            if isinstance(raw_data, list):
                raw_data = np.array(raw_data)
            
            tensor = torch.from_numpy(raw_data).to(torch.bfloat16)
            
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                if ref_shape:
                    selected_tensors.append(torch.zeros(ref_shape, dtype=torch.bfloat16))
                continue
            
            if ref_shape is None:
                ref_shape = tensor.shape
            elif tensor.shape != ref_shape:
                selected_tensors.append(torch.zeros(ref_shape, dtype=torch.bfloat16))
                continue
                
            selected_tensors.append(tensor)
            
        if not selected_tensors:
            return None
            
        raw_galaxy = torch.stack(selected_tensors, dim=0)

        patch_size = config.patch_size if config else 16

        # Apply augmentations if we are training
        if self.aug_pipeline:
             raw_galaxy = self.aug_pipeline(raw_galaxy)
        
        # Then external augmentations (if any in transform)
        if "images_aug" in transform:
             raw_galaxy = transform["images_aug"](raw_galaxy)
        
        patch_galaxy = einops.rearrange(
            raw_galaxy,
            "c (h p1) (w p2) -> (h w) (p1 p2 c)",
            p1=patch_size,
            p2=patch_size,
        )

        if "images_norm" in transform:
            patch_galaxy = transform["images_norm"](patch_galaxy)
            
        if self.spiral:
            patch_galaxy = self._spiralise(patch_galaxy)

        if self.stochastic:
            mask = torch.rand(patch_galaxy.shape[0], device=patch_galaxy.device) < self.mask_prob
            patch_galaxy[mask] = 0.0

        return EuclidImage(flux=patch_galaxy, patch_size=patch_size)
