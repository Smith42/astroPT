import torch
import math
from dataclasses import dataclass
from typing import ClassVar, List
from .base import Modality

# Images Classes

@dataclass
class Image(Modality):
    """Base class for Image type modalities."""
    generic_token: ClassVar[str] = "<IMG>"
    flux: torch.Tensor  # Tensor with shape (bands, height, width)
    patch_size: int   # <--- Now this is saved per instance
    pixel_scale: ClassVar[float] = 0.1  # Default pixel scale in arcsec/pixel
    aperture_radius_arcsec: ClassVar[float] = 1.5  # Fiber radius in arcsec

    @property
    def num_patches(self) -> int:
        """Dynamically calculates the number of patches."""
        # Note: flux is already patchified in ImageProcessor.process
        # so shape is (num_patches, patch_vector)
        return self.flux.shape[0]

    @property
    def positions(self) -> torch.Tensor:
        """For images, positions are usually sequential (or spiral) indices."""
        return torch.arange(self.num_patches, dtype=torch.long)

    @property
    def aperture_indices(self) -> torch.Tensor:
        """
        Determines which patches are inside the central fiber.
        Since patches are in spiral order (center-to-outskirts):
        We calculate the diameter of the fiber in pixels, convert it to 
        the number of patches needed to cover it, and map it to the 
        first K patches of the spiral sequence.
        """
        fiber_diameter_pixels = 2.0 * self.aperture_radius_arcsec / self.pixel_scale
        patches_needed = fiber_diameter_pixels / self.patch_size
        
        # Round up to the nearest odd integer to form a symmetric grid (1x1, 3x3, 5x5, ...)
        m = math.ceil(patches_needed)
        if m % 2 == 0:
            m += 1
        m = max(1, m)
        k = m * m  # Number of patches in the MxM central grid
        
        ap = torch.ones(self.num_patches, dtype=torch.long)
        ap[:min(k, self.num_patches)] = 0
        return ap

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(flux_shape={list(self.flux.shape)})"

@dataclass
class EuclidImage(Image):
    """Modality for continuous Euclid Images (VIS + NISP)."""
    specific_token: ClassVar[str] = "<EUCLID_IMG>"
    bands: ClassVar[List[str]] = ["VIS", "NISP_Y", "NISP_J", "NISP_H"]
    pixel_scale: ClassVar[float] = 0.1  # Euclid VIS pixel scale is 0.1"

@dataclass
class HSCImage(Image):
    """Modality for continuous HSC Images."""
    specific_token: ClassVar[str] = "<HSC_IMG>"
    bands: ClassVar[List[str]] = ["g", "r", "i", "z", "y"]
    pixel_scale: ClassVar[float] = 0.168  # HSC pixel scale is 0.168"