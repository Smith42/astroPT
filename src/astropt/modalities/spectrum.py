import torch
from dataclasses import dataclass
from typing import ClassVar
from .base import Modality

@dataclass
class Spectrum(Modality):
    """Base class for Spectrum type modalities."""
    generic_token: ClassVar[str] = "<SPEC>"
    flux: torch.Tensor
    wavelength: torch.Tensor
    patch_size: int

    @property
    def num_patches(self) -> int:
        length = self.flux.shape[0] # flux shape is (num_patches, patch_size)
        return length

    @property
    def positions(self) -> torch.Tensor:
        """For spectra, positions are the wavelengths."""
        return self.wavelength

    @property
    def aperture_indices(self) -> torch.Tensor:
        """
        All spectrum patches come from the central fiber, so they are all inside (0).
        """
        return torch.zeros(self.num_patches, dtype=torch.long)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(flux_shape={list(self.flux.shape)}, wave_range=[{self.wavelength.min().item():.1f}, {self.wavelength.max().item():.1f}])"

@dataclass
class DESISpectrum(Spectrum):
    """Modality for continuous DESI Spectra."""
    specific_token: ClassVar[str] = "<DESI_SPEC>"