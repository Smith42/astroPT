import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from .base import ModalityProcessor
from astropt.modalities.spectrum import DESISpectrum

class SpectrumProcessor(ModalityProcessor):
    """Base class for spectrum processors."""
    pass


class DESISpectrumProcessor(SpectrumProcessor):
    def __init__(self, stochastic: bool = True, mask_prob: float = 0.0, inverse: bool = False):
        self.stochastic = stochastic
        self.mask_prob = mask_prob
        self.inverse = inverse

    def required_columns(self) -> List[str]:
        return ['spectrum_flux', 'spectrum_wave']

    def is_available(self, item: Dict[str, Any]) -> bool:
        return item.get('spectrum_flux') is not None and len(item.get('spectrum_flux', [])) > 0

    def process(self, item: Dict[str, Any], transform: Dict[str, Any], config: Any = None) -> Optional[DESISpectrum]:
        raw_flux = item['spectrum_flux']
        raw_wave = item['spectrum_wave']
        
        if len(raw_flux) != len(raw_wave):
            return None
            
        if isinstance(raw_flux, list): raw_flux = np.array(raw_flux) 
        if isinstance(raw_wave, list): raw_wave = np.array(raw_wave) 
            
        raw_flux = torch.from_numpy(raw_flux).to(torch.bfloat16)
        raw_wave = torch.from_numpy(raw_wave).to(torch.bfloat16)                    
        
        if torch.isnan(raw_flux).any():
            return None
            
        if self.inverse:
            raw_flux = torch.flip(raw_flux, dims=[0])
            raw_wave = torch.flip(raw_wave, dims=[0])
            
        # Normalize wavelength
        cont_wave = raw_wave.clone()
        cont_wave = (cont_wave - 3000.0) / (10000.0 - 3000.0)

        patch_size = config.patch_size if config else 16
        
        # Padding
        seq_len = raw_flux.shape[0]
        remainder = seq_len % patch_size
        pad_len = (patch_size - remainder) % patch_size
        
        if pad_len > 0:
            raw_flux = F.pad(raw_flux, (0, pad_len))
            cont_wave = F.pad(cont_wave, (0, pad_len))

        # Patchify
        patch_spectra = raw_flux.view(-1, patch_size)
        patch_wl = cont_wave.view(-1, patch_size)

        if self.stochastic:
            mask = torch.rand(patch_spectra.shape[0], device=patch_spectra.device) < self.mask_prob
            patch_spectra[mask] = 0.0

        if "spectra" in transform:
            patch_spectra = transform["spectra"](patch_spectra)

        return DESISpectrum(flux=patch_spectra, wavelength=patch_wl, patch_size=patch_size)
