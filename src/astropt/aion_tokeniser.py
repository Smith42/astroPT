"""Multiprocess-safe Codec Manager for AION in AstroPT.

This module provides the MultiprocessCodecManager, a utility designed to handle
multimodal tokenization within AstroPT's data pipelines. It supports:
1. Loading and caching frozen AION Image and Spectrum codecs.
2. Domain adaptation for Euclid imagery using a ResNet-based adapter to translate
   Euclid bands (VIS, Y, J, H) into HSC-like bands (g, r, i, z, y).
3. Handling domain-specific normalization (e.g., arcsinh to linear translation 
   for the AION ImageCodec).
4. Ensuring thread/process safety when used with PyTorch's DataLoader workers by
   caching codecs per instance.

The manager integrates with the 'resnet_adapter' to bridge the gap between 
Euclid data and AION's pre-trained foundation model weights.

Author: Victor Alonso
Date: May 2026
"""

from dataclasses import asdict
import torch
import torchvision.transforms.functional as TF
import os
import dataclasses

from aion.codecs.base import Codec
from aion.codecs.config import MODALITY_CODEC_MAPPING, HF_REPO_ID
from aion.modalities import Modality, Image as AionImage

# AION's native modalities
from aion.modalities import HSCImage as AionHSCImage, DESISpectrum

from astropt.resnet_adapter import EuclidImage, EuclidToHSC_ResNet, HSCToEuclid_ResNet, HSC_BANDS
from aion.codecs.image import ImageCodec

# Ensure AION knows about EuclidImage for codec lookup
import aion.modalities
aion.modalities.EuclidImage = EuclidImage
aion.modalities.DESISpectrum = DESISpectrum

# Map EuclidImage to ImageCodec
if EuclidImage not in MODALITY_CODEC_MAPPING:
    MODALITY_CODEC_MAPPING[EuclidImage] = ImageCodec

# Map DESISpectrum to SpectrumCodec
from aion.codecs.spectrum import SpectrumCodec
if DESISpectrum not in MODALITY_CODEC_MAPPING:
    MODALITY_CODEC_MAPPING[DESISpectrum] = SpectrumCodec


class ModalityTypeError(TypeError):
    """Error raised when a modality type is not supported."""

class TokenKeyError(ValueError):
    """Error raised when a token key is not found in the tokens dictionary."""

def load_frozen_image_codec(device: torch.device) -> Codec:
    """Manually load frozen AION ImageCodec from HF Hub."""
    from huggingface_hub import hf_hub_download
    import json
    import safetensors.torch as st
    
    cfg_path = hf_hub_download(HF_REPO_ID, "codecs/image/config.json")
    weights_path = hf_hub_download(HF_REPO_ID, "codecs/image/model.safetensors")
    
    with open(cfg_path) as f:
        codec_cfg = json.load(f)
        
    from aion.codecs.preprocessing.band_to_index import BAND_TO_INDEX
    original_bands = dict(BAND_TO_INDEX)
    try:
        keys_to_remove = [k for k in list(BAND_TO_INDEX.keys()) if "EUCLID" in k]
        for k in keys_to_remove:
            del BAND_TO_INDEX[k]
            
        codec = ImageCodec(
            quantizer_levels=codec_cfg["quantizer_levels"],
            hidden_dims=codec_cfg["hidden_dims"],
            multisurvey_projection_dims=codec_cfg["multisurvey_projection_dims"],
            n_compressions=codec_cfg["n_compressions"],
            num_consecutive=codec_cfg["num_consecutive"],
            embedding_dim=codec_cfg["embedding_dim"],
            range_compression_factor=codec_cfg["range_compression_factor"],
            mult_factor=codec_cfg["mult_factor"],
        ).to(device)
    finally:
        BAND_TO_INDEX.clear()
        BAND_TO_INDEX.update(original_bands)
        
    state = st.load_file(weights_path, device="cpu")
    codec.load_state_dict(state, strict=False)
    
    for p in codec.parameters():
        p.requires_grad = False
    codec.eval()
    
    # Replace CenterCrop with Identity to allow dynamic sizes
    codec.center_crop = torch.nn.Identity()
    
    return codec


class MultiprocessCodecManager:
    """Manager for loading and using codecs for different modalities."""

    def __init__(self, device: str | torch.device = "cpu", resnet_weights_path: str = None, aion_image_size: int = 112, aion_image_transform: str = "crop", resnet_hidden: int = 128):
        self.device = device
        self._codec_cache = {}
        self.resnet_weights_path = resnet_weights_path
        self.aion_image_size = aion_image_size
        self.aion_image_transform = aion_image_transform
        self.resnet_hidden = resnet_hidden  
        self.euclid_to_hsc = None
        self.hsc_to_euclid = None

    def _init_resnet_adapter(self):
        """Initialise or load the ResNet adapter for Euclid Imagery."""
        if self.euclid_to_hsc is not None and self.hsc_to_euclid is not None:
            return
            
        # Initialize ResNet-based adapters (Direct and Inverse)
        self.euclid_to_hsc = EuclidToHSC_ResNet(hidden_dim=self.resnet_hidden, num_blocks=4).to(self.device)
        self.hsc_to_euclid = HSCToEuclid_ResNet(hidden_dim=self.resnet_hidden, num_blocks=4).to(self.device)
        
        if self.resnet_weights_path and os.path.exists(self.resnet_weights_path):
            try:
                ckpt = torch.load(self.resnet_weights_path, map_location=self.device)
                if "euclid_to_hsc" in ckpt:
                    self.euclid_to_hsc.load_state_dict(ckpt["euclid_to_hsc"])
                else:
                    self.euclid_to_hsc.load_state_dict(ckpt)
                
                if "hsc_to_euclid" in ckpt:
                    self.hsc_to_euclid.load_state_dict(ckpt["hsc_to_euclid"])
                
                print(f"[info] AION Manager: Loaded ResNet adapter weights from {self.resnet_weights_path}")
            except Exception as e:
                print(f"[warning] AION Manager: Failed to load EuclidToHSC weights: {e}")
        else:
            print("[warning] AION Manager: No valid ResNet weights path found. Initializing with random weights.")
            
        self.euclid_to_hsc.eval()
        self.hsc_to_euclid.eval()
        for p in list(self.euclid_to_hsc.parameters()) + list(self.hsc_to_euclid.parameters()):
            p.requires_grad = False

    def _load_codec(self, modality_type: type[Modality]) -> Codec:
        if modality_type in self._codec_cache:
            return self._codec_cache[modality_type]
        
        if modality_type not in MODALITY_CODEC_MAPPING:
            raise ModalityTypeError(
                f"No codec configuration found for modality type: {modality_type.__name__}"
            )
        
        codec_class = MODALITY_CODEC_MAPPING[modality_type]
        hf_mod = AionImage if codec_class == ImageCodec else modality_type
        
        if codec_class == ImageCodec:
            codec = load_frozen_image_codec(self.device)
        else:
            codec = codec_class.from_pretrained(HF_REPO_ID, modality=hf_mod)
            codec = codec.eval().to(self.device)
            for p in codec.parameters():
                p.requires_grad = False
            
        self._codec_cache[modality_type] = codec
        return codec

    @torch.no_grad()
    def encode(self, *modalities: Modality) -> dict[str, torch.Tensor]:
        tokens = {}

        for modality in modalities:
            if not hasattr(modality, "token_key"):
                raise ModalityTypeError(
                    f"Modality {type(modality).__name__} does not have a token_key attribute"
                )
            
            # Store original key to ensure it's respected after domain translation
            original_token_key = modality.token_key

            if isinstance(modality, EuclidImage):
                self._init_resnet_adapter()   
                flux = modality.flux
                squeezed = flux.dim() == 3
                if squeezed:
                    flux = flux.unsqueeze(0)
                flux = flux.to(self.device)
                
                if self.aion_image_transform == "crop":
                    flux = TF.center_crop(flux, output_size=self.aion_image_size)
                elif self.aion_image_transform == "resize":
                    flux = TF.resize(flux, size=[self.aion_image_size, self.aion_image_size], antialias=True)
                
                # ResNet predicts in arcsinh domain; convert to linear for AION Codec
                hsc_flux_arcsinh = self.euclid_to_hsc(flux)   
                hsc_flux_linear = torch.sinh(hsc_flux_arcsinh)
                
                if squeezed:
                    hsc_flux_linear = hsc_flux_linear.squeeze(0)     
                
                # Substitute the modality with AION's native HSCImage for encoding
                modality = AionHSCImage(flux=hsc_flux_linear, bands=HSC_BANDS)

            # Ensure all tensor fields are on the correct device
            if dataclasses.is_dataclass(modality):
                for field in dataclasses.fields(modality):
                    val = getattr(modality, field.name)
                    if isinstance(val, torch.Tensor):
                        setattr(modality, field.name, val.to(self.device))

            codec = self._load_codec(type(modality))
            tokenized = codec.encode(modality)
            
            # Use the original token key (e.g., tok_image_euclid) for the output dictionary
            tokens[original_token_key] = tokenized.to(torch.long)

        return tokens

    @torch.no_grad()
    def decode(self, tokens: dict[str, torch.Tensor], modality_type: type[Modality], **metadata) -> Modality:
        """Decode tokens back to a modality."""
        if getattr(modality_type, "token_key", None) is None:
            raise ModalityTypeError(f"Modality type {modality_type} does not have a token_key attribute")

        token_key = modality_type.token_key
        if token_key not in tokens:
            raise TokenKeyError(f"Token key '{token_key}' not found in tokens dictionary")

        # EuclidImage implicitly decodes via the HSC Codec
        lookup_type = AionHSCImage if modality_type == EuclidImage else modality_type
        codec = self._load_codec(lookup_type)
        
        # For EuclidImage, we must patch BAND_TO_INDEX during decoding so the AION
        # codec produces 5 HSC bands (not the default 9). This mirrors what
        # load_frozen_image_codec does during construction.
        if modality_type == EuclidImage:
            from aion.codecs.preprocessing.band_to_index import BAND_TO_INDEX
            original_bands = dict(BAND_TO_INDEX)
            BAND_TO_INDEX.clear()
            BAND_TO_INDEX.update({b: i for i, b in enumerate(HSC_BANDS)})
            try:
                decoded_modality = codec.decode(tokens[token_key], **metadata)
            finally:
                BAND_TO_INDEX.clear()
                BAND_TO_INDEX.update(original_bands)
        else:
            decoded_modality = codec.decode(tokens[token_key], **metadata)
        
        # If we requested EuclidImage, we must translate the HSC output back to Euclid
        if modality_type == EuclidImage:
            self._init_resnet_adapter()
            hsc_flux_linear = decoded_modality.flux.to(self.device)
            
            # The inverse adapter was trained in arcsinh space:
            #   encode: Euclid(arcsinh) -> ResNet_fwd -> HSC(arcsinh) -> sinh -> HSC(linear) -> AION
            #   decode: AION -> HSC(linear) -> asinh -> HSC(arcsinh) -> ResNet_inv -> Euclid(arcsinh) -> sinh -> Euclid(linear)
            hsc_flux_arcsinh = torch.asinh(hsc_flux_linear)
            euclid_flux_arcsinh = self.hsc_to_euclid(hsc_flux_arcsinh)
            
            # Convert back to linear space to match raw Arrow data
            euclid_flux_linear = torch.sinh(euclid_flux_arcsinh)
            decoded_modality.flux = euclid_flux_linear

        # Cast to the requested modality type
        decoded_modality = modality_type(**asdict(decoded_modality))
        return decoded_modality