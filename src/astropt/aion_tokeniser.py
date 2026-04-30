"""Multiprocess-safe Codec Manager for AION in AstroPT.
"""

from dataclasses import asdict
import torch
import torchvision.transforms.functional as TF
import os

from aion.codecs.base import Codec
from aion.codecs.config import MODALITY_CODEC_MAPPING, HF_REPO_ID
from aion.modalities import Modality, Image as AionImage

# AION's native HSCImage class (the one codecs recognise via isinstance)
from aion.modalities import HSCImage as AionHSCImage

# FMB's EuclidImage for detecting Euclid data and the U-Net adapter
from fmb.models.aion.modalities import EuclidImage
from fmb.models.aion.model import EuclidToHSC, HSC_BANDS
from aion.codecs.image import ImageCodec

# Ensure AION knows about EuclidImage for codec lookup
import aion.modalities
aion.modalities.EuclidImage = EuclidImage

# Map EuclidImage to ImageCodec (HSCImage already mapped by AION natively)
if EuclidImage not in MODALITY_CODEC_MAPPING:
    MODALITY_CODEC_MAPPING[EuclidImage] = ImageCodec


class ModalityTypeError(TypeError):
    """Error raised when a modality type is not supported."""

def load_frozen_image_codec(device: torch.device) -> Codec:
    """Manually load frozen AION ImageCodec from HF Hub.
    
    This bypasses the broken ImageCodec.from_pretrained() method which fails due to
    inspect.signature mismatches with the quantizer_levels argument.
    """
    from huggingface_hub import hf_hub_download
    import json
    import safetensors.torch as st
    
    # Download config and weights
    cfg_path = hf_hub_download(HF_REPO_ID, "codecs/image/config.json")
    weights_path = hf_hub_download(HF_REPO_ID, "codecs/image/model.safetensors")
    
    with open(cfg_path) as f:
        codec_cfg = json.load(f)
        
    # Patch band registry to avoid collisions with Euclid bands during codec init
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
    
    # CRITICAL: AION has a hardcoded CenterCrop(96) inside _encode which overrides our
    # dynamic target sizes. We replace it with Identity since we already do our own
    # dynamic transform in the dataset / tokenizer.
    codec.center_crop = torch.nn.Identity()
    
    return codec

class TokenKeyError(ValueError):
    """Error raised when a token key is not found in the tokens dictionary."""


class MultiprocessCodecManager:
    """Manager for loading and using codecs for different modalities.
    
    This version is safe to use with PyTorch multiprocessing DataLoaders.
    Unlike AION's original CodecManager, this does not use @lru_cache at
    the class level, instead caching per-instance to allow each worker
    process to maintain its own codec instances.
    """

    def __init__(self, device: str | torch.device = "cpu", unet_weights_path: str = None, aion_image_size: int = 112, aion_image_transform: str = "crop"):
        """Initialize the codec manager.

        Args:
            device: Device to load codecs on ('cpu', 'cuda', etc.)
            unet_weights_path: Optional path to the U-Net adapter weights
            aion_image_size: Target resolution size for AION processing
            aion_image_transform: "crop" or "resize"
        """
        self.device = device
        self._codec_cache = {}
        self.unet_weights_path = unet_weights_path
        self.aion_image_size = aion_image_size
        self.aion_image_transform = aion_image_transform
        self.euclid_to_hsc = None

    def _init_unet_adapter(self):
        """Initialise or load the U-Net adapter for Euclid Imagery."""
        if self.euclid_to_hsc is not None:
            return
            
        # Hardcoding matching dimension from retrain script -> hidden=16
        self.euclid_to_hsc = EuclidToHSC(hidden=16, use_checkpointing=False).to(self.device)
        
        if self.unet_weights_path and os.path.exists(self.unet_weights_path):
            try:
                ckpt = torch.load(self.unet_weights_path, map_location=self.device)
                if "euclid_to_hsc" in ckpt:
                    self.euclid_to_hsc.load_state_dict(ckpt["euclid_to_hsc"])
                else:
                    self.euclid_to_hsc.load_state_dict(ckpt)
                print(f"[info] AION Manager: Loaded EuclidToHSC U-Net weights from {self.unet_weights_path}")
            except Exception as e:
                print(f"[warning] AION Manager: Failed to load EuclidToHSC weights: {e}")
        else:
            print("[warning] AION Manager: No valid U-Net weights path found. Initializing with random weights.")
            
        self.euclid_to_hsc.eval()
        for p in self.euclid_to_hsc.parameters():
            p.requires_grad = False

    def _load_codec(self, modality_type: type[Modality]) -> Codec:
        """Load a codec for the given modality type.
        
        Caches per-instance rather than globally to allow multiprocessing.
        """
        # Return cached codec if already loaded in this instance
        if modality_type in self._codec_cache:
            return self._codec_cache[modality_type]
        
        # Look up codec class for this modality
        if modality_type not in MODALITY_CODEC_MAPPING:
            raise ModalityTypeError(
                f"No codec configuration found for modality type: {modality_type.__name__}"
            )
        
        codec_class = MODALITY_CODEC_MAPPING[modality_type]
        
        # Determine the base modality for HF weight lookup
        hf_mod = AionImage if codec_class == ImageCodec else modality_type
        
        # Load codec from HuggingFace (bypasses global lru_cache)
        if codec_class == ImageCodec:
            codec = load_frozen_image_codec(self.device)
        else:
            codec = codec_class.from_pretrained(HF_REPO_ID, modality=hf_mod)
            codec = codec.eval().to(self.device)
            for p in codec.parameters():
                p.requires_grad = False
            
        # Cache in this instance only
        self._codec_cache[modality_type] = codec
        
        return codec

    @torch.no_grad()
    def encode(self, *modalities: Modality) -> dict[str, torch.Tensor]:
        """Encode multiple modalities into discrete tokens (indices).

        Args:
            *modalities: Variable number of modality instances to encode

        Returns:
            Dictionary mapping token keys to encoded tensors (Long)
        """
        tokens = {}

        for modality in modalities:
            if not isinstance(modality, Modality):
                raise ModalityTypeError(
                    f"Modality {type(modality).__name__} does not have a token_key attribute"
                )
            
            # Use U-Net Adapter for Euclid Images
            if isinstance(modality, EuclidImage):
                self._init_unet_adapter()
                # flux may be (C,H,W) or (B,C,H,W); normalise to (B,C,H,W) for the U-Net
                flux = modality.flux
                squeezed = flux.dim() == 3
                if squeezed:
                    flux = flux.unsqueeze(0)
                flux = flux.to(self.device)
                
                # Apply configured spatial transformation to match token sequence lengths
                if self.aion_image_transform == "crop":
                    flux = TF.center_crop(flux, output_size=self.aion_image_size)
                elif self.aion_image_transform == "resize":
                    flux = TF.resize(flux, size=[self.aion_image_size, self.aion_image_size], antialias=True)
                
                hsc_flux = self.euclid_to_hsc(flux)   # (B, 5, H, W)
                if squeezed:
                    hsc_flux = hsc_flux.squeeze(0)     # back to (5, H, W) — ImageCodec handles both
                # Substitute the modality with AION's native HSCImage
                # (must use aion.modalities.HSCImage, not FMB's, for codec isinstance check)
                modality = AionHSCImage(flux=hsc_flux, bands=HSC_BANDS)

            # Get the appropriate codec
            codec = self._load_codec(type(modality))

            # Tokenize the modality -> (Batch/Seq, Tokens) or (Tokens)
            tokenized = codec.encode(modality)

            # Enforce that output is long for Cross-Entropy processing
            tokens[modality.token_key] = tokenized.to(torch.long)

        return tokens

    @torch.no_grad()
    def decode(
        self,
        tokens: dict[str, torch.Tensor],
        modality_type: type[Modality],
        **metadata,
    ) -> Modality:
        """Decode tokens back to a modality.

        Args:
            tokens: Dictionary mapping token keys to tokenized tensors
            modality_type: The modality type to decode into
            **metadata: Additional metadata required by the specific codec
                       (e.g., wavelength for spectra, bands for images)

        Returns:
            Decoded modality instance
        """
        if not issubclass(modality_type, Modality):
            raise ModalityTypeError(
                f"Modality type {modality_type} does not have a token_key attribute"
            )

        token_key = modality_type.token_key
        if token_key not in tokens:
            raise TokenKeyError(
                f"Token key '{token_key}' for modality {modality_type} not found in tokens dictionary"
            )

        # Handle backward path for EuclidImage (which reconstructs to HSC implicitly unless translated)
        lookup_type = AionHSCImage if modality_type == EuclidImage else modality_type

        # Get the appropriate codec
        codec = self._load_codec(lookup_type)

        # Decode using the codec with any provided metadata
        # Expects continuous vectors inside the codec logic, so quantizer.decode will interpret the tokens
        decoded_modality = codec.decode(tokens[token_key], **metadata)

        # Cast decoded modality to the correct type (Be mindful that EuclidImage will receive HSC decoded properties unless reversed)
        # Note: True translation backwards requires hsc_to_euclid U-net, simplified here to use target type.
        decoded_modality = modality_type(**asdict(decoded_modality))

        return decoded_modality
