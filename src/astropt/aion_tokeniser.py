"""Multiprocess-safe Codec Manager for AION in AstroPT.
"""

from dataclasses import asdict
import torch

from aion.codecs.base import Codec
from aion.codecs.config import MODALITY_CODEC_MAPPING, HF_REPO_ID
from aion.modalities import Modality


class ModalityTypeError(TypeError):
    """Error raised when a modality type is not supported."""


class TokenKeyError(ValueError):
    """Error raised when a token key is not found in the tokens dictionary."""


class MultiprocessCodecManager:
    """Manager for loading and using codecs for different modalities.
    
    This version is safe to use with PyTorch multiprocessing DataLoaders.
    Unlike AION's original CodecManager, this does not use @lru_cache at
    the class level, instead caching per-instance to allow each worker
    process to maintain its own codec instances.
    """

    def __init__(self, device: str | torch.device = "cpu"):
        """Initialize the codec manager.

        Args:
            device: Device to load codecs on ('cpu', 'cuda', etc.)
        """
        self.device = device
        self._codec_cache = {}

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
        
        # Load codec from HuggingFace (bypasses global lru_cache)
        codec = codec_class.from_pretrained(HF_REPO_ID, modality=modality_type)
        codec = codec.eval()
        codec = codec.to(self.device)
        
        # Cache in this instance only
        self._codec_cache[modality_type] = codec
        
        return codec

    @torch.no_grad()
    def encode(self, *modalities: Modality) -> dict[str, torch.Tensor]:
        """Encode multiple modalities.

        Args:
            *modalities: Variable number of modality instances to encode

        Returns:
            Dictionary mapping token keys to encoded tensors
        """
        tokens = {}

        for modality in modalities:
            if not isinstance(modality, Modality):
                raise ModalityTypeError(
                    f"Modality {type(modality).__name__} does not have a token_key attribute"
                )
            
            # Get the appropriate codec
            codec = self._load_codec(type(modality))

            # Tokenize the modality
            tokenized = codec.encode(modality)

            tokens[modality.token_key] = tokenized

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

        # Get the appropriate codec
        codec = self._load_codec(modality_type)

        # Decode using the codec with any provided metadata
        decoded_modality = codec.decode(tokens[token_key], **metadata)

        # Cast decoded modality to the correct type
        decoded_modality = modality_type(**asdict(decoded_modality))

        return decoded_modality
