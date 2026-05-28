from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import torch
import numpy as np
from numpy.typing import NDArray
import einops
import torch.nn.functional as F

from torchvision import transforms
import random

from astropt.modalities import Modality, EuclidImage, DESISpectrum

class ModalityProcessor(ABC):
    """
    Base interface for all modality processors.
    """
    
    @abstractmethod
    def required_columns(self) -> List[str]:
        """Returns the list of columns needed to be extracted from Arrow."""
        pass

    @abstractmethod
    def is_available(self, item: Dict[str, Any]) -> bool:
        """Checks if valid data for the modality exists in this particular sample."""
        pass

    @abstractmethod
    def process(self, item: Dict[str, Any], transform: Dict[str, Any], config: Any = None) -> Optional[Modality]:
        """Converts raw Arrow data into a Modality object."""
        pass
