from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, List, Optional
import torch

class Modality(ABC):
    """Base class for all continuous data types."""
    generic_token: ClassVar[str] = ""
    specific_token: ClassVar[str] = ""

    @property
    @abstractmethod
    def positions(self) -> torch.Tensor:
        """Returns the positions (coordinates or indices) of the patches."""
        pass
