from typing import ClassVar
import torch
import torch.nn as nn
import torch.nn.functional as F
from aion.modalities import Image

# Standard Euclid bands: VIS (optical) + Y, J, H (Near-Infrared)
EUCLID_BANDS = ["EUCLID-VIS", "EUCLID-Y", "EUCLID-J", "EUCLID-H"]
HSC_BANDS = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"]

class EuclidImage(Image):
    """Euclid image modality data representation for AION."""
    token_key: ClassVar[str] = "tok_image_euclid" 
    num_tokens: ClassVar[int] = 576
    
class ResBlock(nn.Module):
    """Robust Residual Block with GroupNorm for astronomical imagery."""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, dim)
        )

    def forward(self, x):
        return x + self.block(x)

class EuclidToHSC_ResNet(nn.Module):
    """Adapter to translate Euclid imagery (4 bands) to HSC format (5 bands)."""
    def __init__(self, hidden_dim: int = 128, num_blocks: int = 4):
        super().__init__()
        self.in_proj = nn.Conv2d(4, hidden_dim, kernel_size=3, padding=1)
        
        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_dim) for _ in range(num_blocks)]
        )
        
        self.out_proj = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden_dim, 5, kernel_size=3, padding=1)
        )
        nn.init.normal_(self.out_proj[1].weight, std=0.001)
        if self.out_proj[1].bias is not None:
            nn.init.zeros_(self.out_proj[1].bias)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.res_blocks(x)
        x = self.out_proj(x)
        return torch.clamp(x, min=-15.0, max=15.0)

class HSCToEuclid_ResNet(nn.Module):
    """Inverse adapter from HSC (5 bands) back to Euclid (4 bands) for cycle loss."""
    def __init__(self, hidden_dim: int = 128, num_blocks: int = 4):
        super().__init__()
        self.in_proj = nn.Conv2d(5, hidden_dim, kernel_size=3, padding=1)
        
        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_dim) for _ in range(num_blocks)]
        )
        
        self.out_proj = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden_dim, 4, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.in_proj(x)
        x = self.res_blocks(x)
        return self.out_proj(x)