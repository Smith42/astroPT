import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinyFlow1DConfig:
    """Configuration for TinyFlow1D operating on patch tokens."""

    dim: int
    steps: int = 4
    hidden_dim: int = 128


class CouplingMLP(nn.Module):
    """RealNVP-style affine coupling over feature dimension for 1D tokens."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.split = dim // 2
        self.net = nn.Sequential(
            nn.Linear(self.split, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * (dim - self.split)),
        )

    def forward(
        self, x: torch.Tensor, reverse: bool = False, flip: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: [*, D]
        flip=False: x1 is identity, x2 is transformed.
        flip=True: x2 is identity, x1 is transformed.
        """
        if not flip:
            x1 = x[..., : self.split]
            x2 = x[..., self.split :]
            st = self.net(x1)
            s, t = st.chunk(2, dim=-1)
            s = torch.tanh(s) * 1.5
            if not reverse:
                y2 = x2 * torch.exp(s) + t
                y = torch.cat([x1, y2], dim=-1)
                logdet = s.sum(dim=-1)
            else:
                y2 = (x2 - t) * torch.exp(-s)
                y = torch.cat([x1, y2], dim=-1)
                logdet = -s.sum(dim=-1)
        else:
            x1 = x[..., : self.split]
            x2 = x[..., self.split :]
            st = self.net(x2)
            s, t = st.chunk(2, dim=-1)
            s = torch.tanh(s) * 1.5
            if not reverse:
                y1 = x1 * torch.exp(s) + t
                y = torch.cat([y1, x2], dim=-1)
                logdet = s.sum(dim=-1)
            else:
                y1 = (x1 - t) * torch.exp(-s)
                y = torch.cat([y1, x2], dim=-1)
                logdet = -s.sum(dim=-1)
        return y, logdet


class TinyFlow1D(nn.Module):
    """
    A small RealNVP-style normalizing flow over patch tokens.

    Operates on [B, T, D] where each token is a D-dimensional vector.
    The log-determinant is aggregated over all tokens to produce a
    per-sample scalar logdet [B].
    """

    def __init__(self, dim: int, steps: int = 4, hidden_dim: int = 128) -> None:
        super().__init__()
        self.dim = dim
        self.steps = steps
        self.blocks = nn.ModuleList(
            [
                CouplingMLP(dim, hidden_dim)
                for _ in range(steps)
            ]
        )

    def forward(
        self, x: torch.Tensor, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, D]
        Returns:
            z: [B, T, D]
            logdet: [B]
        """
        B, T, D = x.shape
        assert D == self.dim

        z = x
        logdet = x.new_zeros(B)

        if not reverse:
            for i, block in enumerate(self.blocks):
                flip = (i % 2) == 1
                z_flat = z.reshape(B * T, D)
                z_flat, ld_flat = block(z_flat, reverse=False, flip=flip)
                z = z_flat.view(B, T, D)
                ld = ld_flat.view(B, T).sum(dim=1)
                logdet = logdet + ld
        else:
            for i, block in reversed(list(enumerate(self.blocks))):
                flip = (i % 2) == 1
                z_flat = z.reshape(B * T, D)
                z_flat, ld_flat = block(z_flat, reverse=True, flip=flip)
                z = z_flat.view(B, T, D)
                ld = ld_flat.view(B, T).sum(dim=1)
                logdet = logdet + ld

        return z, logdet


class CouplingNet2D(nn.Module):
    """A small convolutional network that predicts scale (s) and shift (t) for 2D image flows."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, channels * 2, kernel_size=3, padding=1),  # Output has 2x channels for s and t
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        st = self.net(x)
        C = x.size(1)
        s, t = st[:, :C], st[:, C:]  # Split the output into scale and shift
        s = torch.tanh(s) * 1.5  # Bound the scale for numerical stability
        return s, t


class AffineCoupling2D(nn.Module):
    """
    An affine coupling layer for 2D images. It splits the input using a mask.
    One part is left unchanged (identity), and this part is used to predict
    the scale/shift that will transform the *other* part of the input.
    """

    def __init__(self, in_ch: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)  # A binary mask (e.g., checkerboard)
        self.net = CouplingNet2D(in_ch)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        x_id = x * self.mask  # The part that remains unchanged
        s, t = self.net(x_id)  # Predict s and t from the unchanged part

        if not reverse:  # Forward pass: x -> z
            # Transform the other part of x: y = x * scale + shift
            y = x_id + (1 - self.mask) * (x * torch.exp(s) + t)
            # The log-determinant is just the sum of the logs of the scale factors
            logdet = ((1 - self.mask) * s).flatten(1).sum(dim=1)
            return y, logdet
        else:  # Inverse pass: z -> x
            # The inverse is cheap to compute: x = (y - shift) / scale
            y = x_id + (1 - self.mask) * ((x - t) * torch.exp(-s))
            logdet = -((1 - self.mask) * s).flatten(1).sum(dim=1)
            return y, logdet


def checker_mask(C: int, H: int, W: int, flip: bool = False, device: str = "cpu") -> torch.Tensor:
    """Creates a checkerboard mask where half the channels are masked."""
    m = torch.zeros(1, C, H, W, device=device)
    m[:, ::2, :, :] = 1.0  # Mask even-indexed channels
    return 1.0 - m if flip else m


class TinyFlow2D(nn.Module):
    """
    A stack of Affine Coupling layers for 2D images.
    By alternating the mask between layers, we ensure all dimensions get transformed.
    
    Operates on [B, C, H, W] image tensors.
    """

    def __init__(self, in_ch: int, img_size: int, steps: int = 4):
        super().__init__()
        self.in_ch = in_ch
        self.img_size = img_size
        self.steps = steps
        self.blocks = nn.ModuleList(
            [
                AffineCoupling2D(
                    in_ch, checker_mask(in_ch, img_size, img_size, flip=(k % 2 == 1))
                )
                for k in range(steps)
            ]
        )

    def forward(self, x: torch.Tensor, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] image tensor
            reverse: If True, apply inverse transformation
        
        Returns:
            z: [B, C, H, W] transformed image
            logdet: [B] per-sample log-determinant
        """
        logdet = x.new_zeros(x.size(0))
        z = x

        # Apply the sequence of transformations
        if not reverse:
            for b in self.blocks:
                z, ld = b(z, reverse=False)
                logdet += ld
        else:  # Apply in reverse for the inverse pass
            for b in reversed(self.blocks):
                z, ld = b(z, reverse=True)
                logdet += ld
        return z, logdet


class GMMHead(nn.Module):
    """
    GMM head for continuous tokens.

    Takes Transformer hidden states [B, T, d_model] and predicts parameters
    of a Gaussian Mixture Model over token space of dimension D.
    """

    def __init__(self, d_model: int, d_token: int, K: int) -> None:
        super().__init__()
        self.K = K
        self.D = d_token
        self.proj = nn.Linear(d_model, K * (1 + 2 * d_token))

    def forward(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        h: [B, T, d_model]
        Returns:
            logits_pi: [B, T, K]
            mu: [B, T, K, D]
            log_sigma: [B, T, K, D]
        """
        B, T, _ = h.shape
        out = self.proj(h).view(B, T, self.K, 1 + 2 * self.D)
        logits_pi = out[..., 0]
        mu = out[..., 1 : 1 + self.D]
        log_sigma = out[..., 1 + self.D :]
        log_sigma = torch.clamp(log_sigma, -7, 2)
        return logits_pi, mu, log_sigma


def gmm_nll(
    y: torch.Tensor,
    logits_pi: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Negative log-likelihood of targets y under predicted GMM.

    Args:
        y: [B, T, D] target tokens (e.g. latent z at positions 1..T-1)
        logits_pi: [B, T, K]
        mu: [B, T, K, D]
        log_sigma: [B, T, K, D]

    Returns:
        nll: [B] per-sample negative log-likelihood summed over tokens.
    """
    B, T, D = y.shape
    K = logits_pi.size(-1)

    y_exp = y.unsqueeze(2)  # [B, T, 1, D]
    inv_var = torch.exp(-2 * log_sigma)
    logp = (
        -0.5 * ((y_exp - mu) ** 2 * inv_var).sum(dim=-1)
        - log_sigma.sum(dim=-1)
        - 0.5 * D * math.log(2 * math.pi)
    )  # [B, T, K]

    logmix = F.log_softmax(logits_pi, dim=-1) + logp  # [B, T, K]
    loglik = torch.logsumexp(logmix, dim=-1).sum(dim=-1)  # [B]
    return -loglik


def uniform_dequantize(x: torch.Tensor) -> torch.Tensor:
    """
    Takes a tensor of pixel values (scaled 0-1) and makes them continuous.
    It adds a tiny amount of uniform noise, breaking the discrete nature of pixel values.
    This is a crucial step for training continuous models like normalizing flows.
    
    Args:
        x: Tensor in [0, 1] range (can be uint8 or float)
    
    Returns:
        Dequantized tensor in [0, 1] range with uniform noise added.
    """
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    return (x + torch.rand_like(x) / 256.0).clamp(0.0, 1.0)


def patchify(x: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    Converts a batch of images into a sequence of flattened patches (tokens).
    It slices the image into a grid and then flattens each patch.
    
    Args:
        x: [B, C, H, W] image tensor
        patch_size: Size of each patch (default 16)
    
    Returns:
        tokens: [B, N, D] where N = (H//patch_size) * (W//patch_size),
               D = C * patch_size * patch_size
    """
    B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0, (
        f"Image dimensions must be divisible by the patch size. "
        f"Got H={H}, W={W}, patch_size={patch_size}"
    )
    
    # Use 'unfold' to create sliding blocks (patches) across height and width
    x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    
    # Reshape and flatten to get the final sequence of tokens
    x = x.contiguous().permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * patch_size * patch_size)
    return x


def depatchify(tokens: torch.Tensor, C: int = 3, H: int = 256, W: int = 256, patch_size: int = 16) -> torch.Tensor:
    """
    The exact inverse of the 'patchify' function.
    Converts a sequence of tokens back into an image format.
    
    Args:
        tokens: [B, N, D] where N = (H//patch_size) * (W//patch_size),
               D = C * patch_size * patch_size
        C: Number of channels
        H: Image height
        W: Image width
        patch_size: Size of each patch (default 16)
    
    Returns:
        x: [B, C, H, W] image tensor
    """
    B, N, D = tokens.shape
    hp, wp = H // patch_size, W // patch_size  # Number of patches along height and width
    
    # Reshape the sequence back into a grid of patches
    x = tokens.reshape(B, hp, wp, C, patch_size, patch_size)
    
    # Permute and reshape to reconstruct the final image
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
    return x


