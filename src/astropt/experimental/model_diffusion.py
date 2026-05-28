"""
Conditional Denoising Diffusion Probabilistic Model (DDPM) for 1D Spectrum Generation.

Generates astronomical spectra (flux, length 7781) conditioned on image embeddings
extracted from a foundation model (AstroPT). The conditioning vector (e.g. the [CLS]
token of AstroPT, dim 512 or 768) guides generation via Feature-wise Linear Modulation
(FiLM) applied inside every residual block of a lightweight 1D U-Net.

Architecture overview
---------------------
SpectrumDiffusionModel
  ├── LinearNoiseScheduler   — DDPM β-schedule and forward/reverse process helpers
  └── SpectrumUNet1D         — Encoder–Decoder backbone
        ├── DownBlock1D × 4  — Spatial reduction  (AvgPool1d)
        ├── MidBlock1D       — Bottleneck (optional self-attention)
        └── UpBlock1D   × 4  — Spatial expansion  (Upsample + Conv1d)
        Each block contains ResidualBlock1D with FiLM conditioning.

References
----------
[1] Ho et al., Denoising Diffusion Probabilistic Models, NeurIPS 2020.
[2] Perez et al., FiLM: Visual Reasoning with a General Conditioning Layer, AAAI 2018.
[3] Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation, 2015.

Author: Antigravity (Senior ML Research Engineer)
Date: May 2026
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DiffusionModelConfig:
    """Hyperparameters for the spectrum diffusion model."""

    # Spectrum geometry
    spectrum_length: int = 7781          # Raw DESI spectrum length
    spectrum_channels: int = 1           # Flux is single-channel

    # Conditioning
    context_dim: int = 512               # Image embedding dimension (AstroPT n_embd)
    time_emb_dim: int = 256              # Sinusoidal timestep embedding dimension
    redshift_emb_dim: int = 64           # Sinusoidal redshift embedding dimension
    use_redshift: bool = True            # Enable joint conditioning on redshift

    # U-Net channel progression (encoder side; decoder mirrors)
    channel_dims: tuple[int, ...] = (64, 128, 256, 512)
    num_groups: int = 8                  # Groups for GroupNorm

    # Noise schedule
    num_timesteps: int = 1000            # Diffusion steps T
    beta_start: float = 1e-4             # β₁
    beta_end: float = 0.02               # β_T

    # Optional self-attention in mid block (can be disabled for memory)
    use_mid_attention: bool = True


# ---------------------------------------------------------------------------
# Noise Schedule
# ---------------------------------------------------------------------------

class LinearNoiseScheduler(nn.Module):
    """
    DDPM linear variance schedule.

    Pre-computes all scalar quantities needed for the forward diffusion
    process q(xₜ | x₀) and the reverse sampling step p(xₜ₋₁ | xₜ).
    """

    def __init__(self, num_timesteps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        super().__init__()
        self.num_timesteps = num_timesteps

        # Linear β schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alpha_cum = torch.cumprod(alphas, dim=0)
        alpha_cum_prev = F.pad(alpha_cum[:-1], (1, 0), value=1.0)

        # Register as buffers (moved to device with the module, but not trained)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alpha_cum", alpha_cum.float())
        self.register_buffer("sqrt_alpha_cum", torch.sqrt(alpha_cum).float())
        self.register_buffer("sqrt_one_minus_alpha_cum", torch.sqrt(1.0 - alpha_cum).float())

        # Posterior variance for reverse process
        posterior_var = betas * (1.0 - alpha_cum_prev) / (1.0 - alpha_cum)
        self.register_buffer("posterior_variance", posterior_var.float())
        self.register_buffer("posterior_log_variance_clipped",
                             torch.log(torch.clamp(posterior_var, min=1e-20)).float())

        # Coefficients for computing x₀ from xₜ and predicted noise
        self.register_buffer("sqrt_recip_alpha", (1.0 / torch.sqrt(alphas)).float())
        self.register_buffer("beta_over_sqrt_one_minus_alpha_cum",
                             (betas / torch.sqrt(1.0 - alpha_cum)).float())

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor,
                  noise: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: q(xₜ | x₀) = √ᾱₜ · x₀ + √(1 - ᾱₜ) · ε.

        Args:
            x0: Clean data [B, C, L].
            t:  Timestep indices [B] (long tensor, values in [0, T-1]).
            noise: Optional pre-sampled noise; generated if None.

        Returns:
            (xt, noise): Noised data and the noise that was added.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # Gather coefficients for each sample in the batch
        sqrt_alpha = self.sqrt_alpha_cum[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cum[t].view(-1, 1, 1)

        xt = sqrt_alpha * x0 + sqrt_one_minus * noise
        return xt, noise

    @torch.no_grad()
    def sample_prev_timestep(self, xt: torch.Tensor, noise_pred: torch.Tensor,
                             t: int) -> torch.Tensor:
        """
        Reverse sampling step: compute xₜ₋₁ from xₜ and predicted noise.

        For t > 0 we add stochastic noise scaled by the posterior variance.
        For t == 0 we return the deterministic mean.
        """
        coeff = self.beta_over_sqrt_one_minus_alpha_cum[t]
        mean = self.sqrt_recip_alpha[t] * (xt - coeff * noise_pred)

        if t == 0:
            return mean

        variance = self.posterior_variance[t]
        z = torch.randn_like(xt)
        return mean + torch.sqrt(variance) * z


# ---------------------------------------------------------------------------
# Timestep Embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps a scalar timestep to a high-dimensional vector via sinusoidal
    positional encoding, followed by a two-layer MLP with SiLU activation.
    """

    def __init__(self, time_emb_dim: int = 256):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Integer timesteps [B] (long or float).

        Returns:
            Embedding [B, time_emb_dim].
        """
        half_dim = self.time_emb_dim // 2
        emb_scale = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, time_emb_dim]
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# Redshift Embedding
# ---------------------------------------------------------------------------

class SinusoidalRedshiftEmbedding(nn.Module):
    """
    Maps a continuous redshift z to a high-dimensional vector via sinusoidal
    positional encoding, followed by a two-layer MLP with SiLU activation.
    """

    def __init__(self, redshift_emb_dim: int = 64):
        super().__init__()
        self.redshift_emb_dim = redshift_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(redshift_emb_dim, redshift_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(redshift_emb_dim * 4, redshift_emb_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Float tensor [B] or [B, 1] containing redshifts.

        Returns:
            Embedding [B, redshift_emb_dim].
        """
        if z.ndim == 2:
            z = z.squeeze(-1)
        half_dim = self.redshift_emb_dim // 2
        emb_scale = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=z.device, dtype=torch.float32) * -emb_scale)
        emb = z.float().unsqueeze(1) * emb.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, redshift_emb_dim]
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# FiLM Conditioning
# ---------------------------------------------------------------------------

class FiLMConditioningBlock(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).

    Projects the concatenation of the timestep embedding and the context
    (image) embedding into per-channel scale (γ) and shift (β) parameters.
    Applied after GroupNorm inside each residual block.

    FiLM(x) = γ · x + β
    """

    def __init__(self, cond_dim: int, out_channels: int):
        """
        Args:
            cond_dim:     Dimension of the conditioning vector (time_emb_dim + context_dim).
            out_channels: Number of feature channels to modulate.
        """
        super().__init__()
        self.projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    Feature map [B, C, L].
            cond: Conditioning vector [B, cond_dim].

        Returns:
            Modulated feature map [B, C, L].
        """
        gamma_beta = self.projection(cond)  # [B, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each [B, C]
        gamma = gamma.unsqueeze(-1)  # [B, C, 1]
        beta = beta.unsqueeze(-1)    # [B, C, 1]
        return x * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------

class ResidualBlock1D(nn.Module):
    """
    Core building block for the 1D U-Net.

    Structure:
        Conv1d → GroupNorm → SiLU → Conv1d → GroupNorm → FiLM → SiLU → Residual Add

    The FiLM layer injects timestep and context information after the
    second GroupNorm, enabling the block to adapt its behaviour to both
    the diffusion step and the galaxy morphology embedding.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 cond_dim: int, num_groups: int = 8):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)

        self.film = FiLMConditioningBlock(cond_dim, out_channels)
        self.act = nn.SiLU()

        # Learnable skip projection when channel dims change
        if in_channels != out_channels:
            self.skip_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    Input features [B, C_in, L].
            cond: Conditioning vector [B, cond_dim].

        Returns:
            Output features [B, C_out, L].
        """
        residual = self.skip_proj(x)

        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        h = self.film(h, cond)
        h = self.act(h)

        return h + residual


# ---------------------------------------------------------------------------
# Self-Attention for Bottleneck
# ---------------------------------------------------------------------------

class SelfAttention1D(nn.Module):
    """
    Multi-head self-attention operating on the spatial (sequence) dimension
    of a 1D feature map. Used in the bottleneck to capture long-range
    dependencies across the compressed spectrum representation.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(1, channels)  # Instance norm style
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map [B, C, L].

        Returns:
            Attended feature map [B, C, L].
        """
        B, C, L = x.shape
        residual = x
        x = self.norm(x)

        qkv = self.qkv(x)  # [B, 3C, L]
        q, k, v = qkv.chunk(3, dim=1)

        head_dim = C // self.num_heads
        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, head_dim, L)
        k = k.view(B, self.num_heads, head_dim, L)
        v = v.view(B, self.num_heads, head_dim, L)

        # Scaled dot-product attention via Flash Attention kernel
        # PyTorch expects [B, heads, seq_len, head_dim]
        q = q.permute(0, 1, 3, 2)  # [B, heads, L, head_dim]
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)  # [B, heads, L, head_dim]
        attn_out = attn_out.permute(0, 1, 3, 2).reshape(B, C, L)  # [B, C, L]

        return residual + self.proj_out(attn_out)


# ---------------------------------------------------------------------------
# Down / Mid / Up Blocks
# ---------------------------------------------------------------------------

class DownBlock1D(nn.Module):
    """Encoder block: two residual blocks followed by spatial downsampling."""

    def __init__(self, in_channels: int, out_channels: int,
                 cond_dim: int, num_groups: int = 8):
        super().__init__()
        self.res1 = ResidualBlock1D(in_channels, out_channels, cond_dim, num_groups)
        self.res2 = ResidualBlock1D(out_channels, out_channels, cond_dim, num_groups)
        self.downsample = nn.Sequential(
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (downsampled, skip): The downsampled output and the pre-downsample
            skip connection for the decoder.
        """
        h = self.res1(x, cond)
        h = self.res2(h, cond)
        skip = h
        h = self.downsample(h)
        return h, skip


class MidBlock1D(nn.Module):
    """
    Bottleneck block: two residual blocks with an optional self-attention
    layer in between for global context modelling.
    """

    def __init__(self, channels: int, cond_dim: int,
                 num_groups: int = 8, use_attention: bool = True):
        super().__init__()
        self.res1 = ResidualBlock1D(channels, channels, cond_dim, num_groups)
        self.use_attention = use_attention
        if use_attention:
            self.attn = SelfAttention1D(channels)
        self.res2 = ResidualBlock1D(channels, channels, cond_dim, num_groups)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.res1(x, cond)
        if self.use_attention:
            h = self.attn(h)
        h = self.res2(h, cond)
        return h


class UpBlock1D(nn.Module):
    """Decoder block: upsample, concatenate skip connection, two residual blocks."""

    def __init__(self, in_channels: int, out_channels: int,
                 skip_channels: int, cond_dim: int, num_groups: int = 8):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1),
        )
        # After concatenation with skip, input channels double
        self.res1 = ResidualBlock1D(in_channels + skip_channels, out_channels, cond_dim, num_groups)
        self.res2 = ResidualBlock1D(out_channels, out_channels, cond_dim, num_groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        h = self.upsample(x)

        # Handle length mismatch from rounding during downsampling
        if h.shape[-1] != skip.shape[-1]:
            h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))

        h = torch.cat([h, skip], dim=1)
        h = self.res1(h, cond)
        h = self.res2(h, cond)
        return h


# ---------------------------------------------------------------------------
# Full U-Net
# ---------------------------------------------------------------------------

class SpectrumUNet1D(nn.Module):
    """
    Lightweight 1D U-Net for conditional spectrum denoising.

    The network predicts the noise ε added to a spectrum at diffusion step t,
    conditioned on an image embedding (context) and the timestep.

    Input shape:  [B, 1, spectrum_length]
    Output shape: [B, 1, spectrum_length]

    Internally the spectrum is padded to the smallest multiple of 2^(num_levels)
    that is ≥ spectrum_length (rounding UP to avoid information loss).
    AstroPT uses patch_size=10 for spectra, so the padded length is also
    chosen to be compatible with multiples of 10 when possible.
    """

    def __init__(self, config: DiffusionModelConfig):
        super().__init__()
        self.config = config
        self.spectrum_length = config.spectrum_length
        dims = config.channel_dims
        num_levels = len(dims)
        
        if config.use_redshift:
            cond_dim = config.time_emb_dim + config.context_dim + config.redshift_emb_dim
        else:
            cond_dim = config.time_emb_dim + config.context_dim

        # Compute the padded length (smallest multiple of 2^num_levels >= spectrum_length)
        divisor = 2 ** num_levels
        self.padded_length = math.ceil(config.spectrum_length / divisor) * divisor

        # Timestep and context embeddings
        self.time_emb = SinusoidalTimeEmbedding(config.time_emb_dim)
        self.context_proj = nn.Sequential(
            nn.Linear(config.context_dim, config.time_emb_dim),
            nn.SiLU(),
            nn.Linear(config.time_emb_dim, config.context_dim),
        )
        
        if config.use_redshift:
            self.redshift_emb = SinusoidalRedshiftEmbedding(config.redshift_emb_dim)

        # Input projection
        self.conv_in = nn.Conv1d(config.spectrum_channels, dims[0], kernel_size=3, padding=1)

        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        ch_in = dims[0]
        for i in range(num_levels):
            ch_out = dims[i]
            self.down_blocks.append(
                DownBlock1D(ch_in, ch_out, cond_dim, config.num_groups)
            )
            ch_in = ch_out

        # Bottleneck
        self.mid_block = MidBlock1D(
            dims[-1], cond_dim, config.num_groups,
            use_attention=config.use_mid_attention,
        )

        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(num_levels)):
            ch_out = dims[i] if i > 0 else dims[0]
            skip_ch = dims[i]
            self.up_blocks.append(
                UpBlock1D(ch_in, ch_out, skip_ch, cond_dim, config.num_groups)
            )
            ch_in = ch_out

        # Output projection
        self.conv_out = nn.Sequential(
            nn.GroupNorm(config.num_groups, dims[0]),
            nn.SiLU(),
            nn.Conv1d(dims[0], config.spectrum_channels, kernel_size=3, padding=1),
        )

        # Weight initialisation
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Kaiming-uniform for Conv/Linear, zeros for biases."""
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, a=0, mode="fan_in", nonlinearity="linear")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                context: torch.Tensor, redshift: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict the noise ε for a noisy spectrum xₜ.

        Args:
            x:        Noisy spectrum [B, 1, spectrum_length].
            t:        Timestep indices [B].
            context:  Image embedding  [B, context_dim].
            redshift: Redshift tensor  [B] or [B, 1].

        Returns:
            Predicted noise [B, 1, spectrum_length].
        """
        # Pad input to internal resolution using reflect mode to avoid an
        # artificial discontinuity at the red edge of the DESI spectrum.
        # Zero-padding would create a hard jump that the network could learn
        # to exploit, contaminating the real border pixels.
        pad_amount = self.padded_length - self.spectrum_length
        if pad_amount > 0:
            x = F.pad(x, (0, pad_amount), mode="reflect")

        # Build conditioning vector
        t_emb = self.time_emb(t)                  # [B, time_emb_dim]
        ctx = self.context_proj(context)           # [B, context_dim]
        
        if self.config.use_redshift and redshift is not None:
            z_emb = self.redshift_emb(redshift)   # [B, redshift_emb_dim]
            cond = torch.cat([t_emb, ctx, z_emb], dim=-1)     # [B, cond_dim]
        else:
            cond = torch.cat([t_emb, ctx], dim=-1)     # [B, cond_dim]

        # Encoder
        h = self.conv_in(x)
        skips = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, cond)
            skips.append(skip)

        # Bottleneck
        h = self.mid_block(h, cond)

        # Decoder
        for up_block in self.up_blocks:
            skip = skips.pop()
            h = up_block(h, skip, cond)

        # Output
        h = self.conv_out(h)

        # Remove padding
        if pad_amount > 0:
            h = h[..., :self.spectrum_length]

        return h


# ---------------------------------------------------------------------------
# Top-Level Diffusion Model
# ---------------------------------------------------------------------------

class SpectrumDiffusionModel(nn.Module):
    """
    Conditional DDPM for astronomical spectrum generation.

    Combines the U-Net denoiser with the linear noise scheduler to provide
    a clean training and inference interface.

    Training:
        loss = model(spectra, context)  # MSE on predicted noise

    Inference:
        generated = model.sample(context, device)  # Full reverse process
    """

    def __init__(self, config: DiffusionModelConfig):
        super().__init__()
        self.config = config
        self.unet = SpectrumUNet1D(config)
        self.scheduler = LinearNoiseScheduler(
            num_timesteps=config.num_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        )

    def forward(self, x0: torch.Tensor, context: torch.Tensor, redshift: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Training forward pass.

        Samples a random timestep t for each sample in the batch, adds noise,
        predicts the noise with the U-Net, and returns the MSE loss.

        Args:
            x0:       Clean spectra [B, 1, 7781].
            context:  Image embeddings [B, context_dim].
            redshift: Redshift tensor [B] or [B, 1] (optional).

        Returns:
            Scalar MSE loss between predicted and true noise.
        """
        B = x0.shape[0]
        device = x0.device

        # Sample random timesteps uniformly for each batch element
        t = torch.randint(0, self.config.num_timesteps, (B,), device=device, dtype=torch.long)

        # Forward diffusion: add noise
        xt, noise = self.scheduler.add_noise(x0, t)

        # Predict noise
        noise_pred = self.unet(xt, t, context, redshift)

        # MSE loss on noise prediction (ε-prediction objective)
        loss = F.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, context: torch.Tensor,
               redshift: Optional[torch.Tensor] = None,
               device: Optional[torch.device] = None,
               return_intermediates: bool = False) -> torch.Tensor:
        """
        Generate spectra via the full DDPM reverse process.

        Starting from pure Gaussian noise, iteratively denoise for T steps
        conditioned on the provided image embeddings and redshift.

        Args:
            context:              Image embeddings [B, context_dim].
            redshift:             Redshift tensor [B] or [B, 1] (optional).
            device:               Target device (optional, defaults to context device).
            return_intermediates: If True, returns all intermediate xₜ as well.

        Returns:
            Generated spectra [B, 1, 7781].
            If return_intermediates is True, returns (final, list_of_intermediates).
        """
        B = context.shape[0]
        spectrum_length = self.config.spectrum_length
        if device is None:
            device = context.device

        # Start from pure noise
        xt = torch.randn(B, self.config.spectrum_channels, spectrum_length, device=device)
        intermediates = []

        # Reverse diffusion loop: T-1 → 0
        for t in reversed(range(self.config.num_timesteps)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = self.unet(xt, t_tensor, context, redshift)
            xt = self.scheduler.sample_prev_timestep(xt, noise_pred, t)

            if return_intermediates and t % 100 == 0:
                intermediates.append(xt.clone())

        if return_intermediates:
            return xt, intermediates
        return xt

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
