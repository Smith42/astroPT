"""
train_jetformer_ddp.py

JetFormer (Flow + AR Transformer + GMM) training script with DDP + HF streaming dataset.

Key edits in THIS version (requested):
1) Noise curriculum is tied DIRECTLY to (step, max_iters) and goes to ~0 at max_iters.
   - Image/RGB noise uses paper-style sigma in [0,255] (default 64 -> 0).
   - Latent z noise uses paper-style std (default 0.3 -> 0).
2) Removed the constant latent noise (z += N(0, 0.3)) and replaced it with a decaying schedule.
3) Forward signature changed to: forward(x, step, max_iters)
4) Fixed a few structural/indent issues in the original paste (HF shard indent, ViTFlow.forward indent, etc.)

Notes:
- This keeps your architecture and training logic intact, only changing noise scheduling + small code fixes.
- If you want "almost zero but not exact" at the end, set CFG.noise_floor = 1e-6.
"""

import math
import os
import csv
import time
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

# --- PIL Fix for Truncated Images ---
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm

# Hugging Face Datasets
from datasets import load_dataset

# This must be done BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np


# ======================================================================================
# Block 1: DDP Setup & Configuration
# ======================================================================================
def setup_ddp():
    """Initializes the distributed process group."""
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(backend="nccl")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


@dataclass
class CFG:
    # --- Model Config (Scaled for RTX 4090 24GB) ---
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12

    # --- AstroPT Specific Configs ---
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = False
    is_causal: bool = True

    # --- Flow Specification ---
    flow_steps: int = 16

    # --- Training Config ---
    max_iters: int = 80_000
    save_interval: int = 5000
    batch_size: int = 8
    val_check_interval: int = 5000

    # --- Optimizer Config ---
    lr: float = 1e-4
    wd: float = 1e-4
    beta2: float = 0.95
    warmup_steps: int = 10000

    # --- Data Params ---
    img_size: int = 256
    patch: int = 8
    in_ch: int = 3

    # Derived Dimensions
    n_tokens: int = (img_size // patch) ** 2
    d_token: int = in_ch * patch * patch

    # --- GMM Head ---
    gmm_K: int = 256

    # --- Noise curriculum (paper-style, tied to max_iters) ---
    # JetFormer paper uses σ0 = 64 in pixel space [0,255] (≈ 0.251 in [0,1]).
    rgb_sigma0_255: float = 64.0   # start noise in [0,255]
    rgb_sigmaT_255: float = 0.0    # final noise at max_iters (0 => sharpest end)

    # Latent noise in flow token space (paper mentions std=0.3)
    z_sigma0: float = 0.3          # start latent noise
    z_sigmaT: float = 0.0          # final latent noise at max_iters

    # If 1.0: reaches final exactly at max_iters.
    # If <1.0: reaches final earlier and stays there.
    noise_decay_frac: float = 1.0

    # Optional: set to 1e-6 to avoid EXACT zero (sometimes smoother numerically)
    noise_floor: float = 0.0

    # --- System ---
    grad_clip_val: float = 0.5

    # Paths
    dataset_name: str = "final_sogol_image_patch_8"
    checkpoint_path: str = ""
    samples_dir: str = ""
    loss_csv_path: str = ""
    loss_plot_path: str = ""

    # --- Data Sources (Hugging Face) ---
    hf_repo: str = "Smith42/galaxies"
    val_steps: int = 100

    # DDP Placeholders
    rank: int = 0
    world_size: int = 1
    device: str = "cuda"


# ======================================================================================
# Block 2: Logging Utilities (Rank 0 Only)
# ======================================================================================
def append_losses_to_csv(step, train_loss, val_loss, filename):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['step', 'train_loss', 'val_loss'])
        writer.writerow([step, train_loss, val_loss])


def plot_loss_from_csv(csv_path, output_path):
    if not os.path.isfile(csv_path):
        return
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['step'], df['train_loss'], label='Train Loss', color='blue')

    df_val = df.dropna(subset=['val_loss'])
    if not df_val.empty:
        ax.plot(
            df_val['step'], df_val['val_loss'],
            label='Validation Loss', color='orange',
            linestyle='--', marker='o'
        )

    ax.set_title('Training and Validation Loss per Step')
    ax.set_xlabel('Step')
    ax.set_ylabel('Average Loss')
    ax.legend()
    ax.grid(True)
    fig.savefig(output_path)
    plt.close(fig)


# ======================================================================================
# Block 3: Data Loading
# ======================================================================================
def process_hf_item(item):
    img = item['image_crop']
    to_tensor = transforms.ToTensor()
    img_t = to_tensor(img)
    if img_t.shape[0] == 1:
        img_t = img_t.repeat(3, 1, 1)
    return {"img": img_t}


def get_train_dataloader(cfg: CFG):
    if cfg.rank == 0:
        print(f"Loading streaming dataset: {cfg.hf_repo} (Split: train)")
    ds = load_dataset(cfg.hf_repo, split="train", streaming=True)

    # shard across ranks so each GPU sees a different stream
    if cfg.world_size > 1:
        ds = ds.shard(num_shards=cfg.world_size, index=cfg.rank)

    ds = ds.map(process_hf_item, remove_columns=["image", "image_crop", "survey", "ra", "dec"])

    nw = min(6, max(2, (os.cpu_count() // max(cfg.world_size, 1)) - 1))
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        num_workers=nw,
        pin_memory=True,
    )


def get_val_dataloader(cfg: CFG):
    if cfg.rank == 0:
        print(f"Loading streaming dataset: {cfg.hf_repo} (Split: test)")
    ds = load_dataset(cfg.hf_repo, split="test", streaming=True)

    # No validation sharding to prevent empty shards on small val sets
    ds = ds.map(process_hf_item, remove_columns=["image", "image_crop", "survey", "ra", "dec"])

    nw = min(4, max(2, (os.cpu_count() // max(cfg.world_size, 1)) - 1))
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        num_workers=nw,
        pin_memory=True,
    )


# ======================================================================================
# Block 4: Checkpointing (Rank 0 Only)
# ======================================================================================
def save_checkpoint(step, model, optimizer, cfg, is_latest=True):
    """
    Saves the checkpoint.
    1) Always overwrites 'checkpoint_latest.pt' for easy resuming.
    2) If is_latest=False, saves a numbered file like 'checkpoint_step_005000.pt'.
    """
    if cfg.rank != 0:
        return

    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()

    checkpoint = {
        'step': step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict()
    }

    latest_path = os.path.join(cfg.samples_dir, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)

    if not is_latest:
        history_path = os.path.join(cfg.samples_dir, f"checkpoint_step_{step:07d}.pt")
        torch.save(checkpoint, history_path)
        print(f"Saved historical checkpoint: {history_path}")
    else:
        print(f"Updated latest checkpoint: {latest_path}")


def load_checkpoint(model, optimizer, cfg):
    latest_path = os.path.join(cfg.samples_dir, "checkpoint_latest.pt")

    if not os.path.exists(latest_path):
        if cfg.rank == 0:
            print(f"No checkpoint found at {latest_path}. Starting from scratch.")
        return 0

    map_location = {'cuda:%d' % 0: 'cuda:%d' % cfg.rank}
    checkpoint = torch.load(latest_path, map_location=map_location)

    model_unwrap = model.module if isinstance(model, DDP) else model
    model_unwrap.load_state_dict(checkpoint['model_state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']

    if cfg.rank == 0:
        print(f"Checkpoint loaded from {latest_path}. Resuming from step {step}")
    return step


# ======================================================================================
# Block 5: Model Definitions
# ======================================================================================
def uniform_dequantize(x: torch.Tensor) -> torch.Tensor:
    # Standard dequantization for 8-bit images
    return (x + torch.rand_like(x) / 256.0).clamp(0.0, 1.0)


def patchify(x: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    B, C, H, W = x.shape
    x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    x = x.contiguous().permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * patch_size * patch_size)
    return x


def depatchify(tokens: torch.Tensor, C: int = 3, H: int = 256, W: int = 256, patch_size: int = 16) -> torch.Tensor:
    B, N, D = tokens.shape
    hp, wp = H // patch_size, W // patch_size
    x = tokens.reshape(B, hp, wp, C, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
    return x


def cosine_decay(step: int, T: int, start: float, end: float) -> float:
    """
    Cosine decay from start -> end over steps [0, T].
    Returns exactly end for step >= T.
    """
    if T <= 0:
        return end
    if step >= T:
        return end
    x = step / T  # in [0,1)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * x))


class ViTCouplingBlock(nn.Module):
    def __init__(self, in_channels: int, n_tokens: int, width: int = 512, depth: int = 4, heads: int = 8):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, width)
        self.pos_emb = nn.Parameter(torch.randn(1, n_tokens, width) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=2048, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_proj = nn.Linear(width, in_channels * 2)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.in_proj(x) + self.pos_emb
        h = self.transformer(h)
        st = self.out_proj(h)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s)
        return s, t


class ViTAffineCoupling(nn.Module):
    def __init__(self, d_token: int, n_tokens: int):
        super().__init__()
        self.half_d = d_token // 2
        self.register_buffer('perm', torch.randperm(d_token))
        self.register_buffer('inv_perm', torch.argsort(self.perm))
        self.net = ViTCouplingBlock(self.half_d, n_tokens)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if not reverse:
            x = x[..., self.perm]
            x_a, x_b = x[..., :self.half_d], x[..., self.half_d:]
            s, t = self.net(x_a)
            y_b = x_b * torch.exp(s) + t
            y = torch.cat([x_a, y_b], dim=-1)
            logdet = s.sum(dim=(1, 2))
            return y, logdet
        else:
            x_a, x_b = x[..., :self.half_d], x[..., self.half_d:]
            s, t = self.net(x_a)
            y_b = (x_b - t) * torch.exp(-s)
            y = torch.cat([x_a, y_b], dim=-1)
            y = y[..., self.inv_perm]
            logdet = -s.sum(dim=(1, 2))
            return y, logdet


class ViTFlow(nn.Module):
    def __init__(self, d_token: int, n_tokens: int, steps: int = 32):
        super().__init__()
        self.blocks = nn.ModuleList([ViTAffineCoupling(d_token, n_tokens) for _ in range(steps)])

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet = x.new_zeros(x.size(0))
        z = x
        if not reverse:
            for b in self.blocks:
                z, ld = b(z, reverse=False)
                logdet += ld
        else:
            for b in reversed(self.blocks):
                z, ld = b(z, reverse=True)
                logdet += ld
        return z, logdet


def compute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    if freqs_cis.dtype not in (torch.complex64, torch.complex128):
        if freqs_cis.dim() == 2:
            freqs_cis = freqs_cis.view(*freqs_cis.shape[:-1], -1, 2)
        freqs_cis = torch.view_as_complex(freqs_cis)
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x_c.size(-1))
    x_out = torch.view_as_real(x_c * freqs_cis).flatten(3)
    return x_out.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class GemmaMLP(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.hidden_dim = 4 * cfg.d_model
        self.gate_proj = nn.Linear(cfg.d_model, self.hidden_dim, bias=cfg.bias)
        self.up_proj = nn.Linear(cfg.d_model, self.hidden_dim, bias=cfg.bias)
        self.down_proj = nn.Linear(self.hidden_dim, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


class GemmaAttention(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.head_dim = cfg.d_model // cfg.n_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)

        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_heads
        self.dropout = cfg.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self.register_buffer("freqs_cis", compute_freqs_cis(self.head_dim, cfg.block_size), persistent=False)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim)

        freqs_cis = self.freqs_cis[:T]
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))
        return y


class GemmaBlock(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.ln_1 = RMSNorm(cfg.d_model)
        self.attn = GemmaAttention(cfg)
        self.ln_2 = RMSNorm(cfg.d_model)
        self.mlp = GemmaMLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class AstroPTBackbone(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList([GemmaBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.d_model)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.drop(x)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return x


class GMMHead(nn.Module):
    def __init__(self, d_model: int, d_token: int, K: int):
        super().__init__()
        self.K, self.D = K, d_token
        self.proj = nn.Linear(d_model, K * (1 + 2 * d_token))

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = h.shape
        out = self.proj(h).view(B, N, self.K, 1 + 2 * self.D)

        logits_pi = out[..., 0]
        mu = out[..., 1:1 + self.D]
        log_sigma = out[..., 1 + self.D:]

        log_sigma = torch.clamp(log_sigma, -7, 2)
        return logits_pi, mu, log_sigma


def gmm_nll(y: torch.Tensor, logits_pi: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    B, N, D = y.shape
    K = logits_pi.size(-1)

    y = y.unsqueeze(2)
    inv_var = torch.exp(-2 * log_sigma)
    logp = -0.5 * ((y - mu) ** 2 * inv_var).sum(-1) - log_sigma.sum(-1) - 0.5 * D * math.log(2 * math.pi)
    logmix = F.log_softmax(logits_pi, dim=-1) + logp
    return -torch.logsumexp(logmix, dim=-1).sum(dim=1)


class JetFormer(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.flow = ViTFlow(cfg.d_token, cfg.n_tokens, cfg.flow_steps)
        self.in_proj = nn.Linear(cfg.d_token, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(1, cfg.n_tokens, cfg.d_model) * 0.02)
        self.gpt = AstroPTBackbone(cfg)
        self.head = GMMHead(cfg.d_model, cfg.d_token, cfg.gmm_K)

    def forward(self, x: torch.Tensor, step: int, max_iters: int) -> torch.Tensor:
        """
        Noise curriculum is tied to (step, max_iters) and decays to final values at max_iters.
        - RGB noise: sigma in [0,255] (paper-style)
        - z-noise: token-space Gaussian std (paper-style)
        """
        x = uniform_dequantize(x)

        # Curriculum length T (end exactly at max_iters if noise_decay_frac=1.0)
        T = int(max_iters * self.cfg.noise_decay_frac)
        T = max(T, 1)

        # ---- RGB noise schedule ----
        rgb_sigma_255 = cosine_decay(step, T, self.cfg.rgb_sigma0_255, self.cfg.rgb_sigmaT_255)
        rgb_sigma = rgb_sigma_255 / 255.0
        if self.cfg.noise_floor > 0:
            rgb_sigma = max(rgb_sigma, self.cfg.noise_floor)

        if self.training and rgb_sigma > 0.0:
            x = (x + torch.randn_like(x) * rgb_sigma).clamp(0.0, 1.0)

        # ---- Flow encode ----
        tokens_in = patchify(x, self.cfg.patch)
        z, logdet = self.flow(tokens_in, reverse=False)

        # ---- Latent z noise schedule (decays to ~0 by max_iters) ----
        z_sigma = cosine_decay(step, T, self.cfg.z_sigma0, self.cfg.z_sigmaT)
        if self.cfg.noise_floor > 0:
            z_sigma = max(z_sigma, self.cfg.noise_floor)

        if self.training and z_sigma > 0.0:
            z = z + torch.randn_like(z) * z_sigma

        # ---- AR transformer + GMM ----
        h = self.in_proj(z) + self.pos
        h = self.gpt(h)

        logits_pi, mu, log_sigma = self.head(h[:, :-1])
        target = z[:, 1:]

        nll_gmm = gmm_nll(target, logits_pi, mu, log_sigma)
        loss = (nll_gmm - logdet).mean()
        return loss

    @torch.no_grad()
    def sample(self, n: int = 16, x_real_batch: torch.Tensor = None):
        self.eval()
        B = n
        N = self.cfg.n_tokens
        device = next(self.parameters()).device

        if x_real_batch is None:
            z_seq = torch.zeros(B, N, self.cfg.d_token, device=device)
            for t in range(N - 1):
                h_in = self.in_proj(z_seq) + self.pos
                h_out = self.gpt(h_in)
                logits_pi, mu, log_sigma = self.head(h_out[:, t:t + 1])

                pi = F.softmax(logits_pi.squeeze(1), dim=-1)
                comp_idx = torch.multinomial(pi, 1)
                gather_idx = comp_idx[..., None].expand(-1, -1, self.cfg.d_token)

                sel_mu = mu.squeeze(1).gather(1, gather_idx).squeeze(1)
                sel_sigma = log_sigma.squeeze(1).gather(1, gather_idx).squeeze(1).exp()

                z_next = sel_mu + torch.randn_like(sel_mu) * sel_sigma
                z_seq[:, t + 1] = z_next

            x_rec_tokens, _ = self.flow(z_seq, reverse=True)
            x_rec = depatchify(x_rec_tokens, self.cfg.in_ch, self.cfg.img_size, self.cfg.img_size, self.cfg.patch)
            return x_rec.clamp(0, 1)

        else:
            x_real = x_real_batch.to(device)
            x_real_proc = uniform_dequantize(x_real)

            z_real, _ = self.flow(patchify(x_real_proc, self.cfg.patch), reverse=False)
            h_in = self.in_proj(z_real) + self.pos
            h_out = self.gpt(h_in)

            logits_pi, mu, log_sigma = self.head(h_out)
            best_comp_idx = torch.argmax(logits_pi, dim=-1, keepdim=True)
            gather_idx = best_comp_idx.unsqueeze(-1).expand(-1, -1, -1, self.cfg.d_token)

            z_pred_next = torch.gather(mu, 2, gather_idx).squeeze(2)

            z_rec = torch.zeros_like(z_real)
            z_rec[:, 0] = z_real[:, 0]
            z_rec[:, 1:] = z_pred_next[:, :-1]

            x_rec_tokens, _ = self.flow(z_rec, reverse=True)
            x_rec = depatchify(x_rec_tokens, self.cfg.in_ch, self.cfg.img_size, self.cfg.img_size, self.cfg.patch)

            combined = torch.stack([x_real, x_rec.clamp(0, 1)], dim=1).view(
                -1, self.cfg.in_ch, self.cfg.img_size, self.cfg.img_size
            )
            return combined


# ======================================================================================
# Block 6: Main Training Loop (DDP Aware)
# ======================================================================================
def train():
    # --- 1. DDP Setup ---
    rank, local_rank, world_size = setup_ddp()

    cfg = CFG()
    cfg.rank = rank
    cfg.world_size = world_size
    cfg.device = f"cuda:{local_rank}"

    # ### PATH SETUP ###
    cfg.samples_dir = f"samples_{cfg.dataset_name}_256"
    cfg.loss_csv_path = f"loss_log_{cfg.dataset_name}_256.csv"
    cfg.loss_plot_path = f"loss_plot_{cfg.dataset_name}_256.png"

    if rank == 0:
        os.makedirs(cfg.samples_dir, exist_ok=True)
        print(f"--- DDP CONFIGURATION ---")
        print(f"  World Size: {world_size}")
        print(f"  Per-GPU Batch: {cfg.batch_size}")
        print(f"  Global Batch: {cfg.batch_size * world_size}")
        print(f"  Dataset: {cfg.hf_repo} (Streaming + Sharded)")
        print(f"  Saving Checkpoints to: {cfg.samples_dir}")
        print(f"-------------------------")

        print("Noise curriculum:")
        print(f"  RGB sigma: {cfg.rgb_sigma0_255} -> {cfg.rgb_sigmaT_255} (in [0,255])")
        print(f"  z sigma:   {cfg.z_sigma0} -> {cfg.z_sigmaT} (token space)")
        print(f"  decay_frac: {cfg.noise_decay_frac} (T = {int(cfg.max_iters*cfg.noise_decay_frac)})")
        print(f"  floor: {cfg.noise_floor}")

    # --- 2. Model Setup ---
    model = JetFormer(cfg).to(cfg.device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # --- 3. Optimizer Setup ---
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.wd,
        betas=(0.9, cfg.beta2)
    )

    # --- 4. Checkpoint Loading ---
    start_step = load_checkpoint(model, opt, cfg)

    # --- 5. Data Loading ---
    train_loader = get_train_dataloader(cfg)
    val_loader = get_val_dataloader(cfg)

    # Pre-load fixed batch for viz
    viz_batch = None
    if rank == 0:
        print("Fetching visualization batch...")
        try:
            viz_batch = next(iter(val_loader))['img'][:16].to(cfg.device)
        except Exception as e:
            print(f"Warning: Could not load viz batch: {e}")

    # --- 6. Scheduler ---
    def get_lr_schedule(step):
        if step < cfg.warmup_steps:
            return step / cfg.warmup_steps
        else:
            progress = (step - cfg.warmup_steps) / (cfg.max_iters - cfg.warmup_steps)
            progress = max(0.0, min(1.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    if rank == 0:
        print(f"Starting training loop from step {start_step}...")

    # --- 7. Main Loop ---
    model.train()
    train_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm(range(start_step, cfg.max_iters), initial=start_step, total=cfg.max_iters)
    else:
        pbar = range(start_step, cfg.max_iters)

    train_loss_accum = 0.0
    accum_steps = 0

    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        img = batch["img"].to(cfg.device)

        # LR Update
        lr_scale = get_lr_schedule(step)
        for param_group in opt.param_groups:
            param_group['lr'] = cfg.lr * lr_scale

        # Forward Pass (BFloat16 for H100 / Ampere+)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(img, step=step, max_iters=cfg.max_iters)

        # Backward
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_val)
        opt.step()

        current_loss = float(loss.item())

        if rank == 0:
            train_loss_accum += current_loss
            accum_steps += 1
            if isinstance(pbar, tqdm):
                pbar.set_postfix(loss=f"{current_loss:.3f}", lr=f"{opt.param_groups[0]['lr']:.2e}")

        if step > 0:
            # 1. Validation and Image Sampling
            if step % cfg.val_check_interval == 0:
                if rank == 0:
                    avg_train_loss = train_loss_accum / max(accum_steps, 1)
                    train_loss_accum = 0.0
                    accum_steps = 0

                    # Generate Samples
                    if viz_batch is not None:
                        model.eval()
                        try:
                            with torch.no_grad():
                                fake_images = model.module.sample(n=16, x_real_batch=viz_batch)
                                sample_path = os.path.join(cfg.samples_dir, f"step_{step:07d}.png")
                                save_image(fake_images, sample_path, nrow=2)
                        except Exception as e:
                            print(f"Interval Sampling Error: {e}")
                        model.train()

                # Run Validation
                model.eval()
                val_iter = iter(val_loader)
                local_val_loss = 0.0

                with torch.no_grad():
                    for _ in range(cfg.val_steps):
                        try:
                            vbatch = next(val_iter)
                            vimg = vbatch["img"].to(cfg.device)
                            vloss = model(vimg, step=step, max_iters=cfg.max_iters)
                            local_val_loss += float(vloss.item())
                        except StopIteration:
                            break

                avg_local_val = local_val_loss / max(cfg.val_steps, 1)
                val_tensor = torch.tensor([avg_local_val], device=cfg.device)
                dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
                avg_val_loss = val_tensor.item() / world_size

                if rank == 0:
                    save_checkpoint(step, model, opt, cfg, is_latest=True)
                    append_losses_to_csv(step, avg_train_loss, avg_val_loss, cfg.loss_csv_path)
                    plot_loss_from_csv(cfg.loss_csv_path, cfg.loss_plot_path)
                    print(f"\nStep {step}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

                model.train()

            # 2. Historical checkpoint saving
            if step % cfg.save_interval == 0:
                if rank == 0:
                    save_checkpoint(step, model, opt, cfg, is_latest=False)

    # Final checkpoint and logging at max_iters (after training loop completes)
    final_step = cfg.max_iters - 1
    
    # Calculate final average train loss (rank 0 only)
    if rank == 0:
        avg_train_loss = train_loss_accum / max(accum_steps, 1) if accum_steps > 0 else 0.0
    
    # Final validation (all ranks)
    model.eval()
    val_iter = iter(val_loader)
    local_val_loss = 0.0
    
    with torch.no_grad():
        for _ in range(cfg.val_steps):
            try:
                vbatch = next(val_iter)
                vimg = vbatch["img"].to(cfg.device)
                vloss = model(vimg, step=final_step, max_iters=cfg.max_iters)
                local_val_loss += float(vloss.item())
            except StopIteration:
                break
    
    avg_local_val = local_val_loss / max(cfg.val_steps, 1)
    val_tensor = torch.tensor([avg_local_val], device=cfg.device)
    dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
    avg_val_loss = val_tensor.item() / world_size
    
    # Final sampling, checkpoint and logging (rank 0 only)
    if rank == 0:
        # Final sampling
        if viz_batch is not None:
            try:
                with torch.no_grad():
                    fake_images = model.module.sample(n=16, x_real_batch=viz_batch)
                    sample_path = os.path.join(cfg.samples_dir, f"step_{cfg.max_iters:07d}.png")
                    save_image(fake_images, sample_path, nrow=2)
            except Exception as e:
                print(f"Final Sampling Error: {e}")
        
        # Final checkpoint and logging
        save_checkpoint(final_step, model, opt, cfg, is_latest=True)
        save_checkpoint(final_step, model, opt, cfg, is_latest=False)  # Also save as historical
        append_losses_to_csv(final_step, avg_train_loss, avg_val_loss, cfg.loss_csv_path)
        plot_loss_from_csv(cfg.loss_csv_path, cfg.loss_plot_path)
        print(f"\nFinal Step {final_step}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

    cleanup_ddp()
    if rank == 0:
        print("Training finished.")


if __name__ == "__main__":
    train()
