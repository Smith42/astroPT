"""
AstroPT Causal Attention Inspector.

This script extracts and visualizes the cross-modal attention maps from a
pre-trained AstroPT model, plotting per-object attention heatmaps and a
population average across N galaxies.

Features:
- Per-object attention maps with ID and redshift in title (same as plot_images_spectra.py).
- _R suffix for randomly selected objects (vs explicitly targeted).
- Population-averaged cross-modal attention across N galaxies.
- Bidirectional cross-modal plots (Spectra->Images and Images->Spectra).
- Per-layer cross-modal attention strength barplot.

Author: Victor Alonso Rodriguez
Date: April 2026
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow
from astropt.model_utils import load_local_model

# Logger Configuration
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-Attention")

# Plotting Global Configuration
plt.rcParams['text.latex.preamble'] = r'''
            \usepackage{siunitx}
            \usepackage{bm}
            \usepackage{amsmath} 
            \sisetup{
            detect-family,
            separate-uncertainty=true,
            output-decimal-marker={.},
            exponent-product=\cdot,
            inter-unit-product=\cdot,
            }
            \DeclareSIUnit{\cts}{cts}
            '''
plt.rc('text', usetex=True)
plt.rc('font', family='serif', weight='bold')

plt.rcParams.update({
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titlesize': 20,
    'figure.titleweight': 'bold',
})


def get_patched_forward():
    def patched_forward(self, x, block_mask=None):
        B, T, C = x.size()
        head_dim = C // self.n_head

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Build causal mask on-the-fly (self.bias only exists when flash=False)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~causal_mask.view(1, 1, T, T), float("-inf"))
        att = F.softmax(att, dim=-1)

        # Save explicitly for extraction!
        self.extracted_attention = att.detach().cpu()

        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
    return patched_forward


def reorder_modal_inputs(
    inputs: dict[str, torch.Tensor],
    modality_order: list[str],
) -> dict[str, torch.Tensor]:
    """Helper to swap modality order for bidirectional analysis."""
    ordered = {}
    for mod in modality_order:
        pos_key = f"{mod}_positions"
        if mod in inputs and pos_key in inputs:
            ordered[mod] = inputs[mod]
            ordered[pos_key] = inputs[pos_key]
    for key, val in inputs.items():
        if key not in ordered:
            ordered[key] = val
    return ordered


def plot_cross_attention_single(
    attn_matrix: np.ndarray,
    n_img_tokens: int,
    n_spec_tokens: int,
    n_layers: int,
    target_id: int,
    z_val: float,
    train_name: str,
    save_dir: Path,
    filename: str,
):
    """Plots per-object cross-modal attention maps (2 directions)."""
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle(
        rf"\textbf{{AstroPT Attention Map | ID: {target_id} | z={z_val:.3f}}}"
        + f"\n[{train_name}] -- Layer {n_layers} (Heads Averaged)",
        fontsize=20, y=1.01
    )

    # Spectra -> Images (from pass 1 or bidirectional assembly)
    cross_s2i = attn_matrix[n_img_tokens:, :n_img_tokens]
    # Images -> Spectra (from pass 2 or bidirectional assembly)
    cross_i2s = attn_matrix[:n_img_tokens, n_img_tokens:]

    # Use a unified scale for both cross-modal subplots
    # We use the absolute max to avoid clamping structure into 'yellow' saturation
    vmax = max(cross_s2i.max(), cross_i2s.max(), 1e-4)
    vmin = 1e-7 # Floor to keep the purple 'deep'

    # Subplot 1: Spectra looking at Images
    im0 = axes[0].imshow(
        np.clip(cross_s2i, vmin, None), cmap='inferno', aspect='auto', origin='lower',
        norm=LogNorm(vmin=vmin, vmax=vmax)
    )
    plt.colorbar(im0, ax=axes[0], label='Attention Weight (log scale)')
    axes[0].set_title(r"\textbf{Queries: Spectra $\rightarrow$ Look at: Images}")
    axes[0].set_xlabel(r"Image Patch Tokens (Keys)")
    axes[0].set_ylabel(r"Spectra Tokens (Queries)")

    # Subplot 2: Images looking at Spectra
    im1 = axes[1].imshow(
        np.clip(cross_i2s, vmin, None), cmap='inferno', aspect='auto', origin='lower',
        norm=LogNorm(vmin=vmin, vmax=vmax)
    )
    plt.colorbar(im1, ax=axes[1], label='Attention Weight (log scale)')
    axes[1].set_title(r"\textbf{Queries: Images $\rightarrow$ Look at: Spectra}")
    axes[1].set_xlabel(r"Spectra Tokens (Keys)")
    axes[1].set_ylabel(r"Image Patch Tokens (Queries)")

    save_path = save_dir / filename
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f" --> Saved: {save_path}")


def plot_cross_attention_average(
    avg_attn: np.ndarray,
    avg_per_layer_s2i: list,
    avg_per_layer_i2s: list,
    n_img_tokens: int,
    n_spec_tokens: int,
    n_layers: int,
    num_samples: int,
    train_name: str,
    save_dir: Path,
    run_suffix: str,
):
    """Plots population-averaged attention maps."""
    from matplotlib.colors import LogNorm

    sample_label = f"(Avg. over {num_samples} galaxies)"

    # --- Full Matrix ---
    attn_c = np.clip(avg_attn, 1e-8, None)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(
        attn_c, cmap='magma', aspect='auto', origin='lower',
        norm=LogNorm(vmin=attn_c[attn_c > 0].min(), vmax=attn_c.max())
    )
    plt.colorbar(im, ax=ax, label='Attention Weight (log scale)')
    ax.axvline(x=n_img_tokens - 0.5, color='cyan', lw=1.5, ls='--', alpha=0.8)
    ax.axhline(y=n_img_tokens - 0.5, color='cyan', lw=1.5, ls='--', alpha=0.8)
    ax.set_title(rf"\textbf{{Full Causal Attention -- Layer {n_layers}}} {sample_label}")
    ax.set_xlabel(r"Source Tokens (Keys)")
    ax.set_ylabel(r"Destination Tokens (Queries)")
    fig.savefig(save_dir / f"avg_attention_full{run_suffix}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Compute SHARED scale so both cross-modal plots are directly comparable
    cross_s2i = avg_attn[n_img_tokens:, :n_img_tokens]
    cross_i2s = avg_attn[:n_img_tokens, n_img_tokens:]

    # Use S2I range as reference (it has real signal); I2S should be ~0 if images come first
    shared_vmax = np.clip(cross_s2i, 1e-8, None).max()
    shared_vmin = np.clip(cross_s2i, 1e-8, None)[np.clip(cross_s2i, 1e-8, None) > 0].min()

    # --- Spectra -> Images average ---
    cross_s2i_c = np.clip(cross_s2i, 1e-8, None)
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    im2 = ax2.imshow(
        cross_s2i_c, cmap='inferno', aspect='auto', origin='lower',
        norm=LogNorm(vmin=shared_vmin, vmax=shared_vmax)
    )
    plt.colorbar(im2, ax=ax2, label='Attention Weight (log scale)')
    ax2.set_title(rf"\textbf{{Avg. Queries: Spectra $\rightarrow$ Look at: Images -- Layer {n_layers}}} {sample_label}")
    ax2.set_xlabel(r"Image Patch Tokens (Keys)")
    ax2.set_ylabel(r"Spectra Tokens (Queries)")
    fig2.savefig(save_dir / f"avg_attention_spectra_to_images{run_suffix}.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # --- Images -> Spectra average ---
    cross_i2s_c = np.clip(cross_i2s, 1e-8, None)
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    im3 = ax3.imshow(
        cross_i2s_c, cmap='inferno', aspect='auto', origin='lower',
        norm=LogNorm(vmin=shared_vmin, vmax=shared_vmax)
    )
    plt.colorbar(im3, ax=ax3, label=f'Attention Weight (log scale)')
    ax3.set_title(rf"\textbf{{Avg. Queries: Images $\rightarrow$ Look at: Spectra -- Layer {n_layers}}} {sample_label}")
    ax3.set_xlabel(r"Spectra Tokens (Keys)")
    ax3.set_ylabel(r"Image Patch Tokens (Queries)")
    fig3.savefig(save_dir / f"avg_attention_images_to_spectra{run_suffix}.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)

    # --- Per-layer bidirectional barplot ---
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(n_layers)
    width = 0.35
    ax4.bar(x_pos - width/2, avg_per_layer_s2i, width,
            label=r'Spectra $\rightarrow$ Images', color='dodgerblue', edgecolor='black', lw=0.5)
    ax4.bar(x_pos + width/2, avg_per_layer_i2s, width,
            label=r'Images $\rightarrow$ Spectra', color='coral', edgecolor='black', lw=0.5)
    ax4.set_xlabel(r"Transformer Layer")
    ax4.set_ylabel(r"Mean Cross-Modal Attention Weight")
    ax4.set_title(rf"\textbf{{Bidirectional Cross-Modal Attention per Layer}} {sample_label}")
    ax4.set_xticks(x_pos)
    ax4.legend()
    fig4.savefig(save_dir / f"avg_attention_per_layer{run_suffix}.png", dpi=300, bbox_inches='tight')
    plt.close(fig4)

    logger.info(f"Cross-modal stats (Last Layer, population avg):")
    logger.info(f"  Spectra->Images mean: {cross_s2i.mean():.6e} | max: {cross_s2i.max():.6e}")
    logger.info(f"  Images->Spectra mean: {cross_i2s.mean():.6e} | max: {cross_i2s.max():.6e}")
    logger.info(f"  Asymmetry Ratio S2I/I2S: {cross_s2i.mean() / max(cross_i2s.mean(), 1e-12):.1f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AstroPT Causal Attention Inspector")
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing training weights")
    parser.add_argument("--data_dir", type=str, required=True, help="Arrow data root directory")
    parser.add_argument("--save_dir", type=str, default=None, help="Plot saving directory. Defaults to weights_dir/../plots/attention_maps")
    parser.add_argument("--ckpt_name", type=str, default="ckpt_best.pt", help="Checkpoint filename")
    parser.add_argument("--target_ids", nargs="+", type=int, help="Specific Target IDs to plot")
    parser.add_argument("--num_plot", type=int, default=10, help="Total number of galaxies to process (including targeted)")
    parser.add_argument("--split", type=str, default="test", help="Data split (train/test)")
    parser.add_argument("--train_name", type=str, default=None, help="Custom title for plots (defaults to folder name)")
    parser.add_argument("--bidirectional", action="store_true", default=True, help="Perform bidirectional extraction (two passes) to eliminate order bias")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_dir = Path(args.weights_dir)
    save_dir = Path(args.save_dir) if args.save_dir else weights_dir.parent / "plots" / "attention_maps"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Extract run suffix from checkpoint name (same logic as plot_images_spectra.py)
    ckpt_filename = weights_dir / args.ckpt_name
    name_no_ext = ckpt_filename.stem
    run_suffix = f"_{name_no_ext.split('_')[-1]}" if '_' in name_no_ext else f"_{name_no_ext}"
    logger.info(f"Run suffix: {run_suffix}")

    # Extract train_name from config.json
    config_path = weights_dir / "config.json"
    json_name = None
    if config_path.is_file():
        try:
            with open(config_path) as f:
                json_name = json.load(f).get("train_name", None)
        except Exception:
            pass
    train_name = args.train_name or json_name or weights_dir.parent.name

    # Load model via shared utility (same as all other scripts)
    ckpt_path = weights_dir / args.ckpt_name
    if not ckpt_path.is_file():
        all_ckpts = list(weights_dir.glob("*.pt"))
        best = [c for c in all_ckpts if "best" in c.name]
        ckpt_path = sorted(best or all_ckpts, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        logger.info(f"Using checkpoint: {ckpt_path.name}")

    model, config, registry, raw_config_dict = load_local_model(ckpt_path, device)
    model.eval()

    # Patch attention heads to extract matrices
    for block in model.transformer.h:
        block.attn.forward = get_patched_forward().__get__(block.attn, type(block.attn))

    n_layers = len(model.transformer.h)
    use_token_mixing = raw_config_dict.get("use_token_mixing", False)

    # Load dataset
    ds = EuclidDESIDatasetArrow(
        arrow_folder_root=args.data_dir,
        split=args.split,
        modality_registry=registry,
        spiral=False,
        stochastic=False,
    )

    # Build index list: targeted IDs first, then random fill (same as plot_images_spectra.py)
    indices_to_plot = []
    specific_tids = set()

    if args.target_ids:
        all_ids = ds.ds['targetid']
        id_map = {int(tid): idx for idx, tid in enumerate(all_ids)}
        for tid in args.target_ids:
            if tid in id_map:
                indices_to_plot.append(id_map[tid])
                specific_tids.add(int(tid))

    if len(indices_to_plot) < args.num_plot:
        pool = list(set(range(len(ds))) - set(indices_to_plot))
        needed = args.num_plot - len(indices_to_plot)
        indices_to_plot.extend(np.random.choice(pool, min(needed, len(pool)), replace=False))

    indices_to_plot = [int(i) for i in indices_to_plot]
    subset = Subset(ds, indices_to_plot)
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    logger.info(f"Processing {len(indices_to_plot)} galaxies ({len(specific_tids)} targeted, rest random)...")

    # Accumulators for population average
    avg_attn_last = None
    avg_per_layer_s2i = [0.0] * n_layers
    avg_per_layer_i2s = [0.0] * n_layers
    n_img_tokens = None
    n_spec_tokens = None

    for batch_idx, batch in enumerate(loader):
        # 1. Extract metadata (ID and Redshift)
        try:
            arrow_idx = int(batch['idx'].item())
        except Exception:
            arrow_idx = int(batch['idx'][0])
        raw_record = ds.ds[arrow_idx]
        target_id = int(raw_record['targetid'])
        z_val = float(raw_record.get('redshift', 0.0))

        # 2. Process data
        B = EuclidDESIDatasetArrow.process_modes(batch, registry, device, use_token_mixing=use_token_mixing)

        if n_img_tokens is None:
            n_img_tokens = B["X"]["images"].shape[1]
            n_spec_tokens = B["X"]["spectra"].shape[1]
            logger.info(f"Token layout: Images={n_img_tokens}, Spectra={n_spec_tokens}")

        # 3. Forward Pass Logic
        if args.bidirectional and not use_token_mixing:
            # --- Pass 1: Images -> Spectra (Extraction of Spectra looking at Images) ---
            X1 = reorder_modal_inputs(B["X"], ["images", "spectra"])
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(X1, targets=None)
            
            # Save ALL layers from Pass 1 for the average plot
            layers_attn1 = [block.attn.extracted_attention[0].mean(dim=0).float().numpy() for block in model.transformer.h]
            
            # --- Pass 2: Spectra -> Images (Extraction of Images looking at Spectra) ---
            X2 = reorder_modal_inputs(B["X"], ["spectra", "images"])
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(X2, targets=None)
            
            # Save ALL layers from Pass 2 for the average plot
            layers_attn2 = [block.attn.extracted_attention[0].mean(dim=0).float().numpy() for block in model.transformer.h]
            
            # Assembly of Fused Bidirectional Matrix for the single-object heatmap
            # Spectra -> Images: from Pass 1 (Spectra come 2nd)
            # Images -> Spectra: from Pass 2 (Images come 2nd)
            fused_attn = np.zeros((n_img_tokens + n_spec_tokens, n_img_tokens + n_spec_tokens))
            fused_attn[n_img_tokens:, :n_img_tokens] = layers_attn1[-1][n_img_tokens:, :n_img_tokens]
            fused_attn[:n_img_tokens, n_img_tokens:] = layers_attn2[-1][n_spec_tokens:, :n_spec_tokens]
            
            # Self-attention blocks
            fused_attn[:n_img_tokens, :n_img_tokens] = layers_attn1[-1][:n_img_tokens, :n_img_tokens]
            fused_attn[n_img_tokens:, n_img_tokens:] = layers_attn2[-1][:n_spec_tokens, :n_spec_tokens]
            last_attn = fused_attn

            # Update per-layer averages CORRECTLY (Bidirectional)
            for i in range(n_layers):
                # Pass 1 direction: Spectra -> Images (n_img_tokens are the queries)
                avg_per_layer_s2i[i] += layers_attn1[i][n_img_tokens:, :n_img_tokens].mean()
                # Pass 2 direction: Images -> Spectra (n_spec_tokens are the queries)
                avg_per_layer_i2s[i] += layers_attn2[i][n_spec_tokens:, :n_spec_tokens].mean()

        else:
            # Single pass / Token Mixing logic
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(B["X"], targets=None)
            last_attn = model.transformer.h[-1].attn.extracted_attention[0].mean(dim=0).float().numpy()
            
            # Update per-layer averages (Standard Causal)
            for i, block in enumerate(model.transformer.h):
                layer_attn = block.attn.extracted_attention[0].mean(dim=0).float().numpy()
                avg_per_layer_s2i[i] += layer_attn[n_img_tokens:, :n_img_tokens].mean()
                avg_per_layer_i2s[i] += layer_attn[:n_img_tokens, n_img_tokens:].mean()

        # 4. Accumulate population average for the heatmap
        if avg_attn_last is None:
            avg_attn_last = np.zeros_like(last_attn)
        avg_attn_last += last_attn

        # 5. Save per-object plots
        random_suffix = "" if target_id in specific_tids else "_R"
        filename = f"attn_ID_{target_id}{run_suffix}{random_suffix}.png"
        logger.info(f"[{batch_idx+1}/{len(indices_to_plot)}] ID={target_id} z={z_val:.3f}{' [random]' if random_suffix else ''}")

        plot_cross_attention_single(
            attn_matrix=last_attn,
            n_img_tokens=n_img_tokens,
            n_spec_tokens=n_spec_tokens,
            n_layers=n_layers,
            target_id=target_id,
            z_val=z_val,
            train_name=train_name,
            save_dir=save_dir,
            filename=filename,
        )

    # Normalize accumulators
    n_processed = len(indices_to_plot)
    avg_attn_last /= n_processed
    avg_per_layer_s2i = [x / n_processed for x in avg_per_layer_s2i]
    avg_per_layer_i2s = [x / n_processed for x in avg_per_layer_i2s]

    # Population average plots
    logger.info("Generating population-averaged plots...")
    plot_cross_attention_average(
        avg_attn=avg_attn_last,
        avg_per_layer_s2i=avg_per_layer_s2i,
        avg_per_layer_i2s=avg_per_layer_i2s,
        n_img_tokens=n_img_tokens,
        n_spec_tokens=n_spec_tokens,
        n_layers=n_layers,
        num_samples=n_processed,
        train_name=train_name,
        save_dir=save_dir,
        run_suffix=run_suffix,
    )

    logger.info(f"Done. All plots saved to {save_dir}")


if __name__ == "__main__":
    main()
