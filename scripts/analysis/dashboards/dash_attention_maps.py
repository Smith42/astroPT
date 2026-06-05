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

import dataclasses
from astropt.dataloader_multimodal import MultimodalDatasetArrow
from astropt.training_utils import create_dataloaders
from astropt.config import TrainingConfig
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
    """Patches the attention forward to store the attention matrix."""
    def patched_forward(self, x, block_mask=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        
        # Apply causal mask if no block_mask is provided
        if block_mask is None:
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float('-inf'))
        
        att = torch.softmax(att, dim=-1)
        self.extracted_attention = att.detach().cpu()
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
    return patched_forward

def get_patched_forward_block_with_mask():
    """Patches model's _forward_block_with_mask to capture the attention matrix during unimodal isolation."""
    def patched_forward_block_with_mask(self, block, x, attn_mask):
        B = x.shape[0]
        C = x.shape[2]
        n_head = block.attn.n_head
        head_dim = C // n_head

        # Pre-norm
        x_ln = block.ln_1(x)

        # QKV projection
        q, k, v = block.attn.c_attn(x_ln).split(block.attn.n_embd, dim=2)
        q = q.reshape(B, -1, n_head, head_dim).transpose(1, 2)
        k = k.reshape(B, -1, n_head, head_dim).transpose(1, 2)
        v = v.reshape(B, -1, n_head, head_dim).transpose(1, 2)

        # Convert boolean mask to float mask
        float_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        float_mask.masked_fill_(~attn_mask, float('-inf'))

        # Manual attention calculation for extraction
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        att = att + float_mask
        att = torch.softmax(att, dim=-1)
        block.attn.extracted_attention = att.detach().cpu()

        y = att @ v
        y = y.transpose(1, 2).contiguous().reshape(B, -1, C)

        # Output projection + residual
        y = block.attn.resid_dropout(block.attn.c_proj(y))
        x = x + y

        # MLP + residual
        x = x + block.mlp(block.ln_2(x))
        return x
    return patched_forward_block_with_mask

def get_patched_interleaved_indices(model, original_method):
    """Patches the index generator to capture the sequence permutation."""
    def wrapper(*args, **kwargs):
        idx = original_method(*args, **kwargs)
        model._last_interleaved_idx = idx
        return idx
    return wrapper


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


def cast_batch_for_cpu(X, registry):
    for k, v in X.items():
        if isinstance(v, torch.Tensor):
            is_discrete = False
            try:
                mod_name = k.replace("_positions", "")
                if mod_name in registry.modalities:
                    mc = registry.get_config(mod_name)
                    is_discrete = getattr(mc, 'vocab_size', 0) > 0
            except Exception:
                pass
            
            if 'positions' in k or is_discrete:
                X[k] = v.to(torch.long)
            else:
                X[k] = v.to(torch.float32)


def get_active_modalities(batch: dict):
    """Detects all modality keys in the batch, excluding metadata and positions."""
    return [k for k in batch.keys() if not k.endswith("_positions") and k not in ["targetid", "redshift", "idx", "token_mixing_seed"]]

def get_interleaved_masks(mod_lengths: list[int], block_size: int, has_cls: bool = False, cls_position: str = "last"):
    """
    Reconstructs modality masks for interleaved sequences with N modalities.
    Returns:
        interleaved_idx: The actual index mapping
        is_cls: Mask for CLS token
        mod_masks: List of masks for each modality
    """
    idx = []
    ptrs = [0] * len(mod_lengths)
    offsets = [sum(mod_lengths[:i]) for i in range(len(mod_lengths))]
    
    if has_cls and cls_position == "first":
        idx.append(-1)

    while any(p < l for p, l in zip(ptrs, mod_lengths)):
        for i in range(len(mod_lengths)):
            if ptrs[i] < mod_lengths[i]:
                take = min(block_size, mod_lengths[i] - ptrs[i])
                idx.extend(range(offsets[i] + ptrs[i], offsets[i] + ptrs[i] + take))
                ptrs[i] += take
                
    if has_cls and cls_position == "last":
        idx.append(-1)
    
    interleaved_idx = np.array(idx)
    is_cls = (interleaved_idx == -1) if has_cls else None
    
    mod_masks = []
    for i in range(len(mod_lengths)):
        mask = (interleaved_idx >= offsets[i]) & (interleaved_idx < offsets[i] + mod_lengths[i])
        mod_masks.append(mask)
        
    return interleaved_idx, is_cls, mod_masks


def plot_attention_grid_single(
    attn_matrix: np.ndarray,
    modality_info: dict,
    n_layers: int,
    target_id: int,
    z_val: float,
    train_name: str,
    save_dir: Path,
    filename: str,
    title_prefix: str = "Attention Map",
):
    """
    Plots an N x N grid of attention maps between all modality pairs.
    modality_info: dict mapping name -> { 'tokens': int, 'slice': slice }
    """
    from matplotlib.colors import LogNorm
    mod_names = list(modality_info.keys())
    N = len(mod_names)
    
    fig, axes = plt.subplots(N, N, figsize=(6*N, 5*N), squeeze=False)
    fig.suptitle(
        rf"\textbf{{{title_prefix} | ID: {target_id} | z={z_val:.3f}}}"
        + f"\n[{train_name}] -- Layer {n_layers} (Heads Averaged)",
        fontsize=20, y=1.02
    )

    vmin, vmax = 1e-7, 1.0 # Global scale
    
    for i, target_mod in enumerate(mod_names):
        for j, source_mod in enumerate(mod_names):
            ax = axes[i, j]
            q_slice = modality_info[target_mod]['slice']
            k_slice = modality_info[source_mod]['slice']
            
            block = attn_matrix[q_slice, k_slice]
            
            # Local normalization per block if it's very sparse? 
            # No, keep global to see strength relative to others.
            im = ax.imshow(
                np.clip(block, vmin, None), cmap='inferno', aspect='equal', origin='lower',
                norm=LogNorm(vmin=vmin, vmax=max(block.max(), vmin*10))
            )
            
            ax.set_title(rf"\textbf{{{target_mod} $\rightarrow$ {source_mod}}}")
            if i == N - 1: ax.set_xlabel("Keys")
            if j == 0: ax.set_ylabel("Queries")
            
            plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    save_path = save_dir / filename
    fig.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    logger.info(f" --> Saved: {save_path}")


def plot_attention_matrix_average(
    avg_full_attn: np.ndarray,
    strength_matrix: np.ndarray,
    modality_info: dict,
    n_layers: int,
    num_samples: int,
    train_name: str,
    save_dir: Path,
    run_suffix: str,
):
    """
    Plots population-averaged N x N metrics.
    strength_matrix: (N_layers, N_modalities, N_modalities)
    """
    from matplotlib.colors import LogNorm
    mod_names = list(modality_info.keys())
    N = len(mod_names)
    sample_label = f"(Avg. over {num_samples} galaxies)"

    # 1. Full Matrix Heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(np.clip(avg_full_attn, 1e-8, None), cmap='magma', aspect='equal', origin='lower',
                   norm=LogNorm(vmin=1e-6, vmax=avg_full_attn.max()))
    plt.colorbar(im, ax=ax, label='Attention Weight (log scale)', shrink=0.8)
    
    # Draw boundaries
    for mod_name, info in modality_info.items():
        s = info['slice']
        ax.axvline(x=s.stop - 0.5, color='cyan', lw=1, alpha=0.5)
        ax.axhline(y=s.stop - 0.5, color='cyan', lw=1, alpha=0.5)
        
    ax.set_title(rf"\textbf{{Full Population Average Attention}} {sample_label}")
    ax.set_xlabel("Source Tokens")
    ax.set_ylabel("Destination Tokens")
    fig.savefig(save_dir / f"avg_attention_full{run_suffix}.png", dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    # 2. Attention Strength Heatmap (Source -> Target)
    # Average across layers for the final matrix
    mean_strength = strength_matrix.mean(axis=0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mean_strength, cmap='Blues')
    plt.colorbar(im, ax=ax, label='Mean Attention Weight')
    
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(mod_names, rotation=45)
    ax.set_yticklabels(mod_names)
    
    # Add text annotations
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{mean_strength[i, j]:.4f}", ha="center", va="center", color="black")

    ax.set_title(rf"\textbf{{Cross-Modal Attention Strength (Target $\leftarrow$ Source)}}")
    ax.set_xlabel("Source Modality")
    ax.set_ylabel("Target Modality")
    fig.savefig(save_dir / f"avg_attention_strength_matrix{run_suffix}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3. Strength per Layer Barplot
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(n_layers)
    
    # For each pair (Source -> Target) where Source != Target, draw a line/bar?
    # Better: Plot one line per direction
    for i, target_mod in enumerate(mod_names):
        for j, source_mod in enumerate(mod_names):
            if i == j: continue # Skip self-attention for the cross-modal plot
            ax.plot(x_pos, strength_matrix[:, i, j], label=f"{target_mod} $\leftarrow$ {source_mod}", marker='o', alpha=0.7)

    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Mean Cross-Modal Attention Weight")
    ax.set_title(rf"\textbf{{Cross-Modal Attention Strength per Layer}} {sample_label}")
    ax.set_xticks(x_pos)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.savefig(save_dir / f"avg_attention_per_layer{run_suffix}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Cross-modal stats saved to {save_dir}")


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

    if device.type == 'cuda':
        ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
    else:
        import contextlib
        ctx = contextlib.nullcontext()
        model = model.to(torch.float32)

    # Patch attention heads to extract matrices
    for block in model.transformer.h:
        block.attn.forward = get_patched_forward().__get__(block.attn, type(block.attn))

    if hasattr(model, "_forward_block_with_mask"):
        model._forward_block_with_mask = get_patched_forward_block_with_mask().__get__(model, type(model))

    n_layers = len(model.transformer.h)
    use_token_mixing = raw_config_dict.get("use_token_mixing", False)

    # Load dataset
    valid_keys = {f.name for f in dataclasses.fields(TrainingConfig)}
    filtered_config = {k: v for k, v in raw_config_dict.items() if k in valid_keys}
    config_obj = TrainingConfig(**filtered_config)
    config_obj.data_dir = args.data_dir
    config_obj.batch_size = 1
    
    _, val_loader, _ = create_dataloaders(config_obj, ddp=False)
    ds = val_loader.dataset
    
    has_cls = getattr(config, 'use_cls_token', False)
    logger.info(f"Dataset loaded. Total samples: {len(ds)} | CLS Token: {has_cls}")

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
    avg_attn_full = None # Full concatenated matrix
    
    # Detected modalities from the first sample
    temp_batch = ds[0]
    active_modalities = get_active_modalities(temp_batch)
    # Filter active_modalities to only include sequence-level modalities in the model
    active_modalities = [m for m in active_modalities if m in model.mod_names]
    # Sort active_modalities according to model.mod_names to avoid modality order mismatch
    active_modalities = sorted(active_modalities, key=lambda m: model.mod_names.index(m) if m in model.mod_names else 999)
    N = len(active_modalities)
    logger.info(f"Detected {N} active modalities: {active_modalities}")
    
    # (Layers, TargetMod, SourceMod)
    strength_matrix_acc = np.zeros((n_layers, N, N))
    modality_info = {} # Will be filled in the first batch iteration

    if use_token_mixing:
        model._get_interleaved_indices = get_patched_interleaved_indices(model, model._get_interleaved_indices)

    for batch_idx, batch in enumerate(loader):
        target_id = int(batch['targetid'].item() if torch.is_tensor(batch['targetid']) else batch['targetid'][0])
        z_val = float(batch['redshift'].item() if torch.is_tensor(batch['redshift']) else batch['redshift'][0])

        if use_token_mixing:
            # --- SINGLE PASS EXTRACTION FOR MIXED MODELS ---
            B = MultimodalDatasetArrow.process_modes(
                batch, registry, device, 
                use_token_mixing=True,
                use_cls_token=has_cls,
                cls_position=getattr(config, 'cls_position', 'last')
            )
            if device.type == 'cpu':
                cast_batch_for_cpu(B["X"], registry)
            
            with torch.no_grad(), ctx:
                model(B["X"], targets=B["X"].copy())
            
            # Retrieve the interleaving indices used in this forward pass and reconstruct full_interleaved
            patch_interleaved = model._last_interleaved_idx
            total_patches = sum(batch[m].shape[1] if torch.is_tensor(batch[m]) else len(batch[m][0]) for m in active_modalities)
            patch_interleaved = patch_interleaved[:total_patches]
            
            # Reconstruct suffix indices based on model type (V4 has expert tokens, V3 does not)
            is_v4 = hasattr(model.embedding_layer, "expert_tokens")
            n_experts = len(active_modalities) if is_v4 else 0
            suffix_indices = torch.arange(total_patches, total_patches + n_experts + (1 if has_cls else 0), device=patch_interleaved.device, dtype=torch.long)
            full_interleaved = torch.cat([patch_interleaved, suffix_indices])
            
            idx = full_interleaved.cpu().numpy()
            inv_idx = np.argsort(idx)
            total_tokens = len(idx)
            
            if not modality_info:
                # Calculate canonical offsets based on original lengths
                current_offset = 1 if has_cls and getattr(config, 'cls_position', 'last').lower() == 'first' else 0
                for mod in active_modalities:
                    tokens = batch[mod].shape[1] if torch.is_tensor(batch[mod]) else len(batch[mod][0])
                    modality_info[mod] = {
                        'tokens': tokens,
                        'slice': slice(current_offset, current_offset + tokens)
                    }
                    current_offset += tokens
                avg_attn_full = np.zeros((total_tokens, total_tokens))

            fused_attn = np.zeros_like(avg_attn_full)
            
            for l, block in enumerate(model.transformer.h):
                # A is (T, T) in interleaved space
                layer_attn_interleaved = block.attn.extracted_attention[0].mean(dim=0).float().numpy()
                
                # A_can = A[inv_idx, :][:, inv_idx] maps back to sequential space
                layer_attn = layer_attn_interleaved[inv_idx, :][:, inv_idx]

                for i, target_mod in enumerate(active_modalities):
                    can_q_slice = modality_info[target_mod]['slice']
                    for j, source_mod in enumerate(active_modalities):
                        can_k_slice = modality_info[source_mod]['slice']
                        
                        strength = layer_attn[can_q_slice, can_k_slice].mean()
                        strength_matrix_acc[l, i, j] += strength
                        
                        if l == n_layers - 1:
                            fused_attn[can_q_slice, can_k_slice] = layer_attn[can_q_slice, can_k_slice]
            
            last_attn = fused_attn

        else:
            # --- MULTI-PASS EXTRACTION FOR NON-MIXED MODELS ---
            B_base = MultimodalDatasetArrow.process_modes(
                batch, registry, device, 
                use_token_mixing=False,
                use_cls_token=has_cls,
                cls_position=getattr(config, 'cls_position', 'last')
            )
            if device.type == 'cpu':
                cast_batch_for_cpu(B_base["X"], registry)

            if not modality_info:
                current_offset = 1 if has_cls and getattr(config, 'cls_position', 'last').lower() == 'first' else 0
                for mod in active_modalities:
                    tokens = B_base["X"][mod].shape[1]
                    modality_info[mod] = {
                        'tokens': tokens,
                        'slice': slice(current_offset, current_offset + tokens)
                    }
                    current_offset += tokens
                total_tokens = current_offset + (1 if has_cls and getattr(config, 'cls_position', 'last').lower() == 'last' else 0)
                avg_attn_full = np.zeros((total_tokens, total_tokens))

            fused_attn = np.zeros_like(avg_attn_full)
            
            for i, target_mod in enumerate(active_modalities):
                others = [m for m in active_modalities if m != target_mod]
                order = others + [target_mod]
                X_ordered = reorder_modal_inputs(B_base["X"], order)
                
                with torch.no_grad(), ctx:
                    model(X_ordered, targets=X_ordered.copy())
                
                target_len = modality_info[target_mod]['tokens']
                others_len = sum(modality_info[m]['tokens'] for m in others)
                off_q = 1 if has_cls and getattr(config, 'cls_position', 'last').lower() == 'first' else 0
                q_target_idx = slice(off_q + others_len, off_q + others_len + target_len)
                can_q_slice = modality_info[target_mod]['slice']

                for l, block in enumerate(model.transformer.h):
                    layer_attn = block.attn.extracted_attention[0].mean(dim=0).float().numpy()
                    
                    for j, source_mod in enumerate(active_modalities):
                        src_pos_in_order = order.index(source_mod)
                        src_offset_in_order = sum(B_base["X"][order[k]].shape[1] for k in range(src_pos_in_order))
                        k_source_idx = slice(off_q + src_offset_in_order, off_q + src_offset_in_order + modality_info[source_mod]['tokens'])
                        
                        strength = layer_attn[q_target_idx, k_source_idx].mean()
                        strength_matrix_acc[l, i, j] += strength
                        
                        if l == n_layers - 1:
                            can_k_slice = modality_info[source_mod]['slice']
                            fused_attn[can_q_slice, can_k_slice] = layer_attn[q_target_idx, k_source_idx]

            last_attn = fused_attn

        avg_attn_full += last_attn

        # 5. Save per-object plots
        random_suffix = "" if target_id in specific_tids else "_R"
        filename = f"attn_ID_{target_id}{run_suffix}_grid{random_suffix}.png"
        logger.info(f"[{batch_idx+1}/{len(indices_to_plot)}] ID={target_id} z={z_val:.3f}{' [random]' if random_suffix else ''}")

        plot_attention_grid_single(
            attn_matrix=last_attn,
            modality_info=modality_info,
            n_layers=n_layers,
            target_id=target_id,
            z_val=z_val,
            train_name=train_name,
            save_dir=save_dir,
            filename=filename,
        )

    # Final Population Average Plotting
    avg_attn_full /= len(indices_to_plot)
    strength_matrix_acc /= len(indices_to_plot)
    
    plot_attention_matrix_average(
        avg_full_attn=avg_attn_full,
        strength_matrix=strength_matrix_acc,
        modality_info=modality_info,
        n_layers=n_layers,
        num_samples=len(indices_to_plot),
        train_name=train_name,
        save_dir=save_dir,
        run_suffix=run_suffix,
    )

    logger.info(f"Done. All plots saved to {save_dir}")


if __name__ == "__main__":
    main()
