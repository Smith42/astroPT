"""
AstroPT V4: Dual-Phase Attention with Contrastive Alignment

This module extends the base AstroPT GPT model with:
1. Modality Expert Tokens — learnable summary vectors per modality, appended
   at the end of the sequence before the [CLS] token.
2. Dual-Phase Attention — The first `clip_fusion_layer` layers use a
   block-diagonal mask (unimodal isolation) while the remaining layers use
   standard causal attention (full fusion).
3. CLIP-style Contrastive Loss (InfoNCE) — Computed between the projected
   expert tokens at the boundary between phases, forcing cross-modal
   alignment without contamination.

The class `GPT_V4` inherits from `GPT` and overrides only the methods that
need to change. The original `model.py` is NOT modified.

Author: Victor Alonso Rodriguez
Date: May 2026
"""

import math
import itertools
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from .model import (
    GPT,
    GPTConfig,
    ModalityConfig,
    ModalityRegistry,
    AstroPTEmbeddingLayer,
    Block,
    Encoder,
    Decoder,
    Embedder,
    LayerNorm,
    new_gelu,
)

try:
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        flex_attention,
        or_masks,
    )
    _flex_available = True
except ImportError:
    _flex_available = False


# ---------------------------------------------------------------------------
# Helper Components
# ---------------------------------------------------------------------------

class CLIPProjector(nn.Module):
    """Projects a modality expert embedding into a shared contrastive space.

    Architecture: Linear → GELU → Linear → LayerNorm
    This bottleneck is standard in CLIP/SigLIP and prevents the raw hidden
    states from dominating the contrastive loss.
    """

    def __init__(self, n_embd: int, projection_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, projection_dim),
            nn.LayerNorm(projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_embd) → (batch, projection_dim), L2-normalised."""
        z = self.proj(x)
        return F.normalize(z, dim=-1)


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all DDP workers with support for backward propagation.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        world_size = torch.distributed.get_world_size()
        output = [torch.zeros_like(input) for _ in range(world_size)]
        torch.distributed.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        rank = torch.distributed.get_rank()
        grad_out = torch.zeros_like(input)
        grad_out.copy_(grads[rank])
        return grad_out


def info_nce_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.07,
    inv_temperature: Optional[torch.Tensor] = None,
    use_dist: bool = True,
) -> torch.Tensor:
    """Symmetric InfoNCE (CLIP-style) contrastive loss.
    Supports DDP training by gathering embeddings across all GPUs.

    Args:
        z_a: (B, D) L2-normalised embeddings from modality A.
        z_b: (B, D) L2-normalised embeddings from modality B.
        temperature: Softmax temperature (lower = sharper) if inv_temperature is None.
        inv_temperature: Prefolded inverse temperature. If specified, overrides temperature.
        use_dist: If True, uses distributed gather during multi-GPU training.
                  If False (e.g. during validation), bypasses DDP gather.

    Returns:
        Scalar loss averaged over both directions.
    """
    import torch.distributed as dist
    
    if inv_temperature is None:
        inv_temp = 1.0 / temperature
    else:
        inv_temp = inv_temperature
        
    if use_dist and dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Gather all embeddings with gradient flow support
        z_a_gathered = torch.cat(GatherLayer.apply(z_a), dim=0)
        z_b_gathered = torch.cat(GatherLayer.apply(z_b), dim=0)
        
        # Local batch size
        bsz = z_a.size(0)
        
        # Cosine similarity matrix: (B, world_size * B)
        logits_a = (z_a @ z_b_gathered.T) * inv_temp
        logits_b = (z_b @ z_a_gathered.T) * inv_temp
        
        # Targets: each local sample matches the corresponding gathered sample at (rank * bsz + i)
        labels = torch.arange(bsz, device=z_a.device) + rank * bsz
        
        loss_ab = F.cross_entropy(logits_a, labels)
        loss_ba = F.cross_entropy(logits_b, labels)
    else:
        # Standard local-only InfoNCE computation
        logits = (z_a @ z_b.T) * inv_temp
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.T, labels)
        
    return (loss_ab + loss_ba) / 2.0


# ---------------------------------------------------------------------------
# V4 Embedding Layer (extends base with Expert Tokens)
# ---------------------------------------------------------------------------

class AstroPTEmbeddingLayerV4(AstroPTEmbeddingLayer):
    """Extends the base embedding layer with per-modality expert tokens.

    Sequence layout produced:
        [patch₁ patch₂ ... patchN] [EXPERT_mod1] [EXPERT_mod2] ... [CLS]

    The expert tokens are learnable parameters (like the CLS token) that are
    appended BEFORE the CLS token. During the unimodal phase, each expert
    token only attends to patches of its own modality via the block-diagonal
    mask, producing a pure unimodal summary vector.
    """

    def __init__(self, config, modality_registry, encoders, embedders):
        super().__init__(config, modality_registry, encoders, embedders)

        # Create one learnable expert token per modality
        self.expert_tokens = nn.ParameterDict()
        for name in modality_registry.modalities.keys():
            self.expert_tokens[name] = nn.Parameter(
                torch.randn(1, 1, config.n_embd) * 0.02
            )

    @torch.compiler.disable
    def _build_modality_index_list(self, current_pos, seq_len):
        return list(range(current_pos, current_pos + seq_len))

    def forward(self, inputs, batch_modes):
        """Build the full sequence including expert tokens.

        Returns:
            tok_emb: (B, seq_len, n_embd) — full sequence
            expert_positions: dict mapping modality name → index in the sequence
            modality_indices: dict mapping modality name → list of patch indices
        """
        bsz = inputs[batch_modes[0]].size(0)
        sequences = []
        modality_indices = {}  # Track which indices belong to which modality
        current_pos = 0

        for mod_name in batch_modes:
            x_tok = self.encoders[mod_name](inputs[mod_name])

            # Modality Embedding
            if self.modality_embs is not None and mod_name in self.modality_embs:
                x_tok = x_tok + self.modality_embs[mod_name]

            # Aperture Embedding
            if self.aperture_emb is not None and f"{mod_name}_aperture" in inputs:
                x_tok = x_tok + self.aperture_emb(inputs[f"{mod_name}_aperture"])

            # Positional Embedding
            x_pos = self.embedders[mod_name](inputs[mod_name + "_positions"])

            seq_len = x_tok.size(1)
            # Use disabled helper to avoid SymInt loop/range graph breaks
            modality_indices[mod_name] = self._build_modality_index_list(current_pos, seq_len)
            current_pos += seq_len
            sequences.append(x_tok + x_pos)

        # Concatenate all patch sequences
        tok_emb = torch.cat(sequences, dim=1)

        # Append expert tokens (one per modality, in batch_modes order)
        expert_positions = {}
        for mod_name in batch_modes:
            expert = self.expert_tokens[mod_name].expand(bsz, -1, -1)
            tok_emb = torch.cat([tok_emb, expert], dim=1)
            expert_positions[mod_name] = current_pos
            current_pos += 1

        # Append CLS token at the very end
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(bsz, -1, -1)
            tok_emb = torch.cat([tok_emb, cls_tokens], dim=1)

        return tok_emb, expert_positions, modality_indices


# ---------------------------------------------------------------------------
# GPT V4 Model
# ---------------------------------------------------------------------------

class GPT_V4(GPT):
    """AstroPT V4 with Dual-Phase Attention and Contrastive Alignment.

    Inherits from GPT and overrides:
    - _init_native_backbone: adds expert tokens and CLIP projectors.
    - _forward_native: implements dual-phase masking and CLIP loss.
    - get_embeddings: returns expert embeddings in addition to modality ones.
    """

    def _init_native_backbone(self, config):
        """Initialize native backbone with V4 extensions."""
        # Call parent to set up transformers, encoders, decoders, embedders
        super()._init_native_backbone(config)

        # Replace embedding layer with V4 version (adds expert tokens)
        self.embedding_layer = AstroPTEmbeddingLayerV4(
            config, self.modality_registry, self.encoders, self.embedders
        )

        # Resolve the fusion layer boundary
        if config.clip_fusion_layer < 0:
            # Default: n_layer // 2, with odd layers favoring fusion
            self._fusion_layer = config.n_layer // 2
        else:
            self._fusion_layer = min(config.clip_fusion_layer, config.n_layer - 1)

        # CLIP Projection heads (one per modality)
        proj_dim = getattr(config, "clip_projection_dim", 256)
        self.clip_projectors = nn.ModuleDict({
            name: CLIPProjector(config.n_embd, proj_dim)
            for name in self.modality_registry.modalities.keys()
        })

        # Learnable log inverse temperature for CLIP
        init_temp = getattr(config, "clip_temperature", 0.07)
        self.clip_ln_inv_temperature = nn.Parameter(torch.log(torch.tensor(1.0 / init_temp)))

        # Compile flex_attention for performance if available
        if _flex_available:
            self._flex_attention_fn = torch.compile(flex_attention)
        else:
            self._flex_attention_fn = None

    @torch.compiler.disable
    def _build_unimodal_mask_fn(self, modality_indices, expert_positions, total_seq_len):
        """Build a FlexAttention mask function for the unimodal phase.

        Rules:
        - Patch of modality X can only attend to earlier patches of modality X (causal within modality).
        - Expert token of modality X attends to ALL patches of modality X (bidirectional within modality).
        - CLS token attends to everything (bidirectional).
        - Expert tokens and CLS do NOT attend to each other's patches across modalities.

        Args:
            modality_indices: dict[str, list[int]] — patch positions per modality.
            expert_positions: dict[str, int] — expert token position per modality.
            total_seq_len: int — total sequence length including experts and CLS.

        Returns:
            A mask_mod function for create_block_mask.
        """
        # Pre-compute sets for O(1) lookup (converted to tensors for the mask function)
        # We need to know for each position: which modality does it belong to?
        pos_to_mod_id = torch.zeros(total_seq_len, dtype=torch.long)
        mod_name_to_id = {}
        expert_pos_set = set()
        cls_pos = total_seq_len - 1  # CLS is always last

        for idx, mod_name in enumerate(modality_indices.keys()):
            mod_name_to_id[mod_name] = idx
            for pos in modality_indices[mod_name]:
                pos_to_mod_id[pos] = idx
            # Expert token for this modality
            exp_pos = expert_positions[mod_name]
            pos_to_mod_id[exp_pos] = idx
            expert_pos_set.add(exp_pos)

        # CLS gets a special modality ID
        cls_mod_id = len(mod_name_to_id)
        pos_to_mod_id[cls_pos] = cls_mod_id

        # Convert to sets for fast lookup
        expert_positions_tensor = torch.tensor(sorted(expert_pos_set), dtype=torch.long)

        def unimodal_mask(b, h, q_idx, kv_idx):
            q_mod = pos_to_mod_id[q_idx]
            kv_mod = pos_to_mod_id[kv_idx]

            # CLS token (query) can attend to everything
            is_cls_query = (q_idx == cls_pos)

            # Expert token (query) can attend to all patches of its own modality
            is_expert_query = torch.tensor(False)
            for ep in expert_positions_tensor:
                is_expert_query = is_expert_query | (q_idx == ep)

            # Same modality check
            same_mod = (q_mod == kv_mod)

            # Causal within same modality (for patches)
            causal_same = same_mod & (q_idx >= kv_idx)

            # Expert attends to all of its modality (not just causal)
            expert_to_own = is_expert_query & same_mod

            # CLS attends to everything
            cls_to_all = is_cls_query

            return causal_same | expert_to_own | cls_to_all

        unimodal_mask.__name__ = "unimodal_block_diagonal_mask"
        return unimodal_mask

    @torch.compiler.disable
    def _build_unimodal_dense_mask(self, modality_indices, expert_positions, total_seq_len, device):
        """Vectorized construction of the unimodal dense mask."""
        # 1. Map each position to a modality ID (0, 1, 2...)
        mod_to_id = {name: i for i, name in enumerate(self.mod_names)}
        modality_ids = torch.full((total_seq_len,), -1, dtype=torch.long, device=device)
        
        for mod_name, indices in modality_indices.items():
            if indices:
                modality_ids[torch.tensor(indices, device=device)] = mod_to_id[mod_name]
        
        for mod_name, pos in expert_positions.items():
            modality_ids[pos] = mod_to_id[mod_name]

        # 2. Block-diagonal mask: tokens can only see tokens of the same modality
        # (B, T) == (T, B) -> (T, T)
        same_modality = (modality_ids.unsqueeze(0) == modality_ids.unsqueeze(1))
        
        # 3. Causal mask for patches
        # Important: Experts see ALL patches of their modality, but patches are causal
        is_expert = torch.zeros(total_seq_len, dtype=torch.bool, device=device)
        for pos in expert_positions.values():
            is_expert[pos] = True
        
        # Matrix of (query_is_expert)
        q_is_expert = is_expert.unsqueeze(1) # (T, 1)
        
        # Causal part: tril for everyone
        causal_mask = torch.tril(torch.ones(total_seq_len, total_seq_len, dtype=torch.bool, device=device))
        
        # Rule: Attend if (same modality AND (causal OR query is expert))
        mask = same_modality & (causal_mask | q_is_expert)
        
        # 4. CLS token (last) attends to everything
        mask[-1, :] = True
        
        # 5. Ensure self-attention
        mask.fill_diagonal_(True)

        return mask.unsqueeze(0).unsqueeze(0)

    @torch.compiler.disable
    def _forward_block_with_mask(self, block, x, attn_mask):
        """Forward a transformer block with an explicit dense attention mask.

        This bypasses the block's default attention path to inject our
        unimodal mask via scaled_dot_product_attention's attn_mask parameter.
        """
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

        # Apply attention with our custom mask
        # Convert boolean mask to float mask for SDPA: True -> 0.0, False -> -inf
        float_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        float_mask.masked_fill_(~attn_mask, float('-inf'))

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=float_mask,
            dropout_p=block.attn.dropout if block.attn.training else 0,
            is_causal=False,  # We handle causality via the mask
        )
        y = y.transpose(1, 2).contiguous().reshape(B, -1, C)

        # Output projection + residual
        y = block.attn.resid_dropout(block.attn.c_proj(y))
        x = x + y

        # MLP + residual
        x = x + block.mlp(block.ln_2(x))
        return x

    @torch.compiler.disable
    def _update_modality_indices_after_mix(self, batch_modes, mod_lengths, patch_interleaved):
        new_modality_indices = {m: [] for m in batch_modes}
        cumulative = [sum(mod_lengths[:k]) for k in range(len(batch_modes) + 1)]
        for new_pos, old_pos in enumerate(patch_interleaved.tolist()):
            for k, mod in enumerate(batch_modes):
                if cumulative[k] <= old_pos < cumulative[k + 1]:
                    new_modality_indices[mod].append(new_pos)
                    break
        return new_modality_indices

    def _forward_native(
        self,
        inputs,
        targets=None,
        prefix_len=None,
        target_modality=None,
        attention_mask=None,
        dropped_modality=None,
    ):
        """Dual-phase forward pass.

        Phase 1 (layers 0..fusion_layer-1): Unimodal block-diagonal mask.
        Phase 2 (layers fusion_layer..n_layer-1): Standard causal mask.
        CLIP loss computed at the phase boundary from expert tokens.
        """
        batch_modes = [name for name in inputs if name in self.mod_names]
        if not batch_modes:
            batch_modes = [name for name in self.mod_names if name in inputs]

        # --- If only one modality present, fall back to standard V3 forward ---
        # We can't simply call super()._forward_native() because self.embedding_layer
        # is the V4 version that returns a tuple. Instead, we run V4 forward
        # without the CLIP loss (it requires >= 2 modalities).
        if len(batch_modes) < 2:
            x, expert_positions, modality_indices = self.embedding_layer(inputs, batch_modes)
            x = self.transformer.drop(x)
            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.ln_f(x)

            # Decode: simple linear sequencing for the single modality
            outputs = {}
            mod_name = batch_modes[0]
            seq_len = inputs[mod_name].size(1)
            # Hidden states [0..seq_len-2] predict tokens [1..seq_len-1]
            hidden_state = x[:, :seq_len - 1]
            outputs[mod_name] = self.decoders[mod_name](hidden_state)

            if targets is not None:
                target = targets[mod_name]
                pred = outputs[mod_name]
                if target.shape[1] > pred.shape[1]:
                    target = target[:, -pred.shape[1]:]
                if self.loss_type in ["l1", "mae"]:
                    loss = F.l1_loss(pred, target)
                elif self.loss_type == "mse":
                    loss = F.mse_loss(pred, target)
                else:
                    loss = F.huber_loss(pred, target, delta=self.loss_huber_delta)
            else:
                loss = None

            return outputs, loss

        # =====================================================================
        # EMBEDDING (V4: includes expert tokens)
        # =====================================================================
        x, expert_positions, modality_indices = self.embedding_layer(inputs, batch_modes)

        # Apply token mixing (reorder patches) if configured
        interleaved_idx = None
        if self.config.use_token_mixing and len(batch_modes) >= 2:
            stochastic = getattr(self.config, 'token_mixing_stochastic', False)
            min_block = max(1, getattr(self.config, 'token_mixing_min_block_size', 1))
            max_block = max(min_block, getattr(self.config, 'token_mixing_max_block_size',
                           getattr(self.config, 'token_mixing_block_size', 5)))

            # Build lengths for PATCH tokens only (exclude experts and CLS)
            # Ensure sequence lengths are concrete integers to prevent torch.compile tracing issues 
            # with SymInts failing during dynamic block scheduling evaluation.
            mod_lengths = [int(inputs[m].size(1)) for m in batch_modes]
            total_patches = sum(mod_lengths)

            # Check if aperture embedding is enabled and we have both EuclidImage and DESISpectrum
            if self.config.use_aperture_embedding and "EuclidImage" in batch_modes and "DESISpectrum" in batch_modes:
                img_idx = batch_modes.index("EuclidImage")
                spec_idx = batch_modes.index("DESISpectrum")

                # Compute core patches dynamically exactly as in EuclidImage
                pixel_scale = 0.1
                aperture_radius_arcsec = 1.5
                patch_size = getattr(self.config, "images_patch_size", 16)

                fiber_diameter_pixels = 2.0 * aperture_radius_arcsec / pixel_scale
                patches_needed = fiber_diameter_pixels / patch_size
                m = math.ceil(patches_needed)
                if m % 2 == 0:
                    m += 1
                m = max(1, m)
                k_core = m * m
                k_core = min(k_core, mod_lengths[img_idx])

                # Core-only token mixing round-robin block scheduling
                lengths_to_interleave = [k_core, mod_lengths[spec_idx]] if img_idx < spec_idx else [mod_lengths[spec_idx], k_core]

                rel_interleaved = self._get_interleaved_indices(
                    lengths_to_interleave, stochastic, min_block, max_block,
                    self.config.token_mixing_block_size, 'last',  # No CLS in patch region
                    x.device,
                )
                rel_interleaved = rel_interleaved[:-1]  # Remove trailing CLS index

                # Map relative interleaved back to absolute indices
                offsets = [sum(mod_lengths[:i]) for i in range(len(batch_modes))]
                img_offset = offsets[img_idx]
                spec_offset = offsets[spec_idx]

                abs_interleaved_core_spec = []
                for rel_idx in rel_interleaved.tolist():
                    if img_idx < spec_idx:
                        if rel_idx < k_core:
                            abs_interleaved_core_spec.append(img_offset + rel_idx)
                        else:
                            abs_interleaved_core_spec.append(spec_offset + (rel_idx - k_core))
                    else:
                        if rel_idx < mod_lengths[spec_idx]:
                            abs_interleaved_core_spec.append(spec_offset + rel_idx)
                        else:
                            abs_interleaved_core_spec.append(img_offset + (rel_idx - mod_lengths[spec_idx]))

                # Euclid outskirts go at the end of the patch sequence
                abs_outskirts = list(range(img_offset + k_core, img_offset + mod_lengths[img_idx]))
                patch_interleaved = torch.tensor(abs_interleaved_core_spec + abs_outskirts, device=x.device, dtype=torch.long)
            else:
                # Standard full token mixing
                cls_pos_cfg = getattr(self.config, 'cls_position', 'last').lower()
                patch_interleaved = self._get_interleaved_indices(
                    mod_lengths, stochastic, min_block, max_block,
                    self.config.token_mixing_block_size, 'last',
                    x.device,
                )
                # The helper appends a CLS index at the end — remove it
                patch_interleaved = patch_interleaved[:total_patches]

            # Build full reorder: interleaved patches + expert tokens + CLS (unchanged)
            n_experts = len(batch_modes)
            has_cls = getattr(self.config, 'use_cls_token', False)
            suffix_indices = torch.arange(total_patches, total_patches + n_experts + (1 if has_cls else 0), device=x.device, dtype=torch.long)
            full_interleaved = torch.cat([
                patch_interleaved,
                suffix_indices,
            ])
            interleaved_idx = full_interleaved

            # Update modality_indices to reflect the new positions after interleaving
            modality_indices = self._update_modality_indices_after_mix(
                batch_modes, mod_lengths, patch_interleaved
            )

            # Expert positions stay in order after patches
            for i, mod_name in enumerate(batch_modes):
                expert_positions[mod_name] = total_patches + i

            # Reorder the sequence
            x = x[:, full_interleaved, :]

        total_seq_len = x.size(1)

        # =====================================================================
        # BUILD UNIMODAL MASK
        # =====================================================================
        # We need a mask that isolates modalities in the first phase.
        # On GPU with PyTorch >= 2.6: use FlexAttention (compiled, fast).
        # On CPU or older PyTorch: build a dense boolean mask manually.

        unimodal_attn_mask = self._build_unimodal_dense_mask(
            modality_indices, expert_positions, total_seq_len, x.device
        )

        # =====================================================================
        # PHASE 1: Unimodal Isolation (layers 0 .. fusion_layer-1)
        # =====================================================================
        x = self.transformer.drop(x)
        for layer_idx in range(self._fusion_layer):
            block = self.transformer.h[layer_idx]
            x = self._forward_block_with_mask(block, x, unimodal_attn_mask)

        # =====================================================================
        # CLIP LOSS: Extract expert embeddings and compute InfoNCE
        # =====================================================================
        clip_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        expert_embeddings = {}

        for mod_name in batch_modes:
            exp_pos = expert_positions[mod_name]
            expert_embeddings[mod_name] = x[:, exp_pos, :]  # (B, n_embd)

        # Project and compute pairwise InfoNCE
        clip_temperature = getattr(self.config, 'clip_temperature', 0.07)
        projected = {}
        for mod_name, emb in expert_embeddings.items():
            projected[mod_name] = self.clip_projectors[mod_name](emb)

        # Compute dynamic inverse temperature
        if hasattr(self, "clip_ln_inv_temperature"):
            # Exp and clamp to prevent numerical instability
            inv_temp = torch.exp(self.clip_ln_inv_temperature).clamp(max=100.0)
        else:
            inv_temp = torch.tensor(1.0 / clip_temperature, device=x.device, dtype=x.dtype)

        mod_keys = list(projected.keys())
        n_pairs = 0
        for i in range(len(mod_keys)):
            for j in range(i + 1, len(mod_keys)):
                clip_loss = clip_loss + info_nce_loss(
                    projected[mod_keys[i]],
                    projected[mod_keys[j]],
                    inv_temperature=inv_temp,
                    use_dist=self.training,
                )
                n_pairs += 1
        if n_pairs > 0:
            clip_loss = clip_loss / n_pairs

        # =====================================================================
        # PHASE 2: Full Causal Fusion (layers fusion_layer .. n_layer-1)
        # =====================================================================
        for layer_idx in range(self._fusion_layer, self.config.n_layer):
            block = self.transformer.h[layer_idx]
            # Standard causal attention — no block mask needed
            x = block(x)

        x = self.transformer.ln_f(x)

        # =====================================================================
        # DECODING: Route hidden states to modality decoders
        # =====================================================================
        outputs = {}

        if interleaved_idx is not None:
            # --- INTERLEAVED DECODING (same logic as V3 but adapted for expert/CLS suffix) ---
            tgt_idx = interleaved_idx
            has_cls = getattr(self.config, 'use_cls_token', False)
            mod_lengths = [inputs[m].size(1) for m in batch_modes]
            total_mod_tokens = sum(mod_lengths)
            cumulative = [sum(mod_lengths[:k]) for k in range(len(batch_modes) + 1)]

            # Hidden states for next-token prediction (shift by 1)
            # The sequence layout is exactly: [interleaved patches] + [experts] + [CLS]
            # Since we only want to predict patches from patches, we slice up to total_patches - 1.
            # Because patches[0] predicts patches[1] ... patches[total_patches-2] predicts patches[total_patches-1]
            hidden = x[:, :total_patches - 1, :]
            tgt_idx_filtered = tgt_idx[1:total_patches]

            # Route to decoders
            aligned_targets = {} if targets is not None else None
            for k, mod in enumerate(batch_modes):
                is_mod_k = (tgt_idx_filtered >= cumulative[k]) & (tgt_idx_filtered < cumulative[k + 1])
                if is_mod_k.any():
                    outputs[mod] = self.decoders[mod](hidden[:, is_mod_k, :])
                    if targets is not None and mod in targets:
                        local_idx = tgt_idx_filtered[is_mod_k] - cumulative[k]
                        aligned_targets[mod] = targets[mod][:, local_idx]

            if aligned_targets:
                outputs["_aligned_targets"] = aligned_targets

        else:
            # --- STANDARD LINEAR SEQUENCING (no mixing) ---
            current_idx = 0
            for mod_name in batch_modes:
                input_tensor = inputs[mod_name]
                seq_len = input_tensor.size(1)
                hidden_state = x[:, current_idx: current_idx + seq_len - 1]
                if current_idx > 0:
                    # Previous token predicts current first token
                    hidden_state = x[:, current_idx - 1: current_idx + seq_len - 1]
                outputs[mod_name] = self.decoders[mod_name](hidden_state)
                current_idx += seq_len

        # =====================================================================
        # LOSS COMPUTATION
        # =====================================================================
        if targets is not None:
            loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            num_contributors = 0

            for mod_name in batch_modes:
                if mod_name not in outputs or mod_name not in targets:
                    continue

                pred = outputs[mod_name]
                target = targets[mod_name] if "_aligned_targets" not in outputs else outputs["_aligned_targets"].get(mod_name, targets[mod_name])

                # Align shapes
                if target.shape[1] != pred.shape[1]:
                    if target.shape[1] > pred.shape[1]:
                        target = target[:, -pred.shape[1]:]
                    else:
                        pred = pred[:, :target.shape[1]]

                mod_loss_weight = self.mod_loss_weights[mod_name]

                # Cross-reconstruction boost
                if getattr(self.config, 'cross_reconstruction_loss_use', False) and dropped_modality == mod_name:
                    mod_loss_weight *= getattr(self.config, 'cross_reconstruction_weight', 1.0)

                pred = pred.contiguous()
                target = target.contiguous()

                if self.loss_type in ["l1", "mae"]:
                    mod_loss = F.l1_loss(pred, target)
                elif self.loss_type == "mse":
                    mod_loss = F.mse_loss(pred, target)
                else:
                    mod_loss = F.huber_loss(pred, target, delta=self.loss_huber_delta)

                loss = loss + mod_loss * mod_loss_weight
                num_contributors += 1

            if num_contributors > 0:
                loss = loss / num_contributors

            # Add CLIP loss
            clip_weight = getattr(self.config, 'clip_loss_weight', 0.5)
            total_loss = loss + clip_weight * clip_loss

            # Store component losses for diagnostics
            outputs["_clip_loss"] = clip_loss.detach()
            outputs["_reconstruction_loss"] = loss.detach()
        else:
            total_loss = None

        return outputs, total_loss

    def get_embeddings(self, inputs, draw_from_centre=True, prefix_len=None, batch_modes=None):
        """Get embeddings including expert token embeddings (pure unimodal).

        Returns a dict with:
        - Per-modality patch embeddings (mean-pooled).
        - Expert embeddings (pure unimodal, from fusion boundary).
        - CLS embedding (joint, from final layer).
        """
        if batch_modes is None:
            batch_modes = [k for k in inputs if k in self.mod_names]

        # Build sequence with expert tokens
        x, expert_positions, modality_indices = self.embedding_layer(inputs, batch_modes)

        # Build the unimodal mask for Phase 1
        total_seq_len = x.size(1)
        unimodal_attn_mask = self._build_unimodal_dense_mask(
            modality_indices, expert_positions, total_seq_len, x.device
        )

        # Phase 1: Unimodal layers (applying mask to keep modalities isolated)
        for layer_idx in range(self._fusion_layer):
            block = self.transformer.h[layer_idx]
            x = self._forward_block_with_mask(block, x, unimodal_attn_mask)

        # Capture expert embeddings at the fusion boundary (Phase 1: Pure Unimodal)
        expert_result_p1 = {}
        for mod_name in batch_modes:
            exp_pos = expert_positions[mod_name]
            expert_result_p1[f"{mod_name}_phase1"] = x[:, exp_pos:exp_pos + 1, :]

        if draw_from_centre:
            embeddings_out = x
        else:
            # Phase 2: Fusion layers
            for layer_idx in range(self._fusion_layer, self.config.n_layer):
                x = self.transformer.h[layer_idx](x)
            embeddings_out = self.transformer.ln_f(x)

        # Split modality patch embeddings
        result = {}
        for mod_name in batch_modes:
            indices = modality_indices[mod_name]
            if indices:
                result[mod_name] = embeddings_out[:, indices, :]
            
            # Capture expert embeddings at the final layer (Phase 2: Multimodal)
            exp_pos = expert_positions[mod_name]
            result[f"{mod_name}_phase2"] = embeddings_out[:, exp_pos:exp_pos + 1, :]

        # CLS embedding (last position)
        has_cls = getattr(self.config, 'use_cls_token', False)
        if has_cls:
            result["cls"] = embeddings_out[:, -1:, :]

        # Merge Phase 1 expert embeddings
        result.update(expert_result_p1)

        return result