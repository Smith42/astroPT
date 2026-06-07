# Jetformer Integration into AstroPT

## Table of Contents
1. [Overview](#overview)
2. [What is Jetformer?](#what-is-jetformer)
3. [What is AstroPT?](#what-is-astropt)
4. [Integration Goals](#integration-goals)
5. [Architectural Challenges](#architectural-challenges)
6. [Design Decisions](#design-decisions)
7. [Implementation Details](#implementation-details)
8. [Changes to Jetformer](#changes-to-jetformer)
9. [Changes to AstroPT](#changes-to-astropt)
10. [Data Flow Comparison](#data-flow-comparison)
11. [Usage](#usage)
12. [Technical Notes](#technical-notes)

---

## Overview

This document describes the integration of **Jetformer** (a continuous tokenization approach using normalizing flows) into **AstroPT** (a multimodal GPT framework for astronomical data). The integration allows AstroPT to support three tokenization methods: **AIM**, **Affine**, and **Jetformer**, all sharing the same GPT backbone while maintaining full backward compatibility.

---

## What is Jetformer?

Jetformer is a continuous tokenization approach for images that uses **normalizing flows** and **Gaussian Mixture Models (GMM)** instead of discrete tokens. The original implementation (`scripts/jetformer/train_jetformer.py`) was a standalone model called `JetFormerLite`.

### Key Components of Original Jetformer

1. **Normalizing Flow (TinyFlow)**: 
   - RealNVP-style affine coupling layers operating on **image space** `[B, C, H, W]`
   - Uses 2D convolutions with checkerboard masking
   - Transforms raw images → latent `z` with log-determinant `logdet`

2. **Patchification**:
   - After flow, converts latent images `z [B, C, H, W]` → patch tokens `[B, T, D]`
   - Each patch is a flattened 16×16×3 region (768-dimensional)

3. **Autoregressive Transformer (TinyGPT)**:
   - Standard decoder-only GPT architecture
   - Processes sequence of patchified latent tokens

4. **GMM Output Head**:
   - Predicts parameters of a Gaussian Mixture Model for each token
   - Outputs: mixture weights `logits_pi`, means `mu`, log-std `log_sigma`

5. **Loss Function**:
   - Negative log-likelihood: `loss = (NLL_GMM(z) - logdet).mean()`
   - Can be **negative** when `logdet` is large (expected behavior)

6. **Noise Curriculum**:
   - Anneals noise added to tokens during training for stability
   - `sigma = noise_max + (noise_min - noise_max) * epoch_frac`

### Original Workflow

```
Raw Image [B,C,H,W]
  ↓ uniform_dequantize
  ↓ TinyFlow (image space) → z [B,C,H,W], logdet [B]
  ↓ patchify → tokens [B,T,D]
  ↓ add noise (curriculum)
  ↓ project to embeddings
  ↓ GPT
  ↓ GMM Head
  ↓ Loss: NLL_GMM(tokens[:,1:]) - logdet
```

---

## What is AstroPT?

AstroPT is a **multimodal GPT framework** designed for astronomical data (galaxy images, spectra, etc.). It provides:

1. **Unified GPT Backbone**: Single transformer architecture for all modalities
2. **Modality Registry**: Flexible system for managing different data types
3. **Multiple Tokenizers**: Originally supported **AIM** and **Affine** tokenizers
4. **Training Infrastructure**: Complete training loop with validation, checkpointing, logging

### Original AstroPT Architecture

- **Input**: Patch tokens `[B, T, D]` (produced by dataset)
- **Encoder**: Projects patches → embeddings `[B, T, n_embd]`
- **GPT**: Standard transformer with causal attention
- **Decoder**: Projects embeddings → patch predictions
- **Loss**: Huber loss for regression

### Original Tokenizers

1. **AIM Tokenizer**:
   - Two-layer MLP: `Linear → GELU → Linear`
   - Used for continuous patch tokens

2. **Affine Tokenizer**:
   - Single linear layer
   - Simpler projection

Both operate on **already-patchified** tokens `[B, T, D]` from the dataset.

---

## Integration Goals

The primary goal was to integrate Jetformer into AstroPT **without breaking existing functionality**, allowing users to choose between:
- `tokeniser="aim"` (original AIM)
- `tokeniser="affine"` (original Affine)
- `tokeniser="jetformer"` (new continuous tokenization)

All three should:
- Share the same GPT backbone
- Use the same training infrastructure
- Support the same modalities (initially images-only for Jetformer)
- Maintain backward compatibility

---

## Architectural Challenges

### Challenge 1: Input Format Mismatch

**Problem**: 
- AstroPT expects **patch tokens** `[B, T, D]` as input
- Jetformer needs **raw images** `[B, C, H, W]` to apply the flow

**Solution**: 
- Created `JetformerImageEncoder` that accepts raw images
- Modified data loading to pass raw images when `tokeniser=="jetformer"`
- Added `prepare_batch_for_jetformer()` to swap patches for raw images

### Challenge 2: Flow Operation Space

**Problem**: 
- Original Jetformer flows on **image space** `[B, C, H, W]`
- AstroPT's architecture assumes patch tokens `[B, T, D]`

**Decision**: 
- **Option A**: Flow on patch tokens (simpler integration)
- **Option B**: Flow on image space (matches original, better quality)

**Chosen**: **Option B** - Flow on image space to match original JetFormerLite exactly.

**Rationale**: 
- Preserves original architecture and quality
- More principled (flows operate on natural image structure)
- Better matches the original paper's design

### Challenge 3: Loss Function Difference

**Problem**: 
- AstroPT uses **Huber loss** (regression)
- Jetformer uses **GMM NLL - logdet** (likelihood-based, can be negative)

**Solution**: 
- Added conditional branch in `GPT._forward_native()`
- When `tokeniser=="jetformer"` and `modalities==["images"]`, use Jetformer loss path
- Otherwise, use original Huber loss path

### Challenge 4: Teacher Forcing / Cheating Protection

**Problem**: 
- AstroPT's `process_modes()` slices inputs/targets for teacher forcing
- For Jetformer, we pass full raw images (no slicing)

**Solution**: 
- Added `images_is_raw` flag to skip slicing for Jetformer
- GPT's causal attention still enforces autoregressive property
- Loss uses `tokens[:, 1:]` as target (next-token prediction)

### Challenge 5: Memory Usage

**Problem**: 
- Raw images `[B, C, H, W]` use more memory than patches `[B, T, D]`
- Multiple DataLoader workers multiply memory usage

**Solution**: 
- Reduced `num_workers` for Jetformer (2-4 instead of 16)
- Added `prefetch_factor` control
- Flow operations are efficient (invertible, no gradients stored)

### Challenge 6: Reconstruction / Validation

**Problem**: 
- AstroPT validation expects patch tokens for visualization
- Jetformer needs to reconstruct from raw images

**Solution**: 
- Added `jetformer_reconstruct_images()` method
- Converts raw images → flow → patchify → GPT → GMM → depatchify → inverse flow
- Validation code branches to handle both formats

---

## Design Decisions

This section details the critical design decisions made during integration, explaining **why** each decision was necessary, **what options** were considered, and **what changes** each option would require.

---

### Decision 1: Flow Location (F1 vs F2)

**The Problem**: 
Where should the normalizing flow live in the architecture? The flow transforms raw images `[B,C,H,W]` → latent `z [B,C,H,W]`, but AstroPT's encoder expects to process tokens. We need to decide where this transformation happens.

**Options Considered**:

**F1: Flow inside `JetformerImageEncoder`**
- Flow is part of the encoder's `forward()` method
- Encoder accepts raw images, applies flow internally
- Flow → patchify → project happens sequentially in encoder

**F2: Flow as separate preprocessing step**
- Flow exists as standalone module before encoder
- Training loop calls flow explicitly: `z = flow(x)`, then `tokens = patchify(z)`, then `embeddings = encoder(tokens)`
- Encoder only sees patchified tokens

**What Each Option Changes**:

**F1 Changes**:
- `JetformerImageEncoder.forward()` becomes: `x [B,C,H,W] → flow → z → patchify → tokens → project → embeddings`
- Encoder is responsible for entire transformation pipeline
- Training loop just calls `encoder(x)` with raw images
- **Code location**: All logic in `src/astropt/model.py` `JetformerImageEncoder` class

**F2 Changes**:
- Training loop becomes: `z, logdet = flow(x)`, `tokens = patchify(z)`, `embeddings = encoder(tokens)`
- Encoder remains simple (just projects tokens)
- Flow and patchify are separate steps
- **Code location**: Flow logic in training script, encoder in model.py

**Trade-offs**:

| Aspect | F1 (Inside Encoder) | F2 (Separate Step) |
|--------|---------------------|---------------------|
| **Encapsulation** | ✅ All transformation logic in one place | ❌ Logic scattered across files |
| **Training Loop Complexity** | ✅ Simple: just `encoder(x)` | ❌ Complex: multiple steps |
| **Reusability** | ✅ Encoder is self-contained | ⚠️ Flow can be reused separately |
| **Matches Original** | ✅ Matches JetFormerLite design | ❌ Different from original |
| **Code Organization** | ✅ Follows AstroPT patterns | ❌ Breaks encoder abstraction |

**Chosen**: **F1** - Flow inside `JetformerImageEncoder`

**Why**: 
- **Encapsulation**: All image→token transformation logic lives in one place, following AstroPT's encoder abstraction
- **Matches Original**: Original JetFormerLite has flow as part of the model, not separate preprocessing
- **Cleaner Training Loop**: Training script doesn't need to know about flow/patchify details
- **Consistency**: Other encoders (AIM/Affine) also handle their own transformations internally

**Implementation Impact**:
```python
# With F1 (chosen):
class JetformerImageEncoder(Encoder):
    def forward(self, x):  # x is [B,C,H,W]
        z, logdet = self.flow(x)  # Flow inside encoder
        tokens = patchify(z)
        # ... rest of transformation

# Training loop:
embeddings = encoder(raw_images)  # Simple!

# With F2 (not chosen):
# Training loop would be:
z, logdet = flow(raw_images)  # Flow outside encoder
tokens = patchify(z)
embeddings = encoder(tokens)  # Encoder only projects
```

---

### Decision 2: Noise Schedule Management (L1 vs L2 vs L3)

**The Problem**: 
Jetformer needs a noise curriculum that anneals from `noise_max` to `noise_min` over training. The noise fraction `epoch_frac` must be computed from `iter_num` and `max_iters`, then passed to the encoder. How should this be managed?

**Options Considered**:

**L1: Pass `epoch_frac` as parameter to `GPT.forward()`**
- `GPT.forward(inputs, targets, epoch_frac=None)`
- Training loop computes `epoch_frac = iter_num / max_iters`
- Passes it through: `model(B["X"], targets=B["Y"], epoch_frac=epoch_frac)`
- GPT forwards it to encoder

**L2: Compute `epoch_frac` inside GPT using global iteration counter**
- GPT maintains `self.current_iter` and `self.max_iters`
- Training loop calls `model.set_iteration(iter_num, max_iters)` before forward
- GPT computes `epoch_frac` internally and passes to encoder

**L3: Dedicated setter method `set_jetformer_schedule()`**
- GPT has `set_jetformer_schedule(iter_num, max_iters)` method
- Training loop calls it before forward pass
- GPT computes `epoch_frac` and stores in `self.jet_epoch_frac`
- Encoder reads from `self.jet_epoch_frac` (threaded via `set_jet_epoch_frac()`)

**What Each Option Changes**:

**L1 Changes**:
- `GPT.forward()` signature: `forward(self, inputs, targets=None, epoch_frac=None, ...)`
- All forward calls must pass `epoch_frac` (even when None for non-Jetformer)
- Encoder receives `epoch_frac` as parameter
- **Code location**: Changes in `model.py` forward signature, all call sites

**L2 Changes**:
- GPT has `self.current_iter`, `self.max_iters` attributes
- Training loop: `model.set_iteration(iter_num, max_iters)` before each forward
- GPT computes `epoch_frac` and passes to encoder
- **Code location**: New method in GPT, training loop calls it

**L3 Changes**:
- GPT has `self.jet_epoch_frac` attribute
- New method: `GPT.set_jetformer_schedule(iter_num, max_iters)`
- Training loop calls setter before forward
- Encoder reads from cached value
- **Code location**: New setter method, training loop calls it, encoder reads cached value

**Trade-offs**:

| Aspect | L1 (Parameter) | L2 (Internal Counter) | L3 (Setter Method) |
|--------|----------------|----------------------|-------------------|
| **Forward Signature** | ❌ Changes signature (breaking) | ✅ Clean signature | ✅ Clean signature |
| **Training Loop** | ⚠️ Must pass parameter always | ⚠️ Must call setter always | ⚠️ Must call setter for Jetformer |
| **Flexibility** | ✅ Can vary per batch | ❌ Fixed per forward | ❌ Fixed per forward |
| **Backward Compat** | ❌ Changes all forward calls | ✅ No signature change | ✅ No signature change |
| **Clarity** | ⚠️ Parameter might be None | ✅ Explicit setter call | ✅ Explicit setter call |
| **State Management** | ✅ No state needed | ❌ GPT tracks iteration | ⚠️ GPT caches fraction |

**Chosen**: **L3** - Dedicated setter method `set_jetformer_schedule()`

**Why**: 
- **Clean Signature**: `GPT.forward()` signature remains unchanged, maintaining backward compatibility
- **Explicit Control**: Training loop explicitly sets schedule, making curriculum visible
- **No Global State**: Doesn't require GPT to track iteration counter (simpler)
- **Flexible**: Can be called conditionally only for Jetformer runs
- **Matches Patterns**: Similar to how other hyperparameters are set (e.g., learning rate)

**Implementation Impact**:
```python
# With L3 (chosen):
# Training loop:
if tokeniser == "jetformer":
    model.set_jetformer_schedule(iter_num, max_iters)
loss = model(B["X"], targets=B["Y"])  # Clean signature

# Inside GPT:
def set_jetformer_schedule(self, iter_num, max_iters):
    self.jet_epoch_frac = iter_num / max_iters
    # Propagate to encoder
    for enc in self.encoders.values():
        if hasattr(enc, 'set_jet_epoch_frac'):
            enc.set_jet_epoch_frac(self.jet_epoch_frac)

# With L1 (not chosen):
loss = model(B["X"], targets=B["Y"], epoch_frac=iter_num/max_iters)  # Signature change!

# With L2 (not chosen):
model.set_iteration(iter_num, max_iters)  # GPT tracks iteration
loss = model(B["X"], targets=B["Y"])
```

---

### Decision 3: What Space Does GPT Model? (Z1 vs Z2)

**The Problem**: 
After the flow transforms images `x → z`, we have latent images `z [B,C,H,W]`. But GPT operates on sequences of tokens. What should GPT actually model?

**Options Considered**:

**Z1: GPT models patchified `z` tokens**
- Flow: `x [B,C,H,W] → z [B,C,H,W]`
- Patchify: `z [B,C,H,W] → tokens [B,T,D]` (patchified z)
- GPT processes: `tokens [B,T,D]`
- GMM head predicts: parameters for `tokens[:,1:]`
- Loss targets: `tokens[:,1:]` (patchified z)

**Z2: GPT models raw `z` images (before patchification)**
- Flow: `x [B,C,H,W] → z [B,C,H,W]`
- GPT processes: `z` directly (somehow flattened or reshaped)
- GMM head predicts: parameters for `z`
- Loss targets: `z` directly

**What Each Option Changes**:

**Z1 Changes**:
- Flow output `z [B,C,H,W]` is immediately patchified
- GPT sees patch tokens `[B,T,D]` (standard sequence format)
- GMM head outputs parameters for token space
- Loss compares predicted tokens to actual patchified z tokens
- **Code location**: Encoder patchifies after flow, GPT processes tokens normally

**Z2 Changes**:
- Flow output `z [B,C,H,W]` must be reshaped for GPT (e.g., `z.view(B, -1, C*H*W)`)
- GPT processes flattened z images as sequence
- GMM head outputs parameters for image space
- Loss compares predicted z images to actual z images
- **Code location**: Encoder doesn't patchify, GPT processes reshaped z

**Trade-offs**:

| Aspect | Z1 (Patchified z) | Z2 (Raw z) |
|--------|-------------------|------------|
| **Matches Original** | ✅ Exact match to JetFormerLite | ❌ Different from original |
| **GPT Architecture** | ✅ Standard sequence processing | ❌ Non-standard input format |
| **Token Dimensionality** | ✅ Fixed: `D = C*patch*patch` | ❌ Variable: `C*H*W` (very large) |
| **Sequence Length** | ✅ Fixed: `T = (H/patch)*(W/patch)` | ❌ Very short: `T = 1` (or reshaped) |
| **GMM Head** | ✅ Predicts token-space distributions | ⚠️ Predicts image-space distributions |
| **Loss Computation** | ✅ Standard token-wise NLL | ⚠️ Image-wise NLL (different scale) |
| **Implementation Complexity** | ✅ Standard AstroPT patterns | ❌ Requires custom reshaping |

**Chosen**: **Z1** - GPT models patchified `z` tokens

**Why**: 
- **Matches Original**: Original JetFormerLite patchifies z before GPT, this is the exact design
- **Standard Architecture**: GPT processes sequences of tokens, which is what patchified z provides
- **Consistent Dimensionality**: Token dimension `D = 768` (for 16×16×3 patches) is manageable for GMM
- **Proper Sequence Modeling**: Sequence length `T = 256` (for 256×256 images) allows proper autoregressive modeling
- **Implementation Simplicity**: Uses standard AstroPT token processing patterns

**Implementation Impact**:
```python
# With Z1 (chosen):
# In JetformerImageEncoder:
z, logdet = self.flow(x)  # z is [B,C,H,W]
tokens = patchify(z, patch_size)  # tokens is [B,T,D] where T=256, D=768
# GPT processes tokens normally
# GMM head predicts: (logits_pi, mu, log_sigma) for tokens
# Loss: NLL_GMM(tokens[:,1:], ...) - logdet

# With Z2 (not chosen):
# In JetformerImageEncoder:
z, logdet = self.flow(x)  # z is [B,C,H,W]
z_flat = z.view(B, 1, C*H*W)  # Reshape to [B,1,196608] - huge!
# GPT processes z_flat (non-standard)
# GMM head predicts for 196608-dim space (impractical!)
# Loss: NLL_GMM(z, ...) - logdet
```

---

### Decision 4: Reconstruction API Design (S1 vs S2 vs S3)

**The Problem**: 
For validation/visualization, we need to reconstruct images. The reconstruction process involves: flow → patchify → GPT → GMM → depatchify → inverse flow. Where should this logic live and how should it be called?

**Options Considered**:

**S1: Reconstruction logic in training script `validate()` function**
- All reconstruction code in `scripts/train_jetformer.py`
- `validate()` function has full reconstruction pipeline
- Model only provides forward pass, script handles reconstruction

**S2: Dedicated method `jetformer_reconstruct_images()` + branch in `validate()`**
- GPT has method: `jetformer_reconstruct_images(x_real)`
- Method encapsulates full reconstruction pipeline
- `validate()` calls this method when `tokeniser=="jetformer"`
- For other tokenizers, uses existing reconstruction logic

**S3: Unified reconstruction method that handles all tokenizers**
- GPT has method: `reconstruct_images(x, tokeniser)`
- Method branches internally based on tokenizer
- `validate()` always calls same method

**What Each Option Changes**:

**S1 Changes**:
- Reconstruction code in `scripts/train_jetformer.py` `validate()` function
- Script directly calls: `flow()`, `patchify()`, `model()`, `GMM_head()`, `depatchify()`, `flow(reverse=True)`
- Script needs access to model internals (flow, GMM head)
- **Code location**: All in training script

**S2 Changes**:
- GPT has method: `@torch.no_grad() def jetformer_reconstruct_images(self, x_real)`
- Method encapsulates: flow → patchify → GPT → GMM → depatchify → inverse flow
- `validate()` branches: if Jetformer, call `jetformer_reconstruct_images()`, else use existing logic
- **Code location**: Method in `model.py`, call in training script

**S3 Changes**:
- GPT has unified method: `reconstruct_images(x, tokeniser)`
- Method has internal branches for each tokenizer
- `validate()` always calls `reconstruct_images()` regardless of tokenizer
- **Code location**: Unified method in `model.py`

**Trade-offs**:

| Aspect | S1 (Script Logic) | S2 (Dedicated Method) | S3 (Unified Method) |
|--------|-------------------|----------------------|---------------------|
| **Encapsulation** | ❌ Logic in script | ✅ Logic in model | ✅ Logic in model |
| **Code Reusability** | ❌ Hard to reuse | ✅ Easy to reuse | ✅ Easy to reuse |
| **Access to Internals** | ❌ Script needs model internals | ✅ Method has access | ✅ Method has access |
| **Validation Complexity** | ⚠️ Script handles branching | ⚠️ Script branches on tokenizer | ✅ No branching in script |
| **Backward Compat** | ✅ No model changes | ✅ New method, old code unchanged | ⚠️ Changes existing patterns |
| **Clarity** | ❌ Reconstruction scattered | ✅ Clear Jetformer-specific API | ⚠️ Unified but more complex |

**Chosen**: **S2** - Dedicated method `jetformer_reconstruct_images()` + branch in `validate()`

**Why**: 
- **Encapsulation**: Reconstruction logic lives with the model, not scattered in training script
- **Access to Internals**: Method can access `self.encoders["images"].flow`, `self.jetformer_images_head`, etc.
- **Backward Compatibility**: Doesn't change existing reconstruction patterns for AIM/Affine
- **Clear API**: Method name makes it obvious this is Jetformer-specific
- **Flexibility**: Can easily extend to other reconstruction modes later

**Implementation Impact**:
```python
# With S2 (chosen):
# In model.py:
@torch.no_grad()
def jetformer_reconstruct_images(self, x_real):
    # Full reconstruction pipeline
    z_real, _ = self.encoders["images"].flow(x_real, reverse=False)
    tokens_real = patchify(z_real, ...)
    # ... GPT forward ...
    # ... GMM head ...
    # ... depatchify ...
    x_pred, _ = self.encoders["images"].flow(z_pred, reverse=True)
    return x_pred

# In validate():
if tokeniser == "jetformer":
    x_recon = model.jetformer_reconstruct_images(B["Y"]["images"])
else:
    # Existing AIM/Affine reconstruction
    P = model(B["X"], B["Y"])

# With S1 (not chosen):
# In validate() - all logic here:
z_real, _ = model.encoders["images"].flow(B["Y"]["images"], reverse=False)
tokens_real = patchify(z_real, ...)
# ... need access to model internals ...
# Script becomes very complex

# With S3 (not chosen):
# Unified method with internal branching:
def reconstruct_images(self, x, tokeniser):
    if tokeniser == "jetformer":
        # Jetformer reconstruction
    elif tokeniser == "aim":
        # AIM reconstruction
    # ... but AIM/Affine don't have reconstruction methods currently
```

---

### Decision 5: Configuration Management (C1 vs C2)

**The Problem**: 
Jetformer has hyperparameters: `flow_steps`, `gmm_K`, `noise_max`, `noise_min`, `img_size`. Where should these be stored and how should they be passed to the model?

**Options Considered**:

**C1: All Jetformer fields on `GPTConfig`**
- `GPTConfig` dataclass has: `jetformer_flow_steps`, `jetformer_gmm_K`, `jetformer_noise_max`, `jetformer_noise_min`, `img_size`
- Training script passes via `model_args = dict(..., jetformer_flow_steps=4, ...)`
- Model reads from `config.jetformer_flow_steps`, etc.

**C2: Separate `JetformerConfig` dataclass**
- New dataclass: `@dataclass class JetformerConfig: flow_steps, gmm_K, ...`
- `GPTConfig` has: `jetformer_config: JetformerConfig | None = None`
- Training script: `jetformer_config = JetformerConfig(...)`, then `GPTConfig(..., jetformer_config=jetformer_config)`

**C3: Pass as parameters to `JetformerImageEncoder` directly**
- `GPTConfig` doesn't have Jetformer fields
- `_init_native_backbone()` passes values directly: `JetformerImageEncoder(config, in_size, img_size=256, n_chan=3, ...)`
- Values come from training script via closure or global

**What Each Option Changes**:

**C1 Changes**:
- `GPTConfig` dataclass gets 5 new fields (all optional with defaults)
- Training script: `model_args = dict(..., jetformer_flow_steps=4, ...)`
- Model initialization: `GPTConfig(**model_args)` automatically includes fields
- Model code: `self.encoders["images"] = JetformerImageEncoder(..., steps=config.jetformer_flow_steps)`
- **Code location**: Fields in `model.py` `GPTConfig`, passed via `model_args`

**C2 Changes**:
- New file or section: `JetformerConfig` dataclass
- `GPTConfig` has: `jetformer_config: JetformerConfig | None`
- Training script: Creates `JetformerConfig`, passes to `GPTConfig`
- Model code: `if config.jetformer_config: steps = config.jetformer_config.flow_steps`
- **Code location**: New config class, nested in `GPTConfig`

**C3 Changes**:
- `GPTConfig` unchanged
- Training script: Stores values in variables
- Model initialization: Hardcodes or reads from somewhere: `JetformerImageEncoder(..., steps=4)`
- **Code location**: Values passed directly, not via config

**Trade-offs**:

| Aspect | C1 (GPTConfig Fields) | C2 (Separate Config) | C3 (Direct Parameters) |
|--------|----------------------|---------------------|----------------------|
| **Centralization** | ✅ All config in one place | ⚠️ Nested config | ❌ Scattered |
| **Type Safety** | ✅ Dataclass fields | ✅ Separate dataclass | ❌ No type checking |
| **Checkpointing** | ✅ Saved in config | ✅ Saved in config | ❌ Not saved |
| **Default Values** | ✅ Easy with dataclass | ✅ Easy with dataclass | ⚠️ Manual defaults |
| **Code Complexity** | ✅ Simple, flat structure | ⚠️ Nested access | ✅ Simple but not saved |
| **Consistency** | ✅ Matches existing patterns | ⚠️ New pattern | ❌ Different pattern |

**Chosen**: **C1** - All Jetformer fields on `GPTConfig`

**Why**: 
- **Consistency**: Matches how other AstroPT config is structured (flat dataclass)
- **Checkpointing**: Config is saved/loaded automatically, Jetformer hyperparameters preserved
- **Simplicity**: No nested configs, easy to access `config.jetformer_flow_steps`
- **Type Safety**: Dataclass provides type hints and validation
- **Default Values**: Easy to provide defaults: `jetformer_flow_steps: int = 4`

**Implementation Impact**:
```python
# With C1 (chosen):
# In model.py:
@dataclass
class GPTConfig:
    # ... existing fields ...
    jetformer_flow_steps: int = 4
    jetformer_gmm_K: int = 4
    jetformer_noise_max: float = 0.1
    jetformer_noise_min: float = 0.0
    img_size: int = 256

# Training script:
model_args = dict(
    # ... other args ...
    jetformer_flow_steps=4,
    jetformer_gmm_K=4,
    # ...
)
gptconf = GPTConfig(**model_args)

# Model code:
self.encoders["images"] = JetformerImageEncoder(
    config, in_size, 
    img_size=config.img_size,  # From config!
    # ...
)

# With C2 (not chosen):
@dataclass
class JetformerConfig:
    flow_steps: int = 4
    # ...

@dataclass
class GPTConfig:
    # ...
    jetformer_config: JetformerConfig | None = None

# More complex, nested access

# With C3 (not chosen):
# No config fields, values passed directly:
JetformerImageEncoder(config, in_size, img_size=256, ...)  # Hardcoded!
# Not saved in checkpoints
```

---

### Decision 6: Modality Support (M1 vs M2)

**The Problem**: 
Should Jetformer support multiple modalities (images + spectra) in the same run, or should it be images-only initially?

**Options Considered**:

**M1: Multi-modal Jetformer (images + spectra in same run)**
- Jetformer can be used for images while other modalities use AIM/Affine
- Single forward pass processes mixed modalities
- Loss combines Jetformer loss (images) + Huber loss (spectra)

**M2: Jetformer-only runs (images modality only initially)**
- When `tokeniser=="jetformer"`, only images modality is supported
- Other modalities are not included in the run
- Simpler implementation, can extend later

**What Each Option Changes**:

**M1 Changes**:
- `_forward_native()` must handle mixed modalities
- Jetformer loss path only for images, Huber loss for other modalities
- Need to combine losses: `loss = loss_images + loss_spectra`
- Encoder selection: `JetformerImageEncoder` for images, `Encoder` for spectra
- **Code location**: Complex branching in `_forward_native()`, loss combination logic

**M2 Changes**:
- `_forward_native()` checks: `if tokeniser=="jetformer" and modalities==["images"]`
- Only images modality allowed when using Jetformer
- Single loss path (Jetformer loss only)
- Simpler encoder selection (always `JetformerImageEncoder` for images)
- **Code location**: Simple conditional in `_forward_native()`

**Trade-offs**:

| Aspect | M1 (Multi-modal) | M2 (Images-only) |
|--------|------------------|------------------|
| **Flexibility** | ✅ Can mix modalities | ❌ Only images |
| **Implementation Complexity** | ❌ Complex loss combination | ✅ Simple, single path |
| **Testing** | ❌ Need to test all combinations | ✅ Test images only |
| **Extensibility** | ✅ Already supports extension | ⚠️ Need to add later |
| **User Confusion** | ⚠️ Which tokenizer for which modality? | ✅ Clear: Jetformer = images |
| **Code Maintainability** | ❌ More complex branching | ✅ Simpler code |

**Chosen**: **M2** - Jetformer-only runs (images modality only initially)

**Why**: 
- **Simplicity**: Much simpler initial implementation, easier to debug and test
- **Clear Semantics**: When `tokeniser=="jetformer"`, it's clear this is for images
- **Incremental Development**: Can add multi-modal support later once images work perfectly
- **Reduced Risk**: Fewer code paths = fewer bugs, easier to maintain
- **Matches Original**: Original JetFormerLite was images-only

**Implementation Impact**:
```python
# With M2 (chosen):
# In _forward_native():
if (self.config.backbone == "native" 
    and self.config.tokeniser == "jetformer"
    and self.modality_registry.names() == ["images"]):
    # Jetformer loss path - simple!
    loss = (nll_gmm - logdet).mean()
    return outputs, loss

# Training script:
modalities = [ModalityConfig(name="images", ...)]  # Only images

# With M1 (not chosen):
# In _forward_native():
if self.config.tokeniser == "jetformer":
    loss_images = None
    loss_spectra = None
    if "images" in modalities:
        # Compute Jetformer loss
        loss_images = (nll_gmm - logdet).mean()
    if "spectra" in modalities:
        # Compute Huber loss
        loss_spectra = huber_loss(...)
    # Combine losses - complex!
    loss = loss_images + loss_spectra
    return outputs, loss
```

---

### Decision 7: Training Loop Modifications (T1 vs T2)

**The Problem**: 
The training loop needs to call `set_jetformer_schedule()` to update the noise curriculum. How much should the training loop change?

**Options Considered**:

**T1: Minimal changes + `set_jetformer_schedule()` call**
- Training loop adds: `if tokeniser=="jetformer": model.set_jetformer_schedule(...)`
- All other logic unchanged
- Forward pass, loss computation, etc. all handled in model

**T2: Extensive changes to handle Jetformer-specific logic**
- Training loop branches on tokenizer
- Different forward calls, different loss handling, different validation
- More Jetformer-specific code in training script

**What Each Option Changes**:

**T1 Changes**:
- Training loop adds ~3 lines: check tokenizer, call setter
- All forward/backward logic unchanged: `loss = model(B["X"], targets=B["Y"])`
- Model handles all Jetformer-specific logic internally
- **Code location**: Minimal changes in training script, logic in model

**T2 Changes**:
- Training loop has branches: `if tokeniser=="jetformer": ... else: ...`
- Different forward calls for different tokenizers
- Different loss handling, different validation logic
- **Code location**: Extensive changes in training script

**Trade-offs**:

| Aspect | T1 (Minimal Changes) | T2 (Extensive Changes) |
|--------|---------------------|----------------------|
| **Training Script Complexity** | ✅ Simple, minimal changes | ❌ Complex, many branches |
| **Code Maintainability** | ✅ Logic in model, script simple | ❌ Logic scattered |
| **Backward Compatibility** | ✅ Existing code mostly unchanged | ⚠️ More changes to review |
| **Encapsulation** | ✅ Model handles its own logic | ❌ Script knows tokenizer details |
| **Flexibility** | ⚠️ Less control in script | ✅ More control in script |

**Chosen**: **T1** - Minimal changes + `set_jetformer_schedule()` call

**Why**: 
- **Encapsulation**: All Jetformer logic lives in the model, training script stays simple
- **Maintainability**: Easier to maintain - changes to Jetformer don't require training script changes
- **Consistency**: Matches AstroPT pattern - model handles complexity, script is simple
- **Backward Compatibility**: Existing training scripts mostly unchanged

**Implementation Impact**:
```python
# With T1 (chosen):
# Training loop - minimal changes:
raw_model = model.module if ddp else model
if raw_model.config.tokeniser == "jetformer":
    raw_model.set_jetformer_schedule(iter_num, max_iters)  # Only addition!

loss = model(B["X"], targets=B["Y"])  # Same as before
loss.backward()  # Same as before
# ... rest unchanged

# With T2 (not chosen):
# Training loop - extensive changes:
if tokeniser == "jetformer":
    raw_model.set_jetformer_schedule(iter_num, max_iters)
    # Different forward?
    # Different loss handling?
    # Different validation?
else:
    # Original logic
    # ...
# Many branches, complex code
```

---

## Summary of Design Decisions

Each decision was made to balance:
1. **Matching Original**: Preserving JetFormerLite's design and quality
2. **AstroPT Integration**: Fitting cleanly into existing architecture
3. **Backward Compatibility**: Not breaking AIM/Affine tokenizers
4. **Code Maintainability**: Keeping code simple and maintainable
5. **Future Extensibility**: Allowing easy extension later

The chosen options (F1, L3, Z1, S2, C1, M2, T1) create a clean, maintainable integration that preserves both the original JetFormerLite quality and AstroPT's flexibility.

---

## Implementation Details

### New Files Created

1. **`src/astropt/jetformer.py`**:
   - `TinyFlow2D`: 2D image-space flow (from original)
   - `TinyFlow1D`: 1D patch-space flow (kept for potential future use)
   - `GMMHead`: GMM output head
   - `gmm_nll()`: Negative log-likelihood computation
   - `patchify()` / `depatchify()`: Image ↔ token conversion
   - `uniform_dequantize()`: Continuous dequantization

### Modified Files

1. **`src/astropt/model.py`**:
   - Added `JetformerImageEncoder` class
   - Added `jetformer_images_head` in `_init_native_backbone()`
   - Added Jetformer loss path in `_forward_native()`
   - Added `set_jetformer_schedule()` method
   - Added `jetformer_reconstruct_images()` method
   - Added Jetformer config fields to `GPTConfig`

2. **`scripts/train_jetformer.py`**:
   - Added `prepare_batch_for_jetformer()` helper
   - Modified `process_galaxy_wrapper()` to return raw images
   - Updated validation to handle raw images
   - Added schedule setter call in training loop
   - Fixed loss plotting for negative values (symlog scale)

3. **`src/astropt/local_datasets.py`**:
   - Modified `process_modes()` to skip slicing for raw images
   - Added `images_is_raw` flag handling

---

## Changes to Jetformer

### 1. Modularization

**Original**: Monolithic `JetFormerLite` class with everything inside

**Changed**: Split into reusable components:
- `TinyFlow2D` in `jetformer.py`
- `GMMHead` in `jetformer.py`
- `JetformerImageEncoder` in `model.py`

**Reason**: Better integration with AstroPT's modular architecture

### 2. Encoder Integration

**Original**: `in_proj` + `pos` embeddings inside `JetFormerLite`

**Changed**: Flow → patchify → project in `JetformerImageEncoder`, then standard AstroPT embedding

**Reason**: Reuse AstroPT's embedding infrastructure

### 3. Loss Computation

**Original**: Loss computed inside `JetFormerLite.forward()`

**Changed**: Loss computed in `GPT._forward_native()` with conditional branching

**Reason**: Unified loss handling across all tokenizers

### 4. Noise Curriculum

**Original**: `epoch_frac` passed as parameter to `forward()`

**Changed**: `set_jetformer_schedule()` method + cached `jet_epoch_frac`

**Reason**: Cleaner API, matches AstroPT patterns

### 5. Reconstruction

**Original**: `sample()` method with reconstruction path

**Changed**: `jetformer_reconstruct_images()` method

**Reason**: Better separation, matches AstroPT validation patterns

### 6. Data Loading

**Original**: Custom dataloader with `.pt` shards

**Changed**: Works with AstroPT's `GalaxyImageDataset` and HuggingFace datasets

**Reason**: Unified data infrastructure

---

## Changes to AstroPT

### 1. Encoder System

**Original**: Single `Encoder` class for all tokenizers

**Changed**: 
- `Encoder` remains for AIM/Affine
- `JetformerImageEncoder` inherits from `Encoder` for Jetformer

**Impact**: Backward compatible, AIM/Affine unchanged

### 2. Input Handling

**Original**: Always expects patch tokens `[B, T, D]`

**Changed**: 
- `JetformerImageEncoder` accepts raw images `[B, C, H, W]`
- Data loading branches to provide appropriate format

**Impact**: Requires `prepare_batch_for_jetformer()` in training script

### 3. Loss Computation

**Original**: Single Huber loss path

**Changed**: 
- Conditional branch: Jetformer uses GMM NLL - logdet
- Other tokenizers use original Huber loss

**Impact**: Loss values can be negative for Jetformer (expected)

### 4. Configuration

**Original**: `GPTConfig` with AIM/Affine fields

**Changed**: Added Jetformer fields:
- `jetformer_flow_steps`: Number of coupling layers (default: 4)
- `jetformer_gmm_K`: Number of GMM components (default: 4)
- `jetformer_noise_max`: Maximum noise (default: 0.1)
- `jetformer_noise_min`: Minimum noise (default: 0.0)
- `img_size`: Image size for flow (default: 256)

**Impact**: New config fields, defaults provided

### 5. Data Pipeline

**Original**: `process_modes()` always slices for teacher forcing

**Changed**: 
- Checks `images_is_raw` flag
- Skips slicing for raw images (Jetformer)
- Still slices for patch tokens (AIM/Affine)

**Impact**: Maintains teacher forcing for all tokenizers

### 6. Validation / Visualization

**Original**: Expects patch tokens for plotting

**Changed**: 
- Branches based on tokenizer
- Jetformer: works with raw images, converts for visualization
- AIM/Affine: original patch-based visualization

**Impact**: Validation code handles both formats

### 7. Loss Plotting

**Original**: Log scale (only positive values)

**Changed**: Symlog scale (handles negative values)

**Impact**: Can visualize negative losses (expected for Jetformer)

---

## Data Flow Comparison

### AIM / Affine Tokenizer (Original)

```
Dataset → Patches [B,T,D]
  ↓
Encoder (AIM/Affine projection)
  ↓
Embeddings [B,T,n_embd]
  ↓
GPT (causal attention)
  ↓
Hidden States [B,T,n_embd]
  ↓
Decoder (AIM/Affine projection)
  ↓
Predictions [B,T,D]
  ↓
Huber Loss
```

### Jetformer Tokenizer (New)

```
Dataset → Raw Images [B,C,H,W]
  ↓
JetformerImageEncoder:
  - uniform_dequantize
  - TinyFlow2D → z [B,C,H,W], logdet [B]
  - patchify → tokens [B,T,D]
  - add noise (curriculum)
  - project to embeddings
  ↓
Embeddings [B,T,n_embd]
  ↓
GPT (causal attention)
  ↓
Hidden States [B,T,n_embd]
  ↓
GMM Head → (logits_pi, mu, log_sigma)
  ↓
Loss: NLL_GMM(tokens[:,1:]) - logdet
```

**Key Differences**:
1. Jetformer operates on raw images initially
2. Flow happens before patchification
3. Loss is likelihood-based (can be negative)
4. Noise curriculum for training stability

---

## Usage

### Training with Jetformer

```python
# In train_jetformer.py or similar
tokeniser = "jetformer"

model_args = dict(
    # ... standard AstroPT args ...
    tokeniser=tokeniser,
    jetformer_flow_steps=4,
    jetformer_gmm_K=4,
    jetformer_noise_max=0.1,
    jetformer_noise_min=0.0,
    img_size=256,
)

# Model initialization
gptconf = GPTConfig(**model_args)
model = GPT(gptconf, modality_registry)
```

### Training Loop

```python
# Before forward pass
raw_model = model.module if ddp else model
if raw_model.config.tokeniser == "jetformer":
    raw_model.set_jetformer_schedule(iter_num, max_iters)

# Forward pass (handles Jetformer automatically)
logits, loss = model(B["X"], targets=B["Y"])
```

### Data Loading

```python
# Prepare batch (swaps patches for raw images if Jetformer)
batch_raw = next(dataloader)
batch_raw = prepare_batch_for_jetformer(batch_raw, tokeniser)
B = dataset.process_modes(batch_raw, modality_registry, device)
```

### Validation / Reconstruction

```python
# Reconstruction for visualization
if tokeniser == "jetformer":
    x_recon = model.jetformer_reconstruct_images(B["Y"]["images"])
    # x_recon is [B, C, H, W]
```

---

## Technical Notes

### Why Flow on Image Space?

The original JetFormerLite flows on image space `[B, C, H, W]` before patchification. This is more principled because:
1. Flows operate on natural image structure (spatial relationships)
2. Better matches the original paper's design
3. Preserves image-level statistics before tokenization

Alternative (flow on patches) would be simpler but loses these benefits.

### Why Negative Loss?

Jetformer loss is `NLL_GMM - logdet`. When the flow's log-determinant is large (good flow fit), `logdet` can exceed `NLL_GMM`, making loss negative. This is **expected and correct** - it indicates the model is learning a good latent representation.

### Memory Considerations

Raw images use more memory than patches:
- Image: `B × 3 × 256 × 256 × 4 bytes = B × 786KB`
- Patches: `B × 256 × 768 × 4 bytes = B × 786KB` (same size, but different access pattern)

The main issue is DataLoader workers - reduce `num_workers` for Jetformer (2-4 instead of 16).

### Teacher Forcing

Even though Jetformer receives full raw images (no slicing), teacher forcing is maintained via:
1. GPT's causal attention mask (can't see future tokens)
2. Loss targets are `tokens[:, 1:]` (next-token prediction)
3. First token is copied from real data in reconstruction

### Compatibility

- **AIM/Affine tokenizers**: Completely unchanged, work exactly as before
- **Other modalities**: Unaffected (Jetformer is images-only initially)
- **Training infrastructure**: Shared, no changes needed
- **Checkpointing**: Compatible (Jetformer fields in config)

---

## Summary

The integration of Jetformer into AstroPT successfully:

1. ✅ Preserves original JetFormerLite architecture (flow on image space)
2. ✅ Maintains full backward compatibility (AIM/Affine unchanged)
3. ✅ Shares GPT backbone across all tokenizers
4. ✅ Uses unified training infrastructure
5. ✅ Handles negative losses correctly
6. ✅ Supports proper reconstruction/validation

The key insight was to **encapsulate Jetformer-specific logic** in `JetformerImageEncoder` and conditional branches, while keeping the core GPT architecture unchanged. This allows all three tokenizers to coexist seamlessly.

---

## Future Work

Potential extensions:
1. **Multi-modal Jetformer**: Extend to spectra and other modalities
2. **Mixed tokenizers**: Use different tokenizers for different modalities
3. **Flow improvements**: Experiment with different flow architectures
4. **GMM head variants**: Try different mixture models
5. **Memory optimization**: Further reduce memory footprint for large batches

---

*This integration was completed with careful attention to preserving both the original JetFormerLite design and AstroPT's flexibility. All changes are backward compatible and well-documented.*

