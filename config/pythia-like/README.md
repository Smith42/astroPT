# Scaling-study run matrix

12 pretraining runs for the "physics discovery over training" study:

**{objective: AR, MAE} × {tokeniser: aim, affine} × {size: 1M, 21M, 100M}**

All 12 use the **single unified training loop** (`scripts/train.py`) — the
objective, tokeniser and size are config flags. Each run saves **64 log-spaced
checkpoints** for probing how representations emerge across training, and uses a
**Pythia-style per-size learning rate**.

| size (n_layer, n_head, n_embd) | LR | AR+aim | AR+affine | MAE+aim | MAE+affine |
|---|---|---|---|---|---|
| 1M (4, 8, 128)    | 1e-3 | `ar_aim_001M` | `ar_affine_001M` | `mae_aim_001M` | `mae_affine_001M` |
| 21M (6, 8, 512)   | 1e-3 | `ar_aim_021M` | `ar_affine_021M` | `mae_aim_021M` | `mae_affine_021M` |
| 100M (12, 12, 768)| 6e-4 | `ar_aim_100M` | `ar_affine_100M` | `mae_aim_100M` | `mae_affine_100M` |

## Running

```bash
python scripts/train.py config/pythia-like/<obj>_<tok>_<size>.py
# or, multi-GPU:
torchrun --standalone --nproc_per_node=N scripts/train.py config/pythia-like/<name>.py
```

## What varies vs. what's fixed

- **vary:** size (`n_layer/n_head/n_embd`), `tokeniser`, `objective`, and
  `learning_rate` (scaled with size).
- **fixed:** batch size, accumulation, total steps, LR-schedule shape, and the
  checkpoint schedule — so the four axes above are the only experimental
  variables.

## Notes

- **Checkpoints:** `num_checkpoints=64`, `checkpoint_schedule="log"`. Pure log
  over 30k steps yields ~58 *distinct* steps (early integer collisions); the
  first (random-init) and last (final) step are always pinned. Use `"even"` for
  exactly 64 evenly-spaced, or raise `max_iters` for 64 distinct log steps.
- **Learning rate** is scaled with model size following Pythia/GPT-3
  (1M/21M → 1e-3, 100M → 6e-4); each model trains near its own optimum so size
  isn't confounded with mis-tuning.
- **Storage:** 64 checkpoints × 12 runs is ~370 GB with optimizer state, ~125 GB
  weights-only — upload **slim (weights-only)** checkpoints to HF and keep
  optimizer state local (the wrapper's job).
