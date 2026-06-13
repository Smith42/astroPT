# Scaling-study run: AUTOREGRESSIVE objective, affine tokeniser, 100M model.
# Run (all runs use the unified loop):
#   python scripts/train.py config/study/ar_affine_100M.py

# --- model size: 100M ---
n_layer = 12
n_head = 12
n_embd = 768
n_chan = 3
block_size = 1024
patch_size = 16

# --- tokeniser axis ---
tokeniser = "affine"

# --- objective axis ---
objective = "ar"
attn_type = "causal"

# --- learning rate (Pythia-style: scaled with model size) ---
learning_rate = 0.0006
min_lr = learning_rate / 10

# --- fixed across all 12 runs (only size/objective/tokeniser/LR vary) ---
batch_size = 16
gradient_accumulation_steps = 5 * 8
max_iters = 30000
lr_decay_iters = 27000 * 1.1
weight_decay = 1e-1
num_workers = 32

# --- 64 log-spaced checkpoints, for probing physics emergence across training ---
# (pure log over 30k steps yields ~57 *distinct* steps due to early integer
# collisions; bump max_iters or use "even" for exactly 64.)
num_checkpoints = 64
checkpoint_schedule = "log"

# --- eval ---
eval_interval = 1000
eval_iters = 200
log_interval = 100
log_via_wandb = True

out_dir = "logs/study/ar_affine_100M"
