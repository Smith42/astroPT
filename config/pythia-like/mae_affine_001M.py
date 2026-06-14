# Scaling-study run: BERT-style MAE objective, affine tokeniser, 001M model.
# Run (all runs use the unified loop):
#   python scripts/train.py config/pythia-like/mae_affine_001M.py

# --- model size: 001M ---
n_layer = 4
n_head = 8
n_embd = 128
n_chan = 3
block_size = 1024
patch_size = 16

# --- tokeniser axis ---
tokeniser = "affine"

# --- objective axis ---
objective = "mae"
attn_type = "full"
mae_mask_ratio = 0.5
norm_pix_loss = False

# --- learning rate (Pythia-style: scaled with model size) ---
learning_rate = 0.001
min_lr = learning_rate / 10

# --- fixed across all 12 runs (only size/objective/tokeniser/LR vary) ---
batch_size = 16
gradient_accumulation_steps = 5 * 4  # effective batch = batch_size(16) * 20 = 320
max_iters = 30000
lr_decay_iters = 26500  # cosine reaches min_lr at ~one epoch (floor(N=8,474,566 / 320) = 26483)
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

out_dir = "logs/pythia-like/mae_affine_001M"
