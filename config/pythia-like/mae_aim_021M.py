# Scaling-study run: BERT-style MAE objective, aim tokeniser, 021M model.
# Run (all runs use the unified loop):
#   python scripts/train.py config/pythia-like/mae_aim_021M.py

# --- model size: 021M ---
n_layer = 6
n_head = 8
n_embd = 512
n_chan = 3
block_size = 1024
patch_size = 16

# --- tokeniser axis ---
tokeniser = "aim"

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
gradient_accumulation_steps = 5 * 8
max_iters = 13241  # exactly one pass over the 8,474,566-galaxy train set at eff. batch 640
lr_decay_iters = 13241.0  # decay LR to min over the single epoch
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

out_dir = "logs/pythia-like/mae_aim_021M"
