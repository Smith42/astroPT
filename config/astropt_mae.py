# Example config for masked-autoencoder (MAE) pretraining.
#
# launch as (e.g. in a tmux session):
# $ OMP_NUM_THREADS=32 torchrun --standalone --nproc_per_node=8 scripts/train_mae.py config/astropt_mae.py
#
# don't forget to $(mkdir logs) !

# params
n_layer = 12
n_head = 12
n_chan = 3
n_embd = 768
block_size = 1024

# MAE objective. attn_type must be "full" (bidirectional); the model raises if
# it is left as the autoregressive default of "causal".
objective = "mae"
attn_type = "full"
mae_mask_ratio = 0.75  # fraction of patches hidden from the encoder
mae_decoder_n_layer = 4  # depth of the lightweight MAE decoder
norm_pix_loss = False  # image pipeline already per-patch normalises

# Positional embedding. "learned" is the default 1D index embedding; set
# "2d_sincos" for the fixed ViT/MAE 2D sine-cosine embedding (this also forces
# raster patch order, i.e. spiral=False).
pos_encoding = "learned"
patch_size = 16

# here we follow chinchilla and pythia
learning_rate = 6e-4  # max learning rate
min_lr = learning_rate / 10

init_from = "scratch"
num_workers = 32

# Fixed batch size, identical across model sizes (no VRAM probing) so it is not
# a confound in scaling/probing comparisons. target_batch_size pins the effective
# batch (sequences/optimizer step); gradient_accumulation_steps is derived from
# it and the micro batch_size, so the effective batch is also constant across GPU
# counts.
target_batch_size = 1024
batch_size = 16
gradient_accumulation_steps = 5 * 8

max_iters = 30000
lr_decay_iters = 27000 * 1.1

# eval stuff
eval_interval = 1000
checkpoint_interval = 5000
eval_iters = 200
log_interval = 100
log_via_wandb = True
out_dir = "logs/astropt_mae"

# save 16 checkpoints log-spaced over the run (dense early, to capture emergence,
# Pythia-style), always including the random-init and final steps. NB each
# checkpoint stores optimizer state too, so budget disk (~1.5 GB each at 123M).
num_checkpoints = 16
checkpoint_schedule = "log"
