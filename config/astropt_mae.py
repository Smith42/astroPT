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
batch_size = 16
gradient_accumulation_steps = 5 * 8
num_workers = 32

max_iters = 30000
lr_decay_iters = 27000 * 1.1

# eval stuff
eval_interval = 1000
checkpoint_interval = 5000
eval_iters = 200
log_interval = 100
log_via_wandb = True
out_dir = "logs/astropt_mae"
