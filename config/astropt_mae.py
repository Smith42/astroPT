# Example config for BERT-style masked-autoencoder (MAE) pretraining.
#
# launch as (e.g. in a tmux session):
# $ OMP_NUM_THREADS=32 torchrun --standalone --nproc_per_node=8 scripts/train.py config/astropt_mae.py
#
# don't forget to $(mkdir logs) !

# params
n_layer = 12
n_head = 12
n_chan = 3
n_embd = 768
block_size = 1024

# MAE objective. attn_type must be "full" (bidirectional); the model raises if
# it is left as the autoregressive default of "causal". BERT-style: mask tokens
# flow through the full encoder and the masked patches are reconstructed.
objective = "mae"
attn_type = "full"
# fraction of patches replaced by the mask token. BERT uses 0.15 for text;
# image masked-image-modelling (e.g. SimMIM) typically uses ~0.5-0.6.
mae_mask_ratio = 0.5
norm_pix_loss = False

# chinchilla / pythia schedule
learning_rate = 6e-4
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
