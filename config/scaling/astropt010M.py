# this file will train a ~20m parameter model with the same hyperparams as
# EleutherAI's Pythia-20M (https://arxiv.org/abs/2304.01373)
#
# launch as the following (e.g. in a tmux session) and wait ~5 days:
# $ OMP_NUM_THREADS=32 torchrun --standalone --nproc_per_node=8 src/train.py config/astropt300m.py
#
# don't forget to $(mkdir logs) !

# params
n_layer = 6
n_head = 8
n_embd = 384
block_size = 1024

# here we follow chinchilla and pythia
learning_rate = 10e-4  # max learning rate
min_lr = learning_rate / 10

# these make the total batch size be ~0.328M
# 8 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 327,680
# here we assume a world size of 8!
init_from = "scratch"
batch_size = 8
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 8.5B (one epoch of GZ DESI)
max_iters = 30000
lr_decay_iters = 27000 * 1.1

# eval stuff
eval_interval = 1000
checkpoint_interval = 5000
eval_iters = 200
log_interval = 100
out_dir = "logs/9B_tokens/astropt010M"
