# this file will train a ~2.1B parameter model with the same hyperparams (excluding n_layers) as
# EleutherAI's Pythia-2.8B (https://arxiv.org/abs/2304.01373)
#
# launch as the following (e.g. in a tmux session) and wait ~5 days:
# $ OMP_NUM_THREADS=32 torchrun --standalone --nproc_per_node=4 src/train.py config/astropt300m.py
#
# don't forget to $(mkdir logs) !

# params
n_layer = 26  # the recommended is 32 for EleutherAI's Pythia-2.8B but reduced due to memory limitations
n_head = 32
n_embd = 2560
block_size = 1024

# here we follow chinchilla
learning_rate = 1.6e-4  # max learning rate
min_lr = learning_rate / 10

# these make the total batch size be ~0.328M
# 16 batch size * 1024 block size * 5 gradaccum * 4 GPUs = 327,680
# here we assume a world size of 8!
init_from = "scratch"
batch_size = 16
gradient_accumulation_steps = 5 * 4

# this makes total number of tokens be 8.5B (one epoch of GZ DESI)
max_iters = 30000
lr_decay_iters = 27000 * 1.1

# eval stuff
eval_interval = 1000
checkpoint_interval = 1000
eval_iters = 200
log_interval = 10
out_dir = "logs/9B_tokens/astropt2100M"
