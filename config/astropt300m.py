# this file will train a ~400m parameter model with the same hyperparams as
# EleutherAI's Pythia-410M (https://arxiv.org/abs/2304.01373)
#
# launch as the following (e.g. in a tmux session) and wait ~5 days:
# $ OMP_NUM_THREADS=32 torchrun --standalone --nproc_per_node=8 src/train.py config/astropt300m.py
#
# don't forget to $(mkdir logs) !

# params
n_layer=24
n_head=16
n_embd=1024
block_size=1024

# here we follow chinchilla
learning_rate = 3e-4 # max learning rate
min_lr = learning_rate/10 

# these make the total batch size be ~0.164M
# 4 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 163,840
# here we assume a world size of 8!
init_from = "scratch"
batch_size = 4
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 9B (~chinchilla optimal)
max_iters = 55010
lr_decay_iters = 55010 * 1.1

# eval stuff
eval_interval = 5000
eval_iters = 200 
log_interval = 10
out_dir ='logs/astropt400m'
