# this file will train an AstroPT model with an LLM backbone (SmolLMv3)
#
# launch as the following (e.g. in a tmux session):
# $ OMP_NUM_THREADS=32 torchrun --standalone --nproc_per_node=8 src/train.py config/astropt300m.py
#
# don't forget to $(mkdir logs) !

llm_model_name = "HuggingFaceTB/SmolLM2-360M"
lora_r = 32

tokeniser = "affine"

learning_rate = 6e-4  # max learning rate
max_iters = (
    12000  # total number of training iterations for one pass over our dataset
)
min_lr = learning_rate / 10

init_from = "scratch"
batch_size = 32
gradient_accumulation_steps = 5 # fine for 1 GPU
num_workers = 64

max_iters = 12000 # less than in a full run
lr_decay_iters = 10000 * 1.1

# eval stuff
eval_interval = 100
checkpoint_interval = 1000
eval_iters = 100
log_interval = 50
log_via_wandb = True
out_dir = "logs/smollm-360M"

wandb_project = "smollm"
