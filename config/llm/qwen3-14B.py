# this file will train an AstroPT model with an LLM backbone (SmolLMv3)
#
# launch as the following (e.g. in a tmux session):
# $ OMP_NUM_THREADS=32 torchrun --standalone --nproc_per_node=8 src/train.py config/astropt300m.py
#
# don't forget to $(mkdir logs) !

llm_model_name = "Qwen/Qwen3-14B"
lora_r = 256 
use_qlora = True

tokeniser = "affine"

learning_rate = 1e-4  # max learning rate
min_lr = learning_rate / 10

init_from = "scratch"
batch_size = 32
gradient_accumulation_steps = 4
num_workers = 64

max_iters = 30000 # less than in a full run
lr_decay_iters = 27000 * 1.1
stream_hf_dataset = False  # stream the galaxies from huggingface

# eval stuff
eval_interval = 100
checkpoint_interval = 1000
eval_iters = 100
log_interval = 10
log_via_wandb = True
out_dir = "logs/qwen14B"

wandb_project = "smollm"
