import torch
import loralib as lora
from torch.utils.data import DataLoader

from astropt.model import GPT
from astropt.local_datasets import GalaxyImageDataset

# Config
pretrained_path = "logs/pretrain/ckpt.pt"
out_dir = "logs/finetune"
batch_size = 32
learning_rate = 1e-4
num_epochs = 10

# Load pretrained model
checkpoint = torch.load(pretrained_path)
config = checkpoint["config"]

# Add finetuning configs
config.lora_r = 8  # LoRA rank
config.output_dim = 1  # Your task dimension

# Initialize model
model = GPT(config)
model.load_state_dict(checkpoint["model"])
model.cuda()

# Setup LoRA
lora.mark_only_lora_as_trainable(model)
for param in model.task_head.parameters():
    param.requires_grad = True

# Load datasets
train_dataset = GalaxyImageDataset("train_data.txt")
val_dataset = GalaxyImageDataset("val_data.txt")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Optimizer
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

# Training loop
best_val_loss = float('inf')
for epoch in range(num_epochs):
    # Train
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs, loss = model.get_task_prediction(
            batch['images'].cuda(),
            batch['target'].cuda()
        )
        loss.backward()
        optimizer.step()
    
    # Validate
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            _, val_loss = model.get_task_prediction(
                batch['images'].cuda(),
                batch['target'].cuda()
            )
            val_losses.append(val_loss.item())
    
    val_loss = sum(val_losses) / len(val_losses)
    print(f"Epoch {epoch}: val_loss {val_loss:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'model': model.state_dict(),
            'config': config,
            'val_loss': val_loss
        }, f"{out_dir}/best_model.pt")
