import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import dataclasses
from astropt.dataloader_multimodal import MultimodalDatasetArrow
from astropt.training_utils import create_dataloaders
from astropt.config import TrainingConfig
from astropt.model_utils import load_local_model

# We need the probe models defined exactly as in the probing script to load their weights
import torch.nn as nn
class SequenceAttentionProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, is_sequence: bool = False):
        super().__init__()
        self.is_sequence = is_sequence
        if is_sequence:
            self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        self.head = nn.Linear(input_dim, output_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_sequence:
            B, S, D = x.shape
            q = self.query.expand(B, -1, -1)
            scores = torch.sum(x * q, dim=-1) / math.sqrt(D)
            weights = F.softmax(scores, dim=-1).unsqueeze(-1)
            attended = torch.sum(x * weights, dim=1)
            return self.head(attended)
        else:
            return self.head(x)

class SequenceAttentionMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, is_sequence: bool = False):
        super().__init__()
        self.is_sequence = is_sequence
        if is_sequence:
            self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_sequence:
            B, S, D = x.shape
            q = self.query.expand(B, -1, -1)
            scores = torch.sum(x * q, dim=-1) / math.sqrt(D)
            weights = F.softmax(scores, dim=-1).unsqueeze(-1)
            attended = torch.sum(x * weights, dim=1)
            return self.net(attended)
        else:
            return self.net(x)

# Logger
logging.basicConfig(format="[%(asctime)s] [%(levelname)s] %(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("AstroPT-GradCAM")

def parse_args():
    parser = argparse.ArgumentParser(description="Task-Specific Saliency Maps via Grad-CAM")
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory with AstroPT checkpoint")
    parser.add_argument("--probe_file", type=str, required=True, help="Path to probing_MLP.pt or probing_LP.pt")
    parser.add_argument("--data_dir", type=str, required=True, help="Arrow data root")
    parser.add_argument("--target", type=str, required=True, help="Target task column (e.g. sersic_index_VIS)")
    parser.add_argument("--modality", type=str, required=True, help="Expert Modality used in probing (e.g. EuclidImage_phase1)")
    parser.add_argument("--target_ids", nargs="+", type=int, help="Specific galaxy targetid to interpret")
    parser.add_argument("--num_plot", type=int, default=5, help="Number of random galaxies if target_ids not specified")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save plots")
    return parser.parse_args()

x_val = None
x_grad = None

def get_patched_forward(model):
    def hook(module, input, output):
        global x_val
        x, expert_positions, modality_indices = output
        x_val = x
        x_val.retain_grad()
        def save_grad(g):
            global x_grad
            x_grad = g
        x_val.register_hook(save_grad)
        return output
    model.embedding_layer.register_forward_hook(hook)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    save_dir = Path(args.save_dir) if args.save_dir else Path(args.probe_file).parent / "saliency_maps"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading AstroPT from {args.weights_dir}")
    ckpt_path = sorted(list(Path(args.weights_dir).glob("*.pt")), key=lambda x: x.stat().st_mtime, reverse=True)[0]
    model, config, registry, raw_config_dict = load_local_model(ckpt_path, device)
    
    # We must be in train mode or require_grad=True to do backprop
    model.eval() 
    for param in model.parameters():
        param.requires_grad = False
    
    # Hook embedding layer
    get_patched_forward(model)
    
    logger.info(f"Loading Probe Weights from {args.probe_file}")
    probe_dict = torch.load(args.probe_file, map_location=device, weights_only=False)
    key = f"{args.target}/{args.modality}"
    if key not in probe_dict:
        logger.error(f"Key {key} not found in probe file! Available keys: {list(probe_dict.keys())}")
        sys.exit(1)
        
    pdata = probe_dict[key]
    probe_type = "mlp" if "MLP" in args.probe_file else "lp"
    
    if probe_type == "mlp":
        probe_model = SequenceAttentionMLP(pdata["input_dim"], pdata["output_dim"], is_sequence=False).to(device)
    else:
        probe_model = SequenceAttentionProbe(pdata["input_dim"], pdata["output_dim"], is_sequence=False).to(device)
        
    probe_model.load_state_dict(pdata["model_state_dict"])
    
    # Cast probe model to match AstroPT's dtype
    model_dtype = next(model.parameters()).dtype
    probe_model = probe_model.to(model_dtype)
    probe_model.eval()
    for param in probe_model.parameters():
        param.requires_grad = False

    # Load data
    valid_keys = {f.name for f in dataclasses.fields(TrainingConfig)}
    filtered_config = {k: v for k, v in raw_config_dict.items() if k in valid_keys}
    config_obj = TrainingConfig(**filtered_config)
    config_obj.data_dir = args.data_dir
    config_obj.batch_size = 1
    _, val_loader, _ = create_dataloaders(config_obj, ddp=False)
    ds = val_loader.dataset
    
    # Select subset
    indices_to_plot = []
    if args.target_ids:
        all_ids = ds.ds['targetid']
        id_map = {int(tid): idx for idx, tid in enumerate(all_ids)}
        for tid in args.target_ids:
            if tid in id_map: indices_to_plot.append(id_map[tid])
    else:
        indices_to_plot = np.random.choice(len(ds), args.num_plot, replace=False).tolist()
        
    subset = Subset(ds, indices_to_plot)
    loader = DataLoader(subset, batch_size=1, shuffle=False)
    
    base_modality = args.modality.split('_')[0]
    mod_idx = model.mod_names.index(base_modality) if base_modality in model.mod_names else 0
    phase = 1 if 'phase1' in args.modality else 2 if 'phase2' in args.modality else 0
    
    logger.info(f"Starting Grad-CAM for task {args.target} via {args.modality}")
    
    results = []
    
    for idx, batch in enumerate(loader):
        tid = batch['targetid'].item() if torch.is_tensor(batch['targetid']) else batch['targetid'][0]
        
        B = MultimodalDatasetArrow.process_modes(
            batch, registry, device, 
            use_token_mixing=raw_config_dict.get("use_token_mixing", False),
            use_cls_token=getattr(config, 'use_cls_token', False),
            cls_position=getattr(config, 'cls_position', 'last')
        )
        model_dtype = next(model.parameters()).dtype
        for k, v in B["X"].items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                B["X"][k] = v.to(model_dtype)
                
        # Forward pass AstroPT
        global x_grad, x_val
        x_grad = None
        x_val = None
        
        # We need gradients to flow through x
        out = model._forward_native(B["X"], None)
        
        # Re-extract expert token from x_val or out
        # Actually `model.get_embeddings` is better, but since we intercepted `x` we can do the slicing ourselves or just use out.
        # Wait, get_embeddings returns the expert tokens natively. Let's use get_embeddings directly on the model.
        # However, to flow gradients, we must call get_embeddings inside the graph.
        
        emb_dict = model.get_embeddings(B["X"], batch_modes=None)
        expert_token = emb_dict.get(args.modality)
        
        if expert_token is None:
            logger.error(f"Modality {args.modality} not found in model outputs!")
            continue
            
        # Probing forward
        preds = probe_model(expert_token)
        
        # We want the gradient of the prediction with respect to x_val (the patches)
        # If output is multidimensional (classification), take the max class or sum
        target_score = preds.max() if pdata["task_type"] == 'classification' else preds.sum()
        
        # Backward
        model.zero_grad()
        probe_model.zero_grad()
        target_score.backward()
        
        if x_grad is None:
            logger.error("x_grad was not populated! Hook failed or gradient broken.")
            continue
            
        # x_val is (B, T, D), x_grad is (B, T, D)
        # Get modality indices to filter only the patches of the base_modality
        _, _, mod_indices = model.embedding_layer(B["X"], None)
        
        mask = (mod_indices == mod_idx).squeeze(0) # (T)
        
        val_patches = x_val[0, mask, :] # (N_patches, D)
        grad_patches = x_grad[0, mask, :] # (N_patches, D)
        
        # Grad-CAM: alpha = mean gradient per dimension? Or simple gradient * activation?
        # Usually for sequences gradient * activation works best: sum(grad * act, dim=-1)
        saliency = torch.sum(grad_patches * val_patches, dim=-1)
        saliency = F.relu(saliency) # Only positive contributions
        
        saliency = saliency.to(torch.float32).cpu().numpy()
        
        results.append({
            'targetid': tid,
            'prediction': preds.detach().to(torch.float32).cpu().numpy().tolist(),
            'saliency': saliency.tolist()
        })
        
        # Plot if image
        if 'Image' in base_modality:
            # Assume square image patches
            n_patches = saliency.shape[0]
            side = int(math.sqrt(n_patches))
            if side * side == n_patches:
                sal_map = saliency.reshape(side, side)
                
                fig, ax = plt.subplots()
                im = ax.imshow(sal_map, cmap='magma')
                plt.colorbar(im, ax=ax, label='Grad-CAM Saliency')
                ax.set_title(f"Target: {args.target} | ID: {tid}\nPred: {target_score.item():.3f}")
                plt.axis('off')
                
                fpath = save_dir / f"saliency_{args.target}_{tid}.png"
                fig.savefig(fpath, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved {fpath}")
            else:
                logger.warning(f"Could not reshape {n_patches} patches to square.")
                
    # Save raw array
    with open(save_dir / f"saliency_{args.target}.json", "w") as f:
        json.dump(results, f)
    logger.info(f"Saved all raw Saliency vectors to {save_dir / f'saliency_{args.target}.json'}")

if __name__ == "__main__":
    main()
