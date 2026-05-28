"""
AstroPT Multimodal Embedding Extractor.

Features:
- Uses Numpy Memmap to write directly to disk (Low RAM usage).
- Dynamic folder naming based on run and checkpoint.
- Dynamically detects active modalities (N-modal support).
- Generates both individual .npy files (fast access) and .npz (portable).
- Supports different pooling strategies and cross-modal ordering for probing.

Author: Victor Alonso Rodriguez
Date: May 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import gc
import numpy as np
from dataclasses import fields
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
from contextlib import nullcontext

from astropt.dataloader_multimodal import MultimodalDatasetArrow
from astropt.training_utils import create_dataloaders
from astropt.config import TrainingConfig
from astropt.model_utils import load_local_model
from sklearn.decomposition import IncrementalPCA

# Configure logging
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-Extract")

def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="AstroPT Multimodal Embedding Extractor")
    
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing training weights")
    parser.add_argument("--data_dir", type=str, required=True, help="Arrow data root directory")
    parser.add_argument("--save_dir", type=str, required=True, help="Output Saving Directory")
    parser.add_argument("--ckpt_name", type=str, default="ckpt_best.pt", help="Checkpoint filename")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--pool_method_img", type=str, default="mean", help="Pooling method for images")
    parser.add_argument("--pool_method_spec", type=str, default="rank", help="Pooling method for spectra")
    parser.add_argument("--pca_dim", type=int, default=0, help="PCA reduction dimension (0=disabled)")
    parser.add_argument("--order_mode", type=str, default="isolated", choices=["isolated", "causal_clean", "bidirectional_leaky"], 
                        help="Modality processing strategy for cross-modal contamination control."
    )
    parser.add_argument("--joint_mode", type=str, default="mean", choices=["mean", "l2mean", "weighted"], 
                        help="Joint embedding fusion strategy"
    )
    parser.add_argument("--joint_alpha", type=float, default=0.5, 
                        help="Weight for image embeddings in weighted joint fusion"
    )
    parser.add_argument("--exp_tag", type=str, default="", help="Optional user tag for folder name")
    parser.add_argument("--draw_from_centre", action="store_true", default=False, 
                        help="Use embeddings from middle layer instead of final layer",
    )
    parser.add_argument("--only_cls", action="store_true", default=False,
                        help="Only extract CLS token embeddings"
    )
    
    return parser.parse_args()


def apply_pooling(tensor: torch.Tensor,
                  method: str = "mean", 
                  p_value: float = 2.0, 
                  mixed_lambda: float = 0.5,
                  cls_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
    """Applying different pooling methods to reduce token sequences to vectors."""
    if method == 'cls':
        if cls_token is not None:
            return cls_token.squeeze(1)
        else:
            return tensor.mean(dim=1)
            
    if method == 'mean':
        return tensor.mean(dim=1)
    elif method == 'max':
        return tensor.max(dim=1)[0]
    elif method == 'mixed':
        return mixed_lambda * tensor.max(dim=1)[0] + (1 - mixed_lambda) * tensor.mean(dim=1)
    elif method == 'lp':
        return (torch.mean(tensor.abs()**p_value, dim=1))**(1.0 / p_value)
    elif method == 'rank':
        sorted_tensor, _ = torch.sort(tensor, dim=1, descending=True)
        k = max(1, int(sorted_tensor.size(1) * 0.1))
        return sorted_tensor[:, :k, :].mean(dim=1)
    else:
        raise ValueError(f"Invalid pooling method '{method}'.")

def apply_pca_to_memmap(
        file_path: str | Path, 
        original_shape: Tuple, 
        n_components: int = 0, 
        batch_size: int = 1024
    ) -> Path:
    """Incremental PCA on large memmapped files."""
    file_path = Path(file_path)
    logger.info(f"Starting PCA {file_path.name} (Target: {n_components} dims)")
    
    ipca = IncrementalPCA(n_components=n_components)
    arr_mmap = np.load(file_path, mmap_mode='r')
    
    for i in range(0, original_shape[0], batch_size):
        chunk = arr_mmap[i:i + batch_size]
        if chunk.shape[0] >= n_components:
            ipca.partial_fit(chunk)
    
    pca_path = file_path.with_name(f"{file_path.stem}_pca{n_components}.npy")
    new_mmap = np.lib.format.open_memmap(
        pca_path, mode='w+', dtype='float32', shape=(original_shape[0], n_components)
    )
    
    for i in range(0, original_shape[0], batch_size):
        chunk = arr_mmap[i:i + batch_size]
        new_mmap[i:i + batch_size] = ipca.transform(chunk)
    
    new_mmap.flush()
    return pca_path

def get_output_folder_name(args: argparse.Namespace, is_multimodal: bool) -> Path:
    """Generate a descriptive experiment folder name."""
    ckpt_suffix = args.ckpt_name.replace(".pt", "").replace("ckpt_", "")
    
    parts = [
        ckpt_suffix,
        f"img-{args.pool_method_img}",
        f"spec-{args.pool_method_spec}",
        "mid" if args.draw_from_centre else "final",
    ]

    if is_multimodal:
        parts.append(args.order_mode[:3])
        parts.append(f"j-{args.joint_mode}")

    if args.pca_dim > 0:
        parts.append(f"pca{args.pca_dim}")

    if args.exp_tag:
        parts.append(args.exp_tag)

    return Path("_".join(parts))


def reorder_modal_inputs(
    inputs: Dict[str, torch.Tensor],
    modality_order: List[str],
) -> Dict[str, torch.Tensor]:
    """Helper to swap modality insertion order."""
    ordered = {}
    for mod in modality_order:
        pos_key = f"{mod}_positions"
        if mod in inputs and pos_key in inputs:
            ordered[mod] = inputs[mod]
            ordered[pos_key] = inputs[pos_key]
    for key, val in inputs.items():
        if key not in ordered:
            ordered[key] = val
    return ordered

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model and Config
    ckpt_path = Path(args.weights_dir) / args.ckpt_name
    model, config, registry, raw_config_dict = load_local_model(ckpt_path, device)
    model.eval()

    # 2. Setup Dataloader
    # Update config with provided data_dir
    raw_config_dict['data_dir'] = args.data_dir
    raw_config_dict['batch_size'] = args.batch_size
    
    valid_keys = {f.name for f in fields(TrainingConfig)}
    clean_config_dict = {k: v for k, v in raw_config_dict.items() if k in valid_keys}
    training_config = TrainingConfig(**clean_config_dict)
    
    _, loader, _ = create_dataloaders(training_config, ddp=False)
    ds = loader.dataset
    total_samples = len(ds)
    emb_dim = config.n_embd

    active_modalities = [m for m in training_config.modalities if m in registry.names()]
    has_cls = getattr(config, 'use_cls_token', False)
    is_multimodal = len(active_modalities) > 1

    # 3. Setup Output
    folder_name = get_output_folder_name(args, is_multimodal)
    save_path = Path(args.save_dir) / folder_name
    save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting to: {save_path}")

    mmaps = {
        'ids': np.lib.format.open_memmap(save_path / 'ids.npy', mode='w+', dtype='int64', shape=(total_samples,))
    }
    
    canonical_mods = []
    if not args.only_cls:
        canonical_mods.extend(active_modalities)
    if has_cls:
        canonical_mods.append('cls')
    
    for mod in canonical_mods:
        mmaps[mod] = np.lib.format.open_memmap(save_path / f'{mod}.npy', mode='w+', dtype='float32', shape=(total_samples, emb_dim))
    
    # Define amp autocast context for bfloat16 compatibility
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    # Pre-detect if model will provide experts to initialize mmaps
    sample_batch = next(iter(loader))
    sample_B = MultimodalDatasetArrow.process_modes(sample_batch, registry, device, use_token_mixing=False, use_cls_token=has_cls)
    with torch.no_grad(), ctx:
        sample_emb = model.get_embeddings(sample_B["X"], draw_from_centre=args.draw_from_centre)
    
    expert_keys = [k for k in sample_emb.keys() if "_expert" in k or "_phase" in k]
    for ek in expert_keys:
        mmaps[ek] = np.lib.format.open_memmap(save_path / f'{ek}.npy', mode='w+', dtype='float32', shape=(total_samples, emb_dim))

    if is_multimodal and not args.only_cls:
        mmaps['joint'] = np.lib.format.open_memmap(save_path / 'joint.npy', mode='w+', dtype='float32', shape=(total_samples, emb_dim))

    # 4. Extraction Loop
    start_idx = 0

    with torch.no_grad(), ctx:
        for batch in tqdm(loader, desc="Extracting"):
            B = MultimodalDatasetArrow.process_modes(
                batch, registry, device,
                use_token_mixing=False, # Disable mixing for clean isolated extraction
                use_cls_token=has_cls,
                cls_position=getattr(config, 'cls_position', 'last')
            )
            
            curr_bs = len(batch['targetid'])
            end_idx = start_idx + curr_bs
            mmaps['ids'][start_idx:end_idx] = batch['targetid'].cpu().numpy()

            embeddings = {}
            if is_multimodal and args.order_mode == "isolated":
                # Isolated pass per modality to prevent cross-contamination
                for mod in active_modalities:
                    isolated_X = {mod: B["X"][mod], f"{mod}_positions": B["X"][f"{mod}_positions"]}
                    emb = model.get_embeddings(isolated_X, draw_from_centre=args.draw_from_centre)
                    embeddings[mod] = emb[mod]
                    if has_cls and 'cls' in emb:
                        embeddings['cls'] = emb['cls'] # CLS will be from the last pass, or we could average
                    # Copy phase and expert embeddings if present in this isolated pass
                    for k, v in emb.items():
                        if "_expert" in k or "_phase" in k:
                            embeddings[k] = v
            else:
                # Standard forward
                embeddings = model.get_embeddings(B["X"], draw_from_centre=args.draw_from_centre)

            # Pooling and storage
            pooled = {}
            cls_token = embeddings.get("cls")
            
            for mod in active_modalities:
                if mod in embeddings:
                    method = args.pool_method_img if "Image" in mod else args.pool_method_spec
                    p = apply_pooling(embeddings[mod], method=method, cls_token=cls_token)
                    pooled[mod] = p
                    if mod in mmaps:
                        mmaps[mod][start_idx:end_idx] = p.float().cpu().numpy()

            # Expert tokens storage (Direct access, no pooling needed)
            for ek in expert_keys:
                if ek in embeddings:
                    mmaps[ek][start_idx:end_idx] = embeddings[ek].squeeze(1).float().cpu().numpy()

            if cls_token is not None and 'cls' in mmaps:
                mmaps['cls'][start_idx:end_idx] = cls_token.squeeze(1).float().cpu().numpy()

            # Joint embedding
            if is_multimodal and 'joint' in mmaps:
                # Assume 2 modalities for simplicity in fusion, but could be N
                p_list = [pooled[m] for m in active_modalities if m in pooled]
                if len(p_list) >= 2:
                    if args.joint_mode == "l2mean":
                        joint = (F.normalize(p_list[0], p=2, dim=1) + F.normalize(p_list[1], p=2, dim=1)) / 2.0
                    elif args.joint_mode == "weighted":
                        joint = args.joint_alpha * p_list[0] + (1.0 - args.joint_alpha) * p_list[1]
                    else:
                        joint = sum(p_list) / len(p_list)
                    mmaps['joint'][start_idx:end_idx] = joint.float().cpu().numpy()

            start_idx = end_idx

    # 5. Flush and PCA
    for m in mmaps.values(): m.flush()
    logger.info("Extraction complete. Starting PCA/NPZ packaging...")

    # PCA and NPZ
    final_data = {k: np.load(save_path / f"{k}.npy", mmap_mode='r') for k in mmaps.keys()}
    if args.pca_dim > 0:
        for k in final_data:
            if k != 'ids':
                pca_path = apply_pca_to_memmap(save_path / f"{k}.npy", final_data[k].shape, args.pca_dim)
                final_data[k] = np.load(pca_path, mmap_mode='r')

    np.savez_compressed(save_path / "embeddings_all.npz", **final_data)
    
    # Save Metadata
    with open(save_path / "extraction_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info("Done.")

if __name__ == "__main__":
    main()
