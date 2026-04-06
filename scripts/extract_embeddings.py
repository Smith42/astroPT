"""
AstroPT Production Embedding Extractor.

Features:
- Uses Numpy Memmap to write directly to disk (Low RAM usage).
- Dynamic folder naming based on run and checkpoint.
- Dynamically detects active modalities (Unimodal vs Multimodal) to avoid generating empty/dummy files.
- Generates both individual .npy files (fast access) and .npz (portable).

Author: Victor Alonso Rodriguez
Date: April 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import gc
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple
from contextlib import nullcontext


from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow
from astropt.model_utils import load_local_model
from sklearn.decomposition import IncrementalPCA

# Configure logging
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-ProdExtract")

class SuppressVisWarnings(logging.Filter):
    def filter(self, record):
        return "VIS image is None in Arrow dataset" not in record.getMessage()

for handler in logging.root.handlers:
    handler.addFilter(SuppressVisWarnings())


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="AstroPT Production Embedding Extractor")
    
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing training weights")
    parser.add_argument("--data_dir", type=str, required=True, help="Arrow data root directory")
    parser.add_argument("--save_dir", type=str, required=True, help="Plot Saving Directory")
    parser.add_argument("--ckpt_name", type=str, default="ckpt_best.pt", help="Checkpoint filename")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--pool_method_img", type=str, default="mean", help="Pooling method to reduce dimensionality for images")
    parser.add_argument("--pool_method_spec", type=str, default="rank", help="Pooling method to reduce dimensionality for spectra")
    parser.add_argument("--pca_dim", type=int, default=0, help="Applying PCA analysis with the resulting pca_dim dimension. 0 means disabled")
    parser.add_argument("--order_mode", type=str, default="isolated", choices=["isolated", "causal_clean", "bidirectional_leaky"], 
                        help="Modality processing order strategy. 'isolated' ensures 100% data leak prevention for probing."
    )
    parser.add_argument("--joint_mode", type=str, default="mean", choices=["mean", "l2mean", "weighted"], 
                        help="Joint embedding fusion strategy when both modalities are available"
    )
    parser.add_argument("--joint_alpha", type=float, default=0.5, 
                        help="Weight for image embeddings in weighted joint fusion: alpha*image + (1-alpha)*spectra"
    )
    parser.add_argument("--exp_tag", type=str, default="", help="Optional user tag appended to output folder name")
    parser.add_argument("--draw_from_centre", action="store_true", default=False, 
                        help="Use embeddings from middle layer instead of the final layer (default: final layer)",
    )
    
    return parser.parse_args()


def apply_pooling(tensor: torch.Tensor,
                  method: str = "mean", 
                  p_value: float = 2.0, 
                  mixed_lambda: float = 0.5
    ) -> torch.Tensor:
    """
    Applying different pooling methods.
    Reference: Gholamalinezhad & Khosravi (2020).
    """
    if method == 'mean':
        # Average Pooling
        return tensor.mean(dim=1)
    
    elif method == 'max':
        # Max Pooling
        return tensor.max(dim=1)[0]
    
    elif method == 'mixed':
        # Mixed Pooling
        max_p = tensor.max(dim=1)[0]
        avg_p = tensor.mean(dim=1)
        return mixed_lambda * max_p + (1 - mixed_lambda) * avg_p
    
    elif method == 'lp':
        # L_p Pooling
        return (torch.mean(tensor.abs()**p_value, dim=1))**(1.0 / p_value)
    
    elif method == 'rank':
        # Rank-based Average Pooling (RAP)
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
    """
    Applying PCA to memmaps files
    """
    file_path = Path(file_path)
    logger.info(f"Starting PCA {file_path.name} (Target: {n_components} dims)")
    
    ipca = IncrementalPCA(n_components=n_components)
    
    arr_mmap = np.load(file_path, mmap_mode='r')
    for i in range(0, original_shape[0], batch_size):
        chunk = arr_mmap[i:i + batch_size]
        ipca.partial_fit(chunk)
    
    pca_path = file_path.with_name(f"{file_path.stem}_pca{n_components}.npy")
    new_mmap = np.lib.format.open_memmap(
        pca_path, mode='w+', dtype='float32', shape=(original_shape[0], n_components)
    )
    
    for i in range(0, original_shape[0], batch_size):
        chunk = arr_mmap[i:i + batch_size]
        new_mmap[i:i + batch_size] = ipca.transform(chunk)
    
    new_mmap.flush()
    logger.info(f"PCA completed. New file: {pca_path.name}")
    return pca_path

def _sanitize_folder_token(token: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in ["-", "_"]) else "-" for ch in token)
    cleaned = cleaned.strip("-")
    return cleaned if cleaned else "untagged"


def _float_to_tag(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def get_output_folder_name(
    ckpt_name: str,
    pool_method_img: str,
    pool_method_spec: str,
    pca_dim: int,
    draw_from_centre: bool,
    order_mode: str,
    joint_mode: str,
    joint_alpha: float,
    is_multimodal: bool,
    exp_tag: str,
) -> Path:
    """Generate a descriptive experiment folder name based on extraction flags."""
    ckpt_suffix = ckpt_name.replace(".pt", "").replace("ckpt_", "")

    order_map = {
        "isolated": "ord-iso",
        "causal_clean": "ord-clean",
        "bidirectional_leaky": "ord-bi_leak",
    }

    parts = [
        _sanitize_folder_token(ckpt_suffix),
        f"img-{_sanitize_folder_token(pool_method_img)}",
        f"spec-{_sanitize_folder_token(pool_method_spec)}",
        "layer-mid" if draw_from_centre else "layer-final",
    ]

    if is_multimodal:
        parts.append(order_map[order_mode])
        if joint_mode == "weighted":
            parts.append(f"joint-w{_float_to_tag(joint_alpha)}")
        elif joint_mode == "l2mean":
            parts.append("joint-l2mean")
        else:
            parts.append("joint-mean")

    if pca_dim > 0:
        parts.append(f"pca{pca_dim}")

    if exp_tag.strip():
        parts.append(_sanitize_folder_token(exp_tag.strip()))

    return Path("_".join(parts))


def reorder_modal_inputs(
    inputs: Dict[str, torch.Tensor],
    modality_order: List[str],
) -> Dict[str, torch.Tensor]:
    """Return a copy of inputs where modality tensors follow a specific insertion order."""
    ordered = {}

    # Insert modality tensors first, in requested order.
    for mod in modality_order:
        pos_key = f"{mod}_positions"
        if mod in inputs and pos_key in inputs:
            ordered[mod] = inputs[mod]
            ordered[pos_key] = inputs[pos_key]

    # Preserve any remaining keys (if present).
    for key, val in inputs.items():
        if key not in ordered:
            ordered[key] = val

    return ordered

def main():
    args = parse_args()

    if not 0.0 <= args.joint_alpha <= 1.0:
        logger.error("--joint_alpha must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Load Model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    weights_dir = Path(args.weights_dir)
    save_dir = Path(args.save_dir)
    
    logger.info("Analysis Directories:")
    logger.info(f" --> [Weights]:   {weights_dir}")
    logger.info(f" --> [Saving]:    {save_dir}")
    
    ckpt_path = weights_dir / args.ckpt_name
    
    if not ckpt_path.is_file():
        logger.error(f"Checkpoint not found: {ckpt_path}. Starting smart search.")
        all_ckpts = list(weights_dir.glob("*.pt"))
        if not all_ckpts:
            logger.error(f"FATAL: No .pt file found in {weights_dir}")
            sys.exit(1)

        best_matches = [c for c in all_ckpts if "best" in c.name]
        last_matches = [c for c in all_ckpts if "last" in c.name]
        
        if best_matches:
            ckpt_path = sorted(best_matches, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            logger.info(f"Selected by priority [BEST]: {ckpt_path.name}")
        elif last_matches:
            ckpt_path = sorted(last_matches, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            logger.info(f"Selected by priority [LAST]: {ckpt_path.name}")
        else:
            ckpt_path = sorted(all_ckpts, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            logger.info(f"Selected by date [RECENT]: {ckpt_path.name}")

    logger.info(f"Loading checkpoint: {ckpt_path}")
    
    try:
        model, config, registry, raw_config_dict = load_local_model(ckpt_path, device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # DYNAMIC MODALITY DETECTION
    active_mods = []
    for mod in ['images', 'spectra']:
        try:
            if registry.get_config(mod) is not None:
                active_mods.append(mod)
        except Exception:
            pass 

    if not active_mods:
        logger.warning("Could not read registry. Inspecting config.json directly for training flags...")
        if raw_config_dict.get('images_train', False):
            active_mods.append('images')
        if raw_config_dict.get('spectra_train', False):
            active_mods.append('spectra')


    is_multimodal = len(active_mods) > 1
    logger.info(f"Active modalities detected: {active_mods}")
    logger.info(
        f"Embedding layer source: {'middle layer' if args.draw_from_centre else 'final layer'}"
    )
    logger.info(
        f"Experiment config: order_mode={args.order_mode}, joint_mode={args.joint_mode}, "
        f"joint_alpha={args.joint_alpha}, exp_tag={args.exp_tag if args.exp_tag else 'none'}"
    )

    data_dir = args.data_dir if args.data_dir else raw_config_dict.get('data_dir')
    data_dir = Path(data_dir)
    
    # Retrieve Normalization Constants
    norm_type_img = raw_config_dict.get('images_norm_type', raw_config_dict.get('img_norm_type', 'asinh'))
    norm_scaler_img = raw_config_dict.get('images_norm_scaler', raw_config_dict.get('img_norm_scaler', 1.0))
    norm_const_img = raw_config_dict.get('images_norm_const', raw_config_dict.get('img_norm_const', 1.0))
    
    inverse_spec = raw_config_dict.get('spectra_inverse', False)
    norm_type_spec = raw_config_dict.get('spectra_norm_type', 'constant')
    norm_scaler_spec = raw_config_dict.get('spectra_norm_scaler', 1.0)
    norm_const_spec = raw_config_dict.get('spectra_norm_const', 1.0)
    
    data_tf = EuclidDESIDatasetArrow.data_transforms(
        norm_type_img=norm_type_img, norm_scaler_img=norm_scaler_img, norm_const_img=norm_const_img,
        norm_type_spec=norm_type_spec, norm_scaler_spec=norm_scaler_spec, norm_const_spec=norm_const_spec,
    )
    
    logger.info(f"Initializing Dataset from: {data_dir}")
    ds = EuclidDESIDatasetArrow(
        arrow_folder_root=data_dir, split="test", modality_registry=registry,
        spiral=True, stochastic=False, transform=data_tf,
        spectra_inverse=inverse_spec,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    total_samples = len(ds)
    emb_dim = config.n_embd
    
    folder_name = get_output_folder_name(
        ckpt_name=args.ckpt_name,
        pool_method_img=args.pool_method_img,
        pool_method_spec=args.pool_method_spec,
        pca_dim=args.pca_dim,
        draw_from_centre=args.draw_from_centre,
        order_mode=args.order_mode,
        joint_mode=args.joint_mode,
        joint_alpha=args.joint_alpha,
        is_multimodal=is_multimodal,
        exp_tag=args.exp_tag,
    )
    save_path = save_dir / folder_name
    save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Embeddings directory created: {save_path}")

    # DYNAMIC MEMMAP ALLOCATION
    mmaps = {}
    mmaps['ids'] = np.lib.format.open_memmap(
        save_path / 'ids.npy', mode='w+', dtype='int64', shape=(total_samples,)
    )
    
    for mod in active_mods:
        mmaps[mod] = np.lib.format.open_memmap(
            save_path / f'{mod}.npy', mode='w+', dtype='float32', shape=(total_samples, emb_dim)
        )
        
    if is_multimodal:
        mmaps['joint'] = np.lib.format.open_memmap(
            save_path / 'joint.npy', mode='w+', dtype='float32', shape=(total_samples, emb_dim)
        )

    # Extraction loop
    if device.type == "cuda":
        ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype) # type: ignore
    else:
        ctx = nullcontext()
    
    start_idx = 0

    with torch.no_grad(), ctx:
        for batch in tqdm(dl, desc="Extracting Embeddings"):
            if batch is None: continue
                
            current_batch_size = len(batch['targetid'])
            end_idx = start_idx + current_batch_size

            X = EuclidDESIDatasetArrow.prepare_batch(
                batch_data=batch,
                modality_registry=registry,
                device=device,
            )

            # SAFE EMBEDDINGS EXTRACTION 
            embeddings = {}
            if is_multimodal and ("images" in active_mods) and ("spectra" in active_mods):
                if args.order_mode == "isolated":
                    # 100% Causal Isolation: Only pass one modality per forward pass
                    for mod in active_mods:
                        isolated_X = {mod: X[mod], f"{mod}_positions": X[f"{mod}_positions"]}
                        emb = model.get_embeddings(
                            isolated_X,
                            draw_from_centre=args.draw_from_centre,
                        )
                        embeddings[mod] = emb[mod]
                        
                elif args.order_mode == "causal_clean":
                    # Causal Mask Isolation: Disable token mixing, place desired mod first.
                    original_mixing = getattr(model.config, 'use_token_mixing', False)
                    if original_mixing:
                        model.config.use_token_mixing = False
                        
                    emb_img_first = model.get_embeddings(
                        reorder_modal_inputs(X, ["images", "spectra"]),
                        draw_from_centre=args.draw_from_centre,
                    )
                    emb_spec_first = model.get_embeddings(
                        reorder_modal_inputs(X, ["spectra", "images"]),
                        draw_from_centre=args.draw_from_centre,
                    )
                    
                    if original_mixing:
                        model.config.use_token_mixing = True
                        
                    embeddings["images"] = emb_img_first["images"]
                    embeddings["spectra"] = emb_spec_first["spectra"]
                    
                elif args.order_mode == "bidirectional_leaky":
                    if batch_idx == 0:
                        logger.warning("Using bidirectional_leaky! Embeddings are cross-contaminated via averaging. NOT valid for zero-shot downstream probing.")
                        
                    emb_forward = model.get_embeddings(
                        reorder_modal_inputs(X, ["images", "spectra"]),
                        draw_from_centre=args.draw_from_centre,
                    )
                    emb_reverse = model.get_embeddings(
                        reorder_modal_inputs(X, ["spectra", "images"]),
                        draw_from_centre=args.draw_from_centre,
                    )
                    for modality in active_mods:
                        if modality in emb_forward and modality in emb_reverse:
                            embeddings[modality] = 0.5 * (emb_forward[modality] + emb_reverse[modality])
            else:
                embeddings = model.get_embeddings(
                    X,
                    draw_from_centre=args.draw_from_centre,
                )
            
            img_pooled = None
            spec_pooled = None
            
            # DYNAMIC MODALITY PROCESSING
            for modality in active_mods:
                if modality in embeddings:
                    method = args.pool_method_img if modality == 'images' else args.pool_method_spec
                    pooled_tensor = apply_pooling(embeddings[modality], method=method)
                    
                    if modality == 'images': img_pooled = pooled_tensor 
                    else: spec_pooled = pooled_tensor
                        
                    mmaps[modality][start_idx:end_idx] = pooled_tensor.float().cpu().numpy()

            # Hybrid Joint Embedding (Only if multimodal)
            if is_multimodal and 'joint' in mmaps:
                if img_pooled is not None and spec_pooled is not None:

                    if args.joint_mode == "l2mean":
                        img_norm = F.normalize(img_pooled, p=2, dim=1)
                        spec_norm = F.normalize(spec_pooled, p=2, dim=1)
                        joint_tensor = (img_norm + spec_norm) / 2.0
                    elif args.joint_mode == "weighted":
                        joint_tensor = args.joint_alpha * img_pooled + (1.0 - args.joint_alpha) * spec_pooled
                    else:
                        joint_tensor = (img_pooled + spec_pooled) / 2.0

                    mmaps['joint'][start_idx:end_idx] = joint_tensor.float().cpu().numpy()
                elif img_pooled is not None:
                    mmaps['joint'][start_idx:end_idx] = img_pooled.float().cpu().numpy()
                elif spec_pooled is not None:
                    mmaps['joint'][start_idx:end_idx] = spec_pooled.float().cpu().numpy()

            mmaps['ids'][start_idx:end_idx] = batch['targetid'].numpy().astype('int64')
            start_idx = end_idx

    # Flush changes to disk dynamically
    for name, mmap_obj in mmaps.items():
        mmap_obj.flush()
    
    del mmaps
    gc.collect()
    logger.info("Raw .npy extraction complete.")

    # DYNAMIC NPZ PACKAGING & PCA
    logger.info("Preparing files for PCA and .npz packaging...")
    
    final_paths = {'targetid': save_path / "ids.npy"}
    for mod in active_mods:
        final_paths[mod] = save_path / f"{mod}.npy"
    if is_multimodal:
        final_paths['joint'] = save_path / "joint.npy"

    if args.pca_dim > 0:
        logger.info(f"Applying PCA reduction to {args.pca_dim} dimensions...")
        for key in list(final_paths.keys()):
            if key != 'targetid':  # Do not apply PCA to target IDs
                final_paths[key] = apply_pca_to_memmap(final_paths[key], (total_samples, emb_dim), args.pca_dim)
            
    logger.info("Generating portable .npz file with active modalities only...")
    npz_path = save_path / "embeddings_all.npz"
    
    try:
        # Dynamically load the final paths into the npz kwargs
        npz_kwargs = {k: np.load(v, mmap_mode='r') for k, v in final_paths.items()}
        np.savez_compressed(npz_path, **npz_kwargs)
        logger.info(f"Compressed archive created: {npz_path}")
    except Exception as e:
        logger.error(f"Error creating .npz: {e}")

    metadata_path = save_path / "experiment_config.json"
    metadata = {
        "ckpt_name": args.ckpt_name,
        "resolved_checkpoint": str(ckpt_path),
        "active_modalities": active_mods,
        "is_multimodal": is_multimodal,
        "pool_method_img": args.pool_method_img,
        "pool_method_spec": args.pool_method_spec,
        "pca_dim": args.pca_dim,
        "draw_from_centre": args.draw_from_centre,
        "order_mode": args.order_mode,
        "joint_mode": args.joint_mode,
        "joint_alpha": args.joint_alpha,
        "exp_tag": args.exp_tag,
    }
    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Experiment metadata saved: {metadata_path}")
    except Exception as e:
        logger.error(f"Error writing metadata file: {e}")

    logger.info("Extraction Pipeline Finished Successfully.")

if __name__ == "__main__":
    main()