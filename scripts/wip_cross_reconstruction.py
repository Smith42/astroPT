"""
Spectrum Reconstruction Analysis Script.

This script loads a trained AstroPT model checkpoint and performs inference
on either real galaxy data (from Arrow files) or synthetic data (if real data 
is unavailable). It generates a comparison plot between the original input 
spectrum and the model's reconstruction to analyze artifacts (e.g., patching effects).
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from typing import Optional, Dict, Tuple, Any, List

from astropt.model import GPT, GPTConfig, ModalityRegistry, ModalityConfig
from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow

# PATH to data in Teide
BASE_PATH = "/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/"
CKPT_PATH = os.path.join(BASE_PATH, "astropt_100M_250K_arrow_T1_20251214/ckpt.pt")
DATA_DIR = "/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

def reconstruct_image_from_patches(
    patches: np.ndarray, 
    patch_size: int, 
    channels: int,
    is_spiral: bool
) -> np.ndarray:
    """
    Reconstructs a 2D image from flattened token patches using the 
    logic defined in the Dataloader.
    
    Args:
        patches: Flattened patches (N, Patch_Dim).
        patch_size: Pixel size of patches.
        channels: Number of image channels.
        is_spiral: Boolean flag indicating if spiral ordering was used.
        
    Returns:
        np.ndarray: Image (H, W, C) ready for plotting.
    """
    n_patches = patches.shape[0]
    side_len = int(np.sqrt(n_patches))
    
    # 1. Antispiralise if needed
    # We use the static method directly from the class to avoid code duplication.
    if is_spiral:
        # _spiral_sorting is a @staticmethod, so we can call it without instantiating the class
        spiral_indices = EuclidDESIDatasetArrow._spiral_sorting(side_len)
        patches = patches[spiral_indices]
        
    # 2. Reshape flattened vectors to spatial blocks
    # Shape: (Grid_H * Grid_W, P*P*C) -> (Grid_H, Grid_W, P, P, C)
    reshaped = patches.reshape(side_len, side_len, patch_size, patch_size, channels)
    
    # 3. Transpose to assemble full image
    # (Grid_H, Grid_W, P_H, P_W, C) -> (Grid_H, P_H, Grid_W, P_W, C)
    transposed = reshaped.transpose(0, 2, 1, 3, 4)
    
    # 4. Final Merge
    img_h = side_len * patch_size
    img_w = side_len * patch_size
    full_img = transposed.reshape(img_h, img_w, channels)
    
    return full_img


def parse_hparams_log(log_path: str) -> Dict[str, Any]:
    """
    Parses the hparams.log file generated during training to extract
    configuration parameters. Handles type conversion (int, float, bool).
    
    Args:
        log_path: Path to the hparams.log file.
        
    Returns:
        Dict with configuration keys and correctly typed values.
    """
    config = {}
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"hparams.log not found at {log_path}")
        
    print(f"[INFO] Parsing config from: {os.path.basename(log_path)}")
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
                
            key, val_str = line.split(":", 1)
            key = key.strip()
            val_str = val_str.strip()
            
            # Type Inference
            if val_str.lower() == "true":
                val = True
            elif val_str.lower() == "false":
                val = False
            elif val_str.lower() == "none":
                val = None
            else:
                # Try Int -> Float -> String
                try:
                    val = int(val_str)
                except ValueError:
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = val_str # Keep as string
            
            config[key] = val
            
    return config


def get_real_sample(data_dir: str, config: Dict, registry: ModalityRegistry) -> Dict[str, Any]:
    """Loads a real sample using the exact training configuration."""
    
    print(f"[INFO] Loading dataset from: {data_dir}")
    
    # Initialize Dataset with Config parameters
    dataset = EuclidDESIDatasetArrow(
        arrow_folder_root=data_dir,
        split="train", 
        modality_registry=registry,
        spiral=config.get('spiral', False),
        transform=EuclidDESIDatasetArrow.get_default_transforms()
    )
    
    # Find valid sample
    for i in range(100):
        sample = dataset[i]
        if "spectra" in sample and sample["spectra"] is not None:
            print(f"[INFO] Sample loaded. Target ID: {sample['targetid']}")
            return sample
            
    raise RuntimeError("No valid samples found in dataset header.")



def load_checkpoint_and_config(ckpt_path: str, device: str) -> Tuple[GPT, Dict, ModalityRegistry]:
    """
    Loads model weights from .pt and configuration from hparams.log.
    assumes hparams.log is in the same directory as the checkpoint.
    """
    print(f"[INFO] Loading checkpoint: {os.path.basename(ckpt_path)}")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    # Locate and Parse hparams.log
    ckpt_dir = os.path.dirname(ckpt_path)
    log_path = os.path.join(ckpt_dir, "hparams.log")
    
    try:
        train_config = parse_hparams_log(log_path)
    except FileNotFoundError:
        # Try to load from checkpoint if log is missing
        print("[WARNING] hparams.log not found. Attempting to load config from checkpoint dict...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        train_config = checkpoint.get('config', None)
        if train_config is None:
             raise ValueError("Could not find configuration in hparams.log OR .pt file.")

    print(f"[INFO] Config Loaded | Spiral: {train_config.get('spiral', False)} | Patch: {train_config.get('spectra_patch_size', 10)}")

    # Reconstruct Registry
    img_patch_size = train_config.get('images_patch_size', 16)
    img_channels = train_config.get('images_channels', 4)
    spec_patch_size = train_config.get('spectra_patch_size', 10)
    
    img_dim = (img_patch_size**2) * img_channels
    spec_dim = spec_patch_size
    
    modalities = [
        ModalityConfig(
            name="images", 
            input_size=img_dim, 
            patch_size=img_patch_size, 
            pos_input_size=1, 
            embed_pos=True
        ),
        ModalityConfig(
            name="spectra", 
            input_size=spec_dim, 
            patch_size=spec_patch_size, 
            pos_input_size=spec_dim, 
            embed_pos=False
        )
    ]
    registry = ModalityRegistry(modalities)

    # Initialize Model
    model_config = GPTConfig(
        block_size=train_config.get('block_size', 1024),
        n_layer=train_config.get('n_layer', 12),
        n_head=train_config.get('n_head', 12),
        n_embd=train_config.get('n_embd', 768),
        dropout=0.0,
        bias=train_config.get('bias', False),
        modalities=registry.modalities.values()
    )
    
    model = GPT(model_config, registry)
    
    # Load Weights
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model']
    
    clean_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_dict, strict=False)
    
    model.to(device)
    model.eval()
    
    return model, train_config, registry



def main():
    
    # Setup
    try:
        model, config, registry = load_checkpoint_and_config(CKPT_PATH, DEVICE)
        sample = get_real_sample(DATA_DIR, config, registry)
    except Exception as e:
        print(f"[CRITICAL] Setup failed: {e}")
        return

    # Inference
    inputs = {k: v.unsqueeze(0).to(DEVICE) for k, v in sample.items() if isinstance(v, torch.Tensor)}
    
    print("[INFO] Running inference...")
    with torch.no_grad():
        outputs, _ = model(inputs, target_modality=None)

    # Process Spectra
    if "spectra" not in outputs: return
    
    spec_patch_size = config['spectra_patch_size']
    input_spec = inputs["spectra"].cpu().numpy()[0].flatten()
    pred_spec = outputs["spectra"].cpu().numpy()[0].flatten()

    # Process Images
    img_patches = inputs["images"].cpu().numpy()[0]
    
    try:
        reconstructed_img = reconstruct_image_from_patches(
            patches=img_patches,
            patch_size=config['images_patch_size'],
            channels=config.get('images_channels', 4),
            is_spiral=config.get('spiral', False)
        )
        
        # Select RGB (VIS, Y, J) and normalize
        rgb_img = reconstructed_img[:, :, 0:3]
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
        
    except Exception as e:
        print(f"[WARNING] Image reconstruction failed: {e}")
        rgb_img = np.zeros((224, 224, 3))

    # Plotting
    print("[INFO] Generating dashboard...")
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.5])

    # Image
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(rgb_img)
    ax_img.set_title(f"Target Galaxy\nID: {sample['targetid']}", fontsize=12)
    ax_img.axis('off')

    # Spectrum
    ax_spec = fig.add_subplot(gs[0, 1])
    ax_spec.plot(input_spec, 'k', alpha=0.6, linewidth=0.8, label='Ground Truth')
    ax_spec.plot(pred_spec, 'r', alpha=0.8, linewidth=0.8, label='Reconstruction')
    ax_spec.set_title("Full Spectrum", fontsize=12)
    ax_spec.legend()

    # Zoom
    ax_zoom = fig.add_subplot(gs[1, 1])
    peak_idx = np.argmax(input_spec)
    zw = 120
    z0, z1 = max(0, peak_idx - zw // 2), min(len(input_spec), peak_idx + zw // 2)
    
    ax_zoom.plot(range(z0, z1), input_spec[z0:z1], 'k.-', alpha=0.3, label='Original')
    ax_zoom.plot(range(z0, z1), pred_spec[z0:z1], 'r-', linewidth=2, label='Reconstructed')
    
    for i in range(z0, z1):
        if i % spec_patch_size == 0: ax_zoom.axvline(x=i, color='gray', linestyle='--', alpha=0.15)

    ax_zoom.set_title(f"Zoom (Pixel {peak_idx}) | Patch: {spec_patch_size}px", fontsize=10)
    ax_zoom.legend()

    plt.tight_layout()
    plt.savefig("reconstruction_dashboard.png", dpi=150)
    print("[SUCCESS] Saved: reconstruction_dashboard.png")
    plt.show()

if __name__ == "__main__":
    main()