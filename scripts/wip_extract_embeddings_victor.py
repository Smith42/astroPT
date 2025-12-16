import os
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torchvision import transforms
from dataclasses import fields

from astropt.model import GPT, GPTConfig, ModalityConfig, ModalityRegistry
from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow

# Updated paths - using the full dataset path sctructure
DATA_DIR = "/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"

CHECKPOINT_PATH = "./logs/astropt_100M_250K_arrow_T1_20251214/ckpt.pt"
OUTPUT_FILE = "./logs/astropt_100M_250K_arrow_T1_20251214/astropt_100M_250K_arrow_T1_20251214.npz" 
DEVICE = "cuda"

# Auxiliar function to normalize
def normalise(x):
    """This function computes the mean value of all pixels vector of one image"""
    
    std, mean = torch.std_mean(x, dim=1, keepdim=True)
    x_norm = (x - mean) / (std + 1e-8)
    return x_norm.to(torch.float16)

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    return torch.utils.data.dataloader.default_collate(batch)

# Main function
def extract():
    print(f"#--- Initializing embedding extraction ---#")
    print(f"Device: {DEVICE}")
    
    print(f"Loading model from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    print(f"Ckeckpoint keys:\n{list(checkpoint.keys())}")
    
    if 'config' in checkpoint:
        raw_args = checkpoint['config']
        print("Found configuration under key 'config'.")
    elif 'model_args' in checkpoint:
        raw_args = checkpoint['model_args']
        print("Found configuration under key 'model_args'.")
    else:
        raise KeyError("Could not find configuration in checkpoint! Keys available: " + str(checkpoint.keys()))
    
    
    try:
        valid_fields = {f.name for f in fields(GPTConfig)}
        clean_args = {k: v for k, v in raw_args.items() if k in valid_fields}
        print(f"Config arguments filtered: {len(raw_args)} -> {len(clean_args)} keys.")
    except Exception as e:
        print(f"Warning: Could not filter via dataclasses ({e}). Using raw args (might fail).")
        clean_args = raw_args

    config = GPTConfig(**clean_args)
    
    imgages_patch_size = raw_args.get('images_patch_size', raw_args.get('patch_size', 16))
    images_channels = raw_args.get('images_channels', raw_args.get('images_channels', 4))
    images_loss_weight = raw_args.get('images_loss_weight', raw_args.get('images_loss_weight', 1))
    img_input_size = imgages_patch_size * imgages_patch_size * images_channels

    spectra_patch_size = raw_args.get('spectra_patch_size', raw_args.get('spectra_patch_size', 10))
    spectra_loss_weight = raw_args.get('spectra_loss_weight', raw_args.get('spectra_loss_weight', 1))

    modalities = [
        ModalityConfig(
            name="images",
            input_size=img_input_size,
            patch_size=imgages_patch_size,
            pos_input_size=1,           
            loss_weight=images_loss_weight,          
            embed_pos=True,
        ),
        ModalityConfig(
            name="spectra",
            input_size=spectra_patch_size,
            patch_size=spectra_patch_size,
            pos_input_size=spectra_patch_size, 
            loss_weight=spectra_loss_weight,
            embed_pos=False,
        ),
    ]
    
    modality_registry = ModalityRegistry(modalities)
    print("ModalityRegistry created.")

    model = GPT(config, modality_registry)

    state_dict = checkpoint["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "").replace("_orig_mod.", "")
        new_state_dict[clean_k] = v
            
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Model loaded. Status: {msg}")
    model.to(DEVICE)
    model.eval()

    # Creating a dataset
    print("Setting up the Dataloader...")
    
    transform_dict = {
        "images": transforms.Compose([
            transforms.Lambda(normalise)
        ]),
        "spectra": transforms.Compose([
            transforms.Lambda(normalise)
        ])
    }
    
    ds = EuclidDESIDatasetArrow(
        arrow_folder_root=DATA_DIR,
        split="test",
        modality_registry=modality_registry,
        spiral=False,
        stochastic=False,
        transform=transform_dict
    )
    
    print(f"Hybrid Dataset loaded with {len(ds)} samples.")
    
    dl = DataLoader(
        ds, 
        batch_size=32, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True,
    )
    
    # Extraction
    print(f"Processing {len(ds)} galaxies...")
    modalities = ['images', 'spectra']
    all_embeddings = {mod: [] for mod in modalities}
    all_ids = []

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in tqdm(dl):
            if batch is None: 
                continue
            
            X = {}
            for mod in modalities:
                if mod in batch:
                    X[mod] = batch[mod].to(DEVICE)
                    pos_key = f"{mod}_positions"
                    if pos_key in batch:
                        X[pos_key] = batch[pos_key].to(DEVICE)
            
            emb_dict = model.get_embeddings(X, draw_from_centre=True)
            
            for mod, tensor in emb_dict.items():
                if tensor.ndim == 3:
                    pooled = tensor.mean(dim=1).float().cpu().numpy()
                else:
                    pooled = tensor.float().cpu().numpy()
                    
                all_embeddings[mod].append(pooled)
            
            if 'targetid' in batch:
                all_ids.extend(batch['targetid'].numpy())

    print("Saving data...")
    final_arrays = {k: np.concatenate(v, axis=0) for k, v in all_embeddings.items() if len(v) > 0}
    if all_ids:
        final_arrays['targetid'] = np.array(all_ids)
    
    np.savez(OUTPUT_FILE, **final_arrays)
    print(f"Embeddings saved to {OUTPUT_FILE}!")

if __name__ == "__main__":
    extract()