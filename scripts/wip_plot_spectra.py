import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from astropt.model import GPT, GPTConfig
from astropt.wip_euclid_desi_dataloader_victor import EuclidDESIDataset

BASE_DIR = "/home/valonso/iac18_aasensio_shared/euclid_q1_desi_dr1"
BASE_IMG_DIR = "/home/valonso/iac18_aasensio_shared/euclid_dr1"

METADATA_PATH = os.path.join(BASE_DIR, "base_EuclidQ1_DESIDR1_HYBRID.fits")
VIS_FOLDER = os.path.join(BASE_IMG_DIR, "VIS")
NISP_FOLDERS = {
    'H': os.path.join(BASE_IMG_DIR, "NIR-H"),
    'J': os.path.join(BASE_IMG_DIR, "NIR-J"),
    'Y': os.path.join(BASE_IMG_DIR, "NIR-Y"),
}

SPECTRA_FOLDER = "/home/valonso/iac18_aasensio_shared/euclid_q1_desi_dr1/desi_dr1_training_spectra"

CHECKPOINT_PATH = "./logs/astropt0100M_multimodal_17K/ckpt.pt"
OUTPUT_DIR = "./logs/astropt0100M_multimodal_17K"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda"

def normalise(x):
    std, mean = torch.std_mean(x, dim=1, keepdim=True)
    x_norm = (x - mean) / (std + 1e-8)
    return x_norm.to(torch.float16)

def main():
    print("--- Loading model ---")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    config = GPTConfig(**checkpoint["model_args"])
    model = GPT(config, checkpoint["modality_registry"])
    
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    print("--- Loading data ---")
    transform = transforms.Compose([transforms.Lambda(normalise)])
    
    full_ds = EuclidDESIDataset(
        metadata_path=METADATA_PATH,
        vis_folder=VIS_FOLDER,
        nisp_folders=NISP_FOLDERS,
        spectra_folder=SPECTRA_FOLDER,
        spectra_dirs={"dummy": "active"}, 
        transform={"images": transform},
        modality_registry=checkpoint["modality_registry"],
        spiral=checkpoint["model_args"].get('spiral', True)
    )
    

    generator = torch.Generator().manual_seed(61)
    total_size = len(full_ds)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    _, vds, _ = random_split(full_ds, [train_size, val_size, test_size], generator=generator)
    
    vdl = DataLoader(vds, batch_size=5, shuffle=True, num_workers=8, pin_memory=True)
    
    print("--- Creting plots ---")
    batch = next(iter(vdl))
    
    B = EuclidDESIDataset.process_modes(batch, checkpoint["modality_registry"], DEVICE)
    
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        P, _ = model(B["X"], B["Y"])
        
        Y_spectra = B["Y"]["spectra"].float().cpu().numpy()
        P_spectra = P["spectra"].float().cpu().numpy()
        
        target_ids = batch['targetid'].numpy()

    for i in range(len(target_ids)):
        tid = target_ids[i]
        y_seq = np.concatenate(Y_spectra[i])
        p_seq = np.concatenate(P_spectra[i])
        
        plt.figure(figsize=(30, 6))
        
        plt.plot(y_seq, label='Real (DESI)', color='black', alpha=0.5, linewidth=0.5)
        
        plt.plot(p_seq, label='Predicted (AstroPT)', color='red', linewidth=1)
        
        plt.title(f"Spectra simulation - TARGETID: {tid}")
        plt.xlabel("Tokens (Wavelength index)")
        plt.xlim(left=0,right=2500)
        plt.ylabel("Flux (Normalized)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(OUTPUT_DIR, f"spec_{tid}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight') # Alta resolución
        plt.close()
        print(f"Saved: {save_path}")

    print(f"Finished! Saved in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()