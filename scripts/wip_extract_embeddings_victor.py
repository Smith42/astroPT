import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torchvision import transforms
from astropt.model import GPT, GPTConfig
from astropt.wip_euclid_desi_dataloader_victor_40K import EuclidDESIDataset

# Updated paths - using the full dataset path sctructure
BASE_DIR = "/home/valonso/iac18_aasensio_shared/euclid_q1_desi_dr1"
METADATA_PATH = os.path.join(BASE_DIR, "base_EuclidQ1_DESIDR1_TRAIN.fits")
SPECTRA_FOLDER = os.path.join(BASE_DIR, "desi_dr1_training_spectra")

VIS_FOLDER = os.path.join(BASE_DIR, "VIS")
NISP_FOLDER = os.path.join(BASE_DIR, "NISP")
#VIS_FOLDER = os.path.join(BASE_IMG_DIR, "VIS")
#NISP_FOLDER = {
#    'H': os.path.join(BASE_IMG_DIR, "NIR-H"),
#    'J': os.path.join(BASE_IMG_DIR, "NIR-J"),
#    'Y': os.path.join(BASE_IMG_DIR, "NIR-Y"),
#}


CHECKPOINT_PATH = "./logs/astropt0100M_multimodal_40K_T2/ckpt.pt"
OUTPUT_FILE = "./logs/astropt0100M_multimodal_40K_T2/astropt0100M_multimodal_40K_T2.npz" 
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
    
    # Loading the model
    print(f"Loading model from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    config = GPTConfig(**checkpoint["model_args"])
    model = GPT(config, checkpoint["modality_registry"])
    
    # Cleaning names al weights
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # Creating a dataset
    print("Setting up the Dataloader...")
    transform = transforms.Compose([transforms.Lambda(normalise)])
    
    ds = EuclidDESIDataset(
        metadata_path=METADATA_PATH,
        vis_folder=VIS_FOLDER,
        #nisp_folders=NISP_FOLDERS,
        nisp_folder=NISP_FOLDER,
        spectra_folder=SPECTRA_FOLDER,
        spectra_dirs={"main": "dummy"},
        transform={"images": transform},
        modality_registry=checkpoint["modality_registry"],
        spiral=checkpoint["model_args"].get('spiral', True)
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
    all_embeddings = {mod: [] for mod in checkpoint["modality_registry"].names()}
    all_ids = []

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in tqdm(dl):
            if batch is None: continue
            
            index = batch['targetid']
            
            B_batch = EuclidDESIDataset.process_modes(batch, checkpoint["modality_registry"], DEVICE)
            
            emb_dict = model.get_embeddings(B_batch["X"], draw_from_centre=True)
            
            for mod, tensor in emb_dict.items():
                pooled = tensor.mean(dim=1).float().cpu().numpy()
                all_embeddings[mod].append(pooled)
            
            all_ids.extend(index.numpy())

    # Saving
    print("Saving data...")
    
    final_arrays = {k: np.concatenate(v, axis=0) for k, v in all_embeddings.items()}
    final_arrays['targetid'] = np.array(all_ids)
    
    np.savez(OUTPUT_FILE, **final_arrays)
    
    print(f"Finished without errors!")
    print(f"Embedings save into {OUTPUT_FILE}!!")

if __name__ == "__main__":
    extract()