import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms

from astropt.model import GPT, GPTConfig
from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow

DATA_DIR = "/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow"
CHECKPOINT_PATH = "./logs/astropt_100M_250K_arrow_T1_20251214/ckpt.pt"
OUTPUT_DIR = "./logs/astropt_100M_250K_arrow_T1_20251214/"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot spectra reconstructions from AstroPT (Arrow Version)")
    
    parser.add_argument(
        "--target_ids", 
        nargs="+", 
        type=int, 
        help="Target ID's to plot (e.g: --target_ids 39633...)"
    )
    
    parser.add_argument(
        "--num_random", 
        type=int, 
        default=5, 
        help="Random number of spectra to plot"
    )
    
    parser.add_argument(
        "--wl_range",
        nargs=2,
        type=int,
        default=None,
        help="X range to plot MIN MAX (e.g: --wl_range 0 500)"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "val"],
        help="Dataset split to use (train/test)"
    )

    return parser.parse_args()

def load_model():
    print(f"--- Loading model from {CHECKPOINT_PATH} ---")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    config_dict = checkpoint.get("config", checkpoint.get("model_args"))
    
    if config_dict is None:
        raise ValueError("model cofnfiguration not found")

    config = GPTConfig(**config_dict)
    
    # Creating the model
    model = GPT(config, checkpoint["modality_registry"])
    
    state_dict = checkpoint["model"]
    
    # Cleaning prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "").replace("module.", "")
        new_state_dict[new_key] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    
    return model, checkpoint

def main():
    args = parse_arguments()
    
    # Load model
    model, checkpoint = load_model()

    # Load dataset
    print(f"--- Loading {args.split.upper()} Dataset from Arrow ---")
    
    # Transformations
    try:
        transforms_dict = EuclidDESIDatasetArrow.data_transforms()
    except AttributeError:
        print("Advertencia: Usando transformaciones fallback (no encontradas en clase).")
        def normalise(x):
            x_32 = x.float()
            std, mean = torch.std_mean(x_32, dim=1, keepdim=True)
            x_norm = (x_32 - mean) / (std + 1e-8)
            return x_norm.to(x.dtype)
        transforms_dict = {
            "images": transforms.Compose([transforms.Lambda(normalise)]),
            "spectra": transforms.Compose([transforms.Lambda(normalise)])
        }

    full_ds = EuclidDESIDatasetArrow(
        arrow_folder_root=DATA_DIR,
        split=args.split,
        modality_registry=checkpoint["modality_registry"],
        spiral=checkpoint.get("config", {}).get('spiral', True), 
        stochastic=False, 
        transform=transforms_dict
    )

    # Selection logic
    ds_to_plot = None
    shuffle_loader = False
    
    if args.target_ids:
        print(f"--- Searching {len(args.target_ids)} objsects ---")
        indices_found = []
        
        all_ids = np.array(full_ds.ds['targetid'])
        id_map = {tid: idx for idx, tid in enumerate(all_ids)}
        
        for tid in args.target_ids:
            if tid in id_map:
                idx = id_map[tid]
                indices_found.append(idx)
                print(f"ID dound {tid} at the index {idx}")
            else:
                print(f"ID not found {tid} in the split {args.split}.")
        
        if len(indices_found) == 0:
            print("ID's don't exist in the catalog.")
            return

        ds_to_plot = Subset(full_ds, indices_found)
        batch_size_run = len(indices_found)
        shuffle_loader = False 
        
    else:
        print(f"--- Selecting {len(args.target_ids)} objsects ---")
        indices = np.random.choice(len(full_ds), size=args.num_random, replace=False)
        ds_to_plot = Subset(full_ds, indices)
        batch_size_run = args.num_random
        shuffle_loader = False 

    # Dataloader
    vdl = DataLoader(ds_to_plot, batch_size=batch_size_run, shuffle=shuffle_loader, num_workers=2, pin_memory=True)
    
    print("--- Generating Plots ---")
    try:
        batch = next(iter(vdl))
    except StopIteration:
        print("Dataloader is empty.")
        return
    
    B = EuclidDESIDatasetArrow.process_modes(batch, checkpoint["modality_registry"], DEVICE)
    
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            P, _ = model(B["X"], targets=None) 

        if "spectra" not in B["Y"]:
            print("Not found spectra in the batch")
            return

        Y_spectra = B["Y"]["spectra"].float().cpu().numpy() # Ground Truth
        P_spectra = P["spectra"].float().cpu().numpy()      # Prediction
        
        target_ids = batch['targetid'].numpy()
        indices = batch['idx'].numpy()
        
    # Plotting Loop
    for i in range(len(target_ids)):
        tid = target_ids[i]
        idx = indices[i]
        
        try:
            record = full_ds.ds[int(idx)]
            tz = record.get('z', record.get('Z', -1.0))
        except Exception:
            tz = -1.0
        
        y_seq = np.concatenate(Y_spectra[i])
        p_seq = np.concatenate(P_spectra[i])
        
        plt.figure(figsize=(20, 10)) # Ajustado tamaño
        
        plt.plot(y_seq, label='Real (DESI)', color='black', alpha=0.5, linewidth=1)
        plt.plot(p_seq, label='Predicted (AstroPT)', color='red', linewidth=1.2, alpha=0.8)
        
        # Title
        title_str = f"Spectra Simulation\nTARGETID: {tid} | Z: {tz:.4f}"
        if args.wl_range:
            title_str += f" | Zoom: {args.wl_range[0]}-{args.wl_range[1]}"
        
        plt.title(title_str, fontsize=14)
        plt.xlabel("Tokens (Wavelength index)", fontsize=12)
        plt.ylabel("Flux (Normalized)", fontsize=12)
        
        # Applying Zoom
        if args.wl_range:
            plt.xlim(left=args.wl_range[0], right=args.wl_range[1])
            
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        suffix = f"_zoom_{args.wl_range[0]}-{args.wl_range[1]}" if args.wl_range else ""
        save_path = os.path.join(OUTPUT_DIR, f"spec_{tid}{suffix}.png")
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    print(f"Finished! Plots saved in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()