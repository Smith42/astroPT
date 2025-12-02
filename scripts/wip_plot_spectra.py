import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms

from astropt.model import GPT, GPTConfig
from astropt.wip_euclid_desi_dataloader_victor_40K import EuclidDESIDataset

# --- CONFIGURACIÓN ---
BASE_DIR = "/home/valonso/iac18_aasensio_shared/euclid_q1_desi_dr1"
METADATA_PATH = os.path.join(BASE_DIR, "base_EuclidQ1_DESIDR1_TRAIN.fits")
SPECTRA_FOLDER = os.path.join(BASE_DIR, "desi_dr1_training_spectra")
VIS_FOLDER = os.path.join(BASE_DIR, "VIS")
NISP_FOLDER = os.path.join(BASE_DIR, "NISP")

CHECKPOINT_PATH = "./logs/astropt0100M_multimodal_40K_T3/ckpt.pt"
OUTPUT_DIR = "./logs/astropt0100M_multimodal_40K_T3"
DEVICE = "cuda"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot spectra reconstructions from AstroPT")
    
    # Argumento para IDs específicos
    parser.add_argument(
        "--target_ids", 
        nargs="+", 
        type=int, 
        help="Lista de TARGETIDs específicos para plotear (ej: --target_ids 39633... 39644...)"
    )
    
    # Argumento para cantidad de randoms
    parser.add_argument(
        "--num_random", 
        type=int, 
        default=5, 
        help="Número de espectros aleatorios a plotear si no se especifican IDs (Default: 5)"
    )
    
    # --- NUEVO: Argumento de Rango (Zoom) ---
    parser.add_argument(
        "--wl_range",
        nargs=2,
        type=int,
        default=None,
        help="Rango del eje X (índice espectral) para hacer zoom. Uso: --wl_range MIN MAX (ej: --wl_range 0 500)"
    )

    return parser.parse_args()

def normalise(x):
    std, mean = torch.std_mean(x, dim=1, keepdim=True)
    x_norm = (x - mean) / (std + 1e-8)
    return x_norm.to(torch.float16)

def load_model():
    print(f"--- Loading model from {CHECKPOINT_PATH} ---")
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
    return model, checkpoint

def main():
    args = parse_arguments()
    
    # 1. Cargar Modelo
    model, checkpoint = load_model()

    # 2. Cargar Dataset Completo
    print("--- Loading Full Dataset ---")
    transform = transforms.Compose([transforms.Lambda(normalise)])
    
    full_ds = EuclidDESIDataset(
        metadata_path=METADATA_PATH,
        vis_folder=VIS_FOLDER,
        nisp_folder=NISP_FOLDER,
        spectra_folder=SPECTRA_FOLDER,
        spectra_dirs={"dummy": "active"}, 
        transform={"images": transform},
        modality_registry=checkpoint["modality_registry"],
        spiral=checkpoint["model_args"].get('spiral', True)
    )

    # 3. Lógica de Selección
    ds_to_plot = None
    shuffle_loader = False
    
    if args.target_ids:
        print(f"--- MODO ESPECÍFICO: Buscando {len(args.target_ids)} objetos ---")
        indices_found = []
        
        # Optimizamos la búsqueda creando un diccionario temporal ID -> Índice
        # Esto es mucho más rápido que buscar en la tabla 40k veces
        print("Indexando catálogo...")
        id_map = {tid: idx for idx, tid in enumerate(full_ds.meta['TARGETID'])}
        
        for tid in args.target_ids:
            if tid in id_map:
                idx = id_map[tid]
                indices_found.append(idx)
                print(f"✅ Encontrado ID {tid} en índice {idx}")
            else:
                print(f"❌ NO encontrado ID {tid} en el catálogo.")
        
        if len(indices_found) == 0:
            print("Error: Ninguno de los IDs solicitados existe en el catálogo.")
            return

        ds_to_plot = Subset(full_ds, indices_found)
        batch_size_run = len(indices_found)
        shuffle_loader = False 
        
    else:
        print(f"--- MODO ALEATORIO: Seleccionando {args.num_random} muestras del Validation Set ---")
        generator = torch.Generator().manual_seed(61)
        total_size = len(full_ds)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size
        
        _, vds, _ = random_split(full_ds, [train_size, val_size, test_size], generator=generator)
        
        ds_to_plot = vds
        batch_size_run = args.num_random
        shuffle_loader = True

    # 4. Dataloader y Generación
    vdl = DataLoader(ds_to_plot, batch_size=batch_size_run, shuffle=shuffle_loader, num_workers=4, pin_memory=True)
    
    print("--- Generating Plots ---")
    batch = next(iter(vdl))
    
    B = EuclidDESIDataset.process_modes(batch, checkpoint["modality_registry"], DEVICE)
    
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        P, _ = model(B["X"], B["Y"])
        
        Y_spectra = B["Y"]["spectra"].float().cpu().numpy()
        P_spectra = P["spectra"].float().cpu().numpy()
        
        target_ids = batch['targetid'].numpy()
        indices = batch['idx'].numpy()
        
    for i in range(len(target_ids)):
        tid = target_ids[i]
        idx = indices[i]
        
        try:
            tz = full_ds.meta[idx]['Z']
        except Exception:
            tz = -1.0
        
        y_seq = np.concatenate(Y_spectra[i])
        p_seq = np.concatenate(P_spectra[i])
        
        plt.figure(figsize=(30, 15))
        
        plt.plot(y_seq, label='Real (DESI)', color='black', alpha=0.5, linewidth=0.75)
        plt.plot(p_seq, label='Predicted (AstroPT)', color='red', linewidth=1)
        
        # Título con info extra si hay zoom
        title_str = f"Spectra simulation - TARGETID: {tid} - Z: {tz:.4f}"
        if args.wl_range:
            title_str += f" (Zoom: {args.wl_range[0]}-{args.wl_range[1]})"
        
        plt.title(title_str)
        plt.xlabel("Tokens (Wavelength index)")
        plt.ylabel("Flux (Normalized)")
        
        # --- APLICAR ZOOM SI SE PIDE ---
        if args.wl_range:
            plt.xlim(left=args.wl_range[0], right=args.wl_range[1])
            
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Añadimos sufijo al nombre si es zoom para no sobrescribir
        suffix = f"_zoom_{args.wl_range[0]}-{args.wl_range[1]}" if args.wl_range else ""
        save_path = os.path.join(OUTPUT_DIR, f"spec_{tid}{suffix}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    print(f"Finished! Plots saved in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()