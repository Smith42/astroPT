import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Importamos tus módulos (asegúrate de estar en la raíz del proyecto al ejecutar)
from astropt.model import GPT, GPTConfig, ModalityRegistry, ModalityConfig
from astropt.wip_euclid_desi_dataloader_victor import EuclidDESIDataset

# --- CONFIGURACIÓN ---
# Ajusta estas rutas a donde tengas tus datos EN LOCAL
BASE_DIR_LOCAL = "/Users/victor/TFM/data"  # <-- ¡CAMBIA ESTO!
CHECKPOINT_PATH = "logs/astropt0100M_multimodal/ckpt.pt" # <-- ¡Y ESTO!

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu" # Usa 'mps' para Mac M1/M2/M3
print(f"Usando dispositivo: {DEVICE}")

def load_model_from_checkpoint(ckpt_path, device):
    print(f"Cargando checkpoint desde {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Recuperar argumentos y configuración
    model_args = checkpoint["model_args"]
    config = checkpoint["config"] # Diccionario de configuración original
    modality_registry = checkpoint["modality_registry"]
    
    # Crear una instancia limpia del modelo
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, modality_registry)
    
    # Cargar los pesos (limpiando el prefijo '_orig_mod.' si existe por torch.compile)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval() # ¡Importante! Modo evaluación (apaga dropout)
    return model, modality_registry, model_args

def extract_embeddings():
    # 1. Cargar Modelo
    model, modality_registry, model_args = load_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)
    
    # 2. Preparar Dataset (Usamos la configuración multimodal)
    # Reconstruimos las transformaciones necesarias (normalización)
    from torchvision import transforms
    def normalise(x):
        std, mean = torch.std_mean(x, dim=1, keepdim=True)
        x_norm = (x - mean) / (std + 1e-8)
        return x_norm.to(torch.float32) # float32 para MPS/CPU
        
    tf = transforms.Compose([transforms.Lambda(normalise)])
    transforms_dict = {"image": tf}

    print("Cargando Dataset...")
    # Asegúrate de que apuntas al catálogo correcto en tu Mac
    ds = EuclidDESIDataset(
        metadata_path=os.path.join(BASE_DIR_LOCAL, "base_EuclidQ1_DESIDR1_CLEAN.fits"),
        vis_folder=os.path.join(BASE_DIR_LOCAL, "VIS"),
        nisp_folder=os.path.join(BASE_DIR_LOCAL, "NISP"),
        transform=transforms_dict,
        modality_registry=modality_registry,
        spiral=model_args.get('spiral', True) # Recuperamos si era spiral del config
    )
    
    # Cogemos un subconjunto si quieres ir rápido (ej. los primeros 1000)
    # ds = torch.utils.data.Subset(ds, range(1000)) 

    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0) # num_workers=0 para evitar líos en Mac local

    # 3. Bucle de Extracción
    print("Extrayendo embeddings...")
    
    all_embeddings = {mod: [] for mod in modality_registry.names()}
    all_targets = [] # Guardamos TARGETID u otra info útil
    
    with torch.no_grad():
        for batch in tqdm(dl):
            # Extraer TARGETIDs para luego pintar (asumiendo que el dataloader devuelve indices)
            # Nota: Tu dataloader devuelve 'idx', tendríamos que mirar el catálogo para saber el ID real
            # Por ahora guardamos el índice
            indices = batch['idx']
            
            # Procesar batch
            B = EuclidDESIDataset.process_modes(batch, modality_registry, DEVICE)
            
            # Llamada mágica: get_embeddings
            # Esto devuelve un diccionario {modality: [batch, seq_len, n_embd]}
            emb_dict = model.get_embeddings(B["X"], draw_from_centre=True)
            
            # Procesar cada modalidad
            for mod_name, emb_tensor in emb_dict.items():
                # emb_tensor es [Batch, Tokens, 768]
                # Hacemos MEAN POOLING para tener un solo vector por galaxia
                # (Promediamos todos los parches de la imagen/espectro)
                pooled_emb = emb_tensor.mean(dim=1).cpu().numpy()
                all_embeddings[mod_name].append(pooled_emb)
            
            all_targets.extend(indices.numpy())

    # 4. Guardar resultados
    print("Guardando resultados...")
    final_arrays = {}
    for mod_name in all_embeddings:
        final_arrays[mod_name] = np.concatenate(all_embeddings[mod_name], axis=0)
    
    final_arrays['indices'] = np.array(all_targets)
    
    save_path = "embeddings_dump.npz"
    np.savez(save_path, **final_arrays)
    print(f"¡Hecho! Guardado en {save_path}")
    print("Dimensiones:")
    for k, v in final_arrays.items():
        print(f"  {k}: {v.shape}")

if __name__ == "__main__":
    extract_embeddings()