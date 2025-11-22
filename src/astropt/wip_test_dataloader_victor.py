import torch
from torch.utils.data import DataLoader
from astropt.model import ModalityConfig, ModalityRegistry
from astropt.wip_euclid_desi_dataloader_victor import EuclidDESIDataset 

print("--- Iniciando prueba de integración del Data Loader ---")

# Local Paths
METADATA_PATH = "/home/valonso/iac18_aasensio_shared/euclid_q1_desi_dr1/base_EuclidQ1_DESIDR1.fits"
VIS_FOLDER = "/home/valonso/iac18_aasensio_shared/euclid_q1_desi_dr1/VIS"
NISP_FOLDER = "/home/valonso/iac18_aasensio_shared/euclid_q1_desi_dr1/NISP"
# Need to add desi as not harcoded paths

# Modality registry
# From train_multimodal.py
n_chan = 4 # VIS + NISP-H + NISP-J + NISP-Y
modality_registry = ModalityRegistry(
    [
        ModalityConfig(
            name="images",
            input_size=16 * 16 * n_chan,
            patch_size=16,
            loss_weight=1.0,
            embed_pos=True,
            pos_input_size=1,
        ),
        ModalityConfig(
            name="spectra",
            input_size=256,
            patch_size=256,
            pos_input_size=256,
            loss_weight=0.5,
            embed_pos=False,
        ),
    ]
)

# Creating our dataset
print("Initializing EuclidDESIDataset (version for AstroPT)...")
dataset = EuclidDESIDataset(
    metadata_path=METADATA_PATH, 
    vis_folder=VIS_FOLDER, 
    nisp_folder=NISP_FOLDER,
    modality_registry=modality_registry,
    spiral=True
)

# Testing __getitem__ with element 8
for idx in [8,2]:
    print(f"\n--- Testing __getitem__[{idx}] ---")
    try:
        sample = dataset[idx] # Trying a well known element
        print("Output format of __getitem__:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor's shape {value.shape} and type {value.dtype}")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"ERROR while testing __getitem__: {e}")

    print("\n--- Simulating a batch ---")
    
try:

    indices_locales = [8,2]

    dataset_local = torch.utils.data.Subset(dataset, indices_locales)
    
    dataloader = DataLoader(dataset_local, batch_size=2, num_workers=0)
    
    batch = next(iter(dataloader))
    print(f"Batch data loaded: {indices_locales}")
    
    processed_batch = EuclidDESIDataset.process_modes(
        batch, modality_registry, device='cpu'
    )
    
    print("process_modes worked.")
    print("keys for X (model input):", processed_batch['X'].keys())
    print("keys for Y (model output):", processed_batch['Y'].keys())
    print("Shape of X['images']:", processed_batch['X']['images'].shape)
    print("Shape of Y['images']:", processed_batch['Y']['images'].shape)

except Exception as e:
    print(f"ERROR testing process_modes: {e}")
