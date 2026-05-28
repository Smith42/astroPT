import argparse
import json
import logging
import sys
import gc
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow
from astropt.model import ModalityRegistry, ModalityConfig
from scripts.train_spectra_supervised import SpectraSupervisedBaseline, TrainingConfig

# Configure logging
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("SpectraExtr")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the trained Spectra directory (e.g., logs/spectra_supervised_baseline_Z)")
    parser.add_argument("--output_name", type=str, default="test_set_embeddings")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    weights_dir = model_dir / "weights"
    config_path = weights_dir / "config.json"
    ckpt_path = weights_dir / "ckpt_best.pt"

    if not config_path.exists() or not ckpt_path.exists():
        logger.error(f"Missing config or ckpt in {weights_dir}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Output dir setup
    save_path = model_dir / "embeddings" / args.output_name
    save_path.mkdir(parents=True, exist_ok=True)
    spectra_npy_path = save_path / "spectra_embeddings.npy"
    ids_npy_path = save_path / "ids.npy"
    
    # Dataset config
    data_dir = config_dict["data_dir"]
    
    registry = ModalityRegistry([ModalityConfig(
        name="spectra", 
        input_size=1,
        patch_size=1, 
        pos_input_size=1,
        embed_pos=False,
        encoder_type="discrete"
    )])

    # Same transforms as validation
    tf_kwargs = {
        'norm_type_spec': config_dict.get('spectra_norm_type', 'asinh'),
        'norm_scaler_spec': config_dict.get('spectra_norm_scaler', 1.0),
        'norm_const_spec': config_dict.get('spectra_norm_const', 1.0)
    }
    val_tf = EuclidDESIDatasetArrow.data_transforms(stage='val', **tf_kwargs)

    logger.info("Loading test dataset...")
    base_ds = EuclidDESIDatasetArrow(
        arrow_folder_root=data_dir,
        split="test",
        modality_registry=registry,
        spiral=False,
        transform=val_tf
    )

    loader = DataLoader(
        base_ds, 
        batch_size=256, 
        shuffle=False, 
        num_workers=4, 
        drop_last=False
    )

    # Initialize Model from config parameters
    logger.info("Initializing SpectraSupervisedBaseline backbone...")
    # Filter config dictionary to only include valid TrainingConfig attributes
    valid_keys = TrainingConfig.__dataclass_fields__.keys()
    train_cfg_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
    train_cfg = TrainingConfig(**train_cfg_dict)
    
    # Load checkpoint to detect output_dim
    logger.info(f"Loading weights from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)
    
    # Dynamically detect output_dim from head
    # The head is: Linear -> GELU -> Dropout -> Linear
    # So the last layer is head.3
    if 'head.3.weight' in state_dict:
        output_dim = state_dict['head.3.weight'].shape[0]
        logger.info(f"Detected output_dim={output_dim} from checkpoint.")
    else:
        output_dim = 1
        logger.warning("Could not detect head.3.weight in checkpoint. Defaulting to output_dim=1.")

    model = SpectraSupervisedBaseline(config=train_cfg, output_dim=output_dim)
    model.load_state_dict(state_dict, strict=True)
    
    # Swap out the MLP head with Identity to emit the embedding vector
    # The output will be the Global Average Pooled vector of size embed_dim (e.g. 256)
    model.head = nn.Identity()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_samples = len(base_ds)
    emb_dim = train_cfg.embed_dim # Usually 256
    
    # We use temporary memmaps for extraction
    spectra_tmp_path = save_path / "spectra.tmp"
    ids_tmp_path = save_path / "ids.tmp"
    
    spectra_tmp = np.memmap(spectra_tmp_path, dtype='float32', mode='w+', shape=(total_samples, emb_dim))
    ids_tmp = np.memmap(ids_tmp_path, dtype='int64', mode='w+', shape=(total_samples,))

    ptr = 0
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc="Extracting"):
            B = batch['spectra'].shape[0]
            
            spectra_input = batch['spectra'].to(device=device, dtype=torch.float32)
            
            # Forward pass (head is Identity, so this returns the GAP embedding)
            emb = model(spectra_input)
            
            spectra_tmp[ptr:ptr+B] = emb.cpu().numpy()
            ids_tmp[ptr:ptr+B] = batch['targetid'].numpy()
            
            ptr += B
            
    spectra_tmp.flush()
    ids_tmp.flush()
    
    # Convert temporary raw memmaps to proper .npy files with headers
    logger.info("Converting raw buffers to proper .npy files...")
    
    final_spectra = np.array(spectra_tmp)
    np.save(spectra_npy_path, final_spectra)
    
    final_ids = np.array(ids_tmp)
    np.save(ids_npy_path, final_ids)

    # Cleanup
    del spectra_tmp
    del ids_tmp
    gc.collect()
    
    if spectra_tmp_path.exists(): spectra_tmp_path.unlink()
    if ids_tmp_path.exists(): ids_tmp_path.unlink()

    logger.info(f"Extraction completed successfully. Vectors saved at {save_path.resolve()}")

if __name__ == "__main__":
    main()
