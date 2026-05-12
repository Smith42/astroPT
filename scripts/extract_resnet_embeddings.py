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
import einops
from torchvision.models import resnet18

sys.path.append(str(Path(__file__).resolve().parent.parent))

from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow
from astropt.model import ModalityRegistry, ModalityConfig

# Configure logging
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("ResNetExtr")

def unpatchify(patch_image, patch_size=8, c=4):
    num_patches = patch_image.shape[1]
    grid_size = int(np.sqrt(num_patches))
    return einops.rearrange(
        patch_image,
        'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
        h=grid_size, w=grid_size, p1=patch_size, p2=patch_size, c=c
    )

class ResNet18Astro(nn.Module):
    def __init__(self, in_channels=4, output_dim=1):
        super().__init__()
        self.resnet = resnet18(weights=None)
        
        # Modify conv1 for 4 channels
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels, 
            original_conv1.out_channels, 
            kernel_size=original_conv1.kernel_size, 
            stride=original_conv1.stride, 
            padding=original_conv1.padding, 
            bias=original_conv1.bias is not None
        )
        
        # Modify final fc layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)
        
    def forward(self, x):
        return self.resnet(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the trained ResNet directory (e.g., logs/resnet18_images_supervised/Z)")
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
    images_npy_path = save_path / "images.npy"
    ids_npy_path = save_path / "ids.npy"
    
    # We load the dataset
    data_dir = config_dict["data_dir"]
    channels = config_dict["images_channels"]
    patch_size = config_dict["images_patch_size"]
    
    registry = ModalityRegistry([ModalityConfig(
        name="images", 
        input_size=patch_size**2 * channels,
        pos_input_size=2,
        patch_size=patch_size,
        embed_pos=False,
    )])

    # Same transforms as validation
    tf_kwargs = {
        'norm_type_img': config_dict.get('images_norm_type', 'asinh'),
        'norm_scaler_img': config_dict.get('images_norm_scaler', 1.0),
        'norm_const_img': config_dict.get('images_norm_const', 1.0)
    }
    val_tf = EuclidDESIDatasetArrow.data_transforms(stage='val', **tf_kwargs)

    logger.info("Loading test dataset...")
    base_ds = EuclidDESIDatasetArrow(
        arrow_folder_root=data_dir,
        split="test",
        modality_registry=registry,
        spiral=config_dict.get('spiral', False),
        transform=val_tf
    )

    loader = DataLoader(
        base_ds, 
        batch_size=256, 
        shuffle=False, 
        num_workers=4, 
        drop_last=False
    )

    # Initialize Model
    logger.info("Initializing ResNet18Astro backbone...")
    model = ResNet18Astro(in_channels=channels, output_dim=1)
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint, strict=False)
    
    # Swap out FC layer with Identity to emit the 512-dim embedding
    model.resnet.fc = nn.Identity()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_samples = len(base_ds)
    emb_dim = 512
    
    # We use temporary memmaps for extraction
    images_tmp_path = save_path / "images.tmp"
    ids_tmp_path = save_path / "ids.tmp"
    
    images_tmp = np.memmap(images_tmp_path, dtype='float32', mode='w+', shape=(total_samples, emb_dim))
    ids_tmp = np.memmap(ids_tmp_path, dtype='int64', mode='w+', shape=(total_samples,))

    ptr = 0
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc="Extracting"):
            B = batch['images'].shape[0]
            # Unpatchify into 2D image
            imgs_2d = unpatchify(batch['images'], patch_size=patch_size, c=channels)
            # Ensure float32 to match model weights
            imgs_2d = imgs_2d.to(device=device, dtype=torch.float32)
            
            emb = model(imgs_2d)
            
            images_tmp[ptr:ptr+B] = emb.cpu().numpy()
            ids_tmp[ptr:ptr+B] = batch['targetid'].numpy()
            
            ptr += B
            
    images_tmp.flush()
    ids_tmp.flush()
    
    # Convert temporary raw memmaps to proper .npy files with headers
    logger.info("Converting raw buffers to proper .npy files...")
    
    # Load raw data and save as npy
    final_images = np.array(images_tmp)
    np.save(images_npy_path, final_images)
    
    final_ids = np.array(ids_tmp)
    np.save(ids_npy_path, final_ids)

    # Cleanup
    del images_tmp
    del ids_tmp
    gc.collect()
    
    if images_tmp_path.exists(): images_tmp_path.unlink()
    if ids_tmp_path.exists(): ids_tmp_path.unlink()

    logger.info(f"Extraction for {model_dir.name} completed successfully at {save_path.resolve()}")

if __name__ == "__main__":
    main()