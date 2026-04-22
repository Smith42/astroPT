"""
AstroCLIP embedding extractor compatible with AstroPT downstream analysis scripts.

Outputs in embedding folder:
- ids.npy
- targetid.npy
- images.npy
- spectra.npy
- joint.npy
- embeddings_all.npz
- experiment_config.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Tuple

import einops
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import CenterCrop

# Local path setup
REPO_ROOT = Path(__file__).resolve().parent.parent
ASTROCLIP_ROOT = REPO_ROOT.parent / "AstroCLIP"

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.append(str(REPO_ROOT / "src"))
if str(ASTROCLIP_ROOT) not in sys.path:
    sys.path.append(str(ASTROCLIP_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent))

from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow
from astropt.model import ModalityConfig, ModalityRegistry

from astroclip.models.astroclip import ImageHead, SpectrumHead
import train_astroclip_arrow as train_mod

logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("AstroCLIP-Extractor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract AstroCLIP embeddings for AstroPT analysis")
    parser.add_argument("--weights_dir", type=str, required=True, help="Training weights directory")
    parser.add_argument("--data_dir", type=str, required=True, help="Arrow root data directory")
    parser.add_argument("--save_dir", type=str, required=True, help="Output embeddings root directory")
    parser.add_argument("--ckpt_name", type=str, default="last.ckpt", help="Checkpoint filename")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda", help="cuda|cpu")
    parser.add_argument("--exp_tag", type=str, default="", help="Optional output folder suffix")
    return parser.parse_args()


def resolve_checkpoint(weights_dir: Path, ckpt_name: str) -> Path:
    ckpt_dir = weights_dir / "astroclip_checkpoints"
    candidates = []

    explicit = ckpt_dir / ckpt_name
    if explicit.exists():
        return explicit

    if (ckpt_dir / "last.ckpt").exists():
        return ckpt_dir / "last.ckpt"

    candidates.extend(sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True))
    candidates.extend(sorted(weights_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True))

    if len(candidates) == 0:
        raise FileNotFoundError(f"No .ckpt found in {ckpt_dir} or {weights_dir}")

    return candidates[0]


def load_training_config(weights_dir: Path) -> Dict:
    cfg_path = weights_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            return json.load(f)
    logger.warning("config.json not found in weights directory. Using defaults where needed.")
    return {}


def build_registry(images_size: int, images_patch_size: int, images_channels: int, spectra_size: int, spectra_patch_size: int):
    _ = images_size, spectra_size
    img_input_size = images_patch_size * images_patch_size * images_channels
    modalities = [
        ModalityConfig(
            name="images",
            input_size=img_input_size,
            pos_input_size=1,
            patch_size=images_patch_size,
            embed_pos=True,
        ),
        ModalityConfig(
            name="spectra",
            input_size=spectra_patch_size,
            pos_input_size=1,
            patch_size=spectra_patch_size,
            embed_pos=True,
        ),
    ]
    return ModalityRegistry(modalities)


class CollateForAstroCLIP:
    def __init__(self, images_size: int, images_patch_size: int, images_channels: int, center_crop: int):
        self.images_size = images_size
        self.images_patch_size = images_patch_size
        self.images_channels = images_channels
        self.center_crop = CenterCrop(center_crop)

    def __call__(self, batch):
        images = []
        spectra = []
        targetids = []

        grid = self.images_size // self.images_patch_size
        p1 = self.images_patch_size
        p2 = self.images_patch_size

        valid_items = [item for item in batch if "images" in item and "spectra" in item]
        if len(valid_items) == 0:
            raise RuntimeError("Batch without paired image+spectrum samples")

        for item in valid_items:
            patch_img = item["images"]
            orig_img = einops.rearrange(
                patch_img,
                "(h w) (p1 p2 c) -> c (h p1) (w p2)",
                h=grid,
                w=grid,
                p1=p1,
                p2=p2,
                c=self.images_channels,
            )
            img_3ch = self.center_crop(orig_img[:3, :, :])
            images.append(img_3ch)

            patch_spec = item["spectra"]
            spec = patch_spec.flatten().unsqueeze(-1)
            spectra.append(spec)

            targetids.append(int(item["targetid"]))

        return {
            "image": torch.stack(images).to(torch.float32),
            "spectrum": torch.stack(spectra).to(torch.float32),
            "targetid": torch.tensor(targetids, dtype=torch.int64),
        }


def build_model(config_dict: Dict, train_dir: Path) -> train_mod.AstroClipTrainModule:
    root = os.environ.get("ASTROCLIP_ROOT", str(ASTROCLIP_ROOT))

    image_encoder = ImageHead(
        config=os.path.join(root, "astroclip", "astrodino", "config.yaml"),
        model_weights=os.path.join(root, "pretrained", "astrodino.ckpt"),
        save_directory=str(train_dir / "astrodino"),
        freeze_backbone=bool(config_dict.get("freeze_image_backbone", False)),
    )

    spectrum_encoder = SpectrumHead(
        model_path=os.path.join(root, "pretrained", "specformer.ckpt"),
        freeze_backbone=bool(config_dict.get("freeze_spectrum_backbone", True)),
    )

    model = train_mod.AstroClipTrainModule(
        image_encoder=image_encoder,
        spectrum_encoder=spectrum_encoder,
        temperature=15.5,
        lr=float(config_dict.get("learning_rate", 1e-4)),
        weight_decay=float(config_dict.get("weight_decay", 0.05)),
        epochs=int(config_dict.get("max_epochs", 100)),
        eta_min=float(config_dict.get("lr_min", 5e-7)),
        beta1=float(config_dict.get("beta1", 0.9)),
        beta2=float(config_dict.get("beta2", 0.95)),
        lr_warmup_steps=int(config_dict.get("lr_warmup_steps", 1000)),
        lr_decay_steps=int(config_dict.get("lr_decay_steps", 10000)),
    )

    return model


def make_output_dir(save_root: Path, ckpt_path: Path, exp_tag: str) -> Path:
    tag = exp_tag.strip()
    folder = f"{ckpt_path.stem}"
    if tag:
        folder = f"{folder}_{tag}"
    out = save_root / folder
    out.mkdir(parents=True, exist_ok=True)
    return out


def main():
    args = parse_args()

    weights_dir = Path(args.weights_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    save_root = Path(args.save_dir).resolve()
    save_root.mkdir(parents=True, exist_ok=True)

    cfg = load_training_config(weights_dir)

    images_size = int(cfg.get("images_size", 224))
    images_patch_size = int(cfg.get("images_patch_size", 8))
    images_channels = int(cfg.get("images_channels", 4))
    center_crop = int(cfg.get("center_crop", 144))

    spectra_size = int(cfg.get("spectra_size", 7781))
    spectra_patch_size = int(cfg.get("spectra_patch_size", 10))

    registry = build_registry(
        images_size=images_size,
        images_patch_size=images_patch_size,
        images_channels=images_channels,
        spectra_size=spectra_size,
        spectra_patch_size=spectra_patch_size,
    )

    tf = EuclidDESIDatasetArrow.data_transforms(
        norm_type_img="asinh",
        norm_scaler_img=1.0,
        norm_const_img=7.603847,
        norm_type_spec="asinh",
        norm_scaler_spec=1.0,
        norm_const_spec=7.956048,
        stage="val",
    )

    ds = EuclidDESIDatasetArrow(
        arrow_folder_root=str(data_dir),
        split="test",
        modality_registry=registry,
        spiral=False,
        stochastic=False,
        transform=tf,
    )

    collate = CollateForAstroCLIP(
        images_size=images_size,
        images_patch_size=images_patch_size,
        images_channels=images_channels,
        center_crop=center_crop,
    )

    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=collate,
        drop_last=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = build_model(cfg, train_dir=weights_dir.parent)
    ckpt_path = resolve_checkpoint(weights_dir, args.ckpt_name)
    logger.info(f"Loading checkpoint: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = state.get("state_dict", state)
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Checkpoint load summary: {msg}")

    model.to(device)
    model.eval()

    save_dir = make_output_dir(save_root, ckpt_path, args.exp_tag)
    total = len(ds)

    try:
        probe = next(iter(dl))
        with torch.no_grad():
            probe_img = probe["image"][:1].to(device)
            probe_emb = model.image_encoder(probe_img)
            if probe_emb.dim() == 1:
                emb_dim = int(probe_emb.shape[0])
            else:
                emb_dim = int(probe_emb.shape[-1])
    except Exception:
        emb_dim = int(cfg.get("embed_dim", 1024))
        logger.warning(f"Could not infer embedding size from probe batch. Falling back to {emb_dim}.")

    ids_m = np.lib.format.open_memmap(save_dir / "ids.npy", mode="w+", dtype="int64", shape=(total,))
    targetid_m = np.lib.format.open_memmap(save_dir / "targetid.npy", mode="w+", dtype="int64", shape=(total,))
    img_m = np.lib.format.open_memmap(save_dir / "images.npy", mode="w+", dtype="float32", shape=(total, emb_dim))
    spec_m = np.lib.format.open_memmap(save_dir / "spectra.npy", mode="w+", dtype="float32", shape=(total, emb_dim))
    joint_m = np.lib.format.open_memmap(save_dir / "joint.npy", mode="w+", dtype="float32", shape=(total, emb_dim))

    if device.type == "cuda":
        ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
    else:
        ctx = nullcontext()

    start = 0
    with torch.no_grad(), ctx:
        for batch in tqdm(dl, desc="Extracting AstroCLIP embeddings"):
            img = batch["image"].to(device, non_blocking=True)
            sp = batch["spectrum"].to(device, non_blocking=True)
            ids = batch["targetid"].detach().cpu().numpy().astype("int64")

            emb_i = model.image_encoder(img)
            emb_s = model.spectrum_encoder(sp)

            if emb_i.dim() == 1:
                emb_i = emb_i.unsqueeze(0)
            if emb_s.dim() == 1:
                emb_s = emb_s.unsqueeze(0)

            emb_i_np = emb_i.float().cpu().numpy()
            emb_s_np = emb_s.float().cpu().numpy()
            emb_j_np = 0.5 * (emb_i_np + emb_s_np)

            bs = emb_i_np.shape[0]
            end = start + bs

            ids_m[start:end] = ids
            targetid_m[start:end] = ids
            img_m[start:end] = emb_i_np
            spec_m[start:end] = emb_s_np
            joint_m[start:end] = emb_j_np

            start = end

    for mmap_obj in [ids_m, targetid_m, img_m, spec_m, joint_m]:
        mmap_obj.flush()

    del ids_m, targetid_m, img_m, spec_m, joint_m
    gc.collect()

    npz_path = save_dir / "embeddings_all.npz"
    np.savez_compressed(
        npz_path,
        targetid=np.load(save_dir / "targetid.npy", mmap_mode="r"),
        images=np.load(save_dir / "images.npy", mmap_mode="r"),
        spectra=np.load(save_dir / "spectra.npy", mmap_mode="r"),
        joint=np.load(save_dir / "joint.npy", mmap_mode="r"),
    )

    with open(save_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(ckpt_path),
                "weights_dir": str(weights_dir),
                "data_dir": str(data_dir),
                "exp_tag": args.exp_tag,
            },
            f,
            indent=2,
        )

    logger.info(f"Embeddings exported successfully to: {save_dir}")


if __name__ == "__main__":
    main()
