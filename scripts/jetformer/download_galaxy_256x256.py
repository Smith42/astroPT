# make_galaxy_pt_shards.py
# pip install datasets pillow torch

import os, math
import torch
from datasets import load_dataset
from PIL import Image

OUTDIR = "galaxy_pt_256x256_test"   # folder for shards
TOTAL  = 10_000        # how many samples
SHARD  = 1024          # images per shard (last shard may be smaller)
SEED   = 42
BUFFER = 50_000        # shuffle buffer

IMG_SIZE = 256  # match your model

def pil_to_chw_uint8(img: Image.Image):
    img = img.convert("RGB")
    t = torch.tensor(bytearray(img.tobytes()), dtype=torch.uint8)
    t = t.view(IMG_SIZE, IMG_SIZE, 3).permute(2,0,1).contiguous()  # CHW
    return t

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    ds = load_dataset("Smith42/galaxies", split="test", streaming=True)
    ds = ds.shuffle(buffer_size=BUFFER, seed=SEED)

    buf = []
    saved = 0
    shard_idx = 0

    for ex in ds:
        # strictly use image_crop
        if "image_crop" not in ex or ex["image_crop"] is None:
            continue
        try:
            t = pil_to_chw_uint8(ex["image_crop"])
        except Exception:
            continue

        buf.append(t)
        saved += 1

        if len(buf) == SHARD:
            x = torch.stack(buf, dim=0)  # [N,3,256,256] uint8
            torch.save({"images": x}, os.path.join(OUTDIR, f"shard_{shard_idx:04d}_test.pt"))
            print(f"wrote shard {shard_idx:04d} with {x.size(0)} samples")
            shard_idx += 1
            buf.clear()

        if saved >= TOTAL:
            break

    # flush remainder
    if buf:
        x = torch.stack(buf, dim=0)
        torch.save({"images": x}, os.path.join(OUTDIR, f"shard_{shard_idx:04d}_test.pt"))
        print(f"wrote shard {shard_idx:04d} with {x.size(0)} samples")

    print(f"Done. Total saved: {saved} samples into {OUTDIR}/shard_*_test.pt")

if __name__ == "__main__":
    main()
