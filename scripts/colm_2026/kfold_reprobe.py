"""
kfold_reprobe.py  --  k-fold linear probe on already-uploaded AstroPT checkpoints.

Replaces colm.py's single 80/20 random split with K-fold cross-validation
on the test split, per Mike's recommendation. Runs read-only against HF
during probing; one final parquet upload per config at the end.

USAGE (from astroPT repo root):
    python kfold_reprobe.py                          # reprobe all 11 done configs
    python kfold_reprobe.py --configs ar_aim_021M    # single config
    python kfold_reprobe.py --k 5 --n_galaxies 50000
"""

import argparse, gc, io, os, sys
import numpy as np
import torch
import einops
import polars as pl
from huggingface_hub import HfApi
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants (must match colm.py / train.py exactly)
# ---------------------------------------------------------------------------
RESULTS_REPO     = os.environ.get("COLM_RESULTS_REPO", "HCVYM5w6Gn/colm-results")
HF_TOKEN         = os.environ.get("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
DATASET          = "Smith42/galaxies"
DATASET_REVISION = "v2.0"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
CTX              = torch.autocast(DEVICE, dtype=torch.bfloat16)

PATCH_SIZE  = 16
N_CHAN      = 3
BLOCK_SIZE  = 1024
PROBE_BATCH = 256
RIDGE_LAMBDA = 100.0

PROBE_TARGETS = [
    "redshift", "photo_z", "mass_med_photoz", "ssfr_med_photoz",
    "mag_g_desi", "mag_r_desi", "mag_z_desi",
    "mag_abs_g_photoz", "mag_abs_r_photoz", "mag_abs_z_photoz",
    "est_petro_th50", "est_petro_th50_kpc", "galaxy_size",
    "smooth-or-featured_smooth_fraction", "smooth-or-featured_featured-or-disk_fraction",
    "smooth-or-featured_artifact_fraction",
    "merging_none_fraction", "merging_minor-disturbance_fraction",
    "merging_major-disturbance_fraction", "merging_merger_fraction",
    "how-rounded_round_fraction", "how-rounded_in-between_fraction",
    "how-rounded_cigar-shaped_fraction",
]

ALL_CONFIGS = [
    f"{obj}_{tok}_{size}"
    for obj  in ("ar", "mae")
    for tok  in ("aim", "affine")
    for size in ("001M", "021M", "100M")
]

SIZES = {
    "001M": dict(n_layer=4,  n_head=8,  n_embd=128),
    "021M": dict(n_layer=6,  n_head=8,  n_embd=512),
    "100M": dict(n_layer=12, n_head=12, n_embd=768),
}

api = HfApi(token=HF_TOKEN)

# ---------------------------------------------------------------------------
# HF helpers (read-only during probing)
# ---------------------------------------------------------------------------
def _ls(prefix):
    return [f for f in api.list_repo_files(RESULTS_REPO, repo_type="dataset")
            if f.startswith(prefix)]

def _get(path):
    local = api.hf_hub_download(
        repo_id=RESULTS_REPO, filename=path,
        repo_type="dataset", token=HF_TOKEN,
    )
    with open(local, "rb") as f:
        return f.read()

def _put(path, data, msg="update"):
    api.upload_file(
        path_or_fileobj=io.BytesIO(data), path_in_repo=path,
        repo_id=RESULTS_REPO, repo_type="dataset", commit_message=msg,
    )

# ---------------------------------------------------------------------------
# Image preprocessing (identical to colm.py's _patchify + make_stream)
# ---------------------------------------------------------------------------
def _normalise(x):
    x = torch.from_numpy(x).to(torch.float32)
    std, mean = torch.std_mean(x, dim=1, keepdim=True)
    return ((x - mean) / (std + 1e-8)).to(torch.float16)

def _patchify(img_pil, spiral_fn):
    g = np.array(img_pil).swapaxes(0, 2)   # HWC -> CHW, then C(HP1)(WP2)
    g = einops.rearrange(g, "c (h p1) (w p2) -> (h w) (p1 p2 c)",
                         p1=PATCH_SIZE, p2=PATCH_SIZE)
    g = _normalise(g)
    return spiral_fn(g) if spiral_fn is not None else g

def _spiral_fn():
    from astropt.local_datasets import GalaxyImageDataset
    from astropt.model import ModalityConfig, ModalityRegistry
    mr = ModalityRegistry([ModalityConfig(
        name="images", input_size=PATCH_SIZE * PATCH_SIZE * N_CHAN,
        patch_size=PATCH_SIZE, embed_pos=True, pos_input_size=1,
    )])
    gid = GalaxyImageDataset(paths=None, spiral=True, modality_registry=mr)
    return gid.spiralise

def load_test_set(n_galaxies, spiral_fn):
    """Stream n_galaxies from the test split. Returns (patches cpu fp16, label tensors)."""
    from datasets import load_dataset
    cols = ["image"] + PROBE_TARGETS
    ds = load_dataset(DATASET, revision=DATASET_REVISION, split="test", streaming=True)
    ds = ds.select_columns(cols)

    patches, labels = [], {k: [] for k in PROBE_TARGETS}
    pbar = tqdm(total=n_galaxies, desc="loading test set", unit="gal", ascii=True)
    for i, ex in enumerate(ds):
        try:
            p = _patchify(ex["image"], spiral_fn)
        except Exception:
            continue
        patches.append(p)
        for k in PROBE_TARGETS:
            v = ex.get(k)
            try:
                v = float(v)
                labels[k].append(v if np.isfinite(v) and v > -90 else np.nan)
            except Exception:
                labels[k].append(np.nan)
        pbar.update(1)
        if len(patches) >= n_galaxies:
            break
    pbar.close()

    if not patches:
        raise RuntimeError("No patches loaded — check dataset streaming / image format")

    patches = torch.stack(patches)   # (N, n_patches, P²·C) fp16
    labels  = {k: torch.tensor(v, dtype=torch.float32) for k, v in labels.items()}
    print(f"Test set: {patches.shape}, {len(labels)} label columns")
    return patches, labels

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(cfg):
    from astropt.model import GPT, GPTConfig, ModalityConfig, ModalityRegistry
    torch.serialization.add_safe_globals([ModalityConfig])
    mr = ModalityRegistry([ModalityConfig(
        name="images", input_size=PATCH_SIZE * PATCH_SIZE * N_CHAN,
        patch_size=PATCH_SIZE, embed_pos=True, pos_input_size=1,
    )])
    dims = SIZES[cfg["size"]]
    gptconf = GPTConfig(
        block_size=BLOCK_SIZE, dropout=0.0,
        attn_type=("full" if cfg["objective"] == "mae" else "causal"),
        mae_mask_ratio=0.5, norm_pix_loss=False,
        tokeniser=cfg["tokeniser"],
        **dims,
    )
    model = GPT(gptconf, mr, master_process=True).to(DEVICE)
    model.eval()
    return model

@torch.no_grad()
def layerwise_embeddings(model, eval_patches):
    """Mean-pooled residual stream at every depth. Returns (N, L+1, n_embd) fp16 cpu."""
    mod = model.modality_registry.names()[0]
    enc, emb_pos = model.encoders[mod], model.embedders[mod]
    out = []
    for s in range(0, eval_patches.shape[0], PROBE_BATCH):
        chunk = eval_patches[s:s + PROBE_BATCH].to(DEVICE).float()
        pos = torch.arange(chunk.shape[1], device=DEVICE).unsqueeze(0).expand(chunk.shape[0], -1)
        with CTX:
            x = model.transformer.drop(enc(chunk) + emb_pos(pos))
            layers = [x.float().mean(dim=1)]
            for blk in model.transformer.h:
                x = blk(x)
                layers.append(x.float().mean(dim=1))
        out.append(torch.stack(layers, dim=1).half().cpu())
        del chunk, x, layers
    torch.cuda.empty_cache()
    return torch.cat(out, dim=0)

# ---------------------------------------------------------------------------
# K-fold ridge probe
# ---------------------------------------------------------------------------
def kfold_ridge_r2(emb, y, lam=RIDGE_LAMBDA, k=5):
    """K-fold cross-validated ridge R² on GPU.
    Returns: mean_r2 (float), std_r2 (float), per_fold_r2s (list[float])
    """
    N, D = emb.shape
    idx = torch.randperm(N, device=emb.device)
    fold_size = N // k
    r2s = []
    for fold in range(k):
        te_idx = idx[fold * fold_size: (fold + 1) * fold_size]
        tr_idx = torch.cat([idx[:fold * fold_size], idx[(fold + 1) * fold_size:]])
        Xtr, Xte = emb[tr_idx], emb[te_idx]
        ytr, yte = y[tr_idx],   y[te_idx]
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        A = Xtr.T @ Xtr + lam * torch.eye(D, device=emb.device)
        w = torch.linalg.solve(A, Xtr.T @ (ytr - ytr.mean()))
        pred = Xte @ w + ytr.mean()
        ss_tot = ((yte - yte.mean()) ** 2).sum() + 1e-12
        r2s.append((1 - ((yte - pred) ** 2).sum() / ss_tot).item())
    return float(np.mean(r2s)), float(np.std(r2s)), r2s

# ---------------------------------------------------------------------------
# Per-config reprobe
# ---------------------------------------------------------------------------
def reprobe_config(cfg_name, eval_patches, eval_labels, k, out_dir):
    obj, tok, size = cfg_name.split("_")
    cfg = dict(name=cfg_name, objective=obj, tokeniser=tok, size=size)

    ckpt_files = _ls(f"checkpoints/{cfg_name}/step_")
    steps = sorted(int(f.rsplit("step_", 1)[-1][:-3]) for f in ckpt_files)
    if not steps:
        print(f"[{cfg_name}] no checkpoints on HF, skipping")
        return
    print(f"[{cfg_name}] {len(steps)} checkpoints, k={k}")

    rows = []
    for step in tqdm(steps, desc=cfg_name, unit="ckpt", ascii=True):
        raw = _get(f"checkpoints/{cfg_name}/step_{step:08d}.pt")
        from astropt.model import ModalityConfig
        torch.serialization.add_safe_globals([ModalityConfig])
        try:
            ck = torch.load(io.BytesIO(raw), map_location=DEVICE, weights_only=True)
        except Exception:
            ck = torch.load(io.BytesIO(raw), map_location=DEVICE, weights_only=False)

        model = build_model(cfg)
        model.load_state_dict({kk: v.float() for kk, v in ck["model"].items() if not kk.startswith(("encoders.", "decoders.", "mask_token"))}, strict=False)
        emb = layerwise_embeddings(model, eval_patches)   # (N, L+1, D) fp16 cpu
        del model, ck, raw
        gc.collect(); torch.cuda.empty_cache()

        n_layers = emb.shape[1]
        for layer in range(n_layers):
            Xl = emb[:, layer, :].to(DEVICE).float()
            for param in PROBE_TARGETS:
                y  = eval_labels[param]
                m  = torch.isfinite(y)
                if m.sum() < 200:
                    continue
                Xv, yv = Xl[m.to(DEVICE)], y[m].to(DEVICE)
                mean_r2, std_r2, fold_r2s = kfold_ridge_r2(Xv, yv, lam=RIDGE_LAMBDA, k=k)
                rows.append(dict(
                    config=cfg_name, objective=obj, tokeniser=tok, size=size,
                    step=step, layer=layer, param=param,
                    r2_mean=mean_r2, r2_std=std_r2,
                    r2_folds=fold_r2s,
                    n_galaxies=int(m.sum().item()),
                    k_folds=k, ridge_lambda=RIDGE_LAMBDA,
                ))
                del Xv, yv
            del Xl; torch.cuda.empty_cache()
        del emb; gc.collect(); torch.cuda.empty_cache()

    # save locally
    os.makedirs(out_dir, exist_ok=True)
    local_path = os.path.join(out_dir, f"{cfg_name}_kfold_probe.parquet")
    pl.DataFrame(rows).write_parquet(local_path)
    print(f"[{cfg_name}] saved locally -> {local_path}")

    # upload to HF under results_kfold/ (one commit per config)
    buf = io.BytesIO()
    pl.DataFrame(rows).write_parquet(buf)
    _put(f"results_kfold/{cfg_name}/results.parquet", buf.getvalue(),
         msg=f"kfold probe {cfg_name} k={k}")
    print(f"[{cfg_name}] uploaded -> results_kfold/{cfg_name}/results.parquet")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=ALL_CONFIGS)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n_galaxies", type=int, default=50_000)
    parser.add_argument("--out_dir", default="results/kfold_probe")
    args = parser.parse_args()

    print(f"k-fold reprobe | device={DEVICE} | k={args.k} | "
          f"n_galaxies={args.n_galaxies} | configs={args.configs}")

    print("Building spiral function...")
    spiral_fn = _spiral_fn()

    print("Loading test set...")
    eval_patches, eval_labels = load_test_set(args.n_galaxies, spiral_fn)

    for cfg_name in args.configs:
        if cfg_name not in ALL_CONFIGS:
            print(f"Unknown config {cfg_name}, skipping"); continue
        try:
            reprobe_config(cfg_name, eval_patches, eval_labels, args.k, args.out_dir)
        except Exception as e:
            import traceback
            print(f"[{cfg_name}] FAILED: {e}\n{traceback.format_exc()}")

    print("All done.")

if __name__ == "__main__":
    main()
