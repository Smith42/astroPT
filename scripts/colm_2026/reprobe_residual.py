"""
reprobe_residual.py  --  Probe R² and direction cosines for the residual experiment.

Computes for ar_aim_{001M,021M,100M} across all 64 checkpoints x all layers:
  - R²(L_r)        absolute luminosity
  - R²(mass)       stellar mass
  - R²(eps_M|L)    mass residual at fixed luminosity
  - R²(ssfr)       sSFR
  - cos(L, M)      direction cosines at last layer
  - cos(L, eps)
  - cos(M, eps)
  - probe coef_ vectors saved for downstream use

Uploads to: results_residual/<config>/results.parquet
            probe_weights/<config>/probes.npz

USAGE:
    python reprobe_residual.py
    python reprobe_residual.py --configs ar_aim_021M  # single config
"""

import argparse, gc, io, os, shutil
import numpy as np
import torch
import einops
import polars as pl
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from huggingface_hub import HfApi
from tqdm import tqdm

# ---------------------------------------------------------------------------
RESULTS_REPO     = os.environ.get("COLM_RESULTS_REPO", "HCVYM5w6Gn/colm-results")
HF_TOKEN         = os.environ.get("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
DATASET          = "Smith42/galaxies"
DATASET_REVISION = "v2.0"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
CTX              = torch.autocast(DEVICE, dtype=torch.bfloat16)

PATCH_SIZE   = 16
N_CHAN       = 3
BLOCK_SIZE   = 1024
PROBE_BATCH  = 256
RIDGE_LAMBDA = 100.0
N_GALAXIES   = 50_000
K_FOLDS      = 5

TARGET_CONFIGS = ["ar_aim_001M", "ar_aim_021M", "ar_aim_100M"]

SIZES = {
    "001M": dict(n_layer=4,  n_head=8,  n_embd=128),
    "021M": dict(n_layer=6,  n_head=8,  n_embd=512),
    "100M": dict(n_layer=12, n_head=12, n_embd=768),
}

api = HfApi(token=HF_TOKEN)

# ---------------------------------------------------------------------------
def _get(path):
    local = api.hf_hub_download(repo_id=RESULTS_REPO, filename=path,
                                repo_type="dataset", token=HF_TOKEN)
    with open(local, "rb") as f: return f.read()

def _ls(prefix):
    return [f for f in api.list_repo_files(RESULTS_REPO, repo_type="dataset")
            if f.startswith(prefix)]

def _put(path, data, msg="update"):
    api.upload_file(path_or_fileobj=io.BytesIO(data), path_in_repo=path,
                    repo_id=RESULTS_REPO, repo_type="dataset",
                    commit_message=msg)

# ---------------------------------------------------------------------------
def _normalise(x):
    x = torch.from_numpy(x).float()
    std, mean = torch.std_mean(x, dim=1, keepdim=True)
    return ((x - mean) / (std + 1e-8)).half()

def _patchify(img_pil, spiral_fn):
    g = np.array(img_pil).swapaxes(0, 2)
    g = einops.rearrange(g, "c (h p1) (w p2) -> (h w) (p1 p2 c)",
                         p1=PATCH_SIZE, p2=PATCH_SIZE)
    return spiral_fn(_normalise(g))

def _spiral_fn():
    from astropt.local_datasets import GalaxyImageDataset
    from astropt.model import ModalityConfig, ModalityRegistry
    mr = ModalityRegistry([ModalityConfig(
        name="images", input_size=PATCH_SIZE * PATCH_SIZE * N_CHAN,
        patch_size=PATCH_SIZE, embed_pos=True, pos_input_size=1)])
    return GalaxyImageDataset(paths=None, spiral=True, modality_registry=mr).spiralise

def fit_mass_luminosity_on_train(n_train=2000):
    """Fit log M* = a*L_r + b on training split. Returns fitted LinearRegression."""
    from datasets import load_dataset
    ds = load_dataset(DATASET, revision=DATASET_REVISION,
                      split="train", streaming=True)
    ds = ds.select_columns(["mag_abs_r_photoz", "mass_med_photoz"])
    Ls, Ms = [], []
    for ex in ds:
        try:
            L = float(ex.get("mag_abs_r_photoz", "nan"))
            M = float(ex.get("mass_med_photoz",  "nan"))
            if np.isfinite(L) and np.isfinite(M) and L > -90 and M > -90:
                Ls.append(-0.4 * L); Ms.append(M)
        except: pass
        if len(Ls) >= n_train: break
    Ls = np.array(Ls).reshape(-1, 1); Ms = np.array(Ms)
    reg = LinearRegression().fit(Ls, Ms)
    print(f"Train-split fit (n={len(Ms)}): slope={reg.coef_[0]:.3f}, "
          f"intercept={reg.intercept_:.3f}")
    return reg

def load_test_set(n, spiral_fn, reg):
    """Stream n test galaxies. Returns patches + label dict including eps_M|L."""
    from datasets import load_dataset
    ds = load_dataset(DATASET, revision=DATASET_REVISION,
                      split="test", streaming=True)
    ds = ds.select_columns(["image", "mag_abs_r_photoz",
                             "mass_med_photoz", "ssfr_med_photoz",
                             "redshift", "photo_z"])
    patches, L_r, mass, ssfr, redshift = [], [], [], [], []
    pbar = tqdm(total=n, desc="test set", ncols=70)
    for ex in ds:
        try: p = _patchify(ex["image"], spiral_fn)
        except: continue
        patches.append(p)
        def _f(v):
            try:
                v = float(v)
                return v if np.isfinite(v) and v > -90 else np.nan
            except: return np.nan
        L_r.append(-0.4 * _f(ex.get("mag_abs_r_photoz")))
        mass.append(_f(ex.get("mass_med_photoz")))
        ssfr.append(_f(ex.get("ssfr_med_photoz")))
        _z = _f(ex.get("redshift"))
        if not np.isfinite(_z): _z = _f(ex.get("photo_z"))
        redshift.append(_z)
        pbar.update(1)
        if len(patches) >= n: break
    pbar.close()

    patches = torch.stack(patches)
    Lr = np.array(L_r, dtype=np.float32)
    M  = np.array(mass, dtype=np.float32)
    S  = np.array(ssfr, dtype=np.float32)

    # residual: eps = M - (a*L_r + b)
    valid = np.isfinite(Lr) & np.isfinite(M)
    M_pred = np.full(len(M), np.nan, dtype=np.float32)
    M_pred[valid] = reg.predict(Lr[valid].reshape(-1,1)).astype(np.float32)
    eps = M - M_pred

    Z = np.array(redshift, dtype=np.float32)
    labels = {"L_r": Lr, "mass": M, "ssfr": S, "eps_M_given_L": eps, "z": Z}
    print(f"Loaded {len(patches):,} galaxies")
    for k, v in labels.items():
        fin = np.isfinite(v).sum()
        print(f"  {k}: {fin:,} valid, mean={np.nanmean(v):.3f}, std={np.nanstd(v):.3f}")
    return patches, labels

# ---------------------------------------------------------------------------
def build_model(cfg_name):
    obj, tok, size = cfg_name.split("_")
    from astropt.model import GPT, GPTConfig, ModalityConfig, ModalityRegistry
    torch.serialization.add_safe_globals([ModalityConfig])
    mr = ModalityRegistry([ModalityConfig(
        name="images", input_size=PATCH_SIZE * PATCH_SIZE * N_CHAN,
        patch_size=PATCH_SIZE, embed_pos=True, pos_input_size=1)])
    dims = SIZES[size]
    gptconf = GPTConfig(
        block_size=BLOCK_SIZE, dropout=0.0,
        attn_type=("full" if obj == "mae" else "causal"),
        mae_mask_ratio=0.5, norm_pix_loss=False,
        tokeniser=tok, **dims)
    model = GPT(gptconf, mr, master_process=True).to(DEVICE)
    model.eval()
    return model

@torch.no_grad()
def extract_layers(model, patches):
    """(N, L+1, d) fp32 cpu."""
    mod = model.modality_registry.names()[0]
    enc, emb = model.encoders[mod], model.embedders[mod]
    out = []
    for s in range(0, patches.shape[0], PROBE_BATCH):
        chunk = patches[s:s+PROBE_BATCH].to(DEVICE).float()
        pos = torch.arange(chunk.shape[1], device=DEVICE).unsqueeze(0).expand(chunk.shape[0], -1)
        with CTX:
            x = model.transformer.drop(enc(chunk) + emb(pos))
            layers = [x.mean(1)]
            for blk in model.transformer.h:
                x = blk(x)
                layers.append(x.mean(1))
        out.append(torch.stack(layers, 1).float().cpu())
        del chunk, x, layers
    torch.cuda.empty_cache()
    return torch.cat(out, 0)

def kfold_r2_and_coef(X, y, lam=RIDGE_LAMBDA, k=K_FOLDS):
    """Returns (mean_r2, std_r2, coef_unit_norm)."""
    mask = np.isfinite(y)
    if mask.sum() < 100:
        return np.nan, np.nan, None
    Xv, yv = X[mask], y[mask]
    N = len(yv)
    idx = np.random.default_rng(42).permutation(N)
    fold = N // k
    r2s = []
    for ki in range(k):
        te = idx[ki*fold:(ki+1)*fold]
        tr = np.concatenate([idx[:ki*fold], idx[(ki+1)*fold:]])
        mu = Xv[tr].mean(0); sd = Xv[tr].std(0) + 1e-6
        Xtr = (Xv[tr] - mu) / sd; Xte = (Xv[te] - mu) / sd
        ytr = yv[tr]; yte = yv[te]
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        w = np.linalg.solve(A, Xtr.T @ (ytr - ytr.mean()))
        pred = Xte @ w + ytr.mean()
        ss = ((yte - yte.mean())**2).sum() + 1e-12
        r2s.append(float(1 - ((yte - pred)**2).sum() / ss))
    # fit full probe for coef direction
    sc = StandardScaler(); Xs = sc.fit_transform(Xv)
    p = Ridge(alpha=lam).fit(Xs, yv)
    coef = p.coef_ / (np.linalg.norm(p.coef_) + 1e-12)
    return float(np.mean(r2s)), float(np.std(r2s)), coef

def cosine(a, b):
    if a is None or b is None: return np.nan
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

# ---------------------------------------------------------------------------
def reprobe_one(cfg_name, patches, labels, out_dir):
    ckpt_files = _ls(f"checkpoints/{cfg_name}/step_")
    steps = sorted(int(f.rsplit("step_",1)[-1][:-3]) for f in ckpt_files)
    if not steps:
        print(f"[{cfg_name}] no checkpoints, skipping"); return
    print(f"\n[{cfg_name}] {len(steps)} checkpoints on device={DEVICE}")

    model = build_model(cfg_name)
    rows = []
    all_coefs = {}   # {step: {layer: {target: coef}}}

    for step in tqdm(steps, desc=cfg_name, ncols=70, unit="ckpt"):
        raw = _get(f"checkpoints/{cfg_name}/step_{step:08d}.pt")
        from astropt.model import ModalityConfig
        torch.serialization.add_safe_globals([ModalityConfig])
        try:
            ck = torch.load(io.BytesIO(raw), map_location=DEVICE, weights_only=True)
        except Exception:
            ck = torch.load(io.BytesIO(raw), map_location=DEVICE, weights_only=False)
        model.load_state_dict(
            {k: v.float() for k,v in ck["model"].items()
             if not k.startswith(("encoders.","decoders.","mask_token"))},
            strict=False)
        model.eval()

        hidden = extract_layers(model, patches)  # (N, L+1, d)
        n_layers = hidden.shape[1]
        step_coefs = {}

        for li in range(n_layers):
            X = hidden[:, li, :].numpy()
            layer_coefs = {}
            r2s, stds, coefs = {}, {}, {}
            for tgt, y in labels.items():
                r2, std, coef = kfold_r2_and_coef(X, y)
                r2s[tgt] = r2; stds[tgt] = std; coefs[tgt] = coef
                layer_coefs[tgt] = coef

            # direction cosines at this layer
            cos_LM  = cosine(coefs["L_r"], coefs["mass"])
            cos_Le  = cosine(coefs["L_r"], coefs["eps_M_given_L"])
            cos_Me  = cosine(coefs["mass"], coefs["eps_M_given_L"])
            cos_sSFR_M  = cosine(coefs["ssfr"], coefs["mass"])
            cos_z_M     = cosine(coefs.get("z"), coefs["mass"])
            cos_z_L     = cosine(coefs.get("z"), coefs["L_r"])
            cos_z_sSFR  = cosine(coefs.get("z"), coefs["ssfr"])

            step_coefs[li] = layer_coefs
            rows.append({
                "config": cfg_name,
                "step": step, "layer": li,
                "r2_L":    r2s["L_r"],
                "r2_M":    r2s["mass"],
                "r2_eps":  r2s["eps_M_given_L"],
                "r2_ssfr": r2s["ssfr"],
                "std_L":   stds["L_r"],
                "std_M":   stds["mass"],
                "std_eps": stds["eps_M_given_L"],
                "cos_LM":  cos_LM,
                "cos_Le":  cos_Le,
                "cos_Me":  cos_Me,
                "cos_sSFR_M": cos_sSFR_M,
                "cos_z_M":    cos_z_M,
                "cos_z_L":    cos_z_L,
                "cos_z_sSFR": cos_z_sSFR,
                "r2_z":       r2s.get("z", float("nan")),
            })

        all_coefs[step] = step_coefs
        del hidden, ck, raw
        gc.collect(); torch.cuda.empty_cache()

    del model; gc.collect(); torch.cuda.empty_cache()

    # Save parquet locally + upload
    os.makedirs(out_dir, exist_ok=True)
    local = os.path.join(out_dir, f"{cfg_name}_residual_probe.parquet")
    df = pl.DataFrame(rows)
    df.write_parquet(local)
    print(f"[{cfg_name}] saved locally -> {local}")

    buf = io.BytesIO(); df.write_parquet(buf)
    _put(f"results_residual/{cfg_name}/results.parquet", buf.getvalue(),
         msg=f"residual probe {cfg_name}")
    print(f"[{cfg_name}] uploaded -> results_residual/{cfg_name}/results.parquet")

    # Save probe weights (coef_ vectors) as npz
    npz_data = {}
    for step, layer_dict in all_coefs.items():
        for li, tgt_dict in layer_dict.items():
            for tgt, coef in tgt_dict.items():
                if coef is not None:
                    npz_data[f"step{step:06d}_layer{li}_{tgt}"] = coef
    npz_buf = io.BytesIO()
    np.savez_compressed(npz_buf, **npz_data)
    _put(f"probe_weights/{cfg_name}/probes.npz", npz_buf.getvalue(),
         msg=f"probe weights {cfg_name}")
    print(f"[{cfg_name}] uploaded probe weights")

# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=TARGET_CONFIGS)
    parser.add_argument("--n_galaxies", type=int, default=N_GALAXIES)
    parser.add_argument("--out_dir",
                        default="/work/nvme/bfir/ssourav/astroPT-runs/residual_probe")
    args = parser.parse_args()

    print(f"Residual probe | device={DEVICE} | configs={args.configs}")
    print("\nFitting mass~L_r on train split...")
    reg = fit_mass_luminosity_on_train(n_train=2000)

    print("\nLoading test set...")
    spiral_fn = _spiral_fn()
    patches, labels = load_test_set(args.n_galaxies, spiral_fn, reg)

    for cfg in args.configs:
        try:
            reprobe_one(cfg, patches, labels, args.out_dir)
        except Exception as e:
            import traceback
            print(f"[{cfg}] FAILED: {e}\n{traceback.format_exc()}")

    print("\nAll done.")

if __name__ == "__main__":
    main()
