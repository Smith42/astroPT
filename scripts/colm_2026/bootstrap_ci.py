"""
bootstrap_ci.py -- Bootstrap confidence intervals for AstroPT probe R² and cosines.

Implements the three experiments from the spec:
  Exp 1: cheap bootstrap R² over test galaxies (fixed probe)
  Exp 2: bootstrap cosine similarities by retraining probes on bootstrapped training galaxies
  Exp 3: patching CI (bootstrap success rate over matched pairs)

Outputs JSON files to --out_dir, one per config/checkpoint/layer combination.
A companion plot script reads these and adds error bars to the main figures.

USAGE:
    # Final checkpoint, all 3 AR/AIM sizes (main paper figures)
    python bootstrap_ci.py --mode final --n_boot 500

    # Layer-wise, final checkpoint (appendix)
    python bootstrap_ci.py --mode layer --n_boot 200

    # Patching CI for causal figure
    python bootstrap_ci.py --mode patch --causal_json <path>
"""

import argparse, gc, io, json, os, shutil
import numpy as np
import torch
import einops
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
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
RIDGE_ALPHA  = 100.0
N_GALAXIES   = 50_000
TRAIN_FRAC   = 0.8   # 80% train, 20% test within our 50k sample

TARGET_CONFIGS = ["ar_aim_001M", "ar_aim_021M", "ar_aim_100M"]
SIZES = {
    "001M": dict(n_layer=4,  n_head=8,  n_embd=128),
    "021M": dict(n_layer=6,  n_head=8,  n_embd=512),
    "100M": dict(n_layer=12, n_head=12, n_embd=768),
}

api = HfApi(token=HF_TOKEN)

# ---------------------------------------------------------------------------
# HF helpers
# ---------------------------------------------------------------------------
def _get(path):
    local = api.hf_hub_download(repo_id=RESULTS_REPO, filename=path,
                                repo_type="dataset", token=HF_TOKEN)
    with open(local, "rb") as f: return f.read()

def _ls(prefix):
    return [f for f in api.list_repo_files(RESULTS_REPO, repo_type="dataset")
            if f.startswith(prefix)]

# ---------------------------------------------------------------------------
# Image preprocessing
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

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def fit_mass_lum(n_train=2000):
    from datasets import load_dataset
    ds = load_dataset(DATASET, revision=DATASET_REVISION,
                      split="train", streaming=True)
    ds = ds.select_columns(["mag_abs_r_photoz", "mass_med_photoz"])
    Ls, Ms = [], []
    for ex in ds:
        try:
            L = float(ex.get("mag_abs_r_photoz","nan"))
            M = float(ex.get("mass_med_photoz","nan"))
            if np.isfinite(L) and np.isfinite(M) and L>-90 and M>-90:
                Ls.append(-0.4*L); Ms.append(M)
        except: pass
        if len(Ls) >= n_train: break
    Ls = np.array(Ls).reshape(-1,1); Ms = np.array(Ms)
    reg = LinearRegression().fit(Ls, Ms)
    print(f"Mass~L_abs fit: slope={reg.coef_[0]:.3f}, intercept={reg.intercept_:.3f}")
    return reg

def load_test_set(n, spiral_fn, reg):
    from datasets import load_dataset
    ds = load_dataset(DATASET, revision=DATASET_REVISION,
                      split="test", streaming=True)
    ds = ds.select_columns(["image","mag_abs_r_photoz","mag_r_desi",
                             "mass_med_photoz","ssfr_med_photoz",
                             "redshift","photo_z"])
    patches = []; Lr=[]; M=[]; S=[]; Z=[]; Mag_r=[]
    pbar = tqdm(total=n, desc="test set", ncols=70)
    for ex in ds:
        try: p = _patchify(ex["image"], spiral_fn)
        except: continue
        def _f(v):
            try: v=float(v); return v if np.isfinite(v) and v>-90 else np.nan
            except: return np.nan
        patches.append(p)
        Lr.append(-0.4*_f(ex.get("mag_abs_r_photoz")))
        Mag_r.append(_f(ex.get("mag_r_desi")))
        M.append(_f(ex.get("mass_med_photoz")))
        S.append(_f(ex.get("ssfr_med_photoz")))
        z = _f(ex.get("redshift"))
        if not np.isfinite(z): z = _f(ex.get("photo_z"))
        Z.append(z)
        pbar.update(1)
        if len(patches) >= n: break
    pbar.close()

    patches = torch.stack(patches)
    Lr=np.array(Lr,dtype=np.float32); M=np.array(M,dtype=np.float32)
    S=np.array(S,dtype=np.float32);   Z=np.array(Z,dtype=np.float32)
    Mag_r=np.array(Mag_r,dtype=np.float32)

    valid = np.isfinite(Lr) & np.isfinite(M)
    eps = np.full(len(M), np.nan, dtype=np.float32)
    eps[valid] = M[valid] - reg.predict(Lr[valid].reshape(-1,1)).astype(np.float32)

    labels = {"L_r": Lr, "mag_r": Mag_r, "mass": M,
              "ssfr": S, "z": Z, "eps_M_given_L": eps}
    print(f"Loaded {len(patches):,} galaxies")
    for k,v in labels.items():
        print(f"  {k}: {np.isfinite(v).sum():,} valid, std={np.nanstd(v):.3f}")
    return patches, labels

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(cfg_name):
    obj, tok, size = cfg_name.split("_")
    from astropt.model import GPT, GPTConfig, ModalityConfig, ModalityRegistry
    torch.serialization.add_safe_globals([ModalityConfig])
    mr = ModalityRegistry([ModalityConfig(
        name="images", input_size=PATCH_SIZE*PATCH_SIZE*N_CHAN,
        patch_size=PATCH_SIZE, embed_pos=True, pos_input_size=1)])
    dims = SIZES[size]
    gptconf = GPTConfig(
        block_size=BLOCK_SIZE, dropout=0.0,
        attn_type=("full" if obj=="mae" else "causal"),
        mae_mask_ratio=0.5, norm_pix_loss=False,
        tokeniser=tok, **dims)
    model = GPT(gptconf, mr, master_process=True).to(DEVICE)
    model.eval()
    return model

@torch.no_grad()
def extract_layers(model, patches):
    mod = model.modality_registry.names()[0]
    enc, emb = model.encoders[mod], model.embedders[mod]
    out = []
    for s in range(0, patches.shape[0], PROBE_BATCH):
        chunk = patches[s:s+PROBE_BATCH].to(DEVICE).float()
        pos = torch.arange(chunk.shape[1], device=DEVICE).unsqueeze(0).expand(chunk.shape[0],-1)
        with CTX:
            x = model.transformer.drop(enc(chunk) + emb(pos))
            layers = [x.mean(1)]
            for blk in model.transformer.h:
                x = blk(x); layers.append(x.mean(1))
        out.append(torch.stack(layers,1).float().cpu())
        del chunk, x, layers
    torch.cuda.empty_cache()
    return torch.cat(out,0)  # (N, L+1, d)

# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------
def fit_probe_direction(X_train, y_train, alpha=RIDGE_ALPHA):
    """Returns (fitted pipeline, unit-norm weight vector in standardized space)."""
    mask = np.isfinite(y_train)
    if mask.sum() < 50: return None, None
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(X_train[mask], y_train[mask])
    w = model.named_steps["ridge"].coef_.ravel()
    return model, w / (np.linalg.norm(w) + 1e-12)

def cosine(a, b):
    if a is None or b is None: return np.nan
    return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

def bootstrap_r2_ci(y_true, y_pred, n_boot=1000, seed=0):
    """Exp 1: cheap R² bootstrap over test galaxies (fixed probe)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[ok], y_pred[ok]
    n = len(yt)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.var(yt[idx]) < 1e-12: continue
        vals.append(r2_score(yt[idx], yp[idx]))
    vals = np.array(vals)
    return {"mean": float(np.mean(vals)),
            "lo":   float(np.percentile(vals, 2.5)),
            "hi":   float(np.percentile(vals, 97.5)),
            "std":  float(np.std(vals))}

def bootstrap_probe_cosines(X, y_dict, train_idx, test_idx,
                             n_boot=300, seed=0):
    """Exp 2: bootstrap cosines by retraining probes."""
    rng = np.random.default_rng(seed)
    X_tr = X[train_idx]; X_te = X[test_idx]
    n_tr = len(train_idx)
    targets = list(y_dict.keys())

    all_rows = []
    for b in range(n_boot):
        idx = rng.integers(0, n_tr, size=n_tr)
        dirs = {}; r2s = {}
        for name in targets:
            y = np.asarray(y_dict[name])
            yb = y[train_idx][idx]
            yt = y[test_idx]
            model, v = fit_probe_direction(X_tr[idx], yb)
            if model is None: dirs[name]=None; r2s[name]=np.nan; continue
            dirs[name] = v
            ok = np.isfinite(yt)
            if ok.sum() < 10: r2s[name]=np.nan; continue
            pred = model.predict(X_te)
            if np.var(yt[ok]) < 1e-12: r2s[name]=np.nan; continue
            r2s[name] = float(r2_score(yt[ok], pred[ok]))

        row = {"boot": b}
        # cosines
        pairs = [("L_r","mass","cos_L_M"), ("L_r","eps_M_given_L","cos_L_eps"),
                 ("mass","eps_M_given_L","cos_M_eps"), ("ssfr","mass","cos_ssfr_M"),
                 ("z","mass","cos_z_M"), ("z","L_r","cos_z_L"),
                 ("z","ssfr","cos_z_ssfr")]
        for a,b_,key in pairs:
            row[key] = cosine(dirs.get(a), dirs.get(b_))
        for name, val in r2s.items():
            row[f"r2_{name}"] = val
        all_rows.append(row)

    # summarize
    keys = [k for k in all_rows[0] if k != "boot"]
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in all_rows if np.isfinite(r[k])])
        if len(vals) < 10: summary[k] = {"mean":np.nan,"lo":np.nan,"hi":np.nan,"std":np.nan}
        else:
            summary[k] = {"mean": float(np.mean(vals)),
                          "lo":   float(np.percentile(vals, 2.5)),
                          "hi":   float(np.percentile(vals, 97.5)),
                          "std":  float(np.std(vals))}
    return summary, all_rows

def bootstrap_mean_ci(values, n_boot=1000, seed=0):
    """Exp 3: CI for patching success rates."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    boot = [np.mean(values[rng.integers(0,n,size=n)]) for _ in range(n_boot)]
    return {"mean": float(np.mean(values)),
            "lo":   float(np.percentile(boot, 2.5)),
            "hi":   float(np.percentile(boot, 97.5)),
            "std":  float(np.std(boot))}

# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------
def run_final_checkpoint(patches, labels, out_dir, n_boot=500, configs=None):
    if configs is None: configs = TARGET_CONFIGS
    """Main paper: final checkpoint, all 3 sizes, bootstrap R² and cosines."""
    results = {}
    rng_split = np.random.default_rng(42)
    N = len(patches)
    perm = rng_split.permutation(N)
    n_tr = int(TRAIN_FRAC * N)
    train_idx = perm[:n_tr]; test_idx = perm[n_tr:]
    print(f"Train/test split: {n_tr}/{N-n_tr}")

    for cfg in configs:
        print(f"\n=== {cfg} ===", flush=True)
        ckpt_files = _ls(f"checkpoints/{cfg}/step_")
        steps = sorted(int(f.rsplit("step_",1)[-1][:-3]) for f in ckpt_files)
        if not steps: print("  no checkpoints"); continue
        step = steps[-1]
        print(f"  Using step {step}", flush=True)

        raw = _get(f"checkpoints/{cfg}/step_{step:08d}.pt")
        from astropt.model import ModalityConfig
        torch.serialization.add_safe_globals([ModalityConfig])
        try:
            ck = torch.load(io.BytesIO(raw), map_location=DEVICE, weights_only=True)
        except:
            ck = torch.load(io.BytesIO(raw), map_location=DEVICE, weights_only=False)

        model = build_model(cfg)
        model.load_state_dict(
            {k:v.float() for k,v in ck["model"].items()
             if not k.startswith(("encoders.","decoders.","mask_token"))},
            strict=False)
        model.eval()

        hidden = extract_layers(model, patches)  # (N, L+1, d)
        del model, ck, raw; gc.collect(); torch.cuda.empty_cache()

        # Use last layer for final-checkpoint summary
        n_layers = hidden.shape[1]
        cfg_results = {}

        for li in range(n_layers):
            X = hidden[:, li, :].numpy()

            print(f"  Layer {li}: running bootstrap ({n_boot} iters)...", flush=True)
            # Exp 2: retrain bootstrap for cosines + R²
            summary, _ = bootstrap_probe_cosines(
                X, labels, train_idx, test_idx,
                n_boot=n_boot, seed=li)
            cfg_results[f"layer_{li}"] = {
                "step": step, "layer": li, **summary}

        results[cfg] = cfg_results
        del hidden; gc.collect(); torch.cuda.empty_cache()
        os.makedirs(out_dir, exist_ok=True)
        per_path = os.path.join(out_dir, f"bootstrap_{cfg}.json")
        with open(per_path, "w") as f:
            json.dump({cfg: cfg_results}, f)
        print(f"  Saved {cfg} -> {per_path}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bootstrap_final_checkpoint.json")
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"\nSaved -> {out_path}", flush=True)
    return results

def run_patch_ci(causal_json_path, out_dir, n_boot=1000):
    """Exp 3: bootstrap CI for activation patching success rates."""
    with open(causal_json_path) as f:
        data = json.load(f)

    results = {}
    for res in data:
        step = res["step"]
        patch_rows = res.get("patch_rows", [])
        if not patch_rows:
            print(f"  Step {step}: no patch_rows, skipping"); continue

        print(f"  Step {step}: {len(patch_rows)} patch rows")

        # Group by layer
        layers = sorted(set(r["patch_layer"] for r in patch_rows))
        layer_results = {}
        for li in layers:
            matched = [r for r in patch_rows if r["patch_layer"] == li]
            sr_vals = [r["success"] for r in matched]
            dR_vals = [abs(r["delta_R"]) for r in matched]
            dL_vals = [abs(r["delta_L"]) for r in matched]

            layer_results[f"layer_{li}"] = {
                "success_rate": bootstrap_mean_ci(sr_vals, n_boot=n_boot),
                "residual_effect": bootstrap_mean_ci(dR_vals, n_boot=n_boot),
                "luminosity_effect": bootstrap_mean_ci(dL_vals, n_boot=n_boot),
                "n_pairs": len(matched),
            }
        results[f"step_{step}"] = layer_results

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bootstrap_patching.json")
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"Saved -> {out_path}")
    return results

# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--mode", choices=["final","patch","layer"],
                        default="final")
    parser.add_argument("--n_boot",     type=int, default=500)
    parser.add_argument("--n_galaxies", type=int, default=N_GALAXIES)
    parser.add_argument("--out_dir",
                        default="/work/nvme/bfir/ssourav/astroPT-runs/bootstrap")
    parser.add_argument("--causal_json",
                        default="/work/nvme/bfir/ssourav/astroPT-runs/causal/ar_aim_021M_causal_results.json")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Bootstrap CI | mode={args.mode} | n_boot={args.n_boot} | device={DEVICE}")

    if args.mode == "patch":
        run_patch_ci(args.causal_json, args.out_dir, n_boot=args.n_boot)
        return

    print("Fitting mass~L_r on train split...")
    reg = fit_mass_lum()

    print("Loading test set...")
    spiral_fn = _spiral_fn()
    patches, labels = load_test_set(args.n_galaxies, spiral_fn, reg)

    if args.mode == "final":
        run_final_checkpoint(patches, labels, args.out_dir, n_boot=args.n_boot, configs=args.configs)
    elif args.mode == "layer":
        # Same as final but report all layers separately (already does this)
        run_final_checkpoint(patches, labels, args.out_dir, n_boot=min(args.n_boot, 200))

    print("\nAll done.")

if __name__ == "__main__":
    main()
