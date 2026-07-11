"""
causal_tracing.py  --  Causal Galaxy Relationship Tracing in AstroPT.

Implements the full pipeline from the spec:
  1. Load test galaxies + labels, compute L_r and R_M_given_L
  2. Non-causal probe directions + cosine geometry
  3. Matched pair construction (same L_r, different R_M_given_L)
  4. Activation patching across layers
  5. Save results for plotting

USAGE:
    python causal_tracing.py --config ar_aim_021M --n_galaxies 50000 \
        --n_pairs 500 --out_dir results/causal/

    # sweep multiple checkpoints:
    python causal_tracing.py --config ar_aim_021M --n_galaxies 50000 \
        --checkpoints 0 19 226 2656 26483 --n_pairs 500
"""

import argparse, gc, io, os, shutil, json
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
RIDGE_LAMBDA = 100.0

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
# Data loading: images + labels
# ---------------------------------------------------------------------------
MAG_COLS  = ["mag_r_desi", "mag_abs_r_photoz", "mass_med_photoz", "ssfr_med_photoz"]

def load_test_galaxies(n, spiral_fn, reg=None):
    from datasets import load_dataset
    ds = load_dataset(DATASET, revision=DATASET_REVISION,
                      split="test", streaming=True)
    ds = ds.select_columns(["image"] + MAG_COLS)

    patches, mag_r, mag_abs_r, mass, ssfr = [], [], [], [], []
    pbar = tqdm(total=n, desc="loading test set", ncols=70)
    for ex in ds:
        try:
            p = _patchify(ex["image"], spiral_fn)
        except Exception:
            continue
        patches.append(p)
        def _f(v):
            try:
                v = float(v)
                return v if np.isfinite(v) and v > -90 else np.nan
            except: return np.nan
        mag_r.append(_f(ex.get("mag_r_desi")))
        mag_abs_r.append(_f(ex.get("mag_abs_r_photoz")))
        mass.append(_f(ex.get("mass_med_photoz")))
        ssfr.append(_f(ex.get("ssfr_med_photoz")))
        pbar.update(1)
        if len(patches) >= n: break
    pbar.close()

    patches = torch.stack(patches)       # (N, n_patches, patch_dim) fp16
    labels = {
        "mag_r":     np.array(mag_r,     dtype=np.float32),
        "mag_abs_r": np.array(mag_abs_r, dtype=np.float32),
        "mass":      np.array(mass,      dtype=np.float32),
        "ssfr":      np.array(ssfr,      dtype=np.float32),
    }
    # Use ABSOLUTE magnitude for luminosity proxy (removes distance confusion)
    labels["L_r"] = -0.4 * labels["mag_abs_r"]
    labels["L_r_app"] = -0.4 * labels["mag_r"]  # apparent (for reference only)

    # fit mass ~ L_r on TRAIN split only (avoids label leakage into test evaluation)
    if reg is None:
        print("  Fitting mass~L_r on train split (1000 galaxies)...")
        from datasets import load_dataset as _lds
        _ds_tr = _lds(DATASET, revision=DATASET_REVISION,
                      split="train", streaming=True)
        _ds_tr = _ds_tr.select_columns(["mag_abs_r_photoz", "mass_med_photoz"])
        _tr_L, _tr_M = [], []
        for _ex in _ds_tr:
            try:
                _l = float(_ex.get("mag_abs_r_photoz", "nan"))
                _m = float(_ex.get("mass_med_photoz",  "nan"))
                if np.isfinite(_l) and np.isfinite(_m) and _l > -90 and _m > -90:
                    _tr_L.append(-0.4 * _l); _tr_M.append(_m)
            except: pass
            if len(_tr_L) >= 1000: break
        _tr_L = np.array(_tr_L).reshape(-1,1); _tr_M = np.array(_tr_M)
        reg = LinearRegression().fit(_tr_L, _tr_M)
        print(f"  Train-split fit: slope={reg.coef_[0]:.3f}, intercept={reg.intercept_:.3f}")
    else:
        print(f"  Using pre-fitted reg: slope={reg.coef_[0]:.3f}, intercept={reg.intercept_:.3f}")

    valid = np.isfinite(labels["L_r"]) & np.isfinite(labels["mass"])
    print(f"  std(M_pred) diagnostic: {np.std(reg.predict(labels['L_r'][valid].reshape(-1,1))):.3f}  (want >> 0.046)")
    print(f"  std(mass): {np.nanstd(labels['mass']):.3f}")

    labels["M_pred"] = np.full(len(labels["mass"]), np.nan, dtype=np.float32)
    labels["M_pred"][valid] = reg.predict(labels["L_r"][valid].reshape(-1,1)).astype(np.float32)
    labels["R_M_given_L"] = labels["mass"] - labels["M_pred"]
    print(f"  std(R_M_given_L): {np.nanstd(labels['R_M_given_L']):.3f}  (want < std(mass))")

    print(f"Loaded {len(patches)} galaxies")
    for k, v in labels.items():
        fin = np.sum(np.isfinite(v))
        if fin > 0:
            print(f"  {k}: {fin} valid, mean={np.nanmean(v):.3f}, std={np.nanstd(v):.3f}")
    return patches, labels, reg

# ---------------------------------------------------------------------------
# Model
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

def load_checkpoint(model, cfg_name, step):
    raw = _get(f"checkpoints/{cfg_name}/step_{step:08d}.pt")
    from astropt.model import ModalityConfig
    torch.serialization.add_safe_globals([ModalityConfig])
    try:
        ck = torch.load(io.BytesIO(raw), map_location=DEVICE, weights_only=True)
    except Exception:
        ck = torch.load(io.BytesIO(raw), map_location=DEVICE, weights_only=False)
    state = {k: v.float() for k, v in ck["model"].items()
             if not k.startswith(("encoders.", "decoders.", "mask_token"))}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

# ---------------------------------------------------------------------------
# Forward pass with cache — returns per-layer mean-pooled hidden states
# ---------------------------------------------------------------------------
BATCH = 32

@torch.no_grad()
def extract_all_layers(model, patches):
    """Returns (N, n_layers+1, d_model) float32 cpu."""
    mod  = model.modality_registry.names()[0]
    enc  = model.encoders[mod]
    emb  = model.embedders[mod]
    out  = []
    for s in range(0, patches.shape[0], BATCH):
        chunk = patches[s:s+BATCH].to(DEVICE).float()
        pos = torch.arange(chunk.shape[1], device=DEVICE).unsqueeze(0).expand(chunk.shape[0], -1)
        with CTX:
            x = model.transformer.drop(enc(chunk) + emb(pos))
            layers = [x.mean(dim=1)]           # layer 0: embedding
            for blk in model.transformer.h:
                x = blk(x)
                layers.append(x.mean(dim=1))   # layers 1..L
        out.append(torch.stack(layers, dim=1).float().cpu())
        del chunk, x, layers
    torch.cuda.empty_cache()
    return torch.cat(out, dim=0)                # (N, L+1, d)

# ---------------------------------------------------------------------------
# Activation patching — patch one layer's *pre-pooled* token sequence
# ---------------------------------------------------------------------------
@torch.no_grad()
def patch_layer_and_readout(model, base_patch, source_patch, patch_layer):
    """
    Run model on base_patch, replacing layer `patch_layer`'s output
    with source's activation at that layer. Returns per-layer mean-pooled
    hidden states of the patched run.
    patch_layer=0 patches the embedding (before any transformer block).
    """
    mod = model.modality_registry.names()[0]
    enc = model.encoders[mod]
    emb = model.embedders[mod]

    def _forward_cache(p):
        """Returns list of token sequences, one per layer (before mean-pool)."""
        p = p.to(DEVICE).float().unsqueeze(0)   # (1, T, C)
        pos = torch.arange(p.shape[1], device=DEVICE).unsqueeze(0)
        with CTX:
            x = model.transformer.drop(enc(p) + emb(pos))
        states = [x]
        with CTX:
            for blk in model.transformer.h:
                x = blk(x)
                states.append(x)
        return [s.float() for s in states]

    base_states   = _forward_cache(base_patch)
    source_states = _forward_cache(source_patch)

    # Re-run base, but inject source activation at patch_layer
    def _forward_patched():
        p = base_patch.to(DEVICE).float().unsqueeze(0)
        pos = torch.arange(p.shape[1], device=DEVICE).unsqueeze(0)
        with CTX:
            x = model.transformer.drop(enc(p) + emb(pos))
        # layer 0 = embedding
        if patch_layer == 0:
            x = source_states[0].clone()
        out_layers = [x.mean(dim=1).float().cpu().squeeze(0)]
        with CTX:
            for li, blk in enumerate(model.transformer.h):
                x = blk(x)
                if patch_layer == li + 1:
                    x = source_states[li+1].clone()
                out_layers.append(x.mean(dim=1).float().cpu().squeeze(0))
        return torch.stack(out_layers, dim=0)   # (L+1, d)

    patched = _forward_patched()
    base    = torch.stack([s.mean(dim=1).float().cpu().squeeze(0)
                           for s in base_states], dim=0)  # (L+1, d)
    return base, patched

# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------
def train_probe(X, y, lam=RIDGE_LAMBDA):
    """Ridge probe. Returns (Ridge, StandardScaler)."""
    mask = np.isfinite(y)
    if mask.sum() < 50: return None, None
    sc = StandardScaler()
    Xs = sc.fit_transform(X[mask])
    p = Ridge(alpha=lam).fit(Xs, y[mask])
    return p, sc

def probe_r2(probe, sc, X, y):
    mask = np.isfinite(y)
    if mask.sum() < 10 or probe is None: return np.nan
    Xs = sc.transform(X[mask])
    yp = probe.predict(Xs)
    ss_tot = np.var(y[mask]) * mask.sum() + 1e-12
    ss_res = np.sum((y[mask] - yp)**2)
    return float(1 - ss_res / ss_tot)

def probe_direction(probe):
    """Unit-normed weight vector (probe direction in representation space)."""
    w = probe.coef_.ravel()
    return w / (np.linalg.norm(w) + 1e-12)

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# ---------------------------------------------------------------------------
# Matched pair construction
# ---------------------------------------------------------------------------
def make_matched_pairs(labels, lum_tol=0.05, min_resid_gap=0.5, max_pairs=500, seed=42):
    rng = np.random.default_rng(seed)
    Lr  = labels["L_r"]
    R   = labels["R_M_given_L"]
    valid = np.where(np.isfinite(Lr) & np.isfinite(R))[0]

    pairs = []
    # shuffle to avoid biasing toward early galaxies
    order = rng.permutation(valid)
    for i in order:
        if len(pairs) >= max_pairs: break
        lum_close = valid[np.abs(Lr[valid] - Lr[i]) < lum_tol]
        resid_far  = lum_close[np.abs(R[lum_close] - R[i]) > min_resid_gap]
        resid_far  = resid_far[resid_far != i]
        if len(resid_far) == 0: continue
        # pick source with largest residual gap
        j = resid_far[np.abs(R[resid_far] - R[i]).argmax()]
        pairs.append((int(i), int(j)))

    print(f"Matched pairs: {len(pairs)} (lum_tol={lum_tol}, resid_gap>{min_resid_gap})")
    return pairs

def make_random_pairs(labels, n, seed=99):
    """Control 2: random (non-matched) pairs."""
    rng = np.random.default_rng(seed)
    valid = np.where(np.isfinite(labels["L_r"]) & np.isfinite(labels["R_M_given_L"]))[0]
    idx = rng.permutation(valid)[:n*2]
    return list(zip(idx[:n].tolist(), idx[n:2*n].tolist()))

def make_same_residual_pairs(labels, lum_tol=0.05, max_resid_gap=0.1, max_pairs=500, seed=77):
    """Control 1: matched L_r AND similar R_M_given_L. Expected: small ΔR."""
    rng = np.random.default_rng(seed)
    Lr  = labels["L_r"]
    R   = labels["R_M_given_L"]
    valid = np.where(np.isfinite(Lr) & np.isfinite(R))[0]
    pairs = []
    order = rng.permutation(valid)
    for i in order:
        if len(pairs) >= max_pairs: break
        candidates = valid[
            (np.abs(Lr[valid] - Lr[i]) < lum_tol) &
            (np.abs(R[valid]  - R[i])  < max_resid_gap) &
            (valid != i)
        ]
        if len(candidates) == 0: continue
        j = candidates[rng.integers(len(candidates))]
        pairs.append((int(i), int(j)))
    print(f"Same-residual control pairs: {len(pairs)}")
    return pairs

# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_causal_tracing(model, patches, labels, pairs, probes, scalers):
    """
    For each matched pair and each layer: patch and measure ΔL, ΔM, ΔR.
    Returns list of dicts.
    """
    n_layers = len(model.transformer.h) + 1   # +1 for embedding
    rows = []

    for base_idx, src_idx in tqdm(pairs, desc="patching", ncols=70):
        base_p = patches[base_idx]
        src_p  = patches[src_idx]

        for layer in range(n_layers):
            h_base, h_patch = patch_layer_and_readout(
                model, base_p, src_p, patch_layer=layer)
            # use representation from LAST layer for readout
            h_b = h_base[-1].numpy().reshape(1, -1)
            h_p = h_patch[-1].numpy().reshape(1, -1)

            results = {}
            for key in ("L_r", "mass", "R_M_given_L", "ssfr"):
                if probes[key] is None: continue
                Xb_s = scalers[key].transform(h_b)
                Xp_s = scalers[key].transform(h_p)
                pred_b = probes[key].predict(Xb_s)[0]
                pred_p = probes[key].predict(Xp_s)[0]
                results[key] = float(pred_p - pred_b)

            true_dir_R = labels["R_M_given_L"][src_idx] - labels["R_M_given_L"][base_idx]

            rows.append({
                "base_idx": base_idx, "src_idx": src_idx,
                "patch_layer": layer,
                "delta_L": results.get("L_r", np.nan),
                "delta_M": results.get("mass", np.nan),
                "delta_R": results.get("R_M_given_L", np.nan),
                "delta_sSFR": results.get("ssfr", np.nan),
                "true_dir_R": float(true_dir_R),
                "success": float(results.get("R_M_given_L", 0) * true_dir_R > 0),
            })
        del h_base, h_patch
        torch.cuda.empty_cache()

    return rows

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="ar_aim_021M")
    parser.add_argument("--n_galaxies",  type=int, default=50_000)
    parser.add_argument("--n_pairs",     type=int, default=500)
    parser.add_argument("--checkpoints", type=int, nargs="+",
                        default=None,
                        help="Specific steps to run. Default: final only.")
    parser.add_argument("--lum_tol",     type=float, default=0.05)
    parser.add_argument("--resid_gap",   type=float, default=0.3)
    parser.add_argument("--out_dir",     default="results/causal")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Causal tracing | config={args.config} | device={DEVICE}")

    # 1. Load data
    print("\n--- Loading galaxies ---")
    spiral_fn = _spiral_fn()
    patches, labels, reg = load_test_galaxies(args.n_galaxies, spiral_fn)

    # save label scatter data for Fig 1
    scatter = {k: labels[k].tolist() for k in ("L_r","mass","R_M_given_L","ssfr")}
    with open(os.path.join(args.out_dir, "label_scatter.json"), "w") as f:
        json.dump(scatter, f)
    print("Saved label_scatter.json")

    # 2. Matched pairs (and random-control pairs)
    print("\n--- Building pairs ---")
    pairs = make_matched_pairs(labels,
                               lum_tol=args.lum_tol,
                               min_resid_gap=args.resid_gap,
                               max_pairs=args.n_pairs)
    rand_pairs = make_random_pairs(labels, n=len(pairs))

    with open(os.path.join(args.out_dir, "pairs.json"), "w") as f:
        json.dump({"matched": pairs, "random": rand_pairs}, f)

    # 3. Build model
    print("\n--- Building model ---")
    model = build_model(args.config)

    # 4. Determine checkpoints to run
    all_steps = sorted(int(f.rsplit("step_",1)[-1][:-3])
                       for f in _ls(f"checkpoints/{args.config}/step_"))
    if args.checkpoints is not None:
        steps = []
        for req in args.checkpoints:
            nearest = min(all_steps, key=lambda s: abs(s - req))
            if nearest not in steps:
                steps.append(nearest)
                print(f"  Step {req} -> using {nearest} (gap={abs(nearest-req)})")
        steps.sort()
    else:
        steps = [all_steps[-1]]
        print(f"  Using final step: {steps[0]}")

    print(f"Running on steps: {steps}")

    all_results = []

    for step in steps:
        print(f"\n=== Step {step} ===")

        # load weights
        model = load_checkpoint(model, args.config, step)

        # 5. Extract embeddings for all test galaxies
        print("  Extracting embeddings...")
        hidden = extract_all_layers(model, patches)   # (N, L+1, d)
        n_layers = hidden.shape[1]

        # 6. Train probes per layer (use best layer for readout = last layer)
        # For direction geometry, we use the last-layer hidden
        print("  Training probes...")
        h_last = hidden[:, -1, :].numpy()   # (N, d)

        probe_targets = {
            "L_r":        labels["L_r"],
            "mass":       labels["mass"],
            "R_M_given_L": labels["R_M_given_L"],
            "ssfr":       labels["ssfr"],
        }
        probes, scalers, r2s, dirs = {}, {}, {}, {}
        for key, y in probe_targets.items():
            p, sc = train_probe(h_last, y)
            probes[key] = p; scalers[key] = sc
            r2s[key] = probe_r2(p, sc, h_last, y) if p else np.nan
            dirs[key] = probe_direction(p) if p else None

        print(f"  Probe R²: " +
              " | ".join(f"{k}={v:.3f}" for k,v in r2s.items()))

        # Control 3: shuffled residual probe
        R_shuf = labels["R_M_given_L"].copy()
        rng_shuf = np.random.default_rng(42 + step)
        valid_mask = np.isfinite(R_shuf)
        R_shuf[valid_mask] = rng_shuf.permutation(R_shuf[valid_mask])
        probe_R_shuf, sc_R_shuf = train_probe(h_last, R_shuf)
        probes_shuffled = {"R_M_given_L": probe_R_shuf}
        scalers_shuffled = {"R_M_given_L": sc_R_shuf}
        # (we only need the shuffled R probe for the control run)

        # direction cosines
        cos_LM = cosine(dirs["L_r"], dirs["mass"]) if dirs["L_r"] is not None and dirs["mass"] is not None else np.nan
        cos_LR = cosine(dirs["L_r"], dirs["R_M_given_L"]) if dirs["L_r"] is not None and dirs["R_M_given_L"] is not None else np.nan
        cos_MR = cosine(dirs["mass"], dirs["R_M_given_L"]) if dirs["mass"] is not None and dirs["R_M_given_L"] is not None else np.nan
        print(f"  cos(L,M)={cos_LM:.3f}  cos(L,R)={cos_LR:.3f}  cos(M,R)={cos_MR:.3f}")

        # per-layer probe R² (for Fig 3 emergence curves)
        layer_r2s = {key: [] for key in probe_targets}
        for li in range(n_layers):
            h_li = hidden[:, li, :].numpy()
            for key, y in probe_targets.items():
                p_li, sc_li = train_probe(h_li, y)
                layer_r2s[key].append(probe_r2(p_li, sc_li, h_li, y) if p_li else np.nan)

        # 7. Activation patching
        print(f"  Patching {len(pairs)} matched pairs across {n_layers} layers...")
        patch_rows = run_causal_tracing(model, patches, labels, pairs, probes, scalers)

        # Control 2: random pairs
        print(f"  Patching {len(rand_pairs)} random pairs (ctrl 2: random)...")
        rand_rows = run_causal_tracing(model, patches, labels, rand_pairs, probes, scalers)

        # Control 1: same-residual pairs (similar L_r AND similar R)
        same_resid_pairs_local = make_same_residual_pairs(
            labels, lum_tol=args.lum_tol, max_resid_gap=0.1, max_pairs=len(pairs))
        print(f"  Patching {len(same_resid_pairs_local)} same-residual pairs (ctrl 1: null R)...")
        same_resid_rows = run_causal_tracing(model, patches, labels,
                                             same_resid_pairs_local, probes, scalers)

        # Control 3: shuffled residual probe — use matched pairs but broken probe
        shuf_probes_full = {**probes, "R_M_given_L": probes_shuffled["R_M_given_L"]}
        shuf_scalers_full = {**scalers, "R_M_given_L": scalers_shuffled["R_M_given_L"]}
        print(f"  Patching {len(pairs)} matched pairs with shuffled R probe (ctrl 3)...")
        shuf_rows = run_causal_tracing(model, patches, labels, pairs,
                                       shuf_probes_full, shuf_scalers_full)

        # aggregate by layer
        layer_stats = []
        for li in range(n_layers):
            matched_li = [r for r in patch_rows if r["patch_layer"] == li]
            rand_li    = [r for r in rand_rows  if r["patch_layer"] == li]

            def _agg(rows, key): return np.nanmean([abs(r[key]) for r in rows]) if rows else np.nan
            def _sr(rows):       return np.nanmean([r["success"] for r in rows]) if rows else np.nan

            same_li = [r for r in same_resid_rows if r["patch_layer"] == li]
            shuf_li  = [r for r in shuf_rows       if r["patch_layer"] == li]

            layer_stats.append({
                "layer": li,
                # matched pairs (main result)
                "ResidualEffect":        _agg(matched_li, "delta_R"),
                "LuminosityEffect":      _agg(matched_li, "delta_L"),
                "MassEffect":            _agg(matched_li, "delta_M"),
                "RelationSelectivity":   (_agg(matched_li,"delta_R") /
                                          (_agg(matched_li,"delta_L") + 1e-6)),
                "SuccessRate":           _sr(matched_li),
                # control 1: same-residual pairs (expected small ΔR)
                "SuccessRate_same_resid":    _sr(same_li),
                "ResidualEffect_same_resid": _agg(same_li, "delta_R"),
                # control 2: random pairs
                "SuccessRate_random":        _sr(rand_li),
                "ResidualEffect_random":     _agg(rand_li, "delta_R"),
                # control 3: shuffled residual probe
                "SuccessRate_shuf_probe":    _sr(shuf_li),
                # probe R²s at this layer
                **{f"r2_{k}_layer{li}": layer_r2s[k][li] for k in probe_targets},
            })

        all_results.append({
            "step": step,
            "config": args.config,
            "r2": r2s,
            "cos_LM": cos_LM, "cos_LR": cos_LR, "cos_MR": cos_MR,
            "layer_r2s": layer_r2s,
            "layer_stats": layer_stats,
            "patch_rows": patch_rows,
            "dirs": {k: (v.tolist() if v is not None else None) for k,v in dirs.items()},
        })

        del hidden
        gc.collect(); torch.cuda.empty_cache()

    # Save results
    out_file = os.path.join(args.out_dir, f"{args.config}_causal_results.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f)
    print(f"\nSaved -> {out_file}")
    print("Done.")

if __name__ == "__main__":
    main()
