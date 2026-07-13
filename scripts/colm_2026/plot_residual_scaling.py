"""
plot_residual_scaling.py  --  Figures A-D for residual capacity/emergence analysis.

Reads results_residual/<config>/results.parquet from HF.

Fig A: Capacity scaling — R²(L), R²(M), R²(eps) vs model size (final ckpt, best layer)
Fig B: Direction geometry vs model size — cos(L,M), cos(L,eps), cos(M,eps)
Fig C: Residual emergence over training — R²(eps) vs step for 1M, 21M, 100M
Fig D: Layer dependence — R²(L), R²(M), R²(eps) by layer, columns=model size
"""

import argparse, os, shutil
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from huggingface_hub import HfApi
from tqdm import tqdm

RESULTS_REPO = "HCVYM5w6Gn/colm-results"
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
CONFIGS      = ["ar_aim_001M", "ar_aim_021M", "ar_aim_100M"]
SIZE_LABELS  = {"ar_aim_001M": "1M", "ar_aim_021M": "21M", "ar_aim_100M": "100M"}
SIZE_NM      = {"ar_aim_001M": 1,    "ar_aim_021M": 21,    "ar_aim_100M": 100}

C_L   = "#1f77b4"   # blue   — luminosity
C_M   = "#2ca02c"   # green  — mass
C_EPS = "#d62728"   # red    — residual
C_S   = "#9467bd"   # purple — sSFR

SIZE_COLORS = {"ar_aim_001M": "#aec7e8",
               "ar_aim_021M": "#1f77b4",
               "ar_aim_100M": "#08306b"}
SIZE_LW     = {"ar_aim_001M": 1.0,
               "ar_aim_021M": 1.6,
               "ar_aim_100M": 2.2}

def _style():
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Helvetica Neue","Helvetica","Arial","DejaVu Sans"],
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "axes.facecolor":    "white",
        "figure.facecolor":  "white",
        "axes.edgecolor":    "#444",
        "xtick.color":       "#444",
        "ytick.color":       "#444",
        "text.color":        "black",
        "grid.color":        "#e0e0e0",
        "grid.linewidth":    0.6,
        "grid.linestyle":    ":",
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.major.size":  3.5,
        "ytick.major.size":  3.5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,
        "legend.frameon":    True,
        "legend.framealpha": 0.93,
        "legend.edgecolor":  "#ccc",
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.04,
    })

def load_all(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    api = HfApi(token=HF_TOKEN)
    frames = {}
    for cfg in tqdm(CONFIGS, desc="loading", ncols=60):
        local = os.path.join(cache_dir, f"{cfg}_residual.parquet")
        if not os.path.exists(local):
            try:
                dl = api.hf_hub_download(
                    repo_id=RESULTS_REPO,
                    filename=f"results_residual/{cfg}/results.parquet",
                    repo_type="dataset", token=HF_TOKEN)
                shutil.copy(dl, local)
            except Exception as e:
                print(f"  skip {cfg}: {e}"); continue
        if os.path.exists(local):
            frames[cfg] = pl.read_parquet(local)
    print(f"Loaded {len(frames)} configs")
    return frames

def best_layer_at_step(df, step, col):
    """Best value of `col` across layers at a given step."""
    sub = df.filter(pl.col("step") == step)
    if sub.is_empty(): return np.nan
    return sub[col].max()

def best_layer_series(df, col):
    """Best-layer value of `col` at each step. Returns (steps, vals)."""
    best = (df.group_by("step")
              .agg(pl.col(col).max().alias("val"))
              .sort("step"))
    return best["step"].to_numpy(), best["val"].to_numpy()

def final_step(df):
    return df["step"].max()

# ---------------------------------------------------------------------------
# Fig A: Capacity scaling
# ---------------------------------------------------------------------------
def figA_capacity_scaling(frames, out_dir):
    _style()
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0), constrained_layout=True)

    sizes = []
    r2_L, r2_M, r2_eps = [], [], []

    for cfg in CONFIGS:
        if cfg not in frames: continue
        df = frames[cfg]
        step = final_step(df)
        sizes.append(SIZE_NM[cfg])
        r2_L.append(best_layer_at_step(df, step, "r2_L"))
        r2_M.append(best_layer_at_step(df, step, "r2_M"))
        r2_eps.append(best_layer_at_step(df, step, "r2_eps"))

    xa = np.array(sizes)
    for vals, color, label, marker in [
        (r2_L,   C_L,   r"$R^2(L_r)$",              "o"),
        (r2_M,   C_M,   r"$R^2(\log M_\star)$",      "s"),
        (r2_eps, C_EPS, r"$R^2(\epsilon_{M|L})$",    "^"),
    ]:
        ax.plot(xa, vals, color=color, lw=1.8, marker=marker, ms=7,
                label=label)

    ax.set_xscale("log")
    ax.set_xticks([1, 21, 100])
    ax.set_xticklabels(["1M", "21M", "100M"])
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Model size", labelpad=3)
    ax.set_ylabel(r"$R^2$ (best layer, final ckpt)", labelpad=3)
    ax.grid(True)
    ax.legend(fontsize=7.5, loc="lower right")

    for fmt in ("pdf","png"):
        fig.savefig(os.path.join(out_dir, f"residual_figA_capacity_scaling.{fmt}"))
    plt.close(fig)
    print("Saved residual_figA_capacity_scaling")

# ---------------------------------------------------------------------------
# Fig B: Direction geometry vs model size
# ---------------------------------------------------------------------------
def figB_direction_geometry(frames, out_dir):
    _style()
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0), constrained_layout=True)

    n = len([c for c in CONFIGS if c in frames])
    x = np.arange(n)
    width = 0.22
    labels_x = []

    cos_LM_vals, cos_Le_vals, cos_Me_vals = [], [], []
    for cfg in CONFIGS:
        if cfg not in frames: continue
        df = frames[cfg]
        step = final_step(df)
        # use last layer for direction geometry
        n_layers = df["layer"].max()
        sub = df.filter((pl.col("step") == step) & (pl.col("layer") == n_layers))
        if sub.is_empty(): continue
        cos_LM_vals.append(sub["cos_LM"][0])
        cos_Le_vals.append(sub["cos_Le"][0])
        cos_Me_vals.append(sub["cos_Me"][0])
        labels_x.append(SIZE_LABELS[cfg])

    xi = np.arange(len(labels_x))
    ax.bar(xi - width, cos_LM_vals, width, color="#4c72b0", alpha=0.85,
           edgecolor="white", lw=0.3, label=r"$\cos(v_L, v_M)$")
    ax.bar(xi,         cos_Le_vals, width, color="#dd8452", alpha=0.85,
           edgecolor="white", lw=0.3, label=r"$\cos(v_L, v_\epsilon)$")
    ax.bar(xi + width, cos_Me_vals, width, color="#55a868", alpha=0.85,
           edgecolor="white", lw=0.3, label=r"$\cos(v_M, v_\epsilon)$")

    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(xi); ax.set_xticklabels(labels_x)
    ax.set_xlabel("Model size", labelpad=3)
    ax.set_ylabel("Cosine similarity (last layer)", labelpad=3)
    ax.set_ylim(-0.2, 1.1)
    ax.grid(True, axis="y")
    ax.legend(fontsize=7)

    for fmt in ("pdf","png"):
        fig.savefig(os.path.join(out_dir, f"residual_figB_direction_geometry.{fmt}"))
    plt.close(fig)
    print("Saved residual_figB_direction_geometry")

# ---------------------------------------------------------------------------
# Fig C: Training-time emergence
# ---------------------------------------------------------------------------
def figC_emergence(frames, out_dir):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)

    # Left: R²(eps) emergence
    ax = axes[0]
    for cfg in CONFIGS:
        if cfg not in frames: continue
        steps, vals = best_layer_series(frames[cfg], "r2_eps")
        ax.plot(steps, vals, color=SIZE_COLORS[cfg], lw=SIZE_LW[cfg],
                label=SIZE_LABELS[cfg])
    ax.set_xscale("log")
    ax.set_xlim(0.8, 3e4)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))
    ax.set_ylim(-0.02, 0.5)
    ax.set_xlabel("Training step", labelpad=3)
    ax.set_ylabel(r"$R^2(\epsilon_{M|L})$ (best layer)", labelpad=3)
    ax.set_title(r"Residual $\epsilon_{M|L}$ decodability", pad=4)
    ax.grid(True)
    ax.legend(fontsize=7.5, title="Model size", title_fontsize=8.5)

    # Right: cos(L, eps) over training — want it to stay near 0
    ax = axes[1]
    for cfg in CONFIGS:
        if cfg not in frames: continue
        df = frames[cfg]
        # use last layer cosines
        n_layers = df["layer"].max()
        sub = df.filter(pl.col("layer") == n_layers).sort("step")
        steps = sub["step"].to_numpy()
        cos_Le = sub["cos_Le"].to_numpy()
        ax.plot(steps, cos_Le, color=SIZE_COLORS[cfg], lw=SIZE_LW[cfg],
                label=SIZE_LABELS[cfg])
    ax.axhline(0, color="#bbb", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 3e4)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))
    ax.set_ylim(-0.3, 1.05)
    ax.set_xlabel("Training step", labelpad=3)
    ax.set_ylabel(r"$\cos(v_L, v_\epsilon)$ (last layer)", labelpad=3)
    ax.set_title(r"Luminosity–residual alignment", pad=4)
    ax.grid(True)
    ax.legend(fontsize=7.5, title="Model size", title_fontsize=8.5)

    for fmt in ("pdf","png"):
        fig.savefig(os.path.join(out_dir, f"residual_figC_emergence.{fmt}"))
    plt.close(fig)
    print("Saved residual_figC_emergence")

# ---------------------------------------------------------------------------
# Fig D: Layer dependence — R² by layer, columns = model size
# ---------------------------------------------------------------------------
def figD_layer_dependence(frames, out_dir):
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8),
                             constrained_layout=True, sharey=True)

    for col, cfg in enumerate(CONFIGS):
        ax = axes[col]
        if cfg not in frames:
            ax.set_visible(False); continue
        df = frames[cfg]
        step = final_step(df)
        sub = df.filter(pl.col("step") == step).sort("layer")
        layers = sub["layer"].to_numpy()

        ax.plot(layers, sub["r2_L"].to_numpy(),   "o-", color=C_L,
                lw=1.6, ms=5, label=r"$R^2(L_r)$")
        ax.plot(layers, sub["r2_M"].to_numpy(),   "s-", color=C_M,
                lw=1.6, ms=5, label=r"$R^2(\log M_\star)$")
        ax.plot(layers, sub["r2_eps"].to_numpy(), "^-", color=C_EPS,
                lw=1.6, ms=5, label=r"$R^2(\epsilon_{M|L})$")
        ax.plot(layers, sub["r2_ssfr"].to_numpy(),"v-", color=C_S,
                lw=1.2, ms=4, alpha=0.7, label=r"$R^2(\log\,\mathrm{sSFR})$")

        ax.set_xticks(layers)
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("Layer", labelpad=3)
        if col == 0:
            ax.set_ylabel(r"$R^2$ (final checkpoint)", labelpad=3)
        ax.set_title(SIZE_LABELS[cfg], pad=4)
        ax.grid(True)
        if col == 0:
            ax.legend(fontsize=6.5, loc="lower right")

    for fmt in ("pdf","png"):
        fig.savefig(os.path.join(out_dir, f"residual_figD_layer_dependence.{fmt}"))
    plt.close(fig)
    print("Saved residual_figD_layer_dependence")

# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",   default="figures/residual")
    parser.add_argument("--cache_dir", default="/tmp/residual_cache")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    frames = load_all(cache_dir=args.cache_dir)

    if not frames:
        print("No data found. Run reprobe_residual.py first.")
        return

    figA_capacity_scaling(frames, args.out_dir)
    figB_direction_geometry(frames, args.out_dir)
    figC_emergence(frames, args.out_dir)
    figD_layer_dependence(frames, args.out_dir)
    print(f"\nDone. Figures in {args.out_dir}/")

if __name__ == "__main__":
    main()
