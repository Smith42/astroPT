"""
plot_geometry.py -- Galaxy relationship geometry figures.
Fig 1: Label-space correlations
Fig 2: Direction cosines across capacity
Fig 3: Training-time emergence
Fig 4: Layer dependence
"""
import os, shutil, json
import numpy as np
import polars as pl
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from huggingface_hub import HfApi
from tqdm import tqdm

RESULTS_REPO = "HCVYM5w6Gn/colm-results"
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
CONFIGS      = ["ar_aim_001M", "ar_aim_021M", "ar_aim_100M"]
SIZE_LABELS  = {"ar_aim_001M": "1M", "ar_aim_021M": "21M", "ar_aim_100M": "100M"}
SIZE_COLORS  = {"ar_aim_001M": "#aec7e8", "ar_aim_021M": "#1f77b4", "ar_aim_100M": "#08306b"}
SIZE_LW      = {"ar_aim_001M": 1.0, "ar_aim_021M": 1.6, "ar_aim_100M": 2.2}

def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue","Helvetica","Arial","DejaVu Sans"],
        "font.size":         10, "axes.titlesize":    11, "axes.labelsize":    10,
        "xtick.labelsize":    9, "ytick.labelsize":    9, "legend.fontsize":    9,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.edgecolor": "#444", "xtick.color": "#444", "ytick.color": "#444",
        "text.color": "black", "grid.color": "#e0e0e0",
        "grid.linewidth": 0.6, "grid.linestyle": ":",
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "legend.frameon": True,
        "legend.framealpha": 0.93, "legend.edgecolor": "#ccc",
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    })

def load_frames(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    api = HfApi(token=HF_TOKEN)
    frames = {}
    for cfg in tqdm(CONFIGS, desc="loading", ncols=60):
        local = os.path.join(cache_dir, f"{cfg}_residual_new.parquet")
        try:
            dl = api.hf_hub_download(repo_id=RESULTS_REPO,
                filename=f"results_residual/{cfg}/results.parquet",
                repo_type="dataset", token=HF_TOKEN)
            shutil.copy(dl, local)
            df = pl.read_parquet(local)
            if "cos_z_M" in df.columns:
                frames[cfg] = df
                print(f"  {cfg}: {df.shape[0]} rows, has geometry cols")
            else:
                print(f"  {cfg}: missing cos_z_M — old results, skip")
        except Exception as e:
            print(f"  skip {cfg}: {e}")
    return frames

def final_step(df): return df["step"].max()

def last_layer_val(df, col):
    step = final_step(df)
    n_layer = df["layer"].max()
    sub = df.filter((pl.col("step") == step) & (pl.col("layer") == n_layer))
    if sub.is_empty() or col not in sub.columns: return np.nan
    return float(sub[col][0])

def _scatter_panel(ax, x, y, xlabel, ylabel, color, n=3000):
    rng = np.random.default_rng(42)
    ok = np.isfinite(x) & np.isfinite(y)
    idx = rng.choice(np.where(ok)[0], min(n, ok.sum()), replace=False)
    r, _ = scipy_stats.pearsonr(x[idx], y[idx])
    ax.scatter(x[idx], y[idx], s=1.5, alpha=0.35, color=color,
               linewidths=0, rasterized=True)
    ax.set_xlabel(xlabel, labelpad=3)
    ax.set_ylabel(ylabel, labelpad=3)
    ax.text(0.05, 0.95, f"$r = {r:.3f}$", transform=ax.transAxes,
            fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", alpha=0.9))
    ax.grid(True)

def fig1_label_space(out_dir):
    _style()
    from datasets import load_dataset
    ds = load_dataset("Smith42/galaxies", revision="v2.0",
                      split="test", streaming=True)
    ds = ds.select_columns(["redshift","photo_z","mass_med_photoz",
                             "ssfr_med_photoz","mag_abs_r_photoz"])
    zs, ms, ss, ls = [], [], [], []
    for ex in ds:
        def _f(v):
            try: v=float(v); return v if (np.isfinite(v) and v>-90) else np.nan
            except: return np.nan
        z = _f(ex.get("redshift"))
        if not np.isfinite(z): z = _f(ex.get("photo_z"))
        zs.append(z); ms.append(_f(ex.get("mass_med_photoz")))
        ss.append(_f(ex.get("ssfr_med_photoz")))
        ls.append(-0.4*_f(ex.get("mag_abs_r_photoz")))
        if len(zs) >= 5000: break

    Z=np.array(zs); M=np.array(ms); S=np.array(ss); L=np.array(ls)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6), constrained_layout=True)
    _scatter_panel(axes[0], S, M, r"$\log\,\mathrm{sSFR}$", r"$\log M_\star$", "#9467bd")
    _scatter_panel(axes[1], Z, M, "Redshift $z$", r"$\log M_\star$", "#1f77b4")
    _scatter_panel(axes[2], L, M, r"$L_r = -0.4\,M_r$", r"$\log M_\star$", "#2ca02c")
    for fmt in ("pdf","png"):
        fig.savefig(os.path.join(out_dir, f"astropt_geometry_1_label_space.{fmt}"))
    plt.close(fig); print("Saved fig 1")

def fig2_cosines_by_size(frames, out_dir):
    _style()
    fig, ax = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
    cosine_keys = [
        ("cos_LM",     r"$\cos(v_L,v_M)$"),
        ("cos_sSFR_M", r"$\cos(v_\mathrm{sSFR},v_M)$"),
        ("cos_z_M",    r"$\cos(v_z,v_M)$"),
        ("cos_z_L",    r"$\cos(v_z,v_L)$"),
        ("cos_z_sSFR", r"$\cos(v_z,v_\mathrm{sSFR})$"),
    ]
    configs_ok = [c for c in CONFIGS if c in frames]
    n = len(configs_ok); xi = np.arange(len(cosine_keys)); w = 0.22
    for si, cfg in enumerate(configs_ok):
        vals = [last_layer_val(frames[cfg], k) for k,_ in cosine_keys]
        ax.bar(xi + (si - n/2 + 0.5)*w, vals, w,
               color=SIZE_COLORS[cfg], alpha=0.85, edgecolor="white",
               lw=0.3, label=SIZE_LABELS[cfg])
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(xi)
    ax.set_xticklabels([lbl for _,lbl in cosine_keys], fontsize=6.5)
    ax.set_ylabel("Cosine similarity (last layer, final ckpt)", labelpad=3)
    ax.set_ylim(-0.4, 1.1); ax.grid(True, axis="y")
    ax.legend(title="Model size", title_fontsize=8.5, fontsize=7)
    for fmt in ("pdf","png"):
        fig.savefig(os.path.join(out_dir, f"astropt_geometry_2_cosines_by_size.{fmt}"))
    plt.close(fig); print("Saved fig 2")

def fig3_emergence(frames, out_dir):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    for ax, col, ylabel in [
        (axes[0], "cos_sSFR_M", r"$\cos(v_\mathrm{sSFR},v_M)$"),
        (axes[1], "cos_z_M",    r"$\cos(v_z,v_M)$"),
    ]:
        for cfg in CONFIGS:
            if cfg not in frames or col not in frames[cfg].columns: continue
            df = frames[cfg]
            n_layer = df["layer"].max()
            sub = df.filter(pl.col("layer") == n_layer).sort("step")
            ax.plot(sub["step"].to_numpy(), sub[col].to_numpy(),
                    color=SIZE_COLORS[cfg], lw=SIZE_LW[cfg], label=SIZE_LABELS[cfg])
        ax.axhline(0, color="#bbb", lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_xlim(0.8, 3e4)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x/1000)}k" if x>=1000 else str(int(x))))
        ax.set_ylim(-0.5, 0.6); ax.set_xlabel("Training step", labelpad=3)
        ax.set_ylabel(ylabel, labelpad=3); ax.grid(True)
        ax.legend(title="Model size", title_fontsize=8.5, fontsize=7)
    for fmt in ("pdf","png"):
        fig.savefig(os.path.join(out_dir, f"astropt_geometry_3_emergence.{fmt}"))
    plt.close(fig); print("Saved fig 3")

def fig4_layer(frames, out_dir):
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8),
                             constrained_layout=True, sharey=True)
    PAIRS = [
        ("cos_LM",     r"$\cos(v_L,v_M)$",              "#2ca02c", "-"),
        ("cos_sSFR_M", r"$\cos(v_\mathrm{sSFR},v_M)$",  "#9467bd", "-"),
        ("cos_z_M",    r"$\cos(v_z,v_M)$",              "#1f77b4", "-"),
        ("cos_z_L",    r"$\cos(v_z,v_L)$",              "#17becf", "--"),
        ("cos_z_sSFR", r"$\cos(v_z,v_\mathrm{sSFR})$",  "#e377c2", "--"),
    ]
    for ci, cfg in enumerate(CONFIGS):
        ax = axes[ci]
        if cfg not in frames:
            ax.text(0.5,0.5,"no data",ha="center",va="center",
                    transform=ax.transAxes,color="#aaa"); ax.set_axis_off(); continue
        df = frames[cfg]
        sub = df.filter(pl.col("step") == final_step(df)).sort("layer")
        layers = sub["layer"].to_numpy()
        for col, lbl, color, ls in PAIRS:
            if col not in sub.columns: continue
            ax.plot(layers, sub[col].to_numpy(), color=color, lw=1.5,
                    ls=ls, marker="o", ms=4, label=lbl)
        ax.axhline(0, color="#bbb", lw=0.8, ls=":")
        ax.set_xticks(layers); ax.set_xlabel("Layer", labelpad=3)
        if ci == 0:
            ax.set_ylabel("Cosine similarity", labelpad=3)
            ax.legend(fontsize=5.5, loc="upper left")
        ax.set_title(SIZE_LABELS[cfg], pad=4)
        ax.grid(True); ax.set_ylim(-0.5, 1.1)
    for fmt in ("pdf","png"):
        fig.savefig(os.path.join(out_dir, f"astropt_geometry_4_by_layer.{fmt}"))
    plt.close(fig); print("Saved fig 4")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",   default="figures/geometry")
    parser.add_argument("--cache_dir", default="/tmp/geometry_cache")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    _style()
    print("Generating Fig 1 (label space)...")
    fig1_label_space(args.out_dir)
    print("Loading probe results...")
    frames = load_frames(cache_dir=args.cache_dir)
    if frames:
        fig2_cosines_by_size(frames, args.out_dir)
        fig3_emergence(frames, args.out_dir)
        fig4_layer(frames, args.out_dir)
    else:
        print("No geometry data yet — Figs 2-4 will generate once reprobe_residual finishes")
    print(f"\nDone. {args.out_dir}/")

if __name__ == "__main__":
    main()
