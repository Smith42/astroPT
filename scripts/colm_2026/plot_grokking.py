"""
plot_grokking.py -- Physical emergence figures for AstroPT.
Three targets: band magnitude (easy), stellar mass (hard), sSFR (extra hard).
No colored backgrounds. Larger fonts for Overleaf.
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

ALL_CONFIGS = [f"{o}_{t}_{s}" for o in ("ar","mae")
               for t in ("aim","affine") for s in ("001M","021M","100M")]
SIZE_ORDER  = ["001M","021M","100M"]
SIZE_LABELS = {"001M":"1M","021M":"21M","100M":"100M"}
SIZE_NM     = {"001M":1,"021M":21,"100M":100}

TARGETS = [
    ("mag_r_desi",      r"Band magnitude ($r$)",                   "Easy"),
    ("mass_med_photoz", r"Stellar mass ($\log M_\star$)",          "Hard"),
    ("ssfr_med_photoz", r"Specific SFR ($\log\,\mathrm{sSFR}$)", "Extra hard"),
]

C_AR  = "#1f77b4"
C_MAE = "#d62728"
OBJ_COLORS = {"ar": C_AR, "mae": C_MAE}
TOK_LS     = {"aim": "-", "affine": "--"}
SIZE_LW    = {"001M": 0.9, "021M": 1.5, "100M": 2.1}
SIZE_ALPHA = {"001M": 0.45, "021M": 0.72, "100M": 1.0}

ABS_THRESH = 0.5
REL_THRESH = 0.8

# ---------------------------------------------------------------------------
def _style():
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Helvetica Neue","Helvetica","Arial","DejaVu Sans"],
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "figure.titlesize":   12,
        "axes.facecolor":     "white",
        "figure.facecolor":   "white",
        "axes.edgecolor":     "#444",
        "axes.labelcolor":    "black",
        "xtick.color":        "#444",
        "ytick.color":        "#444",
        "text.color":         "black",
        "grid.color":         "#e0e0e0",
        "grid.linewidth":     0.6,
        "grid.linestyle":     ":",
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "xtick.major.size":   3.5,
        "ytick.major.size":   3.5,
        "xtick.major.width":  0.7,
        "ytick.major.width":  0.7,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.8,
        "legend.frameon":     True,
        "legend.framealpha":  0.93,
        "legend.edgecolor":   "#ccc",
        "legend.handlelength": 2.2,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.04,
    })

# ---------------------------------------------------------------------------
def load_data(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    api = HfApi(token=HF_TOKEN)
    frames = []
    for cfg in tqdm(ALL_CONFIGS, desc="loading", ncols=60):
        local = os.path.join(cache_dir, f"{cfg}.parquet")
        if not os.path.exists(local):
            try:
                dl = api.hf_hub_download(
                    repo_id=RESULTS_REPO,
                    filename=f"results_kfold/{cfg}/results.parquet",
                    repo_type="dataset", token=HF_TOKEN)
                shutil.copy(dl, local)
            except Exception as e:
                print(f"  skip {cfg}: {e}"); continue
        if os.path.exists(local):
            frames.append(pl.read_parquet(local))
    return pl.concat(frames)

def best_layer(df, config, param):
    sub = df.filter((pl.col("config")==config) & (pl.col("param")==param))
    if sub.is_empty(): return None, None, None
    best = (sub.group_by("step")
              .agg(pl.col("r2_mean").max().alias("r2"),
                   pl.col("r2_std").first().alias("std"))
              .sort("step"))
    return best["step"].to_numpy(), best["r2"].to_numpy(), best["std"].to_numpy()

def abs_emergence(steps, r2s, thresh=ABS_THRESH, min_stable=3):
    for i, (s, r) in enumerate(zip(steps, r2s)):
        if r >= thresh:
            future = r2s[i:i+min_stable+1]
            if len(future) >= min_stable and np.mean(future >= thresh*0.95) >= 0.75:
                return s
    return None

def rel_emergence(steps, r2s, frac=REL_THRESH, min_stable=3):
    r0, rf = r2s[0], r2s[-1]
    if rf - r0 < 0.05: return None
    thr = r0 + frac*(rf-r0)
    for i, (s, r) in enumerate(zip(steps, r2s)):
        if r >= thr:
            future = r2s[i:i+min_stable+1]
            if len(future) >= min_stable and np.mean(future >= thr*0.95) >= 0.75:
                return s
    return None

def _fmt_x(ax):
    ax.set_xscale("log"); ax.set_xlim(0.8, 4e4)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x,_: f"{int(x/1000)}k" if x>=1000 else str(int(x))))

def _legend_els():
    return [
        Line2D([0],[0], color=C_AR,  lw=2.0, label="AR"),
        Line2D([0],[0], color=C_MAE, lw=2.0, label="MAE"),
        Line2D([0],[0], color="#888", lw=SIZE_LW["001M"], alpha=SIZE_ALPHA["001M"], label="1M"),
        Line2D([0],[0], color="#888", lw=SIZE_LW["021M"], alpha=SIZE_ALPHA["021M"], label="21M"),
        Line2D([0],[0], color="#888", lw=SIZE_LW["100M"], alpha=SIZE_ALPHA["100M"], label="100M"),
        Line2D([0],[0], color="#888", lw=1.5, ls="-",  label="AIM"),
        Line2D([0],[0], color="#888", lw=1.5, ls="--", label="Affine"),
    ]

# ---------------------------------------------------------------------------
# Fig 1: Emergence curves
# ---------------------------------------------------------------------------
def fig1_emergence_curves(df, out_dir):
    """Single panel: all 3 targets on one axes.
    Target differentiated by line style: easy=solid, hard=dashed, extra-hard=dotted.
    Objective (AR/MAE) by color. Model size by line width.
    AIM tokeniser only (cleaner signal).
    """
    _style()
    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)

    TARGET_LS = {
        "mag_r_desi":      ("-",   "Easy"),
        "mass_med_photoz": ("--",  "Hard"),
        "ssfr_med_photoz": (":",   "Extra hard"),
    }

    ax.axhline(ABS_THRESH, color="#aaa", lw=0.8, ls=":", zorder=1)
    ax.text(1.2, ABS_THRESH+0.02, f"$R^2={ABS_THRESH}$",
            fontsize=8.5, color="#aaa", va="bottom")

    for obj in ("ar", "mae"):
        for size in SIZE_ORDER:
            for param, (ls, dlabel) in TARGET_LS.items():
                steps, r2, _ = best_layer(df, f"{obj}_aim_{size}", param)
                if steps is None: continue
                ax.plot(steps, r2,
                        color=OBJ_COLORS[obj],
                        lw=SIZE_LW[size],
                        alpha=SIZE_ALPHA[size],
                        ls=ls, zorder=2)

    _fmt_x(ax)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Training step", labelpad=3)
    ax.set_ylabel(r"$R^2$ (best layer, AIM tokeniser)", labelpad=3)
    ax.grid(True)

    # Build legend with all three dimensions
    legend_els = [
        # Objective (color)
        Line2D([0],[0], color=C_AR,  lw=2.0, ls="-", label="AR"),
        Line2D([0],[0], color=C_MAE, lw=2.0, ls="-", label="MAE"),
        # Target difficulty (line style)
        Line2D([0],[0], color="#555", lw=1.8, ls="-",  label="Easy (mag $r$)"),
        Line2D([0],[0], color="#555", lw=1.8, ls="--", label=r"Hard ($\log M_\star$)"),
        Line2D([0],[0], color="#555", lw=1.8, ls=":",  label=r"Extra hard ($\log$ sSFR)"),
        # Model size (line width)
        Line2D([0],[0], color="#888", lw=SIZE_LW["001M"], alpha=SIZE_ALPHA["001M"], label="1M"),
        Line2D([0],[0], color="#888", lw=SIZE_LW["021M"], alpha=SIZE_ALPHA["021M"], label="21M"),
        Line2D([0],[0], color="#888", lw=SIZE_LW["100M"], alpha=SIZE_ALPHA["100M"], label="100M"),
    ]
    fig.legend(handles=legend_els, fontsize=8.5,
               loc="center left", bbox_to_anchor=(1.01, 0.5),
               frameon=True, ncol=1, handlelength=2.5,
               framealpha=0.95, edgecolor="#ccc")

    for fmt in ("pdf","png"):
        fig.savefig(os.path.join(out_dir, f"grokking_fig1_emergence_curves.{fmt}"))
    plt.close(fig); print("Saved fig1_emergence_curves")

# ---------------------------------------------------------------------------
# Fig 2: Emergence ladder
# ---------------------------------------------------------------------------
def fig2_emergence_ladder(df, out_dir):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), constrained_layout=True)
    params      = [t[0] for t in TARGETS]
    param_labels= [t[1] for t in TARGETS]
    N = len(params)

    for ax, (metric_fn, metric_label) in zip(axes, [
        (abs_emergence,  f"Absolute: $R^2 \\geq {ABS_THRESH}$"),
        (rel_emergence,  "Relative: 80\\% of total gain"),
    ]):
        for obj, oc in [("ar", C_AR), ("mae", C_MAE)]:
            for size in SIZE_ORDER:
                cfg = f"{obj}_aim_{size}"
                xs = []
                for param in params:
                    steps, r2s, _ = best_layer(df, cfg, param)
                    if steps is None or r2s is None: xs.append(np.nan); continue
                    t = metric_fn(steps, r2s)
                    xs.append(t if t is not None else np.nan)
                y = np.arange(N)
                ms = SIZE_NM[size]*0.6 + 20
                ax.scatter(xs, y, s=ms, color=oc,
                           alpha=SIZE_ALPHA[size], zorder=3)
                valid = [(xi,yi) for xi,yi in zip(xs,y) if not np.isnan(xi)]
                if len(valid) > 1:
                    xv, yv = zip(*valid)
                    ax.plot(xv, yv, color=oc, alpha=SIZE_ALPHA[size], lw=1.0, zorder=2)

        _fmt_x(ax)
        ax.set_yticks(range(N))
        ax.set_yticklabels(param_labels, fontsize=9)
        ax.set_xlabel("Training step at emergence", labelpad=3)
        ax.set_title(metric_label, pad=4)
        ax.grid(True, axis="x", alpha=0.4)
        ax.invert_yaxis()
        ax.axvline(3e4, color="#aaa", lw=0.7, ls=":", zorder=1)
        ax.text(3.2e4, N-0.6, "End of\ntraining", fontsize=7, color="#aaa",
                va="top", ha="left")

    handles = []
    for obj, oc in [("ar",C_AR),("mae",C_MAE)]:
        for size in SIZE_ORDER:
            ms = SIZE_NM[size]*0.6+20
            handles.append(plt.scatter([],[],s=ms,color=oc,
                alpha=SIZE_ALPHA[size], label=f"{obj.upper()} {SIZE_LABELS[size]}"))
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5,-0.15), ncol=6,
               fontsize=8.5, frameon=True, columnspacing=0.8)
    for fmt in ("pdf","png"):
        fig.savefig(os.path.join(out_dir, f"grokking_fig2_emergence_ladder.{fmt}"))
    plt.close(fig); print("Saved fig2_emergence_ladder")

# ---------------------------------------------------------------------------
# Fig 3: AR vs MAE comparison
# ---------------------------------------------------------------------------
def fig3_ar_vs_mae(df, out_dir):
    _style()
    fig, axes = plt.subplots(3, 3, figsize=(7.1, 6.5),
                             constrained_layout=True, sharey="row")
    for row, (param, plabel, dlabel) in enumerate(TARGETS):
        for col, size in enumerate(SIZE_ORDER):
            ax = axes[row, col]
            ax.axhline(ABS_THRESH, color="#aaa", lw=0.7, ls=":", zorder=1)
            for obj in ("ar","mae"):
                for tok in ("aim","affine"):
                    steps, r2, std = best_layer(df, f"{obj}_{tok}_{size}", param)
                    if steps is None: continue
                    ax.plot(steps, r2, color=OBJ_COLORS[obj],
                            lw=1.4, ls=TOK_LS[tok], alpha=0.85, zorder=2)
                    ax.fill_between(steps, r2-std, r2+std,
                                    color=OBJ_COLORS[obj], alpha=0.07, zorder=1)
            _fmt_x(ax)
            ax.set_ylim(-0.02, 1.02)
            ax.set_yticks([0, 0.5, 1.0])
            ax.grid(True)
            if row == 0: ax.set_title(SIZE_LABELS[size], pad=4)
            if row == 2: ax.set_xlabel("Training step", labelpad=2)
            if col == 0:
                ax.set_ylabel(r"$R^2$ (best layer)", labelpad=2)
                ax.annotate(f"{dlabel}\n{plabel}",
                            xy=(-0.52,0.5), xycoords="axes fraction",
                            fontsize=8.5, rotation=90, va="center", ha="center")

    legend_els = [
        Line2D([0],[0], color=C_AR,  lw=2.0, label="AR"),
        Line2D([0],[0], color=C_MAE, lw=2.0, label="MAE"),
        Line2D([0],[0], color="#777", lw=1.5, ls="-",  label="AIM"),
        Line2D([0],[0], color="#777", lw=1.5, ls="--", label="Affine"),
    ]
    fig.legend(handles=legend_els, loc="lower center",
               bbox_to_anchor=(0.5,-0.05), ncol=4,
               fontsize=9, frameon=True, columnspacing=1.0)
    for fmt in ("pdf","png"):
        fig.savefig(os.path.join(out_dir, f"grokking_fig3_ar_vs_mae.{fmt}"))
    plt.close(fig); print("Saved fig3_ar_vs_mae")

# ---------------------------------------------------------------------------
# Fig 4: Layer heatmaps
# ---------------------------------------------------------------------------
def fig4_layer_heatmaps(df, out_dir, obj="ar", tok="aim"):
    _style()
    # Override to larger fonts for this dense figure
    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
    })
    fig, axes = plt.subplots(3, 3, figsize=(10.0, 8.5), constrained_layout=True)
    im_ref = None
    for row, (param, plabel, dlabel) in enumerate(TARGETS):
        for col, size in enumerate(SIZE_ORDER):
            ax = axes[row, col]
            cfg = f"{obj}_{tok}_{size}"
            sub = df.filter((pl.col("config")==cfg) & (pl.col("param")==param))
            if sub.is_empty():
                ax.set_axis_off()
                ax.text(0.5,0.5,"—",ha="center",va="center",
                        transform=ax.transAxes,fontsize=9,color="#aaa")
                continue
            steps_s = sorted(sub["step"].unique().to_list())
            layers_s= sorted(sub["layer"].unique().to_list())
            mat = np.full((len(steps_s),len(layers_s)), np.nan)
            si={s:i for i,s in enumerate(steps_s)}
            li={l:i for i,l in enumerate(layers_s)}
            for r in sub.iter_rows(named=True):
                mat[si[r["step"]],li[r["layer"]]] = r["r2_mean"]
            im = ax.imshow(mat.T, aspect="auto", origin="lower",
                           cmap="RdYlGn", vmin=0, vmax=1,
                           extent=[0,len(steps_s),-0.5,len(layers_s)-0.5])
            if im_ref is None: im_ref = im
            ax.tick_params(labelsize=10)
            if row == 0: ax.set_title(SIZE_LABELS[size], pad=3, fontsize=13)
            if row == 2: ax.set_xlabel("Checkpoint index", labelpad=2)
            if col == 0:
                ax.set_ylabel("Layer", labelpad=2)
                ax.annotate(f"{dlabel}: {plabel}",
                            xy=(-0.52,0.5), xycoords="axes fraction",
                            fontsize=11, rotation=90, va="center", ha="center")
    if im_ref:
        cb = fig.colorbar(im_ref, ax=axes, shrink=0.55, pad=0.02, aspect=25)
        cb.set_label(r"Probe $R^2$", fontsize=13)
        cb.ax.tick_params(labelsize=11)
    for fmt in ("pdf","png"):
        fig.savefig(os.path.join(out_dir,
            f"grokking_fig4_layer_heatmaps_{obj}_{tok}.{fmt}"))
    plt.close(fig); print(f"Saved fig4_layer_heatmaps_{obj}_{tok}")

# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",   default="figures/grokking")
    parser.add_argument("--cache_dir", default="/tmp/kfold_cache")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = load_data(cache_dir=args.cache_dir)
    fig1_emergence_curves(df, args.out_dir)
    fig2_emergence_ladder(df, args.out_dir)
    fig3_ar_vs_mae(df, args.out_dir)
    fig4_layer_heatmaps(df, args.out_dir, obj="ar",  tok="aim")
    fig4_layer_heatmaps(df, args.out_dir, obj="mae", tok="aim")
    print(f"\nDone. {args.out_dir}/")

if __name__ == "__main__":
    main()
