"""
plot_scaling.py  --  AstroPT scaling study figures (publication quality).
No main title in figures — captions will be provided separately.
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

ALL_CONFIGS = [
    f"{obj}_{tok}_{size}"
    for obj  in ("ar", "mae")
    for tok  in ("aim", "affine")
    for size in ("001M", "021M", "100M")
]
SIZE_ORDER  = ["001M", "021M", "100M"]
SIZE_LABELS = {"001M": "1M", "021M": "21M", "100M": "100M"}
SIZE_PARAMS = {"001M": 1, "021M": 21, "100M": 100}

KEY_PARAMS = [
    "redshift", "mass_med_photoz", "ssfr_med_photoz", "galaxy_size",
    "smooth-or-featured_smooth_fraction", "mag_r_desi", "est_petro_th50",
]
PARAM_LABELS = {
    "redshift":                           r"Redshift $z$",
    "mass_med_photoz":                    r"$\log M_\star$",
    "ssfr_med_photoz":                    r"$\log\,\mathrm{sSFR}$",
    "galaxy_size":                        "Galaxy size",
    "smooth-or-featured_smooth_fraction": "Morphology",
    "mag_r_desi":                         r"App. mag $r$",
    "est_petro_th50":                     r"$r_{50}$ (arcsec)",
}

C_AR  = "#1f77b4"
C_MAE = "#d62728"
OBJ_COLOR = {"ar": C_AR, "mae": C_MAE}
TOK_LS    = {"aim": "-", "affine": "--"}
TOK_MK    = {"aim": "o", "affine": "s"}
SIZE_LW   = {"001M": 0.9, "021M": 1.5, "100M": 2.1}
SIZE_ALPHA = {"001M": 0.45, "021M": 0.72, "100M": 1.0}

def _style():
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
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
        "legend.edgecolor":   "#cccccc",
        "legend.handlelength": 2.2,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.04,
        "lines.linewidth":    1.3,
    })

def load_all(cache_dir):
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
    df = pl.concat(frames)
    print(f"Loaded {df.shape[0]:,} rows, {df['config'].n_unique()} configs")
    return df

def best_layer_r2(df, config, param):
    sub = df.filter((pl.col("config") == config) & (pl.col("param") == param))
    if sub.is_empty(): return None, None, None
    best = (sub.group_by("step")
              .agg(pl.col("r2_mean").max().alias("r2"),
                   pl.col("r2_std").first().alias("std"))
              .sort("step"))
    return best["step"].to_numpy(), best["r2"].to_numpy(), best["std"].to_numpy()

def final_r2_by_size(df, obj, tok, param):
    sizes, r2s, stds = [], [], []
    for size in SIZE_ORDER:
        sub = df.filter((pl.col("config") == f"{obj}_{tok}_{size}") &
                        (pl.col("param") == param))
        if sub.is_empty(): continue
        last = sub["step"].max()
        best = (sub.filter(pl.col("step") == last)
                   .sort("r2_mean", descending=True).row(0, named=True))
        sizes.append(SIZE_PARAMS[size]); r2s.append(best["r2_mean"]); stds.append(best["r2_std"])
    return sizes, r2s, stds

def _fmt_x(ax):
    ax.set_xscale("log")
    ax.set_xlim(0.8, 4e4)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))

def _legend(fig, elements, **kw):
    fig.legend(handles=elements, frameon=True,
               edgecolor="#cccccc", framealpha=0.93,
               fontsize=7.5, **kw)

# ---------------------------------------------------------------------------
# Fig 1: Scaling curves
# ---------------------------------------------------------------------------
def fig_scaling_curves(df, out_dir):
    _style()
    params = [p for p in KEY_PARAMS if p in df["param"].unique().to_list()]
    ncols = 4; nrows = -(-len(params) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.1, 2.1 * nrows),
                             constrained_layout=True)
    axes = axes.flatten()

    for ax, param in zip(axes, params):
        for obj in ("ar", "mae"):
            for tok in ("aim", "affine"):
                for size in SIZE_ORDER:
                    steps, r2, _ = best_layer_r2(df, f"{obj}_{tok}_{size}", param)
                    if steps is None: continue
                    ax.plot(steps, r2, color=OBJ_COLOR[obj],
                            lw=SIZE_LW[size], alpha=SIZE_ALPHA[size],
                            ls=TOK_LS[tok])
        _fmt_x(ax)
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlabel("Training step", labelpad=2)
        ax.set_ylabel(r"$R^2$", labelpad=2)
        ax.set_title(PARAM_LABELS.get(param, param), pad=3)
        ax.grid(True)

    for ax in axes[len(params):]: ax.set_visible(False)

    els = [
        Line2D([0],[0], color=C_AR,  lw=2.0, label="AR"),
        Line2D([0],[0], color=C_MAE, lw=2.0, label="MAE"),
        Line2D([0],[0], color="#777", lw=0.9, alpha=0.45, label="1M"),
        Line2D([0],[0], color="#777", lw=1.5, alpha=0.72, label="21M"),
        Line2D([0],[0], color="#777", lw=2.1, alpha=1.0,  label="100M"),
        Line2D([0],[0], color="#777", lw=1.5, ls="-",  label="AIM"),
        Line2D([0],[0], color="#777", lw=1.5, ls="--", label="Affine"),
    ]
    if len(params) < len(axes):
        axes[len(params)].legend(handles=els, loc="center",
                                 fontsize=7.5, frameon=False)
    else:
        _legend(fig, els, loc="lower right",
                bbox_to_anchor=(1.0, 0.0), ncol=1)

    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig1_scaling_curves.{fmt}"))
    plt.close(fig); print("Saved fig1_scaling_curves")

# ---------------------------------------------------------------------------
# Fig 2: Final R² vs model size
# ---------------------------------------------------------------------------
def fig_size_scaling(df, out_dir):
    _style()
    params = [p for p in KEY_PARAMS if p in df["param"].unique().to_list()]
    ncols = 4; nrows = -(-len(params) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.1, 2.1 * nrows),
                             constrained_layout=True)
    axes = axes.flatten()

    for ax, param in zip(axes, params):
        for obj in ("ar", "mae"):
            for tok in ("aim", "affine"):
                sizes, r2s, stds = final_r2_by_size(df, obj, tok, param)
                if not r2s: continue
                xa = np.array(sizes); r2a = np.array(r2s); sa = np.array(stds)
                ax.plot(xa, r2a, color=OBJ_COLOR[obj], ls=TOK_LS[tok],
                        marker=TOK_MK[tok], lw=1.6, ms=5,
                        label=f"{obj.upper()}/{tok.capitalize()}")
                ax.fill_between(xa, r2a-sa, r2a+sa,
                                color=OBJ_COLOR[obj], alpha=0.10)
        ax.set_xscale("log")
        ax.set_xticks([1, 21, 100]); ax.set_xticklabels(["1M","21M","100M"])
        ax.set_ylim(-0.02, 1.02); ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlabel("Model size", labelpad=2)
        ax.set_ylabel(r"$R^2$", labelpad=2)
        ax.set_title(PARAM_LABELS.get(param, param), pad=3)
        ax.grid(True)

    for ax in axes[len(params):]: ax.set_visible(False)

    els = [
        Line2D([0],[0], color=C_AR,  lw=1.8, ls="-",  marker="o", ms=5, label="AR / AIM"),
        Line2D([0],[0], color=C_AR,  lw=1.8, ls="--", marker="s", ms=5, label="AR / Affine"),
        Line2D([0],[0], color=C_MAE, lw=1.8, ls="-",  marker="o", ms=5, label="MAE / AIM"),
        Line2D([0],[0], color=C_MAE, lw=1.8, ls="--", marker="s", ms=5, label="MAE / Affine"),
    ]
    if len(params) < len(axes):
        axes[len(params)].legend(handles=els, loc="center",
                                 fontsize=7.5, frameon=False)
    else:
        _legend(fig, els, loc="lower right",
                bbox_to_anchor=(1.0, 0.0), ncol=1)

    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig2_size_scaling.{fmt}"))
    plt.close(fig); print("Saved fig2_size_scaling")

# ---------------------------------------------------------------------------
# Fig 3: Layer heatmap (4 rows × 3 cols: configs × sizes)
# ---------------------------------------------------------------------------
def fig_layer_heatmap(df, out_dir, param="redshift"):
    _style()
    row_configs = [("ar","aim"), ("ar","affine"), ("mae","aim"), ("mae","affine")]
    row_labels  = ["AR / AIM", "AR / Affine", "MAE / AIM", "MAE / Affine"]

    fig, axes = plt.subplots(4, 3, figsize=(7.1, 6.0), constrained_layout=True)
    im_ref = None

    for ri, (obj, tok) in enumerate(row_configs):
        for ci, size in enumerate(SIZE_ORDER):
            ax = axes[ri, ci]
            cfg = f"{obj}_{tok}_{size}"
            sub = df.filter((pl.col("config") == cfg) & (pl.col("param") == param))
            if sub.is_empty():
                ax.set_axis_off()
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#aaa")
                continue

            steps_s  = sorted(sub["step"].unique().to_list())
            layers_s = sorted(sub["layer"].unique().to_list())
            mat = np.full((len(steps_s), len(layers_s)), np.nan)
            si = {s: i for i, s in enumerate(steps_s)}
            li = {l: i for i, l in enumerate(layers_s)}
            for r in sub.iter_rows(named=True):
                mat[si[r["step"]], li[r["layer"]]] = r["r2_mean"]

            im = ax.imshow(mat.T, aspect="auto", origin="lower",
                           cmap="RdYlGn", vmin=0, vmax=1,
                           extent=[0, len(steps_s), -0.5, len(layers_s)-0.5])
            if im_ref is None: im_ref = im
            ax.tick_params(labelsize=6)

            if ri == 0: ax.set_title(SIZE_LABELS[size], fontsize=9, pad=3)
            if ri == 3: ax.set_xlabel("Checkpoint index", fontsize=7.5, labelpad=2)
            if ci == 0:
                ax.set_ylabel("Layer", fontsize=7.5, labelpad=2)
                ax.annotate(row_labels[ri],
                            xy=(-0.45, 0.5), xycoords="axes fraction",
                            fontsize=8, rotation=90, va="center", ha="center")

    if im_ref:
        cb = fig.colorbar(im_ref, ax=axes, shrink=0.5, pad=0.02, aspect=25)
        cb.set_label(r"Probe $R^2$", fontsize=10)
        cb.ax.tick_params(labelsize=9)

    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig3_layer_heatmap_{param}.{fmt}"))
    plt.close(fig); print(f"Saved fig3_layer_heatmap_{param}")

# ---------------------------------------------------------------------------
# Fig 4: Summary bars
# ---------------------------------------------------------------------------
def fig_summary_bars(df, out_dir):
    _style()
    FULL_LABELS = {
        "redshift": r"Redshift $z$", "photo_z": r"Photo-$z$",
        "mass_med_photoz": r"$\log M_\star$", "ssfr_med_photoz": r"$\log$ sSFR",
        "mag_g_desi": r"Mag $g$", "mag_r_desi": r"Mag $r$", "mag_z_desi": r"Mag $z$",
        "mag_abs_g_photoz": r"$M_g$", "mag_abs_r_photoz": r"$M_r$",
        "mag_abs_z_photoz": r"$M_z$",
        "est_petro_th50": r"$r_{50}$ (arcsec)", "est_petro_th50_kpc": r"$r_{50}$ (kpc)",
        "galaxy_size": "Galaxy size",
        "smooth-or-featured_smooth_fraction":         "Morph: smooth",
        "smooth-or-featured_featured-or-disk_fraction": "Morph: featured",
        "smooth-or-featured_artifact_fraction":       "Morph: artifact",
        "merging_none_fraction":              "Merging: none",
        "merging_minor-disturbance_fraction": "Merging: minor",
        "merging_major-disturbance_fraction": "Merging: major",
        "merging_merger_fraction":            "Merging: merger",
        "how-rounded_round_fraction":         "Shape: round",
        "how-rounded_in-between_fraction":    "Shape: in-between",
        "how-rounded_cigar-shaped_fraction":  "Shape: cigar",
    }
    all_params = df["param"].unique().to_list()
    configs_4 = [("ar","aim"),("ar","affine"),("mae","aim"),("mae","affine")]
    colors4   = [C_AR, C_AR, C_MAE, C_MAE]
    hatches4  = ["", "///", "", "///"]
    labels4   = ["AR/AIM","AR/Affine","MAE/AIM","MAE/Affine"]

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 6.0),
                             constrained_layout=True, sharey=False)

    for col, size in enumerate(SIZE_ORDER):
        ax = axes[col]
        rows = []
        for param in all_params:
            r2s = []
            for obj, tok in configs_4:
                sub = df.filter((pl.col("config") == f"{obj}_{tok}_{size}") &
                                (pl.col("param") == param))
                if sub.is_empty(): r2s.append(0.0); continue
                last = sub["step"].max()
                best = (sub.filter(pl.col("step") == last)
                           .sort("r2_mean", descending=True).row(0, named=True))
                r2s.append(best["r2_mean"])
            rows.append((param, r2s))

        rows.sort(key=lambda x: np.mean(x[1]), reverse=True)
        params_sorted = [r[0] for r in rows]
        mat = np.array([r[1] for r in rows])

        y = np.arange(len(params_sorted))
        w = 0.18
        for i, (c, h) in enumerate(zip(colors4, hatches4)):
            ax.barh(y + (i-1.5)*w, mat[:,i], w,
                    color=c, alpha=(0.88 if h=="" else 0.52),
                    hatch=h, edgecolor="white" if h=="" else c,
                    linewidth=0.3)
        ax.set_yticks(y)
        ax.set_yticklabels([FULL_LABELS.get(p, p) for p in params_sorted],
                           fontsize=6.5)
        ax.set_xlabel(r"$R^2$ (best layer)", fontsize=8)
        ax.set_title(f"{SIZE_LABELS[size]} parameters", fontsize=9, pad=4)
        ax.set_xlim(0, 1.0); ax.set_ylim(-0.5, len(params_sorted)-0.5)
        ax.grid(True, axis="x", alpha=0.4)
        ax.axvline(0.5, color="#999", lw=0.7, ls=":")

    # single shared legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, alpha=(0.88 if h=="" else 0.52),
                     hatch=h, edgecolor="white" if h=="" else c,
                     linewidth=0.3, label=lbl)
               for c, h, lbl in zip(colors4, hatches4, labels4)]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), ncol=4,
               fontsize=7.5, frameon=True, edgecolor="#ccc")

    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig4_summary_bars.{fmt}"))
    plt.close(fig); print("Saved fig4_summary_bars")

# ---------------------------------------------------------------------------
# Fig 5: AR vs MAE comparison
# ---------------------------------------------------------------------------
def fig_obj_comparison(df, out_dir):
    _style()
    COMP_PARAMS = [
        ("redshift",        r"Redshift $z$"),
        ("mass_med_photoz", r"$\log M_\star$"),
        ("ssfr_med_photoz", r"$\log\,\mathrm{sSFR}$"),
        ("galaxy_size",     "Galaxy size"),
    ]
    params = [(p, l) for p, l in COMP_PARAMS if p in df["param"].unique().to_list()]
    fig, axes = plt.subplots(len(params), 3,
                             figsize=(7.1, 1.8 * len(params)),
                             constrained_layout=True, sharey="row")
    if len(params) == 1: axes = axes[np.newaxis, :]

    for row, (param, plabel) in enumerate(params):
        for col, size in enumerate(SIZE_ORDER):
            ax = axes[row, col]
            for obj in ("ar", "mae"):
                for tok in ("aim", "affine"):
                    steps, r2, std = best_layer_r2(df, f"{obj}_{tok}_{size}", param)
                    if steps is None: continue
                    ax.plot(steps, r2, color=OBJ_COLOR[obj],
                            ls=TOK_LS[tok], lw=1.5, alpha=0.85)
                    ax.fill_between(steps, r2-std, r2+std,
                                    color=OBJ_COLOR[obj], alpha=0.07)
            _fmt_x(ax)
            ax.set_ylim(-0.02, 1.02)
            ax.set_yticks([0, 0.5, 1.0])
            ax.grid(True)
            if row == 0: ax.set_title(SIZE_LABELS[size], fontsize=9, pad=3)
            if row == len(params)-1:
                ax.set_xlabel("Training step", fontsize=7.5, labelpad=2)
            if col == 0:
                ax.set_ylabel(r"$R^2$", fontsize=7.5, labelpad=2)
                ax.annotate(plabel,
                            xy=(-0.42, 0.5), xycoords="axes fraction",
                            fontsize=8, rotation=90, va="center", ha="center")

    els = [
        Line2D([0],[0], color=C_AR,  lw=2.0, label="AR"),
        Line2D([0],[0], color=C_MAE, lw=2.0, label="MAE"),
        Line2D([0],[0], color="#777", lw=1.5, ls="-",  label="AIM"),
        Line2D([0],[0], color="#777", lw=1.5, ls="--", label="Affine"),
    ]
    _legend(fig, els, loc="lower center", bbox_to_anchor=(0.5, -0.07),
            ncol=4, columnspacing=1.0)

    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig5_obj_comparison.{fmt}"))
    plt.close(fig); print("Saved fig5_obj_comparison")

# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",   default="figures")
    parser.add_argument("--cache_dir", default="/tmp/kfold_cache")
    parser.add_argument("--param",     default="redshift")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_all(cache_dir=args.cache_dir)

    fig_scaling_curves(df, args.out_dir)
    fig_size_scaling(df, args.out_dir)
    fig_layer_heatmap(df, args.out_dir, param=args.param)
    fig_summary_bars(df, args.out_dir)
    fig_obj_comparison(df, args.out_dir)
    print(f"\nDone. Figures in {args.out_dir}/")

if __name__ == "__main__":
    main()
