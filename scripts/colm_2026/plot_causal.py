"""
plot_causal.py  --  Publication figures for causal galaxy relationship tracing.

Four figures (no bold main titles — captions provided separately):
  Fig 1: Galaxy scaling relation (L_r vs log M*, coloured by R_M_given_L)
  Fig 2: Probe direction cosine similarities vs layer
  Fig 3: Probe R² curves (L_r, M, R_M_given_L, sSFR) by layer
  Fig 4: Causal trace heatmap — SuccessRate and RelationSelectivity by layer

USAGE:
    python plot_causal.py --results results/causal/ar_aim_021M_causal_results.json \
        --scatter results/causal/label_scatter.json --out_dir figures/causal/
"""

import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ---------------------------------------------------------------------------
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
        "grid.color":         "#e5e5e5",
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
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.04,
    })

PROBE_COLORS = {
    "L_r":        "#1f77b4",   # blue
    "mass":       "#2ca02c",   # green
    "R_M_given_L": "#d62728",  # red
    "ssfr":       "#9467bd",   # purple
}
SIGMA = {"L_r": 0.493, "mass": 0.695, "R_M_given_L": 0.285, "ssfr": 1.874}

PROBE_LABELS = {
    "L_r":        r"Luminosity $L_r$",
    "mass":       r"Stellar mass $\log M_\star$",
    "R_M_given_L": r"Mass residual $\epsilon_{M|L}$",
    "ssfr":       r"$\log\,\mathrm{sSFR}$",
}

# ---------------------------------------------------------------------------
# Fig 1: Galaxy scaling relation scatter
# ---------------------------------------------------------------------------
def fig1_scaling_relation(scatter, out_dir):
    _style()
    L  = np.array(scatter["L_r"])
    M  = np.array(scatter["mass"])
    R  = np.array(scatter["R_M_given_L"])
    ok = np.isfinite(L) & np.isfinite(M) & np.isfinite(R)
    L, M, R = L[ok], M[ok], R[ok]

    # subsample for legibility
    rng = np.random.default_rng(42)
    idx = rng.choice(len(L), min(8000, len(L)), replace=False)
    L, M, R = L[idx], M[idx], R[idx]

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0), constrained_layout=True)

    # colour range: symmetric around 0 at 98th percentile
    vlim = np.percentile(np.abs(R), 98)
    sc = ax.scatter(L, M, c=R, cmap="RdBu_r",
                    vmin=-vlim, vmax=vlim,
                    s=1.5, alpha=0.5, linewidths=0, rasterized=True)

    # linear fit line
    m, b = np.polyfit(L, M, 1)
    xl = np.linspace(L.min(), L.max(), 200)
    ax.plot(xl, m*xl + b, color="#333", lw=1.5, ls="--", zorder=5,
            label=rf"$\log M_\star = {m:.2f}\,L_r + {b:.2f}$")

    ax.set_xlabel(r"Luminosity $L_r = -0.4\,m_r$", labelpad=3)
    ax.set_ylabel(r"$\log M_\star$", labelpad=3)
    ax.grid(True)
    ax.legend(fontsize=6.5, loc="upper left", frameon=True)

    cb = fig.colorbar(sc, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label(r"Mass residual $\epsilon_{M|L}$", fontsize=10.5)
    cb.ax.tick_params(labelsize=9)

    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"causal_fig1_scaling_relation.{fmt}"))
    plt.close(fig)
    print("Saved causal_fig1_scaling_relation")

# ---------------------------------------------------------------------------
# Fig 2: Direction cosine similarities vs layer (at final checkpoint)
# ---------------------------------------------------------------------------
def fig2_direction_geometry(results, out_dir):
    _style()
    # use last (final) checkpoint
    res = results[-1]

    # per-layer cosines — need per-layer probe directions
    # We stored global (last-layer) cosines; for per-layer geometry,
    # use the stored layer_r2s as a proxy signal and report global cosines
    # as a single-step bar chart if only one checkpoint, else plot vs step
    steps = [r["step"] for r in results]
    cos_LM = [r["cos_LM"] for r in results]
    cos_LR = [r["cos_LR"] for r in results]
    cos_MR = [r["cos_MR"] for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.8), constrained_layout=True)

    if len(steps) == 1:
        # bar chart
        keys = [r"$\cos(v_L, v_M)$", r"$\cos(v_L, v_R)$", r"$\cos(v_M, v_R)$"]
        vals = [cos_LM[0], cos_LR[0], cos_MR[0]]
        colors = ["#4c72b0", "#dd8452", "#55a868"]
        bars = ax.bar(keys, vals, color=colors, width=0.5,
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.axhline(0, color="#888", lw=0.7)
        ax.set_ylabel("Cosine similarity", labelpad=3)
        ax.set_ylim(-0.1, 1.05)
        ax.grid(True, axis="y")
        ax.set_title(f"Probe direction geometry (step {steps[0]:,})", pad=4)
    else:
        # line chart vs training step
        ax.plot(steps, cos_LM, "o-", color="#4c72b0", lw=1.5, ms=5,
                label=r"$\cos(v_L, v_M)$")
        ax.plot(steps, cos_LR, "s-", color="#dd8452", lw=1.5, ms=5,
                label=r"$\cos(v_L, v_R)$")
        ax.plot(steps, cos_MR, "^-", color="#55a868", lw=1.5, ms=5,
                label=r"$\cos(v_M, v_R)$")
        ax.set_xscale("log")
        ax.set_xlabel("Training step", labelpad=3)
        ax.set_ylabel("Cosine similarity", labelpad=3)
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(0, color="#888", lw=0.7, ls=":")
        ax.grid(True)
        ax.legend(fontsize=7)

    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"causal_fig2_direction_geometry.{fmt}"))
    plt.close(fig)
    print("Saved causal_fig2_direction_geometry")

# ---------------------------------------------------------------------------
# Fig 3: Probe R² by layer (at final checkpoint)
# ---------------------------------------------------------------------------
def fig3_probe_curves(results, out_dir):
    _style()
    res = results[-1]
    layer_r2s = res["layer_r2s"]
    n_layers = max(len(v) for v in layer_r2s.values())
    layers = list(range(n_layers))

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.8), constrained_layout=True)

    for key in ("L_r", "mass", "R_M_given_L", "ssfr"):
        vals = layer_r2s.get(key, [np.nan]*n_layers)
        ax.plot(layers, vals,
                "o-", color=PROBE_COLORS[key], lw=1.6, ms=5,
                label=PROBE_LABELS[key])

    ax.set_xlabel("Layer", labelpad=3)
    ax.set_ylabel(r"$R^2$", labelpad=3)
    ax.set_xticks(layers)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"Step {res['step']:,}", pad=4)

    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"causal_fig3_probe_by_layer.{fmt}"))
    plt.close(fig)
    print("Saved causal_fig3_probe_by_layer")

# ---------------------------------------------------------------------------
# Fig 4: Causal trace results by layer
# ---------------------------------------------------------------------------
def fig4_causal_trace(results, out_dir):
    _style()
    if len(results) == 1:
        res = results[0]
        stats = res["layer_stats"]
        layers = [s["layer"] for s in stats]

        fig, axes = plt.subplots(2, 3, figsize=(7.1, 5.2), constrained_layout=True)

        # ---- Row 0, Panel 0: success rate + controls ----
        ax = axes[0, 0]
        sr      = np.array([s["SuccessRate"] for s in stats])
        sr_rand = np.array([s.get("SuccessRate_random", np.nan) for s in stats])
        sr_same = np.array([s.get("SuccessRate_same_resid", np.nan) for s in stats])
        sr_shuf = np.array([s.get("SuccessRate_shuf_probe", np.nan) for s in stats])
        ax.plot(layers, sr,      "o-", color="#d62728", lw=1.8, ms=6, zorder=5, label="Matched")
        ax.plot(layers, sr_rand, "s--", color="#888",    lw=1.2, ms=4, label="Ctrl: random")
        ax.plot(layers, sr_same, "^--", color="#4c72b0", lw=1.2, ms=4, label="Ctrl: same-R")
        ax.plot(layers, sr_shuf, "v--", color="#dd8452", lw=1.2, ms=4, label="Ctrl: shuffled")
        ax.axhline(0.5, color="#bbb", lw=0.8, ls=":", zorder=0)
        ax.set_xlabel("Patched layer", labelpad=3)
        ax.set_ylabel("Success rate", labelpad=3)
        ax.set_ylim(0.3, 1.05); ax.set_xticks(layers)
        ax.grid(True); ax.legend(fontsize=6, loc="lower right")
        ax.set_title("Directional success rate", pad=4)

        # ---- Row 0, Panel 1: normalised effects (A) ----
        ax = axes[0, 1]
        re_n      = np.array([s["ResidualEffect"]   for s in stats]) / SIGMA["R_M_given_L"]
        le_n      = np.array([s["LuminosityEffect"] for s in stats]) / SIGMA["L_r"]
        me_n      = np.array([s["MassEffect"]        for s in stats]) / SIGMA["mass"]
        re_null_n = np.array([s.get("ResidualEffect_same_resid", np.nan)
                               for s in stats]) / SIGMA["R_M_given_L"]
        ax.plot(layers, re_n,      "o-", color="#d62728", lw=1.8, ms=6, zorder=5,
                label=r"$|\Delta\epsilon|/\sigma_R$ (matched)")
        ax.plot(layers, le_n,      "s-", color="#1f77b4", lw=1.5, ms=5,
                label=r"$|\Delta L|/\sigma_L$")
        ax.plot(layers, me_n,      "^-", color="#2ca02c", lw=1.5, ms=5,
                label=r"$|\Delta M|/\sigma_M$")
        ax.plot(layers, re_null_n, "v--", color="#888",   lw=1.2, ms=4,
                label=r"$|\Delta\epsilon|/\sigma_R$ (same-R)")
        ax.set_xlabel("Patched layer", labelpad=3)
        ax.set_ylabel(r"Normalised $|\Delta y|/\sigma_y$", labelpad=3)
        ax.set_xticks(layers); ax.grid(True)
        ax.legend(fontsize=6)
        ax.set_title("Normalised causal effects", pad=4)

        # ---- Row 0, Panel 2: matched-minus-control (C) ----
        ax = axes[0, 2]
        ax.plot(layers, sr - sr_rand, "s-", color="#888",    lw=1.5, ms=5,
                label="Matched - random")
        ax.plot(layers, sr - sr_same, "^-", color="#4c72b0", lw=1.5, ms=5,
                label="Matched - same-R")
        ax.plot(layers, sr - sr_shuf, "v-", color="#dd8452", lw=1.5, ms=5,
                label="Matched - shuffled")
        ax.axhline(0, color="#bbb", lw=0.8, ls=":", zorder=0)
        ax.set_xlabel("Patched layer", labelpad=3)
        ax.set_ylabel("Success rate difference", labelpad=3)
        ax.set_xticks(layers); ax.grid(True)
        ax.legend(fontsize=6.5)
        ax.set_title("Matched vs controls", pad=4)

        # ---- Row 1: residual transfer slope per layer (B) ----
        patch_rows = res.get("patch_rows", None)
        if patch_rows is not None:
            n_show = min(3, len(layers))
            layer_show = [0, len(layers)//2, len(layers)-1][:n_show]
            for col, li in enumerate(layer_show):
                ax = axes[1, col]
                data = [r for r in patch_rows if r["patch_layer"] == li]
                x = np.array([r["true_dir_R"] for r in data])
                y = np.array([r["delta_R"]    for r in data])
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 10:
                    ax.set_visible(False); continue
                slope, intercept = np.polyfit(x[ok], y[ok], 1)
                ax.scatter(x[ok], y[ok], s=4, alpha=0.35, color="#d62728",
                           rasterized=True, linewidths=0)
                xl = np.linspace(x[ok].min(), x[ok].max(), 100)
                ax.plot(xl, slope*xl + intercept, color="#333", lw=1.8,
                        label=f"slope = {slope:.3f}")
                ax.axhline(0, color="#bbb", lw=0.7, ls=":")
                ax.axvline(0, color="#bbb", lw=0.7, ls=":")
                ax.set_xlabel(r"$\epsilon_\mathrm{src} - \epsilon_\mathrm{base}$", labelpad=2)
                ax.set_ylabel(r"$\hat{\epsilon}_\mathrm{patch} - \hat{\epsilon}_\mathrm{base}$", labelpad=2)
                ax.set_title(f"Layer {li} — transfer slope", pad=3)
                ax.legend(fontsize=7); ax.grid(True)
            for col in range(n_show, 3):
                axes[1, col].set_visible(False)
        else:
            # patch_rows not in JSON — show selectivity + note
            ax = axes[1, 0]
            rs = [s["RelationSelectivity"] for s in stats]
            ax.bar(layers, rs, color="#9467bd", alpha=0.8, edgecolor="white", linewidth=0.4)
            ax.set_xlabel("Patched layer", labelpad=3)
            ax.set_ylabel(r"$|\Delta R|/(|\Delta L|+\epsilon)$", labelpad=3)
            ax.set_xticks(layers); ax.grid(True, axis="y")
            ax.set_title("Relation selectivity", pad=4)
            for col in [1, 2]:
                axes[1, col].text(0.5, 0.5,
                    "Rerun causal_tracing.py\nto get per-pair transfer slope",
                    transform=axes[1, col].transAxes,
                    ha="center", va="center", fontsize=7, color="#888")
                axes[1, col].set_axis_off()

    else:
        # multiple checkpoints: heatmaps
        steps = [r["step"] for r in results]
        n_layers = len(results[0]["layer_stats"])
        sr_mat = np.array([[s["SuccessRate"] for s in r["layer_stats"]] for r in results])
        rs_mat = np.array([[s["RelationSelectivity"] for s in r["layer_stats"]] for r in results])

        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)
        for ax, mat, label, cmap in [
            (axes[0], sr_mat, "Success rate",       "RdYlGn"),
            (axes[1], rs_mat, "Relation selectivity","PuRd"),
        ]:
            im = ax.imshow(mat.T, aspect="auto", origin="lower", cmap=cmap,
                           vmin=(0.4 if cmap=="RdYlGn" else 0),
                           vmax=(0.9 if cmap=="RdYlGn" else mat.max()),
                           extent=[0, len(steps), -0.5, n_layers-0.5])
            ax.set_xticks(range(len(steps)))
            ax.set_xticklabels([f"{s:,}" for s in steps], rotation=45,
                               ha="right", fontsize=6)
            ax.set_xlabel("Training step", labelpad=3)
            ax.set_ylabel("Patched layer", labelpad=3)
            ax.set_title(label, pad=4)
            cb = fig.colorbar(im, ax=ax, shrink=0.9)
            cb.set_label(label, fontsize=10); cb.ax.tick_params(labelsize=9)

    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"causal_fig4_trace.{fmt}"))
    plt.close(fig)
    print("Saved causal_fig4_trace")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="Path to *_causal_results.json")
    parser.add_argument("--scatter", required=True,
                        help="Path to label_scatter.json")
    parser.add_argument("--out_dir", default="figures/causal")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    _style()

    with open(args.results) as f:
        results = json.load(f)
    with open(args.scatter) as f:
        scatter = json.load(f)

    fig1_scaling_relation(scatter, args.out_dir)
    fig2_direction_geometry(results, args.out_dir)
    fig3_probe_curves(results, args.out_dir)
    fig4_causal_trace(results, args.out_dir)

    print(f"\nAll causal figures saved to {args.out_dir}/")

if __name__ == "__main__":
    main()
