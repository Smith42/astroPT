"""
plot_with_ci.py -- Add bootstrap error bars to main figures A, B, and causal Fig 4.

Reads:
  /work/nvme/bfir/ssourav/astroPT-runs/bootstrap/bootstrap_final_checkpoint.json
  /work/nvme/bfir/ssourav/astroPT-runs/bootstrap/bootstrap_patching.json

Produces updated versions of:
  residual_figA  -- R²(L), R²(M), R²(eps) vs model size WITH error bars
  geometry_fig2  -- cosines vs model size WITH error bars
  causal_fig4    -- success rate WITH CI bands
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BOOTSTRAP_DIR = "/work/nvme/bfir/ssourav/astroPT-runs/bootstrap"
CONFIGS       = ["ar_aim_001M", "ar_aim_021M", "ar_aim_100M"]
SIZE_LABELS   = {"ar_aim_001M": "1M", "ar_aim_021M": "21M", "ar_aim_100M": "100M"}
SIZE_NM       = {"ar_aim_001M": 1, "ar_aim_021M": 21, "ar_aim_100M": 100}
SIZE_COLORS   = {"ar_aim_001M": "#aec7e8","ar_aim_021M": "#1f77b4","ar_aim_100M": "#08306b"}

C_L   = "#1f77b4"
C_M   = "#2ca02c"
C_EPS = "#d62728"
C_AR  = "#1f77b4"
C_MAE = "#d62728"

def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue","Helvetica","Arial","DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
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

def get_last_layer(cfg_results):
    """Get results from last layer."""
    layers = sorted(int(k.split("_")[1]) for k in cfg_results if k.startswith("layer_"))
    return cfg_results[f"layer_{layers[-1]}"]

# ---------------------------------------------------------------------------
# Fig A with CI: R² vs model size
# ---------------------------------------------------------------------------
def fig_capacity_with_ci(boot_data, out_dir):
    _style()
    fig, ax = plt.subplots(figsize=(3.8, 3.2), constrained_layout=True)

    sizes = []
    vals = {k: {"mean":[],"lo":[],"hi":[]} for k in ("r2_L_r","r2_mass","r2_eps_M_given_L")}
    labels_plot = {
        "r2_L_r":           (r"$R^2(L_r)$",            C_L,   "o"),
        "r2_mass":          (r"$R^2(\log M_\star)$",    C_M,   "s"),
        "r2_eps_M_given_L": (r"$R^2(\epsilon_{M|L})$", C_EPS, "^"),
    }

    for cfg in CONFIGS:
        if cfg not in boot_data: continue
        res = get_last_layer(boot_data[cfg])
        sizes.append(SIZE_NM[cfg])
        for k in vals:
            v = res.get(k, {})
            vals[k]["mean"].append(v.get("mean", np.nan))
            vals[k]["lo"].append(v.get("lo",   np.nan))
            vals[k]["hi"].append(v.get("hi",   np.nan))

    xa = np.array(sizes)
    for k, (lbl, color, marker) in labels_plot.items():
        mn = np.array(vals[k]["mean"])
        lo = np.array(vals[k]["lo"])
        hi = np.array(vals[k]["hi"])
        ax.plot(xa, mn, color=color, lw=1.8, marker=marker, ms=7, label=lbl)
        ax.fill_between(xa, lo, hi, color=color, alpha=0.18)
        # error bars as caps
        for xi, m, l, h in zip(xa, mn, lo, hi):
            ax.plot([xi,xi],[l,h], color=color, lw=1.0, alpha=0.6)

    ax.set_xscale("log")
    ax.set_xticks([1,21,100]); ax.set_xticklabels(["1M","21M","100M"])
    ax.set_ylim(-0.02, 1.02); ax.set_yticks([0,0.25,0.5,0.75,1.0])
    ax.set_xlabel("Model size", labelpad=3)
    ax.set_ylabel(r"$R^2$ (best layer, final ckpt)", labelpad=3)
    ax.grid(True)
    ax.legend(fontsize=9, loc="lower right")
    ax.text(0.02, 0.98, "Shaded: 95% bootstrap CI",
            transform=ax.transAxes, fontsize=7.5, va="top", color="#666")

    for fmt in ("pdf","png"):
        fig.savefig(os.path.join(out_dir, f"ci_figA_capacity_r2.{fmt}"))
    plt.close(fig); print("Saved ci_figA_capacity_r2")

# ---------------------------------------------------------------------------
# Fig B with CI: cosines vs model size
# ---------------------------------------------------------------------------
def fig_cosines_with_ci(boot_data, out_dir):
    _style()
    fig, ax = plt.subplots(figsize=(4.5, 3.2), constrained_layout=True)

    cosine_keys = [
        ("cos_L_M",    r"$\cos(v_L,v_M)$",              "#2ca02c"),
        ("cos_ssfr_M", r"$\cos(v_\mathrm{sSFR},v_M)$",  "#9467bd"),
        ("cos_z_M",    r"$\cos(v_z,v_M)$",              "#1f77b4"),
        ("cos_L_eps",  r"$\cos(v_L,v_\epsilon)$",       "#ff7f0e"),
        ("cos_M_eps",  r"$\cos(v_M,v_\epsilon)$",       "#d62728"),
    ]
    n_cos = len(cosine_keys)
    configs_ok = [c for c in CONFIGS if c in boot_data]
    n_sizes = len(configs_ok)
    w = 0.15; xi = np.arange(n_cos)

    for si, cfg in enumerate(configs_ok):
        res = get_last_layer(boot_data[cfg])
        offset = (si - n_sizes/2 + 0.5) * w
        for ci, (key, lbl, color) in enumerate(cosine_keys):
            v = res.get(key, {})
            mn = v.get("mean", np.nan); lo = v.get("lo", np.nan); hi = v.get("hi", np.nan)
            if si == 0:
                ax.bar(xi[ci]+offset, mn, w, color=SIZE_COLORS[cfg],
                       alpha=0.85, edgecolor="white", lw=0.3,
                       label=SIZE_LABELS[cfg])
            else:
                ax.bar(xi[ci]+offset, mn, w, color=SIZE_COLORS[cfg],
                       alpha=0.85, edgecolor="white", lw=0.3)
            # error bar
            ax.plot([xi[ci]+offset, xi[ci]+offset], [lo, hi],
                    color="#333", lw=1.2, zorder=5)
            ax.plot([xi[ci]+offset-w*0.3, xi[ci]+offset+w*0.3], [lo,lo],
                    color="#333", lw=1.0, zorder=5)
            ax.plot([xi[ci]+offset-w*0.3, xi[ci]+offset+w*0.3], [hi,hi],
                    color="#333", lw=1.0, zorder=5)

    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(xi)
    ax.set_xticklabels([lbl for _,lbl,_ in cosine_keys], fontsize=8)
    ax.set_ylabel("Cosine similarity (last layer, final ckpt)", labelpad=3)
    ax.set_ylim(-0.4, 1.15); ax.grid(True, axis="y")
    ax.legend(title="Model size", title_fontsize=8.5, fontsize=9)
    ax.text(0.02, 0.02, "Error bars: 95% bootstrap CI",
            transform=ax.transAxes, fontsize=7.5, va="bottom", color="#666")

    for fmt in ("pdf","png"):
        fig.savefig(os.path.join(out_dir, f"ci_figB_cosines.{fmt}"))
    plt.close(fig); print("Saved ci_figB_cosines")

# ---------------------------------------------------------------------------
# Causal Fig 4 with CI: success rate by layer
# ---------------------------------------------------------------------------
def fig_causal_with_ci(patch_boot, causal_json_path, out_dir):
    _style()
    with open(causal_json_path) as f:
        causal = json.load(f)

    res = causal[0]  # final checkpoint
    stats = res["layer_stats"]
    layers = [s["layer"] for s in stats]

    # Load bootstrap CI
    step_key = f"step_{res['step']}"
    boot_layers = patch_boot.get(step_key, {})

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)

    # Panel A: success rate with CI
    ax = axes[0]
    sr      = np.array([s["SuccessRate"] for s in stats])
    sr_rand = np.array([s.get("SuccessRate_random", np.nan) for s in stats])
    sr_same = np.array([s.get("SuccessRate_same_resid", np.nan) for s in stats])

    # Get bootstrap CIs
    sr_lo, sr_hi = [], []
    for li in layers:
        bl = boot_layers.get(f"layer_{li}", {}).get("success_rate", {})
        sr_lo.append(bl.get("lo", np.nan)); sr_hi.append(bl.get("hi", np.nan))
    sr_lo = np.array(sr_lo); sr_hi = np.array(sr_hi)

    ax.plot(layers, sr,      "o-", color="#d62728", lw=1.8, ms=6, zorder=5,
            label="Matched")
    ax.fill_between(layers, sr_lo, sr_hi, color="#d62728", alpha=0.15, zorder=3)
    ax.plot(layers, sr_rand, "s--", color="#888",    lw=1.2, ms=4, label="Ctrl: random")
    ax.plot(layers, sr_same, "^--", color="#4c72b0", lw=1.2, ms=4, label="Ctrl: same-R")
    ax.axhline(0.5, color="#bbb", lw=0.8, ls=":", zorder=0)
    ax.text(layers[-1]+0.1, 0.5, "chance", fontsize=7.5, color="#bbb", va="center")
    ax.set_xlabel("Patched layer", labelpad=3)
    ax.set_ylabel("Success rate", labelpad=3)
    ax.set_ylim(0.3, 1.05); ax.set_xticks(layers)
    ax.grid(True); ax.legend(fontsize=8.5)
    ax.set_title("Directional success rate (shaded: 95% CI)", pad=4)

    # Panel B: matched minus controls
    ax = axes[1]
    sr_shuf = np.array([s.get("SuccessRate_shuf_probe", np.nan) for s in stats])
    ax.plot(layers, sr-sr_rand, "s-", color="#888",    lw=1.5, ms=5,
            label="Matched $-$ random")
    ax.plot(layers, sr-sr_same, "^-", color="#4c72b0", lw=1.5, ms=5,
            label="Matched $-$ same-R")
    ax.plot(layers, sr-sr_shuf, "v-", color="#dd8452", lw=1.5, ms=5,
            label="Matched $-$ shuffled")
    ax.axhline(0, color="#bbb", lw=0.8, ls=":")
    ax.set_xlabel("Patched layer", labelpad=3)
    ax.set_ylabel("Success rate difference", labelpad=3)
    ax.set_xticks(layers); ax.grid(True)
    ax.legend(fontsize=8.5)
    ax.set_title("Matched vs controls", pad=4)

    for fmt in ("pdf","png"):
        fig.savefig(os.path.join(out_dir, f"ci_causal_success_rate.{fmt}"))
    plt.close(fig); print("Saved ci_causal_success_rate")

# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot_dir",   default=BOOTSTRAP_DIR)
    parser.add_argument("--causal_json",
        default="/work/nvme/bfir/ssourav/astroPT-runs/causal/ar_aim_021M_causal_results.json")
    parser.add_argument("--out_dir",    default="figures/with_ci")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    _style()

    boot_path = os.path.join(args.boot_dir, "bootstrap_final_checkpoint.json")
    patch_path = os.path.join(args.boot_dir, "bootstrap_patching.json")

    if not os.path.exists(boot_path):
        print(f"Missing {boot_path} — run bootstrap_ci.py --mode final first")
        return

    with open(boot_path) as f: boot_data = json.load(f)
    patch_boot = {}
    if os.path.exists(patch_path):
        with open(patch_path) as f: patch_boot = json.load(f)

    fig_capacity_with_ci(boot_data, args.out_dir)
    fig_cosines_with_ci(boot_data, args.out_dir)

    if os.path.exists(args.causal_json) and patch_boot:
        fig_causal_with_ci(patch_boot, args.causal_json, args.out_dir)

    print(f"\nDone. {args.out_dir}/")

if __name__ == "__main__":
    main()
