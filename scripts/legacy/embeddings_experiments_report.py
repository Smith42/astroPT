"""
AstroPT Embeddings Experiments Report.

Builds a compact decision-oriented report across many embedding experiments.
Outputs:
- CSV summary ranking
- Target-level matrix and delta-vs-baseline matrix
- PNG charts (ranking, heatmap, pareto)
- Single HTML report
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-experiment embeddings report")
    parser.add_argument("--emb_root", type=str, required=True, help="Root directory containing embedding experiment folders")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory for report assets")
    parser.add_argument("--probe", type=str, default="MLP", help="Probe to compare (LP or MLP)")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline experiment folder name for delta heatmap")
    parser.add_argument("--top_k", type=int, default=12, help="Number of top experiments to display in ranking table")
    return parser.parse_args()


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _normalize_probe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Probe"] = out["Probe"].astype(str).str.upper().str.strip()
    out["Modality"] = out["Modality"].astype(str).str.lower().str.strip()
    out["Task"] = out["Task"].astype(str).str.lower().str.strip()
    out["Target"] = out["Target"].astype(str).str.strip()
    return out


def _extract_primary_metrics(seed_df: Optional[pd.DataFrame], raw_df: Optional[pd.DataFrame], probe: str) -> Optional[pd.DataFrame]:
    probe = probe.upper()

    if seed_df is not None and not seed_df.empty:
        sdf = _normalize_probe(seed_df)
        sdf = sdf[sdf["Probe"] == probe].copy()
        if sdf.empty:
            return None

        rows = []
        for _, row in sdf.iterrows():
            task = row["Task"]
            if task == "regression":
                metric_mean = row.get("R2_mean", np.nan)
                metric_std = row.get("R2_std", np.nan)
                metric_name = "R2"
            else:
                metric_mean = row.get("F1_score_mean", np.nan)
                metric_std = row.get("F1_score_std", np.nan)
                metric_name = "F1_score"

            rows.append(
                {
                    "Target": row["Target"],
                    "Task": task,
                    "Modality": row["Modality"],
                    "Probe": row["Probe"],
                    "PrimaryMetricName": metric_name,
                    "PrimaryMetricMean": float(metric_mean) if pd.notna(metric_mean) else np.nan,
                    "PrimaryMetricStd": float(metric_std) if pd.notna(metric_std) else np.nan,
                }
            )

        out = pd.DataFrame(rows)
        return out if not out.empty else None

    if raw_df is not None and not raw_df.empty:
        rdf = _normalize_probe(raw_df)
        rdf = rdf[rdf["Probe"] == probe].copy()
        if rdf.empty:
            return None

        rows = []
        for _, row in rdf.iterrows():
            task = row["Task"]
            if task == "regression":
                metric_mean = row.get("R2", np.nan)
                metric_name = "R2"
            else:
                metric_mean = row.get("F1_score", np.nan)
                metric_name = "F1_score"

            rows.append(
                {
                    "Target": row["Target"],
                    "Task": task,
                    "Modality": row["Modality"],
                    "Probe": row["Probe"],
                    "PrimaryMetricName": metric_name,
                    "PrimaryMetricMean": float(metric_mean) if pd.notna(metric_mean) else np.nan,
                    "PrimaryMetricStd": np.nan,
                }
            )

        out = pd.DataFrame(rows)
        return out if not out.empty else None

    return None


def _infer_family(exp_name: str) -> str:
    name = exp_name.lower()
    if "ord-" in name or "ord_" in name or "imgfirst" in name or "specfirst" in name:
        return "orders"
    if "joint" in name or "fusion" in name or "l2mean" in name or "w0p" in name:
        return "fusion"
    if "baseline" in name or "post_analysis" in name:
        return "baseline"
    return "other"


def _read_cosine_metrics(exp_dir: Path) -> Tuple[float, float]:
    metrics_path = exp_dir / "cosine_metrics.json"
    if not metrics_path.is_file():
        return np.nan, np.nan
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        return float(data.get("top1", np.nan)), float(data.get("top5", np.nan))
    except Exception:
        return np.nan, np.nan


def collect_experiment_data(emb_root: Path, probe: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    experiment_rows: List[Dict[str, object]] = []
    joint_target_rows: List[Dict[str, object]] = []

    for exp_dir in sorted(p for p in emb_root.iterdir() if p.is_dir()):
        downstream_dir = exp_dir / "downstream_tasks"
        seed_path = downstream_dir / "downstream_results_seedstats.csv"
        raw_path = downstream_dir / "downstream_results.csv"
        gap_path = downstream_dir / "cross_modal_transfer_gap.csv"

        seed_df = _safe_read_csv(seed_path)
        raw_df = _safe_read_csv(raw_path)
        metrics_df = _extract_primary_metrics(seed_df, raw_df, probe)

        if metrics_df is None or metrics_df.empty:
            continue

        gap_df = _safe_read_csv(gap_path)
        avg_abs_gap = np.nan
        if gap_df is not None and not gap_df.empty:
            gdf = gap_df.copy()
            gdf["Probe"] = gdf["Probe"].astype(str).str.upper().str.strip()
            gdf = gdf[gdf["Probe"] == probe.upper()]
            if not gdf.empty and "AbsGap" in gdf.columns:
                avg_abs_gap = float(pd.to_numeric(gdf["AbsGap"], errors="coerce").mean())

        joint_df = metrics_df[metrics_df["Modality"] == "joint"]
        images_df = metrics_df[metrics_df["Modality"] == "images"]
        spectra_df = metrics_df[metrics_df["Modality"] == "spectra"]

        joint_mean = float(joint_df["PrimaryMetricMean"].mean()) if not joint_df.empty else np.nan
        images_mean = float(images_df["PrimaryMetricMean"].mean()) if not images_df.empty else np.nan
        spectra_mean = float(spectra_df["PrimaryMetricMean"].mean()) if not spectra_df.empty else np.nan
        joint_std = float(joint_df["PrimaryMetricStd"].mean()) if not joint_df.empty else np.nan

        cos_top1, cos_top5 = _read_cosine_metrics(exp_dir)

        exp_name = exp_dir.name
        experiment_rows.append(
            {
                "Experiment": exp_name,
                "Family": _infer_family(exp_name),
                "N_Targets_Joint": int(joint_df["Target"].nunique()) if not joint_df.empty else 0,
                "JointPrimaryMean": joint_mean,
                "ImagesPrimaryMean": images_mean,
                "SpectraPrimaryMean": spectra_mean,
                "JointPrimaryStd": joint_std,
                "AvgAbsGap": avg_abs_gap,
                "CosineTop1": cos_top1,
                "CosineTop5": cos_top5,
            }
        )

        for _, row in joint_df.iterrows():
            joint_target_rows.append(
                {
                    "Experiment": exp_name,
                    "Target": row["Target"],
                    "PrimaryMetricMean": row["PrimaryMetricMean"],
                }
            )

    summary_df = pd.DataFrame(experiment_rows)
    joint_target_df = pd.DataFrame(joint_target_rows)
    return summary_df, joint_target_df


def _fill_with_median(series: pd.Series) -> pd.Series:
    if series.dropna().empty:
        return series.fillna(0.0)
    return series.fillna(float(series.median()))


def compute_composite_score(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()
    df["JointPrimaryMean"] = _fill_with_median(df["JointPrimaryMean"])
    df["AvgAbsGap"] = _fill_with_median(df["AvgAbsGap"])
    df["JointPrimaryStd"] = _fill_with_median(df["JointPrimaryStd"])
    df["CosineTop1"] = df["CosineTop1"].fillna(0.0)

    df["CompositeScore"] = (
        df["JointPrimaryMean"]
        - 0.25 * df["AvgAbsGap"]
        - 0.10 * df["JointPrimaryStd"]
        + 0.10 * df["CosineTop1"]
    )

    df = df.sort_values("CompositeScore", ascending=False).reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df) + 1)
    return df


def plot_ranking(summary_df: pd.DataFrame, out_path: Path) -> None:
    df = summary_df.sort_values("CompositeScore", ascending=True)
    fig_h = max(5, 0.45 * len(df))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.barh(df["Experiment"], df["CompositeScore"], color="#457b9d", alpha=0.9)
    ax.set_title("Global Embedding Ranking (Composite Score)")
    ax.set_xlabel("Composite Score (higher is better)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_heatmap_delta(joint_target_df: pd.DataFrame, baseline: str, out_path: Path) -> pd.DataFrame:
    pivot = joint_target_df.pivot_table(index="Experiment", columns="Target", values="PrimaryMetricMean", aggfunc="mean")
    if baseline not in pivot.index:
        baseline = pivot.index[0]

    delta = pivot.subtract(pivot.loc[baseline], axis=1)
    delta = delta.dropna(axis=1, how="all")
    delta = delta.fillna(0.0)

    vmax = float(np.abs(delta.values).max()) if delta.size else 1.0
    if vmax == 0.0:
        vmax = 1.0

    fig_w = max(10, 0.75 * max(1, len(delta.columns)))
    fig_h = max(6, 0.45 * max(1, len(delta.index)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(delta.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(delta.columns)))
    ax.set_xticklabels(delta.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(delta.index)))
    ax.set_yticklabels(delta.index)
    ax.set_title(f"Delta vs Baseline ({baseline}) - Joint {delta.shape[1]} Targets")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Delta primary metric")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return delta


def plot_pareto(summary_df: pd.DataFrame, out_path: Path) -> None:
    df = summary_df.copy()
    x = _fill_with_median(df["AvgAbsGap"])
    y = _fill_with_median(df["JointPrimaryMean"])
    s = 80 + 500 * df["CosineTop1"].fillna(0.0)

    color_map = {
        "baseline": "#1d3557",
        "orders": "#2a9d8f",
        "fusion": "#e76f51",
        "other": "#6c757d",
    }
    colors = [color_map.get(fam, "#6c757d") for fam in df["Family"]]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(x, y, s=s, c=colors, alpha=0.85, edgecolors="black", linewidths=0.4)

    top_to_label = df.nsmallest(5, "Rank") if "Rank" in df.columns else df.head(5)
    for _, row in top_to_label.iterrows():
        ax.annotate(row["Experiment"], (row["AvgAbsGap"], row["JointPrimaryMean"]), fontsize=8, alpha=0.9)

    ax.set_xlabel("Average Abs Cross-Modal Gap (lower is better)")
    ax.set_ylabel("Joint Primary Metric Mean (higher is better)")
    ax.set_title("Pareto View: Performance vs Modality Gap")
    ax.grid(alpha=0.25)

    legend_handles = []
    for fam, color in color_map.items():
        legend_handles.append(plt.Line2D([0], [0], marker="o", color="w", label=fam, markerfacecolor=color, markersize=8))
    ax.legend(handles=legend_handles, title="Family", loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_html_report(
    save_dir: Path,
    summary_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    baseline: str,
    probe: str,
    top_k: int,
) -> Path:
    top_df = summary_df.head(top_k).copy()

    float_cols = [
        "JointPrimaryMean",
        "ImagesPrimaryMean",
        "SpectraPrimaryMean",
        "JointPrimaryStd",
        "AvgAbsGap",
        "CosineTop1",
        "CosineTop5",
        "CompositeScore",
    ]
    for col in float_cols:
        if col in top_df.columns:
            top_df[col] = pd.to_numeric(top_df[col], errors="coerce").map(lambda x: f"{x:.4f}" if pd.notna(x) else "-")

    top_html = top_df.to_html(index=False, escape=False)

    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>AstroPT Embeddings Experiments Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .meta {{ color: #4b5563; margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
    img {{ width: 100%; max-width: 1400px; border: 1px solid #d1d5db; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; font-size: 13px; }}
    th {{ background: #f3f4f6; }}
    .note {{ margin-top: 12px; color: #374151; }}
  </style>
</head>
<body>
  <h1>AstroPT Embeddings Experiments Report</h1>
  <div class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Probe: {probe} | Baseline: {baseline}</div>

  <div class="grid">
    <section>
      <h2>Top Experiments</h2>
      {top_html}
      <div class="note">Full table: summary_ranking.csv</div>
    </section>

    <section>
      <h2>Global Ranking</h2>
      <img src="ranking_bar.png" alt="Ranking chart" />
    </section>

    <section>
      <h2>Delta by Target (vs Baseline)</h2>
      <img src="target_delta_heatmap.png" alt="Delta heatmap" />
      <div class="note">Matrix CSV: target_delta_vs_baseline.csv</div>
    </section>

    <section>
      <h2>Pareto View</h2>
      <img src="pareto_gap_vs_performance.png" alt="Pareto chart" />
    </section>
  </div>
</body>
</html>
"""

    out_path = save_dir / "embeddings_report.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()

    emb_root = Path(args.emb_root)
    if not emb_root.is_dir():
        raise FileNotFoundError(f"Embeddings root not found: {emb_root}")

    save_dir = Path(args.save_dir) if args.save_dir else (emb_root / "embeddings_report")
    save_dir.mkdir(parents=True, exist_ok=True)

    summary_df, joint_target_df = collect_experiment_data(emb_root, probe=args.probe)
    if summary_df.empty:
        raise RuntimeError("No valid experiments with downstream metrics were found.")

    summary_df = compute_composite_score(summary_df)

    baseline = args.baseline
    if not baseline or baseline not in set(summary_df["Experiment"].values):
        baseline = str(summary_df.iloc[0]["Experiment"])

    ranking_png = save_dir / "ranking_bar.png"
    heatmap_png = save_dir / "target_delta_heatmap.png"
    pareto_png = save_dir / "pareto_gap_vs_performance.png"

    plot_ranking(summary_df, ranking_png)
    delta_df = plot_heatmap_delta(joint_target_df, baseline=baseline, out_path=heatmap_png)
    plot_pareto(summary_df, pareto_png)

    summary_df.to_csv(save_dir / "summary_ranking.csv", index=False)
    delta_df.to_csv(save_dir / "target_delta_vs_baseline.csv")

    pivot_df = joint_target_df.pivot_table(index="Experiment", columns="Target", values="PrimaryMetricMean", aggfunc="mean")
    pivot_df.to_csv(save_dir / "target_joint_metric_matrix.csv")

    html_path = write_html_report(
        save_dir=save_dir,
        summary_df=summary_df,
        delta_df=delta_df,
        baseline=baseline,
        probe=args.probe.upper(),
        top_k=args.top_k,
    )

    print("-----------------------------------------------")
    print(f"Report generated in: {save_dir}")
    print(f"HTML report: {html_path}")
    print("-----------------------------------------------")


if __name__ == "__main__":
    main()
