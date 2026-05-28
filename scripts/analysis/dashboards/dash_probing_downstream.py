import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json

# Modality Aliases for consistent naming across legacy, new, and AION runs
MODALITY_MAP = {
    'images': 'EuclidImage',
    'spectra': 'DESISpectrum',
    'joint': 'Joint',
    'cls': 'CLS',
    'cls_images': 'CLS (Images)',
    'cls_spectra': 'CLS (Spectra)',
    'aion_images': 'EuclidImage',
    'aion_spectra': 'DESISpectrum'
}

# Logical groups for separate dashboards
TARGET_GROUPS = {
    "Physics": ['Z', 'LOGMSTAR', 'LOGSFR', 'GR', 'flux_detection_total'],
    "Morphology": ['sersic_sersic_vis_radius', 'sersic_sersic_vis_index', 'sersic_sersic_vis_axis_ratio', 'smoothness', 'gini'],
    "Spectroscopy": ['HALPHA_EW', 'HALPHA_FLUX', 'NII_6584_FLUX', 'OIII_5007_FLUX', 'HBETA_FLUX'],
}

PROBE_ORDER = ['KNN', 'LP', 'MLP']

MARKER_CYCLE = ['o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', '>', 'h', '8', 'p']
LINESTYLE_CYCLE = ['-', '--', '-.', ':']

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Generate performance dashboards for downstream tasks.")
    
    parser.add_argument("--save_dir", type=str, default=None, help="Plot Saving Directory")
    parser.add_argument('--save_name', default="downstream_dashboard", help="Saving optional name")
    parser.add_argument('--csv_path', nargs='+', help="Paths to the CSV files to compare. Required if --run_dirs is not used.")
    parser.add_argument('--names', nargs='+', help="Names of the tests for the legend (must match number of CSVs).")
    parser.add_argument('--run_dirs', nargs='+', help="List of model run directories in logs/ to auto-discover. Replaces --csv_path and --names.")
    parser.add_argument('--emb_filter', type=str, default=None, help="Filter for embedding folders to plot across runs (e.g. 'specfirst')")
    parser.add_argument('--targets', nargs='+', default=None, help="List of targets to plot (e.g., Z SPECTYPE HALPHA_FLUX).")
    parser.add_argument('--all_targets', action='store_true', help="Plot all available targets found in the provided CSV files.")
    parser.add_argument('--exclude_modalities', nargs='+', default=['CLS', 'CLS (Images)', 'CLS (Spectra)'], help="Modalities to skip (default: CLS variants). Use 'none' to show all.")
    parser.add_argument('--exclude_metrics', nargs='+', default=['RMSE', 'Bias'], help="Metrics to skip (default: RMSE, Bias). Use 'none' to show all.")
    
    return parser.parse_args()


def build_style_map(test_names: List[str]) -> Dict[str, Dict[str, object]]:
    """Creates deterministic styles for an arbitrary number of test curves."""
    if not test_names:
        return {}

    cmap = plt.get_cmap("tab20", max(1, len(test_names)))
    style_map: Dict[str, Dict[str, object]] = {}

    for idx, test_name in enumerate(test_names):
        style_map[test_name] = {
            "color": cmap(idx),
            "marker": MARKER_CYCLE[idx % len(MARKER_CYCLE)],
            "linestyle": LINESTYLE_CYCLE[(idx // len(MARKER_CYCLE)) % len(LINESTYLE_CYCLE)],
            "markersize": 7,
        }

    return style_map


def get_ordered_configs(df: pd.DataFrame) -> List[str]:
    """Dynamically builds X-axis order based on present modalities and probes."""
    # Sort alphabetically, with 'Joint' or anything containing 'joint' always at the end
    mods = sorted(
        df['Modality'].unique(),
        key=lambda x: (x.lower() == 'joint' or 'joint' in x.lower(), x)
    )
    configs = []
    for m in mods:
        # Get probes present for this modality
        available_probes = sorted(df[df['Modality'] == m]['Probe'].unique(), 
                                 key=lambda x: (x not in PROBE_ORDER, PROBE_ORDER.index(x) if x in PROBE_ORDER else 0))
        for p in available_probes:
            configs.append(f"{m} {p}")
    return configs

def load_and_merge_data(filepaths: List[str | Path], test_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Loads multiple CSV files, assigns a test name to each, cleans empty rows, and merges them.

    Args:
        filepaths (List[str]): List of paths to the CSV files.
        test_names (Optional[List[str]]): List of names for each test to be used in legends.

    Returns:
        pd.DataFrame: A single concatenated DataFrame containing all tests.
    """
    if test_names is None or len(test_names) != len(filepaths):
        test_names = [f"Test {i+1}" for i in range(len(filepaths))]

    df_list = []
    for filepath, name in zip(filepaths, test_names):
        filepath = Path(filepath)
        try:
            df = pd.read_csv(filepath)
            
            # Clean empty rows
            df = df.dropna(subset=['Target', 'Task'])
            
            # Detect if this specific CSV contains phase-based modalities
            has_phases = df['Modality'].astype(str).str.contains('_phase').any()
            
            def process_row(row):
                mod = str(row['Modality'])
                if '_phase1' in mod:
                    row['Modality'] = mod.replace('_phase1', '')
                    row['Test_Name'] = f"{name} (Phase 1)"
                elif '_phase2' in mod:
                    row['Modality'] = mod.replace('_phase2', '')
                    row['Test_Name'] = f"{name} (Phase 2)"
                elif has_phases and mod in ['EuclidImage', 'DESISpectrum', 'images', 'spectra', 'joint', 'cls', 'Joint', 'CLS']:
                    row['Test_Name'] = f"{name} (Final)"
                else:
                    row['Test_Name'] = name
                return row

            df = df.apply(process_row, axis=1)
            
            # Normalize modalities via Aliases
            df['Modality'] = df['Modality'].replace(MODALITY_MAP)
            
            # Replicate supervised baseline points across all probes (KNN, LP, MLP)
            rows_to_add = []
            for idx, row in df.iterrows():
                mod = str(row['Modality'])
                probe = str(row['Probe'])
                
                is_spectra_baseline = (mod in ['DESISpectrum', 'spectra']) and (probe.upper() == 'TRANSFORMER_BASELINE')
                is_images_baseline = (mod in ['EuclidImage', 'images']) and (probe.upper() == 'RESNET18')
                
                if is_spectra_baseline or is_images_baseline:
                    for p in ['KNN', 'LP', 'MLP']:
                        new_row = row.copy()
                        new_row['Probe'] = p
                        rows_to_add.append(new_row)
            
            if rows_to_add:
                df = df[~(((df['Modality'].isin(['DESISpectrum', 'spectra'])) & (df['Probe'].str.upper() == 'TRANSFORMER_BASELINE')) | 
                          ((df['Modality'].isin(['EuclidImage', 'images'])) & (df['Probe'].str.upper() == 'RESNET18')))]
                df_replicated = pd.DataFrame(rows_to_add)
                df = pd.concat([df, df_replicated], ignore_index=True)
                
            df['Config'] = df['Modality'] + ' ' + df['Probe']
            df_list.append(df)
        except Exception as e:
            print(f"[ERROR] Could not read {filepath}: {e}")

    if not df_list:
        raise ValueError("No valid CSV files were loaded.")

    return pd.concat(df_list, ignore_index=True)

def generate_dashboard(
    df: pd.DataFrame,
    targets: List[str],
    save_dir: str,
    save_name: str,
    global_test_order: Optional[List[str]] = None,
) -> None:
    """
    Generates and saves a matplotlib dashboard for a list of targets, 
    potentially mixing regression and classification.
    """
    if not targets:
        return

    # Filter dataset by requested targets
    task_df = df[df['Target'].isin(targets)].copy()
    
    if task_df.empty:
        return

    # Drop columns that are completely NaN
    task_df = task_df.dropna(axis=1, how='all')

    # Identify metric columns 
    metadata_cols = ['Target', 'Task', 'Modality', 'Probe', 'Test_Name', 'Config']
    metrics = [col for col in task_df.columns if col not in metadata_cols and pd.api.types.is_numeric_dtype(task_df[col])]
    
    # Keep only targets that are present in this task subset.
    available_targets = [t for t in targets if t in task_df['Target'].unique()]
    if not available_targets:
        return

    # Priority metrics for sorting columns
    priority_metrics = ['R2', 'F1_score', 'Accuracy', 'RMSE', 'Bias', 'NMAD']
    metrics = sorted(metrics, key=lambda x: (x not in priority_metrics, priority_metrics.index(x) if x in priority_metrics else 0))

    n_rows = len(available_targets)
    n_cols = len(metrics)

    if n_cols == 0:
        return

    # Set up the matplotlib figure
    fig_w = max(12, 4.2 * n_cols)
    fig_h = max(7, 5.0 * n_rows)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(fig_w, fig_h), squeeze=False)

    tests_present = task_df['Test_Name'].dropna().astype(str).unique().tolist()
    if global_test_order:
        tests = [t for t in global_test_order if t in tests_present]
    else:
        tests = tests_present
    style_map = build_style_map(global_test_order or tests)
    config_order = get_ordered_configs(task_df)
    if not config_order:
        plt.close(fig)
        return
    config_to_x = {cfg: idx for idx, cfg in enumerate(config_order)}

    for i, target in enumerate(available_targets):
        target_df = task_df[task_df['Target'] == target]
        
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            
            # Check if this metric is valid for this target (has non-nan values)
            if target_df[metric].isna().all():
                ax.text(0.5, 0.5, "N/A", ha='center', va='center', color='gray', alpha=0.5)
                ax.set_title(f'{target} - {metric}', fontsize=10, alpha=0.5)
                ax.set_axis_off()
                continue

            for test_name in tests:
                test_df = target_df[target_df['Test_Name'] == test_name].copy()
                
                # Ensure the data follows the expected X-axis order
                test_df['Config'] = pd.Categorical(test_df['Config'], categories=config_order, ordered=True)
                test_df = test_df.sort_values('Config')
                test_df = test_df[test_df['Config'].notna()].copy()
                test_df['x_pos'] = test_df['Config'].astype(str).map(config_to_x)
                test_df = test_df[test_df['x_pos'].notna()]

                if test_df.empty or test_df[metric].isna().all():
                    continue

                style = style_map[test_name]
                
                # Split the data into continuous modality segments to avoid lines crossing
                for mod in test_df['Modality'].unique():
                    mod_df = test_df[test_df['Modality'] == mod].copy()
                    if mod_df.empty:
                        continue
                        
                    ax.plot(
                        mod_df['x_pos'], 
                        mod_df[metric], 
                        marker=style['marker'],
                        linestyle=style['linestyle'],
                        linewidth=0.7,
                        markersize=style['markersize'],
                        alpha=0.9,
                        label=test_name if (test_name not in [l.get_label() for l in ax.get_lines()]) else "",
                        color=style['color']
                    )

            # Add background blocks
            mod_colors = plt.cm.Set3(np.linspace(0, 1, len(task_df['Modality'].unique())))
            for idx, mod in enumerate(task_df['Modality'].unique()):
                mod_indices = [idx for idx, cfg in enumerate(config_order) if cfg.startswith(mod)]
                if mod_indices:
                    start = min(mod_indices) - 0.5
                    end = max(mod_indices) + 0.5
                    ax.axvspan(start, end, color=mod_colors[idx], alpha=0.15, zorder=0)

            # Formatting
            ax.set_title(f'{target} - {metric}', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(config_order)))
            ax.set_xticklabels(config_order, rotation=45, ha='right')
            ax.tick_params(axis='x', labelsize=10, pad=6)
            ax.grid(True, linestyle='-', alpha=0.4)
            if hasattr(ax, 'set_box_aspect'):
                ax.set_box_aspect(0.9)
            
            if j == 0:
                ax.set_ylabel('Score / Value', fontsize=12)

    # Fitting legend and title
    fig_height = fig.get_figheight()
    header_space_inches = 1.6
    top_limit = 1.0 - (header_space_inches / fig_height)
    title_y = 1.0 - (0.5 / fig_height)
    legend_y = 1.0 - (0.9 / fig_height)
    
    plt.tight_layout(rect=[0, 0, 1, top_limit], h_pad=2.4, w_pad=1.0)
    fig.suptitle(f'Downstream Performance: {save_name.split("_")[-1].capitalize()}', 
                 fontsize=20, y=title_y, va='top')
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles: # fallback if first plot was empty
        for r in range(n_rows):
            for c in range(n_cols):
                handles, labels = axes[r, c].get_legend_handles_labels()
                if handles: break
            if handles: break

    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=min(len(tests), 4), 
                   fontsize=12, bbox_to_anchor=(0.5, legend_y), frameon=True)
    
    save_path = Path(save_dir) / f"{save_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Combined Dashboard saved at: {save_path}")

def main():
    
    args = parse_args()
    
    csv_paths_list = []
    names_list = []

    if args.run_dirs:
        # Intrasection Mode: If exactly 1 run_dir and no emb_filter, plot all embedding variations inside it.
        if len(args.run_dirs) == 1 and not args.emb_filter and Path(args.run_dirs[0]).is_dir():
            run_dir = Path(args.run_dirs[0])
            csv_candidates = sorted(list(run_dir.glob("embeddings/*/downstream_tasks/downstream_results.csv")))
            for csv_path in csv_candidates:
                emb_folder_name = csv_path.parent.parent.name
                config_path = run_dir / "weights" / "config.json"
                train_name = run_dir.name
                if config_path.is_file():
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            train_name = config.get("train_name", train_name)
                    except Exception:
                        pass
                csv_paths_list.append(csv_path)
                names_list.append(f"{train_name} [{emb_folder_name}]")
        else:
            # Multi-run Mode: Compare multiple runs taking the best/latest/filtered embedding
            for run_dir_str in args.run_dirs:
                run_path = Path(run_dir_str)
                
                # Support direct CSV paths
                if run_path.is_file() and run_path.suffix == '.csv':
                    csv_paths_list.append(run_path)
                    # Try to infer names from parent structure
                    try:
                        name_candidate = run_path.parent.parent.parent.parent.name # run_dir/embeddings/emb_folder/downstream_tasks/res.csv
                        config_path = run_path.parent.parent.parent.parent / "weights" / "config.json"
                        if config_path.is_file():
                            with open(config_path, 'r') as f:
                                name_candidate = json.load(f).get("train_name", name_candidate)
                        names_list.append(f"{name_candidate} [{run_path.parent.parent.name}]")
                    except:
                        names_list.append(run_path.name)
                    continue

                # Directory Search Mode: Find the CSV
                candidates = []
                if (run_path / "downstream_results.csv").is_file():
                    candidates.append(run_path / "downstream_results.csv")
                if (run_path / "downstream_tasks" / "downstream_results.csv").is_file():
                    candidates.append(run_path / "downstream_tasks" / "downstream_results.csv")
                candidates.extend(list(run_path.glob("embeddings/*/downstream_tasks/downstream_results.csv")))
                
                if not candidates:
                    print(f"[WARNING] No downstream_results.csv found in {run_path}. Skipping.")
                    continue
                
                if args.emb_filter:
                    filtered = [c for c in candidates if args.emb_filter in c.parent.parent.name]
                    if filtered:
                        candidates = filtered
                
                best_csvs = [c for c in candidates if "best" in str(c)]
                pool = best_csvs if best_csvs else candidates
                pool.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                chosen_csv = pool[0]
                
                potential_root = run_path
                if not (potential_root / "weights" / "config.json").is_file():
                    try:
                        test_root = chosen_csv.parent.parent.parent.parent
                        if (test_root / "weights" / "config.json").is_file():
                            potential_root = test_root
                    except:
                        pass

                emb_folder_name = chosen_csv.parent.parent.name if "downstream_tasks" in chosen_csv.parent.name else chosen_csv.parent.name
                config_path = potential_root / "weights" / "config.json"
                train_name = potential_root.name
                
                if "resnet18_images_supervised" in str(chosen_csv):
                    target_candidate = chosen_csv.parent.parent.parent.name
                    train_name = f"ResNet18 Supervised ({target_candidate})" if target_candidate != "embeddings" else "ResNet18 Supervised"
                elif "spectra_supervised_baseline" in str(chosen_csv):
                    target_candidate = chosen_csv.parent.parent.parent.name
                    train_name = f"Spectra Supervised ({target_candidate})" if target_candidate != "embeddings" else "Spectra Supervised"
                elif config_path.is_file():
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            train_name = config.get("train_name", train_name)
                    except Exception:
                        pass
                
                if "best_meanrank" not in emb_folder_name and "img-mean" not in emb_folder_name and "spec-rank" not in emb_folder_name and "embeddings_all_targets" not in emb_folder_name:
                    train_name += f" [{emb_folder_name}]"
                
                base_name = train_name
                counter = 1
                while train_name in names_list:
                    train_name = f"{base_name} ({counter})"
                    counter += 1
                    
                csv_paths_list.append(chosen_csv)
                names_list.append(train_name)
    
    elif args.csv_path:
        csv_paths_list = [Path(p) for p in args.csv_path]
        names_list = args.names
    else:
        print("[FATAL ERROR] You must provide either --csv_path or --run_dirs.")
        return

    try:
        df = load_and_merge_data(csv_paths_list, names_list)
    except Exception as e:
        print(f"[FATAL ERROR] Failed to load data: {e}")
        return

    if args.save_dir:
        save_dir = Path(args.save_dir) / "downstream_tasks"
    else:
        save_dir = csv_paths_list[-1].parent if csv_paths_list else Path("results/downstream_tasks")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Apply Exclusions
    if args.exclude_modalities and 'none' not in [m.lower() for m in args.exclude_modalities]:
        df = df[~df['Modality'].isin(args.exclude_modalities)]
    
    if args.exclude_metrics and 'none' not in [m.lower() for m in args.exclude_metrics]:
        # Drop columns that match excluded metrics
        cols_to_drop = [c for c in df.columns if c in args.exclude_metrics]
        df = df.drop(columns=cols_to_drop)

    if args.all_targets:
        targets_to_plot = sorted(df['Target'].dropna().astype(str).unique().tolist())
    else:
        if not args.targets:
            print("[FATAL ERROR] You must provide --targets or enable --all_targets")
            return
        targets_to_plot = args.targets

    # Sort runs alphabetically for a more organized legend
    global_test_order: List[str] = sorted(list(set([str(n) for n in (names_list or [])] + df['Test_Name'].dropna().astype(str).unique().tolist())))

    # Generate Unified Dashboards
    # 1. Classification Dashboard (All classification tasks together)
    task_col = df['Task'].astype(str).str.lower()
    all_classification_targets = sorted(df[task_col == 'classification']['Target'].dropna().unique().tolist())
    # Only keep those requested (or all if --all_targets)
    cls_to_plot = [t for t in all_classification_targets if t in targets_to_plot]
    if cls_to_plot:
        generate_dashboard(df, cls_to_plot, save_dir, f"{args.save_name}_Classification", global_test_order)

    # 2. Categorical Regression Dashboards
    for group_name, group_targets in TARGET_GROUPS.items():
        available_group_targets = [t for t in group_targets if t in targets_to_plot]
        if available_group_targets:
            generate_dashboard(df, available_group_targets, save_dir, f"{args.save_name}_{group_name}", global_test_order)

if __name__ == "__main__":
    main()