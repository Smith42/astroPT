import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

# Expected X-axis categories in strict order (Zero-Shot only)
CONFIG_ORDER = [
    'images->joint', 
    'images->spectra',
    'spectra->joint',
    'spectra->images',
    'joint->images',
    'joint->spectra'
]

MARKER_CYCLE = ['o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', '>', 'h', '8', 'p']
LINESTYLE_CYCLE = ['-', '--', '-.', ':']

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate performance dashboards for latent mapping tasks.")
    
    parser.add_argument("--save_dir", type=str, default=None, help="Plot Saving Directory")
    parser.add_argument('--save_name', default="mapper_dashboard", help="Saving optional name")
    
    parser.add_argument('--input_dirs', nargs='+', required=True, help="Paths to the latent_mapper directories to compare.")
    parser.add_argument('--names', nargs='+', help="Names of the training runs for the legend (must match number of directories).")
    parser.add_argument('--targets', nargs='+', required=True, help="List of targets to plot (e.g., Z SPECTYPE HALPHA_FLUX).")
    
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


def get_ordered_configs(config_values: pd.Series) -> List[str]:
    """Returns X-axis configuration order with known configs first and extras appended."""
    present = [str(v) for v in config_values.dropna().astype(str).unique().tolist()]
    preferred = [cfg for cfg in CONFIG_ORDER if cfg in present]
    extras = sorted([cfg for cfg in present if cfg not in CONFIG_ORDER])
    return preferred + extras

def load_and_merge_data(dirpaths: List[str | Path], test_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Scans directories for Mapper CSVs, merges all mappings found matching a pattern.
    """
    if test_names is None or len(test_names) != len(dirpaths):
        test_names = [f"Run {i+1}" for i in range(len(dirpaths))]

    df_list = []
    for dirpath_str, name in zip(dirpaths, test_names):
        dir_path = Path(dirpath_str)
        
        # 1. Direct CSV file?
        if dir_path.is_file() and dir_path.suffix == '.csv':
            csv_files = [dir_path]
        else:
            # 2. Try looking for mapper files directly in this dir or latent_mapper/
            csv_files = list(dir_path.glob("mapper_*.csv"))
            if not csv_files and (dir_path / "latent_mapper").is_dir():
                csv_files = list((dir_path / "latent_mapper").glob("mapper_*.csv"))
            
            # 3. Standard run-root glob: embeddings/*/downstream_tasks/latent_mapper/ or fallback
            if not csv_files:
                candidates = list(dir_path.glob("embeddings/*/downstream_tasks/latent_mapper/mapper_*.csv"))
                if not candidates:
                    candidates = list(dir_path.glob("embeddings/*/latent_mapper/mapper_*.csv"))
                    
                if candidates:
                    # Pick the folder that was modified last (latest embeddings run)
                    folders = list(set(c.parent for c in candidates))
                    folders.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    csv_files = list(folders[0].glob("mapper_*.csv"))

        if not csv_files:
            print(f"[WARNING] No mapper_*.csv files found for {dir_path}. Skipping.")
            continue
            
        for filepath in csv_files:
            try:
                df = pd.read_csv(filepath)
                df = df.dropna(subset=['Target', 'Task'])
                df['Test_Name'] = name
                df['Config'] = df['Mapping'] 
                df_list.append(df)
            except Exception as e:
                print(f"[ERROR] Could not read {filepath}: {e}")

    if not df_list:
        raise ValueError("No valid CSV files were loaded.")

    return pd.concat(df_list, ignore_index=True)

def generate_dashboard(
    df: pd.DataFrame,
    task_type: str,
    targets: List[str],
    save_dir: Path,
    save_name: str,
    global_test_order: Optional[List[str]] = None,
) -> None:
    if not targets:
        return

    task_df = df[(df['Task'].str.lower() == task_type.lower()) & (df['Target'].isin(targets))].copy()
    
    if task_df.empty:
        return

    task_df = task_df.dropna(axis=1, how='all')
    metadata_cols = ['Target', 'Task', 'Mapping', 'k_Neighbors', 'Test_Name', 'Config']
    metrics = [col for col in task_df.columns if col not in metadata_cols and pd.api.types.is_numeric_dtype(task_df[col])]
    
    available_targets = [t for t in targets if t in task_df['Target'].unique()]
    if not available_targets:
        print(f"[WARNING] No available targets for task {task_type}. Skipping.")
        return

    n_rows = len(available_targets)
    n_cols = len(metrics)
    if n_cols == 0:
        return

    fig_w = max(12, 4.2 * n_cols)
    fig_h = max(7, 5.0 * n_rows)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(fig_w, fig_h), squeeze=False)
    tests_present = task_df['Test_Name'].dropna().astype(str).unique().tolist()
    if global_test_order:
        tests = [t for t in global_test_order if t in tests_present]
    else:
        tests = tests_present
    style_map = build_style_map(global_test_order or tests)

    final_config_order = get_ordered_configs(task_df['Config'])
    if not final_config_order:
        print(f"[WARNING] No configuration labels found for task {task_type}. Skipping.")
        plt.close(fig)
        return
    config_to_x = {cfg: idx for idx, cfg in enumerate(final_config_order)}

    for i, target in enumerate(available_targets):
        target_df = task_df[task_df['Target'] == target]
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            for test_name in tests:
                test_df = target_df[target_df['Test_Name'] == test_name].copy()
                test_df['Config'] = pd.Categorical(test_df['Config'], categories=final_config_order, ordered=True)
                test_df = test_df.sort_values('Config')
                test_df = test_df[test_df['Config'].notna()].copy()
                test_df['x_pos'] = test_df['Config'].astype(str).map(config_to_x)
                test_df = test_df[test_df['x_pos'].notna()]
                if test_df.empty:
                    continue

                style = style_map[test_name]
                
                ax.plot(
                    test_df['x_pos'], 
                    test_df[metric], 
                    marker=style['marker'],
                    linestyle=style['linestyle'],
                    linewidth=1.0,
                    markersize=style['markersize'],
                    alpha=0.8,
                    label=test_name, 
                    color=style['color']
                )
            ax.set_title(f'{target} - {metric}')
            ax.set_xticks(range(len(final_config_order)))
            ax.set_xticklabels(final_config_order, rotation=45, ha='right')
            ax.tick_params(axis='x', labelsize=10, pad=6)
            ax.grid(True, linestyle='-', alpha=0.4)
            if hasattr(ax, 'set_box_aspect'):
                ax.set_box_aspect(0.9)
            if j == 0:
                ax.set_ylabel('Score / Value')

    # Reserve top space for title + legend (same strategy as downstream dashboard)
    fig_height = fig.get_figheight()
    legend_ncol = min(len(tests), 3) if tests else 1
    legend_nrows = ((len(tests) + legend_ncol - 1) // legend_ncol) if tests else 1

    title_offset_inches = 0.4
    legend_offset_inches = 0.9
    header_space_inches = 1.7 + 0.55 * max(0, legend_nrows - 1)

    title_y = 1.0 - (title_offset_inches / fig_height)
    legend_y = 1.0 - (legend_offset_inches / fig_height)
    top_limit = 1.0 - (header_space_inches / fig_height)

    plt.tight_layout(rect=[0, 0, 1, top_limit], h_pad=2.4, w_pad=1.0)
    fig.suptitle(
        f'Latent Mapper Performance Comparison: {task_type.capitalize()}',
        fontsize=20,
        y=title_y,
        va='top',
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc='upper center',
            ncol=legend_ncol,
            fontsize=12,
            bbox_to_anchor=(0.5, legend_y),
            frameon=True,
        )
    
    prefix = save_name or "mapper_dashboard"
    save_path = save_dir / f"{prefix}_{task_type}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] {task_type.capitalize()} Dashboard saved at: {save_dir}")

def main():
    args = parse_args()
    dir_paths_list = [Path(p) for p in args.input_dirs]

    try:
        df = load_and_merge_data(dir_paths_list, args.names)
    except Exception as e:
        print(f"[FATAL ERROR] Failed to load data: {e}")
        return

    # Handle Save Directory
    if args.save_dir:
        save_dir = Path(args.save_dir) / "latent_mapper"
    else:
        # Default to the location of the latest/primary CSV found
        # We search within the last directory provided (usually the current run)
        last_dir = dir_paths_list[-1]
        
        # Exact same logic used to load data
        csv_files = list(last_dir.glob("mapper_*.csv"))
        if not csv_files and (last_dir / "latent_mapper").is_dir():
            csv_files = list((last_dir / "latent_mapper").glob("mapper_*.csv"))
            
        if not csv_files:
            candidates = list(last_dir.glob("embeddings/*/downstream_tasks/latent_mapper/mapper_*.csv"))
            if not candidates:
                candidates = list(last_dir.glob("embeddings/*/latent_mapper/mapper_*.csv"))
            if candidates:
                folders = list(set(c.parent for c in candidates))
                folders.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                csv_files = list(folders[0].glob("mapper_*.csv"))
                
        if csv_files:
            save_dir = csv_files[0].parent
        else:
            save_dir = last_dir / "latent_mapper"
            
    save_dir.mkdir(parents=True, exist_ok=True)

    # Perform plotting for results found
    requested_targets = set(args.targets)
    regression_targets = df[(df['Task'] == 'regression') & (df['Target'].isin(requested_targets))]['Target'].unique().tolist()
    classification_targets = df[(df['Task'] == 'classification') & (df['Target'].isin(requested_targets))]['Target'].unique().tolist()

    # Fixed color/style mapping across tasks based on the global run list.
    global_test_order: List[str] = []
    for n in args.names or []:
        s = str(n)
        if s and s not in global_test_order:
            global_test_order.append(s)
    for n in df['Test_Name'].dropna().astype(str).unique().tolist():
        if n not in global_test_order:
            global_test_order.append(n)
    
    # Generate Dashboards independently
    if regression_targets:
        generate_dashboard(df, 'regression', regression_targets, save_dir, args.save_name, global_test_order)
    if classification_targets:
        generate_dashboard(df, 'classification', classification_targets, save_dir, args.save_name, global_test_order)

if __name__ == "__main__":
    main()