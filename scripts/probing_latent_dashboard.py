import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional

# Expected X-axis categories in strict order (Zero-Shot only)
CONFIG_ORDER = [
    'images->joint', 
    'images->spectra',
    'spectra->joint',
    'spectra->images',
    'joint->images',
    'joint->spectra'
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate performance dashboards for latent mapping tasks.")
    
    parser.add_argument("--save_dir", type=str, default=None, help="Plot Saving Directory")
    parser.add_argument('--save_name', default="mapper_dashboard", help="Saving optional name")
    
    parser.add_argument('--input_dirs', nargs='+', required=True, help="Paths to the latent_mapper directories to compare.")
    parser.add_argument('--names', nargs='+', help="Names of the training runs for the legend (must match number of directories).")
    parser.add_argument('--targets', nargs='+', required=True, help="List of targets to plot (e.g., Z SPECTYPE HALPHA_FLUX).")
    
    return parser.parse_args()

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
            
            # 3. Standard run-root glob: embeddings/*/latent_mapper/
            if not csv_files:
                candidates = list(dir_path.glob("embeddings/*/latent_mapper/mapper_*.csv"))
                if candidates:
                    # Pick the folder that was modified last (latest run)
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

def generate_dashboard(df: pd.DataFrame, task_type: str, targets: List[str], save_dir: Path, save_name: str) -> None:
    if not targets:
        return

    task_df = df[(df['Task'].str.lower() == task_type.lower()) & (df['Target'].isin(targets))].copy()
    
    if task_df.empty:
        return

    task_df = task_df.dropna(axis=1, how='all')
    metadata_cols = ['Target', 'Task', 'Mapping', 'k_Neighbors', 'Test_Name', 'Config']
    metrics = [col for col in task_df.columns if col not in metadata_cols and pd.api.types.is_numeric_dtype(task_df[col])]
    
    n_rows = len(targets)
    n_cols = len(metrics)
    if n_cols == 0:
        return

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    tests = task_df['Test_Name'].unique()
    colors = ["black", "red", "mediumseagreen", "dodgerblue", "darkorchid", "darkorange", "saddlebrown"]
    markers = ['.', '*', 'v', 'P', 's', 'X', 'D']
    sizes = ['10', '9', '8', '8', '7', '8', '7']

    available_configs = [c for c in CONFIG_ORDER if c in task_df['Config'].unique()]
    missing_configs = sorted([c for c in task_df['Config'].unique() if c not in available_configs])
    final_config_order = available_configs + missing_configs

    for i, target in enumerate(targets):
        target_df = task_df[task_df['Target'] == target]
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            for test_idx, test_name in enumerate(tests):
                test_df = target_df[target_df['Test_Name'] == test_name].copy()
                test_df['Config'] = pd.Categorical(test_df['Config'], categories=final_config_order, ordered=True)
                test_df = test_df.sort_values('Config')
                
                ax.plot(
                    test_df['Config'], 
                    test_df[metric], 
                    marker=markers[test_idx % len(markers)], 
                    linestyle='--', 
                    linewidth=0.5,
                    markersize=sizes[test_idx % len(sizes)],
                    alpha=0.5,
                    label=test_name, 
                    color=colors[test_idx % len(colors)]
                )
            ax.set_title(f'{target} - {metric}')
            ax.set_xticks(range(len(final_config_order)))
            ax.set_xticklabels(final_config_order, rotation=45, ha='right')
            ax.grid(True, linestyle='-', alpha=0.4)
            if j == 0:
                ax.set_ylabel('Score / Value')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(f'Latent Mapper Performance Comparison: {task_type.capitalize()}', fontsize=20)
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=min(len(tests), 4), bbox_to_anchor=(0.5, 0.92))
    
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
        mapper_files = list(last_dir.glob("mapper_*.csv"))
        if not mapper_files and (last_dir / "latent_mapper").is_dir():
            mapper_files = list((last_dir / "latent_mapper").glob("mapper_*.csv"))
        
        if mapper_files:
            save_dir = mapper_files[-1].parent
        else:
            save_dir = last_dir / "latent_mapper"
            
    save_dir.mkdir(parents=True, exist_ok=True)

    # Perform plotting for results found
    requested_targets = set(args.targets)
    regression_targets = df[(df['Task'] == 'regression') & (df['Target'].isin(requested_targets))]['Target'].unique().tolist()
    classification_targets = df[(df['Task'] == 'classification') & (df['Target'].isin(requested_targets))]['Target'].unique().tolist()
    
    # Generate Dashboards independently
    if regression_targets:
        generate_dashboard(df, 'regression', regression_targets, save_dir, args.save_name)
    if classification_targets:
        generate_dashboard(df, 'classification', classification_targets, save_dir, args.save_name)

if __name__ == "__main__":
    main()