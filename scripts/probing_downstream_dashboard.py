import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json

# Expected X-axis categories in strict order
CONFIG_ORDER = [
    'images KNN', 
    'spectra KNN',
    'joint KNN',
    'images LP', 
    'spectra LP',
    'joint LP',
    'images MLP',
    'spectra MLP', 
    'joint MLP'
]

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
            
            # Clean empty rows (like trailing commas at the end of the file)
            df = df.dropna(subset=['Target', 'Task'])
            
            df['Test_Name'] = name
            # Create the X-axis category
            df['Config'] = df['Modality'] + ' ' + df['Probe']
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
    save_dir: str,
    save_name: str,
    global_test_order: Optional[List[str]] = None,
) -> None:
    """
    Generates and saves a matplotlib dashboard for a specific task type.
    Dynamically drops empty metric columns based on the task type.

    Args:
        df (pd.DataFrame): The complete dataset containing all tests.
        task_type (str): The type of task to filter by ('regression' or 'classification').
        targets (List[str]): List of target labels belonging to this task.
        output_dir (str): Directory where the output image will be saved.
    """
    if not targets:
        return # Silently skip if no targets were requested for this specific task

    # Filter dataset by task and requested targets
    task_df = df[(df['Task'].str.lower() == task_type.lower()) & (df['Target'].isin(targets))].copy()
    
    if task_df.empty:
        return

    # Drop columns that are completely NaN (e.g., dropping R2, RMSE for classification tasks)
    task_df = task_df.dropna(axis=1, how='all')

    # Identify metric columns 
    metadata_cols = ['Target', 'Task', 'Modality', 'Probe', 'Test_Name', 'Config']
    metrics = [col for col in task_df.columns if col not in metadata_cols and pd.api.types.is_numeric_dtype(task_df[col])]
    
    # Keep only targets that are present in this task subset.
    available_targets = [t for t in targets if t in task_df['Target'].unique()]
    if not available_targets:
        print(f"[WARNING] No available targets for task {task_type}. Skipping.")
        return

    n_rows = len(available_targets)
    n_cols = len(metrics)

    if n_cols == 0:
        print(f"[WARNING] No numeric metrics found for {task_type}. Skipping dashboard.")
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
    config_order = get_ordered_configs(task_df['Config'])
    if not config_order:
        print(f"[WARNING] No configuration labels found for task {task_type}. Skipping dashboard.")
        plt.close(fig)
        return
    config_to_x = {cfg: idx for idx, cfg in enumerate(config_order)}

    for i, target in enumerate(available_targets):
        target_df = task_df[task_df['Target'] == target]
        
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            
            for test_name in tests:
                test_df = target_df[target_df['Test_Name'] == test_name].copy()
                
                # Ensure the data follows the expected X-axis order
                test_df['Config'] = pd.Categorical(test_df['Config'], categories=config_order, ordered=True)
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

            # Formatting the subplot
            ax.set_title(f'{target} - {metric}')
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
    
    # Fixed inches
    title_offset_inches = 0.5    
    legend_offset_inches = 0.9   
    header_space_inches = 1.6
    
    # Relative coordinates
    title_y = 1.0 - (title_offset_inches / fig_height)
    legend_y = 1.0 - (legend_offset_inches / fig_height)
    top_limit = 1.0 - (header_space_inches / fig_height)
    
    plt.tight_layout(rect=[0, 0, 1, top_limit], h_pad=2.4, w_pad=1.0)

    fig.suptitle(f'Downstream Task Performance Comparison: {task_type.capitalize()}', 
                 fontsize=20, y=title_y, va='top')
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=min(len(tests), 4), 
                   fontsize=12, bbox_to_anchor=(0.5, legend_y), frameon=True)
    
    # Save the figure

    prefix = save_name or "downstream_dashboard"
    save_name_dash = f"{prefix}_{task_type}.png"
    save_path = Path(save_dir) / save_name_dash
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] {task_type.capitalize()} Dashboard saved successfully at: {save_dir}")

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
                # We try several levels to be robust to "deep" paths provided by the user
                candidates = []
                # 1. Is it right here?
                if (run_path / "downstream_results.csv").is_file():
                    candidates.append(run_path / "downstream_results.csv")
                # 2. Is it in downstream_tasks/?
                if (run_path / "downstream_tasks" / "downstream_results.csv").is_file():
                    candidates.append(run_path / "downstream_tasks" / "downstream_results.csv")
                # 3. Standard run-root glob
                candidates.extend(list(run_path.glob("embeddings/*/downstream_tasks/downstream_results.csv")))
                
                if not candidates:
                    print(f"[WARNING] No downstream_results.csv found in {run_path}. Skipping.")
                    continue
                
                if args.emb_filter:
                    filtered = [c for c in candidates if args.emb_filter in c.parent.parent.name]
                    if filtered:
                        candidates = filtered
                
                # Tie-breaker logic: 
                # 1. Prioritize files with "best" in the path name
                # 2. Between those (or all), pick the LATEST one modified (user's suggestion)
                best_csvs = [c for c in candidates if "best" in str(c)]
                pool = best_csvs if best_csvs else candidates
                
                # Sort by modification time (most recent first)
                pool.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                chosen_csv = pool[0]
                
                # Infer Name
                # We try to go up from chosen_csv to find the run root for the config.json
                # Structure: run_root / embeddings / [EMB_NAME] / downstream_tasks / results.csv
                # Or just use the provided run_path as root if it has weights/
                potential_root = run_path
                if not (potential_root / "weights" / "config.json").is_file():
                    # Try going up from chosen_csv
                    try:
                        test_root = chosen_csv.parent.parent.parent.parent
                        if (test_root / "weights" / "config.json").is_file():
                            potential_root = test_root
                    except:
                        pass

                emb_folder_name = chosen_csv.parent.parent.name if "downstream_tasks" in chosen_csv.parent.name else chosen_csv.parent.name
                config_path = potential_root / "weights" / "config.json"
                train_name = potential_root.name
                if config_path.is_file():
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            train_name = config.get("train_name", train_name)
                    except Exception:
                        pass
                
                if "best_meanrank" not in emb_folder_name and "img-mean" not in emb_folder_name:
                    train_name += f" [{emb_folder_name}]"
                    
                csv_paths_list.append(chosen_csv)
                names_list.append(train_name)
    
    elif args.csv_path:
        csv_paths_list = [Path(p) for p in args.csv_path]
        names_list = args.names
    else:
        print("[FATAL ERROR] You must provide either --csv_path or --run_dirs.")
        return

    # Load Data
    try:
        df = load_and_merge_data(csv_paths_list, names_list)
    except Exception as e:
        print(f"[FATAL ERROR] Failed to load data: {e}")
        return

    # Handle Save Directory
    if args.save_dir:
        save_dir = Path(args.save_dir) / "downstream_tasks"
    else:
        # Default to the location of the latest/primary CSV found
        # We use [-1] because the current run is typically appended last in the SLURM script
        if csv_paths_list:
            save_dir = csv_paths_list[-1].parent
        else:
            save_dir = Path("results/downstream_tasks")
    save_dir.mkdir(parents=True, exist_ok=True)

    task_col = df['Task'].astype(str).str.lower()

    if args.all_targets:
        regression_targets = sorted(df[task_col == 'regression']['Target'].dropna().astype(str).unique().tolist())
        classification_targets = sorted(df[task_col == 'classification']['Target'].dropna().astype(str).unique().tolist())
        print(
            f"[INFO] --all_targets enabled. Found {len(regression_targets)} regression "
            f"and {len(classification_targets)} classification targets."
        )
    else:
        if not args.targets:
            print("[FATAL ERROR] You must provide --targets or enable --all_targets")
            return

        # Segregate requested targets by task type automatically
        requested_targets = set(args.targets)

        regression_targets = sorted(
            df[(task_col == 'regression') & (df['Target'].isin(requested_targets))]['Target']
            .dropna().astype(str).unique().tolist()
        )
        classification_targets = sorted(
            df[(task_col == 'classification') & (df['Target'].isin(requested_targets))]['Target']
            .dropna().astype(str).unique().tolist()
        )

        # Check for requested targets that weren't found at all
        found_targets = set(regression_targets + classification_targets)
        missing_targets = requested_targets - found_targets
        if missing_targets:
            print(f"[WARNING] The following requested targets were not found in the data: {missing_targets}")

    # Fixed color/style mapping across tasks based on the global run list.
    global_test_order: List[str] = []
    for n in names_list or []:
        s = str(n)
        if s and s not in global_test_order:
            global_test_order.append(s)
    for n in df['Test_Name'].dropna().astype(str).unique().tolist():
        if n not in global_test_order:
            global_test_order.append(n)

    # Generate Dashboards independently
    generate_dashboard(df, 'regression', regression_targets, save_dir, args.save_name, global_test_order)
    generate_dashboard(df, 'classification', classification_targets, save_dir, args.save_name, global_test_order)

if __name__ == "__main__":
    main()