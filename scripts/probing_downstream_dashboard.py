import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional

# Expected X-axis categories in strict order
CONFIG_ORDER = [
    'images LP', 
    'spectra LP',
    'joint LP',
    'images MLP',
    'spectra MLP', 
    'joint MLP'
]

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Generate performance dashboards for downstream tasks.")
    
    parser.add_argument("--save_dir", type=str, required=True, help="Plot Saving Directory")
    parser.add_argument('--save_name', default="donwstream_dashboard", help="Saving optional name")
    parser.add_argument('--csv_path', nargs='+', required=True, help="Paths to the CSV files to compare.")
    parser.add_argument('--names', nargs='+', help="Names of the tests for the legend (must match number of files).")
    parser.add_argument('--targets', nargs='+', required=True, help="List of targets to plot (e.g., Z SPECTYPE HALPHA_FLUX).")
    
    return parser.parse_args()

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

def generate_dashboard(df: pd.DataFrame, task_type: str, targets: List[str], save_dir: str, save_name: str) -> None:
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
    
    n_rows = len(targets)
    n_cols = len(metrics)

    if n_cols == 0:
        print(f"[WARNING] No numeric metrics found for {task_type}. Skipping dashboard.")
        return

    # Set up the matplotlib figure
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    tests = task_df['Test_Name'].unique()
    colors = ["black","red","mediumseagreen","dodgerblue","darkorchid"]
    markers = ['.','*','v','P','s']
    sizes = ['10','8','7','7','5']

    for i, target in enumerate(targets):
        target_df = task_df[task_df['Target'] == target]
        
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            
            for test_idx, test_name in enumerate(tests):
                test_df = target_df[target_df['Test_Name'] == test_name].copy()
                
                # Ensure the data follows the expected X-axis order
                test_df['Config'] = pd.Categorical(test_df['Config'], categories=CONFIG_ORDER, ordered=True)
                test_df = test_df.sort_values('Config')
                
                ax.plot(
                    test_df['Config'], 
                    test_df[metric], 
                    marker=markers[test_idx], 
                    linestyle='--', 
                    linewidth=0.5,
                    markersize=sizes[test_idx],
                    alpha=0.5,
                    label=test_name, 
                    color=colors[test_idx]
                )

            # Formatting the subplot
            ax.set_title(f'{target} - {metric}')
            ax.set_xticks(range(len(CONFIG_ORDER)))
            ax.set_xticklabels(CONFIG_ORDER, rotation=45, ha='right')
            ax.grid(True, linestyle='-', alpha=0.4)
            
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
    
    plt.tight_layout(rect=[0, 0, 1, top_limit])

    fig.suptitle(f'Downstream Task Performance Comparison: {task_type.capitalize()}', 
                 fontsize=20, y=title_y, va='top')
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(tests), 
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

    save_dir = Path(args.save_dir) / "downstream_tasks"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    csv_paths_list = [Path(p) for p in args.csv_path]

    # Load Data
    try:
        df = load_and_merge_data(csv_paths_list, args.names)
    except Exception as e:
        print(f"[FATAL ERROR] Failed to load data: {e}")
        return

    # Segregate targets by task type automatically
    requested_targets = set(args.targets)
    
    # Find which requested targets belong to regression and which to classification
    regression_targets = df[(df['Task'] == 'regression') & (df['Target'].isin(requested_targets))]['Target'].unique().tolist()
    classification_targets = df[(df['Task'] == 'classification') & (df['Target'].isin(requested_targets))]['Target'].unique().tolist()
    
    # Check for requested targets that weren't found at all
    found_targets = set(regression_targets + classification_targets)
    missing_targets = requested_targets - found_targets
    if missing_targets:
        print(f"[WARNING] The following requested targets were not found in the data: {missing_targets}")

    # Generate Dashboards independently
    generate_dashboard(df, 'regression', regression_targets, save_dir, args.save_name)
    generate_dashboard(df, 'classification', classification_targets, save_dir, args.save_name)

if __name__ == "__main__":
    main()