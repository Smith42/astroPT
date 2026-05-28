import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import os

st.set_page_config(layout="wide", page_title="AstroPT Probing Dashboard")

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

def get_ordered_configs(df: pd.DataFrame) -> list:
    """Dynamically builds X-axis order based on present modalities and probes."""
    PROBE_ORDER = ['KNN', 'LP', 'MLP']
    # Sort alphabetically, with 'Joint' or anything containing 'joint' always at the end
    mods = sorted(
        df['Modality'].dropna().unique(),
        key=lambda x: (x.lower() == 'joint' or 'joint' in x.lower(), x)
    )
    configs = []
    for m in mods:
        available_probes = sorted(
            df[df['Modality'] == m]['Probe'].dropna().unique(), 
            key=lambda x: (x not in PROBE_ORDER, PROBE_ORDER.index(x) if x in PROBE_ORDER else 0)
        )
        for p in available_probes:
            configs.append(f"{m} {p}")
    return configs

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive Probing Dashboard args")
    parser.add_argument('--csv_path', nargs='+', help="Paths to downstream_results.csv files to compare")
    parser.add_argument('--names', nargs='+', help="Legend names corresponding to each CSV")
    parser.add_argument('--run_dirs', nargs='+', help="Directories of runs to scan dynamically")
    parser.add_argument('--emb_filter', type=str, default=None, help="Filter for embedding folders to plot across runs")
    return parser.parse_known_args()

def resolve_run_dir(run_dir_str: str):
    """Finds the most appropriate downstream_results.csv for a given run directory."""
    run_path = Path(run_dir_str)
    
    # 1. Is it literally the CSV file?
    if run_path.is_file() and run_path.name == 'downstream_results.csv':
        return str(run_path)
        
    candidates = []
    # 2. Directly inside?
    if (run_path / "downstream_results.csv").is_file():
        candidates.append(run_path / "downstream_results.csv")
    # 3. Inside a downstream_tasks folder?
    if (run_path / "downstream_tasks" / "downstream_results.csv").is_file():
        candidates.append(run_path / "downstream_tasks" / "downstream_results.csv")
    # 4. In an embeddings subfolder?
    candidates.extend(list(run_path.glob("embeddings/*/downstream_tasks/downstream_results.csv")))
    # 5. Fallback glob
    if not candidates:
        candidates.extend(list(run_path.glob("**/downstream_tasks/downstream_results.csv")))
        
    if not candidates:
        return None
        
    # Prefer 'best' checkpoints, then fallback to newest
    best_csvs = [c for c in candidates if "best" in str(c)]
    pool = best_csvs if best_csvs else candidates
    pool.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return str(pool[0])

@st.cache_data
def scan_for_runs(logs_dir: str):
    """Scan the logs directory for runs containing downstream_results.csv"""
    logs_path = Path(logs_dir)
    runs = {}
    if not logs_path.exists():
        return runs
        
    for csv_path in logs_path.glob("**/downstream_tasks/downstream_results.csv"):
        # Basic naming inference similar to the static dashboard
        try:
            potential_root = csv_path.parent.parent.parent.parent
            train_name = potential_root.name
            config_path = potential_root / "weights" / "config.json"
            
            if config_path.is_file():
                with open(config_path, 'r') as f:
                    train_name = json.load(f).get("train_name", train_name)
                    
            emb_folder_name = csv_path.parent.parent.name
            if "resnet18_images_supervised" in str(csv_path):
                 train_name = f"ResNet18 Supervised ({csv_path.parent.parent.parent.name})"
            elif "spectra_supervised_baseline" in str(csv_path):
                 train_name = f"Spectra Supervised ({csv_path.parent.parent.parent.name})"
            else:
                if not any(x in emb_folder_name for x in ["best_meanrank", "img-mean", "spec-rank"]):
                    train_name += f" [{emb_folder_name}]"
            
            # Use absolute path as key to avoid collisions
            runs[str(csv_path)] = train_name
        except Exception as e:
            continue

    # Also discover baseline result aggregates and separate CSV files
    baseline_patterns = [
        "**/resnet_baseline_results_all.csv",
        "**/supervised_baseline_images_results.csv",
        "**/supervised_baseline_spectra_results.csv"
    ]
    for pattern in baseline_patterns:
        for csv_path in logs_path.glob(pattern):
            try:
                name = csv_path.name
                if "resnet18_images_supervised" in str(csv_path) and "resnet_baseline_results_all" in name:
                    train_name = "ResNet18 Supervised (All Targets)"
                elif "spectra_supervised_baseline" in str(csv_path) and "resnet_baseline_results_all" in name:
                    train_name = "Spectra Supervised (All Targets)"
                elif "supervised_baseline_images_results" in name:
                    train_name = "Images Supervised"
                elif "supervised_baseline_spectra_results" in name:
                    train_name = "Spectra Supervised"
                else:
                    train_name = csv_path.parent.name
                runs[str(csv_path)] = train_name
            except Exception:
                continue

    return runs

@st.cache_data
def load_data(selected_csvs: list, runs_dict: dict):
    """Load and merge selected CSVs, applying normalization and phase routing logic"""
    df_list = []
    for csv_path in selected_csvs:
        try:
            df = pd.read_csv(csv_path)
            df = df.dropna(subset=['Target', 'Task'])
            
            # Detect if this specific CSV contains phase-based modalities
            has_phases = df['Modality'].astype(str).str.contains('_phase').any()
            name = runs_dict[csv_path]
            
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
            st.error(f"Error loading {csv_path}: {e}")
            
    if not df_list:
        return pd.DataFrame()
        
    return pd.concat(df_list, ignore_index=True)

def main():
    st.title("🔭 AstroPT Interactive Probing Dashboard")
    
    # Sidebar setup
    st.sidebar.header("Data Selection")
    
    runs_dict = {}
    csv_paths_list = []
    names_list = []
    
    args, unknown = parse_args()
    
    # 1. Parse run_dirs if provided
    if args.run_dirs:
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
            for run_dir_str in args.run_dirs:
                run_path = Path(run_dir_str)
                if run_path.is_file() and run_path.suffix == '.csv':
                    csv_paths_list.append(run_path)
                    try:
                        name_candidate = run_path.parent.parent.parent.parent.name
                        config_path = run_path.parent.parent.parent.parent / "weights" / "config.json"
                        if config_path.is_file():
                            with open(config_path, 'r') as f:
                                name_candidate = json.load(f).get("train_name", name_candidate)
                        names_list.append(f"{name_candidate} [{run_path.parent.parent.name}]")
                    except:
                        names_list.append(run_path.name)
                    continue

                candidates = []
                if (run_path / "downstream_results.csv").is_file():
                    candidates.append(run_path / "downstream_results.csv")
                if (run_path / "downstream_tasks" / "downstream_results.csv").is_file():
                    candidates.append(run_path / "downstream_tasks" / "downstream_results.csv")
                candidates.extend(list(run_path.glob("embeddings/*/downstream_tasks/downstream_results.csv")))
                
                if not candidates:
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
        names_list = args.names if args.names else [f"Run {i+1}" for i in range(len(csv_paths_list))]
        
    for p, n in zip(csv_paths_list, names_list):
        runs_dict[str(p)] = n
        
    # Always scan logs base for additional runs to populate dropdown options
    default_logs = "/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs"
    logs_dir = st.sidebar.text_input("Logs Directory for scanning", value=default_logs)
    scanned_dict = scan_for_runs(logs_dir)
    
    for k, v in scanned_dict.items():
        if k not in runs_dict:
            runs_dict[k] = v

    if not runs_dict:
        st.warning("No run tracking files found.")
        return
        
    # By default, select the runs specified via CLI. If none, select the top 2 scanned.
    default_selection = [str(p) for p in csv_paths_list] if csv_paths_list else list(runs_dict.keys())[:2]
    
    # Multi-select for runs
    selected_csvs = st.sidebar.multiselect(
        "Select Runs specifically", 
        options=list(runs_dict.keys()), 
        format_func=lambda x: runs_dict[x],
        default=[s for s in default_selection if s in runs_dict]
    )
    
    if not selected_csvs:
        st.info("Please select at least one run from the sidebar to continue.")
        return
        
    with st.spinner("Loading Data..."):
        df = load_data(selected_csvs, runs_dict)
        
    if df.empty:
        st.error("No valid data could be loaded.")
        return

    # Extract available targets and metrics
    all_targets = sorted(df['Target'].dropna().unique().tolist())
    metadata_cols = ['Target', 'Task', 'Modality', 'Probe', 'Test_Name', 'Config']
    all_metrics = [col for col in df.columns if col not in metadata_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    # Multi-select for targets
    selected_targets_raw = st.sidebar.multiselect(
        "Select Targets to Compare", 
        options=all_targets, 
        default=all_targets[:2] if len(all_targets) >= 2 else all_targets,
    )
    
    selected_targets = selected_targets_raw
    # Simple list reordering workaround using data_editor
    if len(selected_targets_raw) > 1:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔄 Change Order")
        
        # Check if the ordering has changed to trigger a rerun if needed
        order_df = pd.DataFrame({
            "Order": list(range(len(selected_targets_raw))),
            "Target": selected_targets_raw
        })
        
        edited_df = st.sidebar.data_editor(
            order_df, 
            num_rows="fixed", 
            use_container_width=True,
            hide_index=True,
            key="target_order_editor", # Key to maintain state
            column_config={
                "Order": st.column_config.NumberColumn("Pos", min_value=0, max_value=len(selected_targets_raw)-1, step=1),
                "Target": st.column_config.TextColumn("Target", disabled=True)
            }
        )
        
        # Apply re-sort
        selected_targets = edited_df.sort_values("Order")["Target"].tolist()
    
    # Pre-calculate active tests to avoid redundancy
    active_tests = df[df['Target'].isin(selected_targets)]['Test_Name'].unique()
    active_tests = sorted(active_tests)
    
    # Multi-select for metrics
    selected_metrics = st.sidebar.multiselect("Select Metrics", options=all_metrics, default=["R2", "RMSE"] if "R2" in all_metrics else all_metrics[:2])
    
    # Axis Range Control
    st.sidebar.markdown("---")
    st.sidebar.subheader("Axis Range Control")
    enable_fixed_range = st.sidebar.checkbox("Enable Fixed Y-Axis Range", value=False)
    y_range = st.sidebar.slider("Y-Axis Range", 0.0, 1.0, (0.0, 1.0)) if enable_fixed_range else None

    # Complete possible configs dynamically
    all_possible_configs = get_ordered_configs(df)

    # Multi-select for configs (X-axis)
    default_configs = [cfg for cfg in all_possible_configs if not any(x in cfg for x in ['CLS', 'cls'])]
    selected_configs = st.sidebar.multiselect("Select Configs (X-Axis)", options=all_possible_configs, default=default_configs)

    # Theme and UI setup (Moved to the bottom)
    st.sidebar.markdown("---")
    theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)
    plotly_template = "plotly_white" if theme == "Light" else "plotly_dark"
    
    if not selected_targets or not selected_metrics or not selected_configs:
        st.info("Please select Targets, Metrics, and Configs to visualize.")
        return

    # Define the active ordered configs based on selection
    active_config_order = [cfg for cfg in all_possible_configs if cfg in selected_configs]

    # Inject custom CSS for Theme, Multiselect text wrapping, and Sticky Legend
    custom_css = f"""
    <style>
    /* Force Light or Dark background based on selection */
    .stApp {{
        background-color: {"#FFFFFF" if theme == "Light" else "#0E1117"};
        color: {"#000000" if theme == "Light" else "#FAFAFA"};
    }}
    
    /* Make multiselect dropdown options text wrap so long run names are fully readable */
    .stMultiSelect div[data-baseweb="select"] ul li {{
        white-space: normal !important;
        overflow-wrap: break-word !important;
    }}
    
    /* Sticky Global Legend Container */
    .sticky-legend {{
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        background-color: {"rgba(255, 255, 255, 0.95)" if theme == "Light" else "rgba(14, 17, 23, 0.95)"};
        padding: 10px 20px;
        border-radius: 8px;
        border: 1px solid {"#ddd" if theme == "Light" else "#444"};
        margin-bottom: 2rem;
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        justify-content: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .legend-item {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
        font-weight: 500;
    }}
    .legend-color {{
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Global Plotly Colors and Symbols
    plotly_colors = px.colors.qualitative.Plotly
    plotly_symbols = ['circle', 'diamond', 'square', 'cross', 'star', 'triangle-up', 'x', 'pentagon']
    
    # Filter only the test names that will actually be plotted
    active_tests = df[df['Target'].isin(selected_targets)]['Test_Name'].unique()
    active_tests = sorted(active_tests)
    
    # Create a mapping for consistent styles across all subplots
    style_mapping = {
        test: {
            "color": plotly_colors[i % len(plotly_colors)],
            "symbol": plotly_symbols[i % len(plotly_symbols)]
        } 
        for i, test in enumerate(active_tests)
    }
    
    # Generate the Sticky Legend HTML
    legend_html = '<div class="sticky-legend">'
    symbol_to_css = {
        'circle': 'border-radius: 50%;',
        'diamond': 'transform: rotate(45deg); width: 8px; height: 8px;',
        'square': 'border-radius: 0%;',
        'cross': 'clip-path: polygon(0% 40%, 40% 40%, 40% 0%, 60% 0%, 60% 40%, 100% 40%, 100% 60%, 60% 60%, 60% 100%, 40% 100%, 40% 60%, 0% 60%);',
        'star': 'clip-path: polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%);',
        'triangle-up': 'clip-path: polygon(50% 0%, 0% 100%, 100% 100%);',
        'x': 'clip-path: polygon(20% 0%, 0% 20%, 30% 50%, 0% 80%, 20% 100%, 50% 70%, 80% 100%, 100% 80%, 70% 50%, 100% 20%, 80% 0%, 50% 30%);',
        'pentagon': 'clip-path: polygon(50% 0%, 100% 38%, 82% 100%, 18% 100%, 0% 38%);'
    }

    for test_name in active_tests:
        style = style_mapping[test_name]
        css_shape = symbol_to_css.get(style["symbol"], "border-radius: 50%;")
        
        # Render a symbol-aware marker in the legend
        legend_content = f'<div class="legend-item"><div style="width: 14px; height: 14px; display: flex; align-items: center; justify-content: center;"><div style="width: 10px; height: 10px; background-color: {style["color"]}; {css_shape}"></div></div><span>{test_name}</span></div>'
        legend_html += legend_content
        
    legend_html += '</div>'
    
    st.markdown(legend_html, unsafe_allow_html=True)
    
    # Plotting loop
    for metric in selected_metrics:
        st.subheader(f"Metric: {metric}")
        
        # Inject CSS for horizontal scrolling and maintaining square proportions
        st.markdown(
            f"""
            <style>
            /* Create a scrollable container for the plots */
            div[data-testid="stHorizontalBlock"] {{
                overflow-x: auto;
                flex-wrap: nowrap !important;
                gap: 20px;
                padding-bottom: 25px; /* Added more space for the scrollbar */
                
                /* Custom Scrollbar Styling for visibility */
                scrollbar-width: auto;
                scrollbar-color: {"#888 #eee" if theme == "Light" else "#555 #222"};
            }}
            
            /* Chrome, Edge, and Safari scrollbar styles */
            div[data-testid="stHorizontalBlock"]::-webkit-scrollbar {{
                height: 12px;
            }}
            div[data-testid="stHorizontalBlock"]::-webkit-scrollbar-track {{
                background: {"#f1f1f1" if theme == "Light" else "#1a1c23"};
                border-radius: 10px;
            }}
            div[data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb {{
                background: {"#888" if theme == "Light" else "#555"};
                border-radius: 10px;
                border: 2px solid {"#f1f1f1" if theme == "Light" else "#1a1c23"};
            }}
            div[data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb:hover {{
                background: {"#555" if theme == "Light" else "#888"};
            }}

            /* Ensure each column has a fixed minimum width so they don't squash */
            div[data-testid="stColumn"] {{
                min-width: 500px !important;
                flex: 0 0 auto !important;
            }}
            </style>
            """, 
            unsafe_allow_html=True
        )

        cols = st.columns(len(selected_targets))
        
        for col, target in zip(cols, selected_targets):
            with col:
                # Filter strictly by selected configs
                target_df = df[(df['Target'] == target) & (df[metric].notna()) & (df['Config'].isin(selected_configs))].copy()
                
                if target_df.empty:
                    st.write(f"No data for {target}")
                    continue
                    
                # Create a Line Group column to break lines between modalities
                target_df['Line_Group'] = target_df['Test_Name'] + " " + target_df['Modality']
                    
                # Sort exactly by config_order
                target_df['Config'] = pd.Categorical(target_df['Config'], categories=active_config_order, ordered=True)
                target_df = target_df.sort_values(['Config', 'Test_Name'])

                # Create plotly figure
                fig = px.line(
                    target_df, 
                    x="Config", 
                    y=metric, 
                    color="Test_Name",
                    symbol="Test_Name", # Added unique symbols
                    line_group="Line_Group",
                    markers=True,
                    template=plotly_template,
                    category_orders={"Config": active_config_order, "Test_Name": active_tests},
                    color_discrete_map={test: style["color"] for test, style in style_mapping.items()},
                    symbol_map={test: style["symbol"] for test, style in style_mapping.items()}
                )

                # Set lines thinner and markers slightly larger
                fig.update_traces(
                    line=dict(width=1.5), # Thinner lines
                    marker=dict(size=10)   # More prominent points
                )

                # Dynamically calculate background colors for modalities
                dynamic_shapes = []
                opacity_val = 0.03 if theme == "Dark" else 0.05
                mod_colors = {'EuclidImage': 'blue', 'DESISpectrum': 'green', 'Joint': 'orange'}
                
                for mod, mod_color in mod_colors.items():
                    indices = [i for i, cfg in enumerate(active_config_order) if cfg.startswith(mod)]
                    if indices:
                        x0, x1 = min(indices) - 0.5, max(indices) + 0.5
                        dynamic_shapes.append(
                            dict(type="rect", xref="x", yref="paper", x0=x0, x1=x1, y0=0, y1=1, 
                                 fillcolor=mod_color, opacity=opacity_val, layer="below", line_width=0)
                        )

                fig.update_layout(shapes=dynamic_shapes)
                
                bg_color = "#FFFFFF" if theme == "Light" else "#0E1117"
                grid_color = "#EBF0F8" if theme == "Light" else "#22262F"
                text_color = "#000000" if theme == "Light" else "#FAFAFA"
                
                fig.update_layout(
                    plot_bgcolor=bg_color,
                    paper_bgcolor=bg_color,
                    font=dict(color=text_color),
                    xaxis=dict(
                        gridcolor=grid_color, 
                        zerolinecolor=grid_color, 
                        showline=True, 
                        linewidth=1, 
                        linecolor=text_color, 
                        mirror=True,
                        tickfont=dict(size=16)
                    ),
                    yaxis=dict(
                        gridcolor=grid_color, 
                        zerolinecolor=grid_color, 
                        showline=True, 
                        linewidth=1, 
                        linecolor=text_color, 
                        mirror=True,
                        tickfont=dict(size=18),
                        range=y_range if enable_fixed_range else None
                    ),
                    title=dict(
                        text=f"<b>{target} ({metric})</b>",
                        x=0.5,
                        xanchor='center',
                        font=dict(size=20, color=text_color)
                    ),
                    xaxis_title="",
                    yaxis_title=metric,
                    showlegend=False, # Removed internal legend as requested
                    height=500,
                    margin=dict(t=60, b=80) 
                )
                
                # Plot - explicitly ignoring Streamlit's default theme so our Dark/Light mode overrides persist
                st.plotly_chart(fig, theme=None, width="stretch")

if __name__ == "__main__":
    main()
