import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import os
from astropy.io import fits
from astropy.table import Table

# Page Config
st.set_page_config(layout="wide", page_title="AstroPT Probing Predictions Dashboard")

# Global variables and constants
CLASSIFICATION_TARGETS = ['SPECTYPE', 'data_set_release']
FLUX_TARGETS = ['flux_detection_total', 'HALPHA_EW', 'HALPHA_FLUX', 'NII_6584_FLUX', 'OIII_5007_FLUX', 'HBETA_FLUX']

def apply_scaling(vals: np.ndarray, target: str) -> np.ndarray:
    """Apply asinh scaling for flux/EW targets to improve visualization of long tails."""
    if target in FLUX_TARGETS:
        return np.arcsinh(vals)
    return vals

@st.cache_data
def get_fits_columns(fits_path: str) -> list:
    """Read FITS binary table column names efficiently."""
    try:
        with fits.open(fits_path) as hdul:
            # Table is typically in HDU 1
            cols = hdul[1].columns.names
            return list(cols)
    except Exception as e:
        st.error(f"Error reading column names from FITS catalog: {e}")
        return []

@st.cache_data
def load_fits_columns(fits_path: str, cols_to_load: list) -> pd.DataFrame:
    """Load only the selected columns from FITS catalog to save RAM and time."""
    try:
        # Enforce unique columns
        cols_to_load = list(set(cols_to_load))
        
        with fits.open(fits_path, memmap=True) as hdul:
            data = hdul[1].data
            df_dict = {}
            for col in cols_to_load:
                # Find matching column name in FITS (case-insensitive)
                fit_col = next((c for c in data.columns.names if c.lower() == col.lower()), None)
                if fit_col:
                    # Convert to numpy array safely and convert big-endian data to native host order
                    arr = np.array(data[fit_col])
                    if arr.dtype.byteorder == '>':
                        arr = arr.byteswap().newbyteorder()
                    df_dict[col] = arr
                else:
                    st.warning(f"Column '{col}' not found in FITS catalog.")
            
            df = pd.DataFrame(df_dict)
            
            # Bytes decoding for object columns
            for col in df.columns:
                if df[col].dtype == object and len(df) > 0:
                    try:
                        if isinstance(df[col].iloc[0], bytes):
                            df[col] = df[col].str.decode('utf-8')
                    except Exception:
                        pass
            return df
    except Exception as e:
        st.error(f"Error loading columns from FITS: {e}")
        return pd.DataFrame()

def load_config_from_checkpoint(checkpoint_dir: Path) -> dict:
    """
    Search for config.json or PyTorch checkpoints (.pt) in checkpoint_dir
    and extract training configurations.
    """
    # 1. Try config.json first
    for cand in [checkpoint_dir / "config.json", checkpoint_dir / "weights" / "config.json", checkpoint_dir.parent / "weights" / "config.json"]:
        if cand.is_file():
            try:
                with open(cand, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
                
    # 2. Fallback: Try PyTorch checkpoints (.pt)
    pt_candidates = []
    for d in [checkpoint_dir, checkpoint_dir / "weights", checkpoint_dir.parent / "weights"]:
        if d.exists():
            pt_candidates.extend(list(d.glob("*.pt")))
            
    for cand in pt_candidates:
        if cand.is_file():
            try:
                import torch
                # Load on CPU to avoid allocating GPU RAM
                checkpoint = torch.load(cand, map_location='cpu')
                if isinstance(checkpoint, dict) and "config" in checkpoint:
                    config_data = checkpoint["config"]
                    if isinstance(config_data, dict):
                        return config_data
            except Exception:
                pass
                
    return {}

@st.cache_data
def scan_prediction_runs(logs_dir: str) -> dict:
    """
    Scan the logs directory for runs that contain prediction npz files.
    Returns a dictionary of run_path_str -> Display Name.
    """
    logs_path = Path(logs_dir)
    runs = {}
    if not logs_path.exists():
        return runs

    # Find directories named 'predictions' which contains npz files
    for pred_dir in logs_path.glob("**/downstream_tasks/predictions"):
        try:
            # The run directory is four levels up
            # e.g., logs/run_name/embeddings/emb_name/downstream_tasks/predictions
            run_root = pred_dir.parent.parent.parent.parent
            train_name = run_root.name
            
            # Load config dynamically using our helper (json or pt fallback)
            config_data = load_config_from_checkpoint(run_root)
            metadata_path = config_data.get("metadata_path", None)
            train_name = config_data.get("train_name", train_name)
            
            emb_folder_name = pred_dir.parent.parent.name
            display_name = f"{train_name} [{emb_folder_name}]"
            
            runs[str(pred_dir)] = {
                "display_name": display_name,
                "metadata_path": metadata_path,
                "run_root": str(run_root)
            }
        except Exception:
            continue
            
    return runs

def parse_npz_filename(filename: str) -> tuple:
    """
    Parse the target, mode, and probe from prediction filenames.
    
    The naming convention is: preds_{target}_{mode}_{probe}.npz
    where mode can be compound (e.g. DESISpectrum_phase1).
    
    We parse from right to left using known probe and mode tokens to
    correctly separate multi-underscore targets (e.g. HALPHA_FLUX) from
    multi-underscore modes (e.g. DESISpectrum_phase1).
    """
    KNOWN_PROBES = {'knn', 'lp', 'mlp'}
    KNOWN_MODES = {
        'DESISpectrum', 'EuclidImage', 'cls', 'joint',
        'DESISpectrum_phase1', 'DESISpectrum_phase2',
        'EuclidImage_phase1', 'EuclidImage_phase2',
        'cls_phase1', 'cls_phase2',
        'joint_phase1', 'joint_phase2',
        'images', 'spectra',                # legacy mode names
        'images_phase1', 'images_phase2',
        'spectra_phase1', 'spectra_phase2',
    }
    
    stem = Path(filename).stem  # e.g. preds_HALPHA_FLUX_DESISpectrum_phase1_knn
    if not stem.startswith('preds_'):
        return None, None, None
    
    body = stem[len('preds_'):]  # HALPHA_FLUX_DESISpectrum_phase1_knn
    parts = body.split('_')
    
    if len(parts) < 3:
        return None, None, None
    
    # Last token must be the probe
    probe = parts[-1]
    if probe not in KNOWN_PROBES:
        return None, None, None
    
    # Try matching mode from right-to-left (longest match first)
    # Check 2-token mode first (e.g. DESISpectrum_phase1), then 1-token
    remaining = parts[:-1]  # everything except probe
    
    mode = None
    target_parts = None
    
    # Try 2-token mode (e.g. "DESISpectrum_phase1")
    if len(remaining) >= 3:
        candidate_mode = remaining[-2] + '_' + remaining[-1]
        if candidate_mode in KNOWN_MODES:
            mode = candidate_mode
            target_parts = remaining[:-2]
    
    # Try 1-token mode (e.g. "DESISpectrum", "joint", "cls")
    if mode is None and len(remaining) >= 2:
        candidate_mode = remaining[-1]
        if candidate_mode in KNOWN_MODES:
            mode = candidate_mode
            target_parts = remaining[:-1]
    
    if mode is None or not target_parts:
        return None, None, None
    
    target = '_'.join(target_parts)
    return target, mode, probe

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive Probing Predictions Dashboard args")
    parser.add_argument('--logs_dir', type=str, default=None, help="Root directory to scan for probing runs")
    parser.add_argument('--metadata_path', type=str, default=None, help="Path to FITS metadata catalog")
    parser.add_argument('--pred_dir', type=str, default=None, help="Initial predictions directory to select")
    parser.add_argument('--weights_dir', type=str, default=None, help="Directory containing training weights / config.json")
    return parser.parse_known_args()

def main():
    args, unknown = parse_args()
    
    st.title("🔭 AstroPT Interactive Probing Predictions Dashboard")
    st.markdown(
        """
        Analyze systematically aligned predictions from **Euclid images** and **DESI spectra** downstream probing tasks.
        Assess the **modality gap** and locate potential biases by projecting prediction errors against a third coordinate.
        """
    )

    st.sidebar.header("📁 Data & Run Selection")
    
    # 1. Logs Path Configuration
    default_logs = args.logs_dir if args.logs_dir else "/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs"
    logs_dir = st.sidebar.text_input("Logs Root Directory", value=default_logs)
    
    # Scan logs dynamically
    with st.spinner("Scanning logs for prediction data..."):
        runs_dict = scan_prediction_runs(logs_dir)
        
    if not runs_dict:
        st.warning("No runs with probing predictions found under the specified logs root directory.")
        st.info("Ensure that `probing_downstream_benchmark.py` has run and saved prediction `.npz` files.")
        return

    # Select Run
    runs_options = list(runs_dict.keys())
    default_run_idx = 0
    if args.pred_dir and args.pred_dir in runs_options:
        default_run_idx = runs_options.index(args.pred_dir)
        
    selected_pred_path = st.sidebar.selectbox(
        "Select Probing Run",
        options=runs_options,
        index=default_run_idx,
        format_func=lambda x: runs_dict[x]["display_name"]
    )
    
    run_info = runs_dict[selected_pred_path]
    pred_dir = Path(selected_pred_path)
    
    # 2. Metadata Catalog Selection
    # Dynamically extract metadata path from config.json or fallback
    config_metadata_path = run_info["metadata_path"]
    
    if args.weights_dir:
        w_path = Path(args.weights_dir)
        config_data = load_config_from_checkpoint(w_path)
        if "metadata_path" in config_data:
            config_metadata_path = config_data["metadata_path"]
    
    if args.metadata_path:
        init_catalog = args.metadata_path
    else:
        default_filtered_fits = "/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1_FILTERED.fits"
        default_unfiltered_fits = "/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits"
        
        # Choose catalog based on config, filtered presence or default fallback
        init_catalog = config_metadata_path if config_metadata_path else default_filtered_fits
        if not Path(init_catalog).exists():
            init_catalog = default_filtered_fits if Path(default_filtered_fits).exists() else default_unfiltered_fits

    metadata_path = st.sidebar.text_input("Metadata FITS Catalog", value=init_catalog)
    
    if not Path(metadata_path).exists():
        st.error(f"Selected FITS catalog not found at: {metadata_path}")
        return

    # Scan NPZ files in selected run to populate selection options
    npz_files = list(pred_dir.glob("preds_*.npz"))
    if not npz_files:
        st.error(f"No prediction `.npz` files found in directory: {pred_dir}")
        return

    # Extract target, modality/mode, probe
    targets_found = set()
    modes_found = set()
    probes_found = set()
    file_mapping = {}
    
    for f in npz_files:
        target, mode, probe = parse_npz_filename(f.name)
        if target:
            targets_found.add(target)
            modes_found.add(mode)
            probes_found.add(probe)
            file_mapping[(target, mode, probe)] = f

    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Probing Coordinate Selection")
    
    # Select target, mode, probe
    selected_target = st.sidebar.selectbox("Primary Probing Target", options=sorted(list(targets_found)))
    
    # Filter available modalities and probes for this target
    avail_modes = sorted([m for t, m, p in file_mapping.keys() if t == selected_target])
    selected_mode = st.sidebar.selectbox("Modality / Embedding", options=avail_modes)
    
    avail_probes = sorted([p for t, m, p in file_mapping.keys() if t == selected_target and m == selected_mode])
    selected_probe = st.sidebar.selectbox("Probe Model", options=avail_probes)

    # 3. Third Coordinate Selection
    # Read column names from FITS first (extremely fast)
    with st.spinner("Fetching catalog column headers..."):
        all_catalog_cols = get_fits_columns(metadata_path)
        
    if not all_catalog_cols:
        st.error("Could not fetch column headers from the catalog FITS file.")
        return

    # Determine ID column name inside catalog
    catalog_id_col = 'TARGETID' if 'TARGETID' in all_catalog_cols else 'targetid' if 'targetid' in all_catalog_cols else None
    if not catalog_id_col:
        st.error("FITS catalog does not contain TARGETID or targetid column.")
        return

    # Sort remaining cols alphabetically for selectbox
    candidate_third_cols = sorted([c for c in all_catalog_cols if c not in [catalog_id_col]])
    
    # Default selection to redshift (Z) or stellar mass if present
    default_third_idx = 0
    for name in ['Z', 'z', 'LOGMSTAR', 'logmstar']:
        if name in candidate_third_cols:
            default_third_idx = candidate_third_cols.index(name)
            break

    selected_third_col = st.sidebar.selectbox(
        "Third Coordinate (for bias analysis)",
        options=candidate_third_cols,
        index=default_third_idx
    )

    # Layout Configurations
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Visualizations & Styling")
    
    theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)
    plotly_template = "plotly_white" if theme == "Light" else "plotly_dark"
    
    enable_asinh = st.sidebar.checkbox("Enable asinh Scaling for Fluxes", value=True)
    add_trendline = st.sidebar.checkbox("Add OLS Trendline", value=True)
    
    # Placeholder for Plot Type selector (filled after we determine regression vs classification)
    plot_type_placeholder = st.sidebar.empty()
    
    # Inject CSS for styling
    bg_color = "#FFFFFF" if theme == "Light" else "#0E1117"
    text_color = "#000000" if theme == "Light" else "#FAFAFA"
    custom_css = f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # Load prediction file
    target_npz_file = file_mapping.get((selected_target, selected_mode, selected_probe))
    if not target_npz_file or not target_npz_file.exists():
        st.error(f"Selected prediction file does not exist: {target_npz_file}")
        return

    # Load predictions data
    try:
        pred_data = np.load(target_npz_file)
        preds = pred_data['preds']
        true_vals = pred_data['true_vals']
        pred_ids = pred_data['targetid'].astype(np.int64)
    except Exception as e:
        st.error(f"Error loading prediction data `.npz`: {e}")
        return

    # Load catalog columns (only ID, primary target true values, and third coordinate!)
    # Include original primary target to check values or in case of classification
    cols_to_load = [catalog_id_col, selected_third_col]
    if selected_target in all_catalog_cols:
        cols_to_load.append(selected_target)
        
    with st.spinner(f"Loading selected coordinates from FITS catalog..."):
        meta_df = load_fits_columns(metadata_path, cols_to_load)
        
    if meta_df.empty:
        st.error("No data could be loaded from the FITS catalog. Alignment halted.")
        return

    # Align predictions with catalog on TARGETID
    # Place predictions into a dataframe
    pred_df = pd.DataFrame({
        'TARGETID': pred_ids,
        'pred_vals': preds,
        'true_vals_original': true_vals
    })
    
    # Merge
    # Ensure ID column is aligned
    meta_df = meta_df.rename(columns={catalog_id_col: 'TARGETID'})
    meta_df['TARGETID'] = meta_df['TARGETID'].astype(np.int64)
    
    merged_df = pd.merge(pred_df, meta_df, on='TARGETID', how='inner')
    
    if merged_df.empty:
        st.error("Alignment yields 0 overlapping TARGETID samples between prediction npz and catalog FITS.")
        st.info("Verify if the correct FITS catalog is selected (Filtered vs. Unfiltered version).")
        return

    # Display Metrics summary
    st.markdown(f"### 📊 Experiment Summary: **{selected_target}** ({selected_mode} | {selected_probe})")
    
    total_aligned = len(merged_df)
    
    # Calculate performance metrics
    is_regression = selected_target not in CLASSIFICATION_TARGETS
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Aligned Samples", f"{total_aligned:,}")
        
    if is_regression:
        # Calculate R2 and RMSE on original scale
        y_true = merged_df['true_vals_original'].values
        y_pred = merged_df['pred_vals'].values
        
        r2 = 1.0 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        bias = np.median(y_pred - y_true)
        
        with col2:
            st.metric("Probing R²", f"{r2:.4f}")
        with col3:
            st.metric("RMSE", f"{rmse:.4f}")
        with col4:
            st.metric("Median Bias", f"{bias:.4f}")
    else:
        # Classification
        y_true = merged_df['true_vals_original'].values
        y_pred = merged_df['pred_vals'].values
        acc = np.mean(y_true == y_pred)
        
        with col2:
            st.metric("Probing Accuracy", f"{acc:.2%}")
        with col3:
            correct_cnt = np.sum(y_true == y_pred)
            st.metric("Correct Predictions", f"{correct_cnt:,}")
        with col4:
            incorrect_cnt = np.sum(y_true != y_pred)
            st.metric("Incorrect Predictions", f"{incorrect_cnt:,}")

    # Prepare data for plotting
    plot_df = merged_df.copy()
    
    # Apply asinh scaling if enabled
    if enable_asinh:
        plot_df['true_plot'] = apply_scaling(plot_df['true_vals_original'].values, selected_target)
        plot_df['pred_plot'] = apply_scaling(plot_df['pred_vals'].values, selected_target)
        plot_df['third_plot'] = apply_scaling(plot_df[selected_third_col].values, selected_third_col)
    else:
        plot_df['true_plot'] = plot_df['true_vals_original']
        plot_df['pred_plot'] = plot_df['pred_vals']
        plot_df['third_plot'] = plot_df[selected_third_col]

    # Handle NaNs in third coordinate
    plot_df = plot_df.dropna(subset=['third_plot'])
    if plot_df.empty:
        st.warning(f"All aligned samples had NaN values in selected third coordinate: '{selected_third_col}'")
        return

    st.markdown("---")
    st.subheader("📈 Custom Probing Bias Visualization")
    
    # Fill the Plot Type selector in the sidebar now that we know if it's regression
    if is_regression:
        plot_options = [
            "Residuals (Pred - True) vs. Third Coordinate",
            "True vs. Predicted (Colored by Third Coordinate)",
            "Hybrid: Predictions + Residuals",
            "Overlay: True & Predicted vs. Third Coordinate"
        ]
    else:
        plot_options = [
            "Distribution of Third Coordinate (Correct vs. Incorrect)",
            "Distribution of Third Coordinate by Class"
        ]
        
    selected_plot_type = plot_type_placeholder.selectbox("Plot Type", options=plot_options)

    # Grid details
    grid_color = "#EBF0F8" if theme == "Light" else "#22262F"

    # Render Plotly Charts
    if is_regression:
        # Calculate residuals
        plot_df['residual'] = plot_df['pred_plot'] - plot_df['true_plot']
        
        # Calculate correlation for bias detection
        pearson_val = plot_df['residual'].corr(plot_df['third_plot'], method='pearson')
        spearman_val = plot_df['residual'].corr(plot_df['third_plot'], method='spearman')

        if selected_plot_type == "Residuals (Pred - True) vs. Third Coordinate":
            st.markdown(
                f"""
                **Residual analysis**: Projects the prediction error ($y_{{pred}} - y_{{true}}$) onto the third coordinate.
                - **Pearson Correlation**: `{pearson_val:.4f}`
                - **Spearman Rank Correlation**: `{spearman_val:.4f}`
                
                A non-zero correlation indicates a systematic **modality bias** over the third coordinate (e.g. model struggles more at high redshift).
                """
            )
            
            fig = px.scatter(
                plot_df,
                x='third_plot',
                y='residual',
                opacity=0.4,
                template=plotly_template,
                labels={
                    'third_plot': f"{selected_third_col} {'(asinh scaled)' if enable_asinh and selected_third_col in FLUX_TARGETS else ''}",
                    'residual': f"Residual [Pred - True] {'(asinh scaled)' if enable_asinh and selected_target in FLUX_TARGETS else ''}"
                },
                title=f"Residuals vs. {selected_third_col}"
            )
            
            # Add custom OLS trendline using numpy (avoids statsmodels dependency!)
            if add_trendline:
                x_vals = plot_df['third_plot'].values
                y_vals = plot_df['residual'].values
                mask_valid = ~np.isnan(x_vals) & ~np.isnan(y_vals)
                x_valid = x_vals[mask_valid]
                y_valid = y_vals[mask_valid]
                if len(x_valid) > 1:
                    m, c = np.polyfit(x_valid, y_valid, 1)
                    x_line = np.array([x_valid.min(), x_valid.max()])
                    y_line = m * x_line + c
                    # R² for residual trendline
                    y_pred_line = m * x_valid + c
                    ss_res = np.sum((y_valid - y_pred_line) ** 2)
                    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
                    r2_res = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                    fig.add_trace(go.Scatter(
                        x=x_line, y=y_line,
                        mode='lines',
                        line=dict(color='red', width=2),
                        name=f"OLS Fit (R²={r2_res:.4f}, slope={m:.4f})"
                    ))
            
            # Add horizontal zero bias reference line
            fig.add_shape(
                type="line", line=dict(dash="dash", width=1.5, color="gray"),
                x0=plot_df['third_plot'].min(), x1=plot_df['third_plot'].max(),
                y0=0, y1=0
            )

        elif selected_plot_type == "True vs. Predicted (Colored by Third Coordinate)":
            st.markdown(
                """
                **Accuracy projection map**: Plots actual versus predicted probing values, colored by the third coordinate.
                Points clustered around the diagonal red dashed line ($y=x$) indicate accurate predictions.
                Color gradients along specific axes can reveal where the model's accuracy degrades.
                """
            )
            
            fig = px.scatter(
                plot_df,
                x='true_plot',
                y='pred_plot',
                color='third_plot',
                color_continuous_scale=px.colors.sequential.Hot,
                opacity=0.6,
                template=plotly_template,
                labels={
                    'true_plot': f"True {selected_target} {'(asinh scaled)' if enable_asinh and selected_target in FLUX_TARGETS else ''}",
                    'pred_plot': f"Predicted {selected_target} {'(asinh scaled)' if enable_asinh and selected_target in FLUX_TARGETS else ''}",
                    'third_plot': selected_third_col
                },
                title=f"True vs. Predicted {selected_target} (Colored by {selected_third_col})"
            )
            
            # Diagonal identity line
            min_val = min(plot_df['true_plot'].min(), plot_df['pred_plot'].min())
            max_val = max(plot_df['true_plot'].max(), plot_df['pred_plot'].max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', line=dict(color='red', dash='dash', width=2),
                name="y=x"
            ))
            
            # Custom OLS trendline on True vs Predicted (Colored by Third Coordinate)
            if add_trendline:
                x_vals = plot_df['true_plot'].values
                y_vals = plot_df['pred_plot'].values
                mask_valid = ~np.isnan(x_vals) & ~np.isnan(y_vals)
                x_valid = x_vals[mask_valid]
                y_valid = y_vals[mask_valid]
                if len(x_valid) > 1:
                    m, c = np.polyfit(x_valid, y_valid, 1)
                    x_line = np.array([x_valid.min(), x_valid.max()])
                    y_line = m * x_line + c
                    
                    # Calculate OLS R^2 for this trendline fit
                    y_pred_line = m * x_valid + c
                    ss_res = np.sum((y_valid - y_pred_line) ** 2)
                    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
                    r2_fit = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                    
                    fig.add_trace(go.Scatter(
                        x=x_line, y=y_line,
                        mode='lines',
                        line=dict(color='orange', width=2.5),
                        name=f"OLS Fit (R²={r2_fit:.4f}, slope={m:.4f})"
                    ))

        elif selected_plot_type == "Hybrid: Predictions + Residuals":
            st.markdown(
                f"""
                **Combined diagnostic panel**: Top panel shows True vs. Predicted colored by {selected_third_col}.
                Bottom panel shows Residuals vs. {selected_third_col} for bias detection.
                - **Pearson Correlation (residuals)**: `{pearson_val:.4f}`
                - **Spearman Rank Correlation (residuals)**: `{spearman_val:.4f}`
                """
            )
            
            # Build 2-row subplot: shared x-axis not applicable (different axes), so independent
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.6, 0.4],
                vertical_spacing=0.08,
                subplot_titles=(
                    f"True vs. Predicted {selected_target}",
                    f"Residuals vs. {selected_third_col}"
                )
            )
            
            # ---- TOP: True vs Predicted scatter colored by third coordinate ----
            fig.add_trace(
                go.Scatter(
                    x=plot_df['true_plot'],
                    y=plot_df['pred_plot'],
                    mode='markers',
                    marker=dict(
                        color=plot_df['third_plot'],
                        colorscale='Hot',
                        opacity=0.5,
                        size=4,
                        colorbar=dict(title=selected_third_col, x=1.02, len=0.55, y=0.78)
                    ),
                    name='Predictions',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # y=x identity line (top)
            min_val = min(plot_df['true_plot'].min(), plot_df['pred_plot'].min())
            max_val = max(plot_df['true_plot'].max(), plot_df['pred_plot'].max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines', line=dict(color='red', dash='dash', width=2),
                    name='y=x', showlegend=False
                ),
                row=1, col=1
            )
            
            # OLS trendline (top)
            if add_trendline:
                x_vals_top = plot_df['true_plot'].values
                y_vals_top = plot_df['pred_plot'].values
                mask_top = ~np.isnan(x_vals_top) & ~np.isnan(y_vals_top)
                x_v_top = x_vals_top[mask_top]
                y_v_top = y_vals_top[mask_top]
                if len(x_v_top) > 1:
                    m_top, c_top = np.polyfit(x_v_top, y_v_top, 1)
                    x_line_top = np.array([x_v_top.min(), x_v_top.max()])
                    y_line_top = m_top * x_line_top + c_top
                    y_pred_top = m_top * x_v_top + c_top
                    ss_res_top = np.sum((y_v_top - y_pred_top) ** 2)
                    ss_tot_top = np.sum((y_v_top - np.mean(y_v_top)) ** 2)
                    r2_top = 1.0 - (ss_res_top / ss_tot_top) if ss_tot_top > 0 else 0.0
                    fig.add_trace(
                        go.Scatter(
                            x=x_line_top, y=y_line_top,
                            mode='lines', line=dict(color='orange', width=2.5),
                            name=f'OLS (R²={r2_top:.4f})', showlegend=False
                        ),
                        row=1, col=1
                    )
                    # Annotation for the top panel OLS
                    fig.add_annotation(
                        text=f"OLS R²={r2_top:.4f}, slope={m_top:.4f}",
                        xref="x domain", yref="y domain",
                        x=0.02, y=0.98, showarrow=False,
                        font=dict(size=12, color='orange'),
                        bgcolor='rgba(0,0,0,0.5)' if plotly_template == 'plotly_dark' else 'rgba(255,255,255,0.7)',
                        row=1, col=1
                    )
            
            # ---- BOTTOM: Residuals vs Third Coordinate ----
            fig.add_trace(
                go.Scatter(
                    x=plot_df['third_plot'],
                    y=plot_df['residual'],
                    mode='markers',
                    marker=dict(color='#636EFA', opacity=0.35, size=4),
                    name='Residuals',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Zero-bias reference line (bottom)
            fig.add_shape(
                type="line", line=dict(dash="dash", width=1.5, color="gray"),
                x0=plot_df['third_plot'].min(), x1=plot_df['third_plot'].max(),
                y0=0, y1=0,
                row=2, col=1
            )
            
            # OLS trendline (bottom)
            if add_trendline:
                x_vals_bot = plot_df['third_plot'].values
                y_vals_bot = plot_df['residual'].values
                mask_bot = ~np.isnan(x_vals_bot) & ~np.isnan(y_vals_bot)
                x_v_bot = x_vals_bot[mask_bot]
                y_v_bot = y_vals_bot[mask_bot]
                if len(x_v_bot) > 1:
                    m_bot, c_bot = np.polyfit(x_v_bot, y_v_bot, 1)
                    x_line_bot = np.array([x_v_bot.min(), x_v_bot.max()])
                    y_line_bot = m_bot * x_line_bot + c_bot
                    y_pred_bot = m_bot * x_v_bot + c_bot
                    ss_res_bot = np.sum((y_v_bot - y_pred_bot) ** 2)
                    ss_tot_bot = np.sum((y_v_bot - np.mean(y_v_bot)) ** 2)
                    r2_bot = 1.0 - (ss_res_bot / ss_tot_bot) if ss_tot_bot > 0 else 0.0
                    fig.add_trace(
                        go.Scatter(
                            x=x_line_bot, y=y_line_bot,
                            mode='lines', line=dict(color='red', width=2),
                            name=f'OLS (slope={m_bot:.4f})', showlegend=False
                        ),
                        row=2, col=1
                    )
                    fig.add_annotation(
                        text=f"OLS R²={r2_bot:.4f}, slope={m_bot:.4f}",
                        xref="x2 domain", yref="y2 domain",
                        x=0.02, y=0.95, showarrow=False,
                        font=dict(size=12, color='red'),
                        bgcolor='rgba(0,0,0,0.5)' if plotly_template == 'plotly_dark' else 'rgba(255,255,255,0.7)',
                        row=2, col=1
                    )
            
            # Axis labels for the hybrid subplot
            asinh_tag_target = ' (asinh scaled)' if enable_asinh and selected_target in FLUX_TARGETS else ''
            asinh_tag_third = ' (asinh scaled)' if enable_asinh and selected_third_col in FLUX_TARGETS else ''
            fig.update_xaxes(title_text=f"True {selected_target}{asinh_tag_target}", row=1, col=1)
            fig.update_yaxes(title_text=f"Predicted {selected_target}{asinh_tag_target}", row=1, col=1)
            fig.update_xaxes(title_text=f"{selected_third_col}{asinh_tag_third}", row=2, col=1)
            fig.update_yaxes(title_text=f"Residual [Pred - True]{asinh_tag_target}", row=2, col=1)
            
            # Override height for the hybrid plot (needs more space)
            fig.update_layout(height=850)

        elif selected_plot_type == "Overlay: True & Predicted vs. Third Coordinate":
            st.markdown(
                """
                **Distribution overlay**: Overlaying true values and predicted values as a function of the third coordinate.
                This comparison illustrates whether the predicted distribution (red) accurately mirrors the true distribution (blue) across all coordinates.
                """
            )
            
            # Structure overlay DF
            true_part = pd.DataFrame({
                'third': plot_df['third_plot'],
                'value': plot_df['true_plot'],
                'Series': 'True'
            })
            pred_part = pd.DataFrame({
                'third': plot_df['third_plot'],
                'value': plot_df['pred_plot'],
                'Series': 'Predicted'
            })
            overlay_df = pd.concat([true_part, pred_part], ignore_index=True)
            
            fig = px.scatter(
                overlay_df,
                x='third',
                y='value',
                color='Series',
                color_discrete_map={'True': '#1f77b4', 'Predicted': '#ff7f0e'},
                opacity=0.4,
                template=plotly_template,
                labels={
                    'third': f"{selected_third_col} {'(asinh scaled)' if enable_asinh and selected_third_col in FLUX_TARGETS else ''}",
                    'value': f"Value [True / Pred] {'(asinh scaled)' if enable_asinh and selected_target in FLUX_TARGETS else ''}"
                },
                title=f"Overlay: True vs. Predicted {selected_target} vs. {selected_third_col}"
            )
            
    else:
        # Classification Targets
        plot_df['Is_Correct'] = (plot_df['true_vals_original'] == plot_df['pred_vals']).map({True: 'Correct', False: 'Incorrect'})
        
        if selected_plot_type == "Distribution of Third Coordinate (Correct vs. Incorrect)":
            st.markdown(
                """
                **Error distribution boxplot**: Visualizes whether prediction errors correlate with specific ranges of the third coordinate.
                For example, if the model struggles at higher redshifts, the "Incorrect" box will be systematically shifted towards higher values.
                """
            )
            
            fig = px.box(
                plot_df,
                x='Is_Correct',
                y='third_plot',
                color='Is_Correct',
                color_discrete_map={'Correct': '#2ca02c', 'Incorrect': '#d62728'},
                points="outliers",
                template=plotly_template,
                labels={
                    'Is_Correct': 'Prediction Status',
                    'third_plot': f"{selected_third_col} {'(asinh scaled)' if enable_asinh and selected_third_col in FLUX_TARGETS else ''}"
                },
                title=f"Distribution of {selected_third_col} by Prediction Status"
            )
            
        elif selected_plot_type == "Distribution of Third Coordinate by Class":
            st.markdown(
                """
                **Class-based distribution**: Inspect the values of the third coordinate for each predicted and actual class.
                This highlights whether certain classes only exist or are predicted in specific regimes of the third coordinate.
                """
            )
            
            class_type = st.radio("Group classes by:", ["True Classes", "Predicted Classes"])
            color_col = 'true_vals_original' if class_type == "True Classes" else 'pred_vals'
            
            fig = px.box(
                plot_df,
                x=color_col,
                y='third_plot',
                color=color_col,
                points="outliers",
                template=plotly_template,
                labels={
                    color_col: 'Class Name',
                    'third_plot': f"{selected_third_col} {'(asinh scaled)' if enable_asinh and selected_third_col in FLUX_TARGETS else ''}"
                },
                title=f"Distribution of {selected_third_col} across {class_type}"
            )

    # Style Axis lines and layout nicely
    fig.update_xaxes(showline=True, linewidth=1, linecolor='gray', mirror=True, gridcolor=grid_color)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='gray', mirror=True, gridcolor=grid_color)
    fig.update_layout(
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        title_x=0.5,
        title_font_size=18,
        height=fig.layout.height or 600,
        margin=dict(l=60, r=40, t=80, b=60),
        # Legend: horizontal at the top to avoid wasting horizontal plot space
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )

    st.plotly_chart(fig, theme=None, use_container_width=True)

    # Statistical correlation reporting for bias detection
    if is_regression:
        st.subheader("🧪 Mathematical Bias Metrics")
        
        # Pearson and Spearman analysis
        met_col1, met_col2 = st.columns(2)
        with met_col1:
            st.markdown("**Pearson Correlation (Linear Bias)**")
            st.code(f"Value: {pearson_val:.5f}\nStrength: {get_corr_strength(pearson_val)}")
            
        with met_col2:
            st.markdown("**Spearman Rank Correlation (Monotonic Bias)**")
            st.code(f"Value: {spearman_val:.5f}\nStrength: {get_corr_strength(spearman_val)}")
            
        st.info(
            """
            *Strength categories based on standard statistical guidelines:*
            - $|r| < 0.1$: Negligible
            - $0.1 \le |r| < 0.3$: Weak
            - $0.3 \le |r| < 0.5$: Moderate
            - $|r| \ge 0.5$: Strong
            """
        )

def get_corr_strength(val: float) -> str:
    """Categorize correlation strength."""
    abs_val = abs(val)
    if abs_val < 0.1:
        return "Negligible (No bias)"
    elif abs_val < 0.3:
        return "Weak Bias"
    elif abs_val < 0.5:
        return "Moderate Bias"
    else:
        return "Strong Bias (Systematic issue detected)"

if __name__ == "__main__":
    main()
