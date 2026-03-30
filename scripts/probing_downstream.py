"""
AstroPT Downstream Task Prober (Unified LP & MLP) - Enhanced Logging.

This script performs a battery of evaluations by training Linear Probes (LP)
and/or Multi-Layer Perceptrons (MLP) on pre-computed embeddings (.npy).

Features:
- Unified pipeline for Linear Probes (LP) and MLPs.
- Real-time CSV saving (appending mode).
- Per-Target summary tables in the console.
- Clean logging configuration.
- Input and Target scaling (StandardScaler and Asinh for fluxes).

Author: Victor Alonso Rodriguez
Date: March 2026
"""

import argparse
import glob
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, 
                             r2_score, mean_squared_error, 
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# Logger Configuration
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S", # Simplified time format
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-Prober")

# çRemoving AstroPy Errors
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.filterwarnings('ignore', module='astropy.io.fits')

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Targets
DEFAULT_TARGETS = ['Z', 'LOGMSTAR', 'LOGSFR', 'GR', 
                    'flux_detection_total', 'HALPHA_EW', 'HALPHA_FLUX', 'NII_6584_FLUX', 'OIII_5007_FLUX', 'HBETA_FLUX', 'NII_6584_FLUX', 
                    'sersic_sersic_vis_radius', 'sersic_sersic_vis_index', 'sersic_sersic_vis_axis_ratio',
                    'has_spiral_arms_yes', 'smoothness', 'gini', 'SPECTYPE', 'data_set_release']

CLASSIFICATION_TARGETS = ['SPECTYPE', 'data_set_release'] 

# Column order for the final CSV
CSV_COLUMNS = [
    'Target', 'Task', 'Modality', 'Probe',
    'R2', 'RMSE', 'Bias', 'NMAD', 'Outliers',
    'Accuracy', 'Precision', 'Recall', 'F1_score', 'FPR'
]

def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="AstroPT Downstream Prober (LP & MLP)")
    
    # Data Paths
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to .fits catalog")
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing training weights")
    parser.add_argument("--emb_dir", type=str, required=True, help="Directory containing .npy embedding files")
    parser.add_argument("--save_dir", type=str, required=True, help="Plot Saving Directory")
    
    parser.add_argument("--train_name", type=str, default=None, help="Custom title for plots")
    
    # Task Config
    parser.add_argument("--targets", nargs='+', default=None, help="Overwrites default target list.")
    parser.add_argument("--targets_add", nargs='+', default=None, help="Appends to default list.")
    
    # Probing Config
    parser.add_argument("--probes", nargs='+', default=['lp', 'mlp'], choices=['lp', 'mlp'], 
                        help="Type of probes to run: 'lp', 'mlp' or both.")
    parser.add_argument("--conf_matrix", type=bool, default=False, help="Generates confusion matrix")
    
    # Training Hyperparams
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per task")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    return parser.parse_args()


# Scaler
class AsinhScaler(BaseEstimator, TransformerMixin):
    """
    Inverse hyperbolic sine scaler.
    Useful for heavy-tailed distributions like astronomical fluxes.
    """
    def __init__(self, a: float = 1.0):
        self.a = a 
    def fit(self, X: Any, y: Any = None) -> 'AsinhScaler': return self
    def transform(self, X: np.ndarray) -> np.ndarray: return np.arcsinh(X / self.a)
    def inverse_transform(self, X: np.ndarray) -> np.ndarray: return self.a * np.sinh(X)

# Fixed seed for different experiments
def set_seed(seed=61):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Seed set to {seed} for reproducibility.")


# LINEAR MODEL
class LinearProbe(nn.Module):
    """Linear Probe: Single Linear Layer (Input -> Output)."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# MLP MODEL
class AdaptiveMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for probing.
    Architecture: Input -> 512 -> 256 -> Output.
    Includes BatchNorm, ReLU, and Dropout.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def load_data(
        emb_dir: str | Path, 
        metadata_path: str | Path
    ) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Loads embeddings (.npy) and FITS metadata."""
    emb_dir = Path(emb_dir)
    metadata_path = Path(metadata_path)
    
    logger.info(f"Scanning embeddings directory: {emb_dir}...")
    
    if not emb_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {emb_dir}")
        
    data_dict = {}
    
    # Find IDs
    id_candidates = ['targetid.npy', 'ids.npy', 'target_ids.npy', 'object_ids.npy']
    id_path = next((emb_dir / c for c in id_candidates if (emb_dir / c).exists()), None)
    if not id_path: 
        raise FileNotFoundError(f"Could not find Target IDs in {emb_dir}")
    data_dict['targetid'] = np.load(id_path)

    # Find Modalities (Using memmap for efficient RAM usage)
    for mod in ['images', 'spectra', 'joint']:
        candidates = list(emb_dir.glob(f"{mod}*.npy"))
        if candidates:
            best = next((c for c in candidates if c.name == f"{mod}.npy"), candidates[0])
            logger.info(f"Loading {mod} from: {best.name} (memmap)")
            data_dict[mod] = np.load(best, mmap_mode='r')

    # Load Metadata
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
    try:
        catalog = Table.read(metadata_path)
        df = catalog.to_pandas()
    except Exception as e:
        logger.error(f"Failed to read FITS: {e}")
        sys.exit(1)
    
    # Bytes decoding
    for col in df.columns:
        if df[col].dtype == object and isinstance(df[col].iloc[0], bytes):
            try: df[col] = df[col].str.decode('utf-8')
            except: pass
            
    if 'SURVEY' in df.columns: 
        df['SURVEY'] = df['SURVEY'].str.lower().str.strip()
        
    return data_dict, df


def filter_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Applies strict spectral quality filtering (SNR > 3 & H-alpha)."""
    logger.info("Applying Spectral Quality Filter (SNR > 3)...")
    initial_len = len(df)
    
    # Spectrograph filter
    if 'SNR_SPEC_R' in df.columns and 'SNR_SPEC_Z' in df.columns:
        mask_snr = (df['SNR_SPEC_R'] > 3.0) | (df['SNR_SPEC_Z'] > 3.0)
    else:
        logger.warning("SNR_SPEC_R or SNR_SPEC_Z not found. Skipping global SNR filter.")
        mask_snr = pd.Series(True, index=df.index)

    # Halpha filter
    if 'HALPHA_FLUX' in df.columns and 'HALPHA_FLUX_IVAR' in df.columns:
        halpha_err = np.where(df['HALPHA_FLUX_IVAR'] > 0, 1.0 / np.sqrt(df['HALPHA_FLUX_IVAR'].clip(lower=1e-10)), np.inf)
        mask_halpha = df['HALPHA_FLUX'] > (3.0 * halpha_err)
    else:
        logger.warning("HALPHA_FLUX or HALPHA_FLUX_IVAR not found. Skipping H-alpha quality filter.")
        mask_halpha = pd.Series(True, index=df.index)

    # Combining results
    filtered_df = df[mask_snr & mask_halpha].copy()
    final_len = len(filtered_df)
    
    logger.info(f"Filtered out {initial_len - final_len} noisy spectra. Retained {final_len} ({final_len/initial_len:.1%})")
    return filtered_df


def align_data(embeddings_dict: Dict[str, np.ndarray], catalog_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Aligns filtered metadata with embeddings using TARGETID intersection."""
    logger.info("Aligning embeddings with filtered metadata...")
    emb_ids = embeddings_dict['targetid']
    target_col = 'TARGETID' if 'TARGETID' in catalog_df.columns else 'targetid'
    
    catalog_df[target_col] = catalog_df[target_col].astype('int64')
    catalog_indexed = catalog_df.drop_duplicates(subset=[target_col]).set_index(target_col)
    
    # Intersection
    common_ids = np.intersect1d(emb_ids, catalog_indexed.index.values)
    matched_catalog = catalog_indexed.loc[common_ids].reset_index()
    
    # Indices Mapping
    id_to_idx = {id_val: i for i, id_val in enumerate(emb_ids)}
    valid_indices = np.array([id_to_idx[uid] for uid in matched_catalog[target_col].values])
    
    # Extract valid slices from Memmaps
    aligned_embeddings = {}
    for mod in ['images', 'spectra', 'joint']:
        if mod in embeddings_dict:
            # Fancy indexing triggers RAM loading only for valid items
            aligned_embeddings[mod] = embeddings_dict[mod][valid_indices]
            
    logger.info(f"Final aligned pure Test Set objects available for probing: {len(matched_catalog)}")
    return matched_catalog, aligned_embeddings


def get_task_data(
    raw_data: Dict[str, np.ndarray], 
    df: pd.DataFrame, 
    modality: str, 
    target_col: str
) -> Tuple[np.ndarray, np.ndarray, str, Optional[Any]]:
    """
    Filters aligned data for a specific target (removing NaNs).

    Args:
        raw_data: Dictionary of aligned embeddings.
        df: Metadata DataFrame.
        modality: Current modality key (e.g., 'images').
        target_col: Column name of the target variable.

    Returns:
        Tuple containing filtered X, y, task_type ('regression'/'classification'), and the label processor.
    """
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found.")
        
    vals = df[target_col]
    
    # Task Logic
    if target_col not in CLASSIFICATION_TARGETS or pd.api.types.is_numeric_dtype(vals):
        task_type = 'regression'
        mask = vals.notna() & (vals > -90)
    else:
        task_type = 'classification'
        mask = vals.notna() & (vals != "") & (vals != "N/A")
        
    y_filtered = vals[mask].values
    X_filtered = raw_data[modality][mask]
    
    processor = None
    if task_type == 'regression':
        y_final = y_filtered.astype(np.float32)
    else:
        processor = LabelEncoder()
        y_final = processor.fit_transform(y_filtered)
        
    return X_filtered, y_final, task_type, processor


def compute_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    task_type: str, 
    target_name: str
) -> Dict[str, float]:
    """
    Computes performance metrics based on the task type.
    
    Regression: R2, RMSE, Bias, NMAD, Outliers.
    Classification: Accuracy, F1 Weighted, FPR.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        task_type: 'regression' or 'classification'.
        target_name: Name of the target variable (for specific robust metrics).

    Returns:
        Dictionary containing calculated metrics.
    """
    if task_type == 'regression':
        # REGRESSION
        diff = y_pred - y_true
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        if target_name == 'Z':
            # REDSHIFT (Z)
            # Reference: Gosia's Paper: Normalized error > 0.15
            norm_diff = diff / (1 + y_true)
            bias = np.median(norm_diff) 
            nmad = 1.4826 * np.median(np.abs(norm_diff - np.median(norm_diff)))
            outliers = np.mean(np.abs(norm_diff) > 0.15) * 100
            
        elif target_name == 'LOGMSTAR':
            # STELLAR MASS
            # Reference: Gosia's Paper: Error > 0.25
            bias = np.median(diff)      
            nmad = 1.4826 * np.median(np.abs(diff - np.median(diff)))
            outliers = np.mean(np.abs(diff) > 0.25) * 100 
            
        else:
            bias = np.median(diff)      
            nmad = 1.4826 * np.median(np.abs(diff - np.median(diff)))
            relative_error = np.abs(diff) / (np.abs(y_true) + 1e-5)
            outliers = np.mean(relative_error > 0.30) * 100
            
        return {
            "R2": r2, 
            "RMSE": rmse, 
            "Bias": bias, 
            "NMAD": nmad, 
            "Outliers": outliers
        }
    else:
        # CLASIFICATION
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # FPR Calculation
        cm = confusion_matrix(y_true, y_pred)
        FP = cm.sum(axis=0) - np.diag(cm)
        TN = cm.sum() - (FP + (cm.sum(axis=1) - np.diag(cm)) + np.diag(cm))
        with np.errstate(divide='ignore', invalid='ignore'):
            fpr_macro = np.mean(np.nan_to_num(FP / (FP + TN)))

        return {
            "Accuracy": acc, 
            "Precision": precision, 
            "Recall": recall, 
            "F1_score": f1, 
            "FPR": fpr_macro
        }


def train_engine(
    X: np.ndarray, 
    y: np.ndarray, 
    probe_type: str, 
    task_type: str, 
    output_dim: int, 
    epochs: int, 
    lr: float, 
    batch_size: int, 
    target_name: str = ""
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main Training Loop. Handles Scaling, Model Creation, and Training.
    NO logging inside here to keep the terminal clean.

    Args:
        X: Input features.
        y: Target values.
        probe_type: 'lp' (Linear Probe) or 'mlp'.
        task_type: 'regression' or 'classification'.
        output_dim: Number of output neurons.
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Batch size.
        target_name: Name of the target (used for scaler selection).

    Returns:
        Tuple of (Predictions, True Values) for the test set.
    """
    # Split datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Input Scaling
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    
    # Target Scaling
    scaler_y = None
    if task_type == 'regression':
        is_flux = "flux" in target_name.lower()
        if is_flux:
            scaler_y = AsinhScaler(a=1.0)
        else:
            scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_raw = y_test 
    else:
        y_test_raw = y_test

    # DataLoaders
    train_ds = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train) if task_type == 'regression' else torch.LongTensor(y_train)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # Model Selection
    input_dim = X_train.shape[1]
    if probe_type == 'lp':
        model = LinearProbe(input_dim, output_dim).to(DEVICE)
        weight_decay = 1e-5 
    elif probe_type == 'mlp':
        model = AdaptiveMLP(input_dim, output_dim).to(DEVICE)
        weight_decay = 0.0 
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss() if task_type == 'regression' else nn.CrossEntropyLoss()
    
    # Training Loop
    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            if task_type == 'regression': preds = preds.squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    # Inference
    model.eval()
    test_ds = TensorDataset(torch.FloatTensor(X_test))
    test_loader = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False)
    
    all_preds = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb[0].to(DEVICE)
            logits = model(xb)
            
            if task_type == 'regression':
                batch_preds = logits.cpu().numpy().flatten()
                # Inverse Transform
                batch_preds = scaler_y.inverse_transform(batch_preds.reshape(-1, 1)).flatten()
            else:
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(batch_preds)
            
    return np.concatenate(all_preds), y_test_raw


def save_row_to_csv(row: Dict[str, Any], csv_path: str | Path):
    """
    Appends a single row to the CSV. Writes header if the file is new.
    
    Args:
        row: Dictionary containing metric values.
        csv_path: Path to the output CSV file.
    """
    df = pd.DataFrame([row])
    # Ensure correct column order
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[CSV_COLUMNS]
    
    csv_path = Path(csv_path)
    header = not csv_path.is_file()
    df.to_csv(csv_path, mode='a', header=header, index=False)


def main():
    """Main execution block."""
    args = parse_args()
    
    # Required paths
    weights_dir = Path(args.weights_dir)
    emb_dir = Path(args.emb_dir)
    metadata_path = Path(args.metadata_path)
    
    save_dir = Path(args.save_dir) / "downstream_tasks"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = save_dir / "downstream_results.csv"
    
    # Fixing the seed
    set_seed(61)
    
    # Prepare Initial Targets List
    initial_targets = DEFAULT_TARGETS.copy()
    if args.targets: initial_targets = args.targets
    elif args.targets_add: initial_targets.extend(args.targets_add)

    initial_targets = list(dict.fromkeys(initial_targets))
    
    # Prepare Probes List
    probes_to_run = args.probes
    logger.info(f"Initial Target List: {initial_targets}")
    logger.info(f"Probes to run: {probes_to_run}")
    
    # Load Data
    raw_data_dict, raw_df = load_data(emb_dir, metadata_path)
    
    if not raw_data_dict:
        logger.error("No valid embeddings found. Exiting.")
        sys.exit(1)
    
    # Apply scientific filters
    filtered_df = filter_catalog(raw_df)
    
    # Align and extract to RAM
    aligned_df, aligned_embeddings = align_data(raw_data_dict, filtered_df)
    
    # TITLE LOGIC for confusion matrix
    config_path = weights_dir / "config.json"
    json_name = None
    
    # Try reading config.json
    if config_path.is_file():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                json_name = config.get("train_name", None)
        except Exception:
            pass 

    # Select ID: CLI > JSON > Folder
    train_name = args.train_name or json_name or weights_dir.parent.name
    raw_emb_name = emb_dir.name
    emb_parts = [p.upper() for p in raw_emb_name.split('_')]
    embedding_method = " + ".join(emb_parts)
    
    title_suffix = f"[{train_name} - {embedding_method}]"
    title_suffix = title_suffix.replace('_', r'\_')


    # REORDER TARGETS (Regression First, then Classification)
    reg_list = []
    cls_list = []

    for t in initial_targets:
        if t in aligned_df.columns:
            vals = aligned_df[t]
            if t not in CLASSIFICATION_TARGETS or pd.api.types.is_numeric_dtype(vals):
                reg_list.append(t)
            else:
                cls_list.append(t)
        else:
            cls_list.append(t)

    # Concatenate
    final_targets = reg_list + cls_list
    logger.info(f"Ordered Processing List: {final_targets}")


    # MAIN LOOP
    for target_col in final_targets:
        print("\n" + "-"*40)
        logger.info(f">>> TARGET: {target_col} <<<")
        
        is_flux = "flux" in target_col.lower()
        scaler_name = "AsinhScaler" if is_flux else "StandardScaler"
        logger.info(f"    [Scaler] Strategy: {scaler_name}")
        
        target_results = [] # To store results for this target
        
        for mode in ['images', 'spectra', 'joint']:
            if mode not in aligned_embeddings: continue
            
            # Get Data
            try:
                X, y, task_type, processor = get_task_data(
                    aligned_embeddings, aligned_df, mode, target_col
                )
                output_dim = 1 if task_type == 'regression' else len(processor.classes_)
                
                # Logging valid objects
                n_total = len(X)
                n_train = int(n_total * 0.8) # 80% train split
                n_test = n_total - n_train   # 20% test split
                
                logger.info(f"    [{mode.upper()}] Valid objects: {n_total} (Train: {n_train} | Test: {n_test})")
                
            except Exception as e:
                logger.error(f"Skipping {target_col} ({mode}): {e}")
                continue
            
            # Loop over Probes 
            for probe in probes_to_run:
                print(f"    -> Running {mode.upper()} / {probe.upper()}...", end="", flush=True)
                
                try:
                    preds, true_vals = train_engine(
                        X, y, probe, task_type, output_dim, 
                        args.epochs, args.lr, args.batch_size, target_col
                    )
                    
                    metrics = compute_metrics(true_vals, preds, task_type, target_col)
                    
                    # Store Results
                    row = {
                        "Target": target_col, 
                        "Task": task_type, 
                        "Modality": mode,
                        "Probe": probe.upper()
                    }
                    row.update(metrics)
                    
                    # SAVE TO CSV IMMEDIATELY
                    save_row_to_csv(row, csv_path)
                    
                    # Add to local list for table
                    target_results.append(row)
                    
                    print(f" Done.")
                    
                    # Save Confusion Matrix
                    if task_type == 'classification' and args.conf_matrix:
                        cm = confusion_matrix(true_vals, preds)
                        plt.figure(figsize=(6, 5))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                    xticklabels=processor.classes_, yticklabels=processor.classes_)
                        plt.title(f"CM: {target_col} ({mode} - {probe.upper()})")
                        plt.tight_layout()
                        plt.savefig(save_dir / f"cm_{target_col}_{mode}_{probe}.png")
                        plt.close()
                        
                except Exception as e:
                    print(" Failed.")
                    logger.error(f"Error in {probe.upper()} for {target_col}: {e}")

        # PRINT SUMMARY TABLE
        if target_results:
            df_target = pd.DataFrame(target_results)
            # Filter columns
            cols_to_show = [c for c in CSV_COLUMNS if c in df_target.columns and df_target[c].notna().any()]
            
            print(f"\n--- Results for {target_col} ---")
            print(df_target[cols_to_show].to_string(index=False))
            print("-" * 40 + "\n")

    logger.info(f" --> Results saved in: {csv_path}")

if __name__ == "__main__":
    main()