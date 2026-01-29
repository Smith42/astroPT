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
Date: January 2026
"""

import argparse
import glob
import logging
import os
import sys
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
DEFAULT_TARGETS = ['Z', 'LOGMSTAR', 'LOGSFR', 'GR', 'SPECTYPE', 'data_set_release', 
                   'flux_detection_total', 'SNR_SPEC_R', 'HALPHA_FLUX', 'OIII_5007_FLUX', 'OII_3726_FLUX', 'NII_6584_FLUX']
REGRESSION_TARGETS = ['Z', 'LOGMSTAR', 'LOGSFR', 'GR', 
                      'flux_detection_total', 'SNR_SPEC_R', 'HALPHA_FLUX', 'OIII_5007_FLUX', 'OII_3726_FLUX', 'NII_6584_FLUX'] 

# Column order for the final CSV
CSV_COLUMNS = [
    'Target', 'Task', 'Modality', 'Probe',
    'R2', 'RMSE', 'Bias', 'NMAD', 'Outliers',
    'Accuracy', 'F1_Weighted', 'FPR'
]

def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="AstroPT Downstream Prober (LP & MLP)")
    
    # Data Paths
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Directory containing .npy embeddings")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to .fits catalog")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save results")
    
    # Task Config
    parser.add_argument("--targets", nargs='+', default=None, help="Overwrites default target list.")
    parser.add_argument("--targets_add", nargs='+', default=None, help="Appends to default list.")
    
    # Probing Config
    parser.add_argument("--probes", nargs='+', default=['lp', 'mlp'], choices=['lp', 'mlp'], 
                        help="Type of probes to run: 'lp', 'mlp' or both.")
    
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


# Data Loading
def load_data_global(embeddings_dir: str, metadata_path: str) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """
    Loads embeddings (.npy) from a directory and metadata, aligning them by TARGETID.
    Uses memmap for efficient loading of large embedding files.

    Args:
        embeddings_dir: Directory containing .npy files.
        metadata_path: Path to the FITS catalog.

    Returns:
        Tuple containing the aligned embeddings dictionary and metadata DataFrame.
    """
    # Scan for IDs
    if not os.path.exists(embeddings_dir):
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")
        
    id_candidates = ['targetid.npy', 'ids.npy', 'target_ids.npy', 'object_ids.npy']
    id_path = None
    for cand in id_candidates:
        p = os.path.join(embeddings_dir, cand)
        if os.path.exists(p):
            id_path = p
            break
            
    if not id_path:
        raise FileNotFoundError(f"Could not find IDs ({id_candidates}) in {embeddings_dir}")
        
    logger.info(f"Loading IDs from {os.path.basename(id_path)}...")
    ids = np.load(id_path)
    
    # Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    try:
        t = Table.read(metadata_path)
        df = t.to_pandas()
    except Exception as e:
        logger.error(f"Failed to read FITS: {e}")
        sys.exit(1)
        
    # Bytes decoding
    for col in df.columns:
        if df[col].dtype == object and isinstance(df[col].iloc[0], bytes):
             df[col] = df[col].str.decode('utf-8')

    # Global Alignment
    logger.info(f"Performing global alignment (IDs)...")
    df['TARGETID'] = df['TARGETID'].astype('int64')
    df = df.drop_duplicates(subset=['TARGETID']).set_index('TARGETID')
    
    # Intersection
    common_ids = np.intersect1d(ids, df.index.values)
    df_aligned = df.loc[common_ids]
    
    # Indices Mapping
    id_to_idx = {id_val: i for i, id_val in enumerate(ids)}
    emb_indices = [id_to_idx[uid] for uid in df_aligned.index.values]
    
    # Load & Align Embeddings
    data_aligned = {}
    modalities = ['images', 'spectra', 'joint']
    
    for mod in modalities:
        candidates = glob.glob(os.path.join(embeddings_dir, f"{mod}*.npy"))
        
        if candidates:
            best_cand = candidates[0]
            for c in candidates:
                if os.path.basename(c) == f"{mod}.npy":
                    best_cand = c
                    break
            
            logger.info(f"Loading & Aligning {mod} from {os.path.basename(best_cand)}...")
            # Memmap reading + Fancy indexing (loads subset to RAM)
            raw_emb = np.load(best_cand, mmap_mode='r')
            data_aligned[mod] = raw_emb[emb_indices]
        else:
            logger.warning(f"No embeddings found for modality: {mod}")
        
    logger.info(f"Global alignment complete. {len(df_aligned)} objects available.")
    return data_aligned, df_aligned


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
    if target_col in REGRESSION_TARGETS or pd.api.types.is_numeric_dtype(vals):
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
            outliers = np.mean(np.abs(diff) > 0.5) * 100 
            
        return {
            "R2": r2, 
            "RMSE": rmse, 
            "Bias": bias, 
            "NMAD": nmad, 
            "Outliers": outliers
        }
    else:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # FPR Calculation
        cm = confusion_matrix(y_true, y_pred)
        FP = cm.sum(axis=0) - np.diag(cm)
        TN = cm.sum() - (FP + (cm.sum(axis=1) - np.diag(cm)) + np.diag(cm))
        with np.errstate(divide='ignore', invalid='ignore'):
            fpr_macro = np.mean(np.nan_to_num(FP / (FP + TN)))

        return {"Accuracy": acc, "F1_Weighted": f1, "FPR": fpr_macro}


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


def save_row_to_csv(row: Dict[str, Any], csv_path: str):
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
    
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=header, index=False)


def main():
    """Main execution block."""
    args = parse_args()
    out_dir = args.out_dir if args.out_dir else args.embeddings_dir
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "probing_results_all.csv")
    
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
    raw_data, df_global = load_data_global(args.embeddings_dir, args.metadata_path)
    
    if not raw_data:
        logger.error("No valid embeddings found. Exiting.")
        sys.exit(1)

    # EORDER TARGETS (Regression First, then Classification)
    reg_list = []
    cls_list = []

    for t in initial_targets:
        if t in df_global.columns:
            vals = df_global[t]
            if t in REGRESSION_TARGETS or pd.api.types.is_numeric_dtype(vals):
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
            if mode not in raw_data: continue
            
            # Get Data
            try:
                X, y, task_type, processor = get_task_data(raw_data, df_global, mode, target_col)
                output_dim = 1 if task_type == 'regression' else len(processor.classes_)
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
                    if task_type == 'classification':
                        cm = confusion_matrix(true_vals, preds)
                        plt.figure(figsize=(6, 5))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                    xticklabels=processor.classes_, yticklabels=processor.classes_)
                        plt.title(f"CM: {target_col} ({mode} - {probe.upper()})")
                        plt.tight_layout()
                        plt.savefig(os.path.join(out_dir, f"cm_{target_col}_{mode}_{probe}.png"))
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