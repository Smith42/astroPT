"""
AstroPT Downstream Task Prober (Flexible Target Management).

This script performs a battery of evaluations by training MLPs on pre-computed
embeddings (.npz) to predict multiple physical properties.

It features a flexible argument system:
- Default targets: Z, LOGMSTAR, LOGSFR, SPECTYPE.
- Use --targets_add to append new properties to the defaults.
- Use --targets to completely overwrite the list.

Author: Victor Alonso Rodriguez
Date: January 2026
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any, Tuple, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from astropy.table import Table
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, 
                             r2_score, mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- Logging Configuration ---
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-MLP")

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# --- Constants ---
# The standard "Big 4" properties to check every time
DEFAULT_TARGETS = ['Z', 'LOGMSTAR', 'LOGSFR', 'SPECTYPE']

# Known regression columns to assist auto-detection
REGRESSION_TARGETS = ['Z', 'LOGMSTAR', 'LOGSFR', 'METALLICITY', 'HALPHA_FLUX', 'flux_detection_total'] 


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AstroPT Downstream MLP Battery")
    
    parser.add_argument("--embeddings_path", type=str, required=True, help="Path to .npz embeddings")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to .fits catalog")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results")
    
    # Flexible Target Management
    parser.add_argument("--targets", nargs='+', default=None, 
                        help="Overwrites the default target list completely.")
    parser.add_argument("--targets_add", nargs='+', default=None, 
                        help="Appends these targets to the default list.")
    
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per task")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    return parser.parse_args()


class AdaptiveMLP(nn.Module):
    """
    Simple MLP that adapts its output dimension based on the task.
    Structure: Input -> 256 -> 128 -> Output
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)


def load_data_global(embeddings_path: str, metadata_path: str) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """
    Loads embeddings and metadata ONCE.
    Aligns them by TARGETID intersection but does NOT filter NaNs yet.
    """
    # 1. Load Embeddings
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
    data = np.load(embeddings_path)
    ids = data['targetid']
    
    # 2. Load Metadata
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

    # 3. Global Alignment (Intersection)
    logger.info(f"Performing global alignment (IDs)...")
    
    df['TARGETID'] = df['TARGETID'].astype('int64')
    df = df.drop_duplicates(subset=['TARGETID']).set_index('TARGETID')
    
    common_ids = np.intersect1d(ids, df.index.values)
    
    # Reduce DF to common IDs
    df_aligned = df.loc[common_ids]
    
    # Reduce Embeddings to common IDs (preserving order of df_aligned)
    id_to_idx = {id_val: i for i, id_val in enumerate(ids)}
    emb_indices = [id_to_idx[uid] for uid in df_aligned.index.values]
    
    data_aligned = {}
    for key in data.files:
        if key == 'targetid': continue
        data_aligned[key] = data[key][emb_indices]
        
    logger.info(f"Global alignment complete. {len(df_aligned)} objects available for tasks.")
    return data_aligned, df_aligned


def get_task_data(
    raw_data: Dict[str, np.ndarray], 
    df: pd.DataFrame, 
    modality: str, 
    target_col: str
) -> Tuple[np.ndarray, np.ndarray, str, Any]:
    """
    Filters the globally aligned data for a SPECIFIC target column (removing NaNs).
    Returns: X, y, task_type, encoder/scaler
    """
    
    # 1. Check target existence
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in catalog.")
        
    # 2. Create Mask for Valid Values
    vals = df[target_col]
    
    # Determine Task Type
    if target_col in REGRESSION_TARGETS or pd.api.types.is_numeric_dtype(vals):
        task_type = 'regression'
        # Filter NaNs and nonsensical values (e.g. -99)
        mask = vals.notna() & (vals > -90)
    else:
        task_type = 'classification'
        mask = vals.notna() & (vals != "") & (vals != "N/A")
        
    # 3. Apply Mask
    y_filtered = vals[mask].values
    X_filtered = raw_data[modality][mask]
    
    # 4. Prepare Y (Encode/Format)
    processor = None
    
    if task_type == 'regression':
        y_final = y_filtered.astype(np.float32)
    else:
        processor = LabelEncoder()
        y_final = processor.fit_transform(y_filtered)
        
    return X_filtered, y_final, task_type, processor


def compute_metrics(y_true, y_pred, task_type, target_name):
    """Computes relevant metrics based on task type."""
    
    if task_type == 'regression':
        # Regression Metrics
        diff = y_pred - y_true
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Astrophysics Metrics (NMAD)
        # For Z, normalize by (1+z)
        if target_name == 'Z':
            norm_diff = diff / (1 + y_true)
            bias = np.median(norm_diff)
            nmad = 1.4826 * np.median(np.abs(norm_diff - np.median(norm_diff)))
            outliers = np.mean(np.abs(norm_diff) > 0.15) * 100
        else:
            bias = np.median(diff)
            nmad = 1.4826 * np.median(np.abs(diff - np.median(diff)))
            outliers = np.mean(np.abs(diff) > 0.5) * 100 # > 0.5 dex error
            
        return {
            "R2": r2, "RMSE": rmse, "Bias": bias, "NMAD": nmad, "Outlier_Frac": outliers
        }
    
    else:
        # Classification Metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        return {
            "Accuracy": acc, "F1_Weighted": f1
        }


def train_engine(
    X: np.ndarray, 
    y: np.ndarray, 
    task_type: str, 
    output_dim: int,
    epochs: int,
    lr: float
) -> Tuple[Any, List[float]]:
    
    # Data Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling / Encoding
    if task_type == 'regression':
        # Scale Target for Regression stability
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_raw = y_test # Keep raw for final metric eval
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    else:
        # Labels are already int encoded from main
        y_test_raw = y_test

    # Tensors
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE) if task_type == 'regression' else torch.LongTensor(y_train).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    
    # Model
    model = AdaptiveMLP(input_dim=X_train.shape[1], output_dim=output_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if task_type == 'regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    # Loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_train_t)
        if task_type == 'regression':
            preds = preds.squeeze()
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()

    # Inference
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        
        if task_type == 'regression':
            preds_scaled = test_logits.cpu().numpy().flatten()
            preds_final = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        else:
            preds_final = torch.argmax(test_logits, dim=1).cpu().numpy()
            
    return preds_final, y_test_raw


def main():
    args = parse_args()
    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.embeddings_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Logic for Target Selection ---
    final_targets = DEFAULT_TARGETS.copy()
    
    if args.targets:
        logger.info(f"Target list overwritten by user.")
        final_targets = args.targets
    elif args.targets_add:
        logger.info(f"Appending new targets to default list.")
        final_targets.extend(args.targets_add)
        
    # Remove duplicates and sort
    final_targets = sorted(list(set(final_targets)))
    
    logger.info(f"Starting Multi-Target MLP Battery on {DEVICE}")
    logger.info(f"Final Target List: {final_targets}")
    
    # 1. Load Data Globally
    raw_data, df_global = load_data_global(args.embeddings_path, args.metadata_path)
    
    results_list = []
    
    # 2. Iterate over Targets
    for target_col in final_targets:
        logger.info(f"\n>>> PROCESSING TARGET: {target_col} <<<")
        
        modalities = ['images', 'spectra', 'joint']
        
        for mode in modalities:
            if mode not in raw_data: continue
            
            try:
                # Get Specific Data (Filtered)
                X, y, task_type, processor = get_task_data(raw_data, df_global, mode, target_col)
                
                # Setup Training
                output_dim = 1 if task_type == 'regression' else len(processor.classes_)
                
                # Train & Predict
                preds, true_vals = train_engine(X, y, task_type, output_dim, args.epochs, args.lr)
                
                # Compute Metrics
                metrics = compute_metrics(true_vals, preds, task_type, target_col)
                
                # Log & Store
                row = {"Target": target_col, "Task": task_type, "Modality": mode}
                row.update(metrics)
                results_list.append(row)
                
                metric_summary = f"R2: {metrics.get('R2', 0):.3f}" if task_type == 'regression' else f"Acc: {metrics.get('Accuracy', 0):.3f}"
                logger.info(f"   [{mode.upper()}] completed. {metric_summary}")
                
                # Confusion Matrix for Classification
                if task_type == 'classification':
                    cm = confusion_matrix(true_vals, preds)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=processor.classes_, yticklabels=processor.classes_)
                    plt.title(f"Confusion Matrix: {target_col} ({mode})")
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"cm_{target_col}_{mode}.png"))
                    plt.close()
                    
            except Exception as e:
                logger.error(f"Skipping {target_col} for {mode} due to error (column missing?): {e}")

    # 3. Final Export
    if results_list:
        results_df = pd.DataFrame(results_list)
        
        # Smart Column Ordering
        cols = ['Target', 'Task', 'Modality', 'R2', 'NMAD', 'Outlier_Frac', 'Accuracy', 'F1_Weighted']
        # Add extra cols if present (like Bias or RMSE)
        cols += [c for c in results_df.columns if c not in cols]
        results_df = results_df[cols]
        
        csv_path = os.path.join(output_dir, "mlp_results_all.csv")
        results_df.to_csv(csv_path, index=False)
        logger.info(f"\nBattery complete. Results saved to: {csv_path}")
        print(results_df.to_string())
    else:
        logger.warning("No valid results generated.")

if __name__ == "__main__":
    main()