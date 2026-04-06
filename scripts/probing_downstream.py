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
import json
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
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
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
                    'flux_detection_total', 'HALPHA_EW', 'HALPHA_FLUX', 'NII_6584_FLUX', 'OIII_5007_FLUX', 'HBETA_FLUX',
                    'sersic_sersic_vis_radius', 'sersic_sersic_vis_index', 'sersic_sersic_vis_axis_ratio',
                    'has_spiral_arms_yes', 'smoothness', 'gini', 'SPECTYPE', 'data_set_release']

CLASSIFICATION_TARGETS = ['SPECTYPE', 'data_set_release'] 

# Column order for the final CSV
CSV_COLUMNS = [
    'Target', 'Task', 'Modality', 'Probe',
    'R2', 'RMSE', 'Bias', 'NMAD', 'Outliers',
    'Accuracy', 'Precision', 'Recall', 'F1_score', 'FPR'
]

SEED_STATS_COLUMNS = [
    'Target', 'Task', 'Modality', 'Probe', 'N_Seeds',
    'R2_mean', 'R2_std', 'RMSE_mean', 'RMSE_std', 'Bias_mean', 'Bias_std',
    'NMAD_mean', 'NMAD_std', 'Outliers_mean', 'Outliers_std',
    'Accuracy_mean', 'Accuracy_std', 'Precision_mean', 'Precision_std',
    'Recall_mean', 'Recall_std', 'F1_score_mean', 'F1_score_std',
    'FPR_mean', 'FPR_std', 'AvgBestEpoch'
]

DEFAULT_EASY_TARGETS = {
    'SPECTYPE',
    'data_set_release',
    'has_spiral_arms_yes',
}

def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="AstroPT Downstream Prober (LP & MLP)")
    
    # Data Paths
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to .fits catalog")
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing training weights")
    parser.add_argument("--emb_dir", type=str, required=True, help="Directory containing .npy embedding files")
    parser.add_argument("--save_dir", type=str, default=None, help="Plot Saving Directory")
    
    parser.add_argument("--train_name", type=str, default=None, help="Custom title for plots")
    
    # Task Config
    parser.add_argument("--targets", nargs='+', default=None, help="Overwrites default target list.")
    parser.add_argument("--targets_add", nargs='+', default=None, help="Appends to default list.")
    
    # Probing Config
    parser.add_argument("--probes", nargs='+', default=['lp', 'mlp'], choices=['lp', 'mlp'], 
                        help="Type of probes to run: 'lp', 'mlp' or both.")
    parser.add_argument("--conf_matrix", action="store_true", help="Generates confusion matrix")
    
    # Subsets options
    parser.add_argument("--save_name", type=str, default="downstream_results.csv", 
                        help="Custom name for the output CSV file")
    parser.add_argument("--filter_ids_path", type=str, default=None, 
                        help="Path to an external ids.npy file to restrict the evaluation to a custom ID's subset")
    
    # Training Hyperparams
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per task")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seeds", nargs='+', type=int, default=[61, 21, 278], help="Random seeds for repeated probing")
    parser.add_argument("--easy_targets", nargs='+', default=list(DEFAULT_EASY_TARGETS), help="Targets where MLP runs fewer epochs")
    parser.add_argument("--mlp_easy_epoch_factor", type=float, default=0.6, help="Epoch scaling factor for easy targets in MLP")
    parser.add_argument("--mlp_weight_decay", type=float, default=1e-3, help="MLP weight decay regularization")
    parser.add_argument("--mlp_val_split", type=float, default=0.1, help="Validation split fraction for MLP early stopping")
    parser.add_argument("--mlp_patience", type=int, default=8, help="Early stopping patience for MLP")
    parser.add_argument("--mlp_min_delta", type=float, default=1e-4, help="Minimum validation loss improvement for MLP early stopping")
    
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
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
            #nn.Dropout(0.1),
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
        if len(df) == 0:
            break
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


def align_data(
    embeddings_dict: Dict[str, np.ndarray], 
    catalog_df: pd.DataFrame, 
    external_filter_ids: Optional[np.ndarray] = None
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Aligns filtered metadata with embeddings using TARGETID intersection."""
    logger.info("Aligning embeddings with filtered metadata...")
    emb_ids = embeddings_dict['targetid']
    target_col = 'TARGETID' if 'TARGETID' in catalog_df.columns else 'targetid'
    
    catalog_df[target_col] = catalog_df[target_col].astype('int64')
    catalog_indexed = catalog_df.drop_duplicates(subset=[target_col]).set_index(target_col)
    
    # Intersection
    common_ids = np.intersect1d(emb_ids, catalog_indexed.index.values)
    
    if external_filter_ids is not None:
        initial_count = len(common_ids)
        common_ids = np.intersect1d(common_ids, external_filter_ids)
        logger.info(f"Applied external ID filter: Intersected from {initial_count} down to {len(common_ids)} objects.")
    
    matched_catalog = catalog_indexed.loc[common_ids].reset_index()

    if len(matched_catalog) == 0:
        raise ValueError("No common TARGETIDs between embeddings and filtered catalog.")
    
    # Indices Mapping
    id_to_idx = {id_val: i for i, id_val in enumerate(emb_ids)}
    valid_indices = np.array([id_to_idx[uid] for uid in matched_catalog[target_col].values])
    
    # Extract valid slices from Memmaps
    aligned_embeddings = {}
    for mod in ['images', 'spectra', 'joint']:
        if mod in embeddings_dict:
            # Fancy indexing triggers RAM loading only for valid items
            aligned_embeddings[mod] = embeddings_dict[mod][valid_indices]

    for mod_name, arr in aligned_embeddings.items():
        if len(arr) != len(matched_catalog):
            raise ValueError(
                f"Alignment mismatch for {mod_name}: {len(arr)} embeddings vs {len(matched_catalog)} rows"
            )
            
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
    if modality not in raw_data:
        raise ValueError(f"Modality '{modality}' not found in aligned embeddings.")
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found.")
    if len(raw_data[modality]) != len(df):
        raise ValueError(
            f"Length mismatch for modality '{modality}': {len(raw_data[modality])} vs {len(df)}"
        )
        
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
    target_name: str = "",
    seed: int = 61,
    mlp_val_split: float = 0.1,
    mlp_patience: int = 8,
    mlp_min_delta: float = 1e-4,
    mlp_weight_decay: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, int]:
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
        Tuple of (Predictions, True Values, Best Epoch) for the test set.
    """
    set_seed(seed)

    if len(X) < 20:
        raise ValueError(f"Too few samples for stable probing split: {len(X)}")

    # Split datasets
    stratify_labels = y if task_type == 'classification' and len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_labels,
    )
    
    # Scaling: L2 Normalization preserves hypersphere vector geometry (cosine similarity). 
    # StandardScaler scales variance and shifts mean, distorting geometry.
    X_train = normalize(X_train, norm='l2', axis=1)
    X_test = normalize(X_test, norm='l2', axis=1)
    
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
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=(len(train_ds) > batch_size))

    val_loader = None
    if probe_type == 'mlp' and mlp_val_split > 0.0:
        val_stratify = y_train if task_type == 'classification' and len(np.unique(y_train)) > 1 else None
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train,
            y_train,
            test_size=mlp_val_split,
            random_state=seed,
            stratify=val_stratify,
        )
        train_ds = TensorDataset(
            torch.FloatTensor(X_train_sub),
            torch.FloatTensor(y_train_sub) if task_type == 'regression' else torch.LongTensor(y_train_sub),
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val) if task_type == 'regression' else torch.LongTensor(y_val),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=(len(train_ds) > batch_size))
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    
    # Model Selection
    input_dim = X_train.shape[1]
    if probe_type == 'lp':
        model = LinearProbe(input_dim, output_dim).to(DEVICE)
        weight_decay = 1e-5 
    elif probe_type == 'mlp':
        model = AdaptiveMLP(input_dim, output_dim).to(DEVICE)
        weight_decay = mlp_weight_decay 
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss() if task_type == 'regression' else nn.CrossEntropyLoss()
    scheduler = None
    if probe_type == 'mlp' and val_loader is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=max(2, mlp_patience // 2),
            min_lr=1e-6,
        )
    
    # Training Loop
    model.train()
    best_state = None
    best_val_loss = float('inf')
    best_epoch = epochs
    patience_counter = 0

    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            if task_type == 'regression': preds = preds.squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        if probe_type == 'mlp' and val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    preds = model(xb)
                    if task_type == 'regression':
                        preds = preds.squeeze()
                    val_loss = criterion(preds, yb)
                    val_losses.append(val_loss.item())

            mean_val_loss = float(np.mean(val_losses)) if val_losses else float('inf')
            if scheduler is not None:
                scheduler.step(mean_val_loss)

            if mean_val_loss < (best_val_loss - mlp_min_delta):
                best_val_loss = mean_val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            model.train()

            if patience_counter >= mlp_patience:
                break

    if probe_type == 'mlp' and best_state is not None:
        model.load_state_dict(best_state)

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
            
    return np.concatenate(all_preds), y_test_raw, best_epoch


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
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = not csv_path.is_file()
    df.to_csv(csv_path, mode='a', header=header, index=False)


def save_seed_stats_row_to_csv(row: Dict[str, Any], csv_path: str | Path):
    """Append one row with per-metric mean/std across seeds."""
    df = pd.DataFrame([row])
    for col in SEED_STATS_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[SEED_STATS_COLUMNS]

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = not csv_path.is_file()
    df.to_csv(csv_path, mode='a', header=header, index=False)


def aggregate_seed_metrics(
    metrics_per_seed: List[Dict[str, float]],
    best_epochs: List[int],
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """Aggregate mean/std metric dictionaries and average best epoch."""
    if not metrics_per_seed:
        raise ValueError("No successful seed runs to aggregate.")

    keys = sorted({k for d in metrics_per_seed for k in d.keys()})
    means = {}
    stds = {}
    for k in keys:
        vals = [float(d[k]) for d in metrics_per_seed if k in d]
        if not vals:
            continue
        means[k] = float(np.mean(vals))
        stds[k] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    avg_best_epoch = float(np.mean(best_epochs)) if best_epochs else float('nan')
    return means, stds, avg_best_epoch


def compute_transfer_gap_rows(target_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build per-target/probe transfer gap rows between image and spectra embeddings."""
    if not target_results:
        return []

    rows = []
    df_target = pd.DataFrame(target_results)
    if df_target.empty:
        return rows

    for probe in sorted(df_target['Probe'].dropna().unique()):
        probe_df = df_target[df_target['Probe'] == probe]
        img_row = probe_df[probe_df['Modality'] == 'images']
        spec_row = probe_df[probe_df['Modality'] == 'spectra']
        if img_row.empty or spec_row.empty:
            continue

        img_row = img_row.iloc[0]
        spec_row = spec_row.iloc[0]
        task = img_row['Task']

        if task == 'regression' and pd.notna(img_row.get('R2')) and pd.notna(spec_row.get('R2')):
            primary_metric = 'R2'
            img_val = float(img_row['R2'])
            spec_val = float(spec_row['R2'])
        elif task == 'classification' and pd.notna(img_row.get('F1_score')) and pd.notna(spec_row.get('F1_score')):
            primary_metric = 'F1_score'
            img_val = float(img_row['F1_score'])
            spec_val = float(spec_row['F1_score'])
        else:
            continue

        rows.append({
            'Target': img_row['Target'],
            'Task': task,
            'Probe': probe,
            'PrimaryMetric': primary_metric,
            'ImageValue': img_val,
            'SpectraValue': spec_val,
            'Gap_ImageMinusSpectra': img_val - spec_val,
            'AbsGap': abs(img_val - spec_val),
        })

    return rows


def main():
    """Main execution block."""
    args = parse_args()
    
    # Required paths
    weights_dir = Path(args.weights_dir)
    emb_dir = Path(args.emb_dir)
    metadata_path = Path(args.metadata_path)
    
    save_dir = Path(args.save_dir) / "downstream_tasks" if args.save_dir else emb_dir / "downstream_tasks"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = save_dir / args.save_name
    seed_stats_csv_path = save_dir / "downstream_results_seedstats.csv"
    gap_csv_path = save_dir / "cross_modal_transfer_gap.csv"
    
    # Fixing the seed baseline for deterministic preprocessing order
    set_seed(args.seeds[0])
    
    # Prepare Initial Targets List
    initial_targets = DEFAULT_TARGETS.copy()
    if args.targets: initial_targets = args.targets
    elif args.targets_add: initial_targets.extend(args.targets_add)

    initial_targets = list(dict.fromkeys(initial_targets))
    
    # Prepare Probes List
    # LP is always executed as the official per-target baseline.
    probes_to_run = ['lp'] + [p for p in args.probes if p != 'lp']
    logger.info(f"Initial Target List: {initial_targets}")
    logger.info(f"Probes to run: {probes_to_run}")
    logger.info(f"Seeds: {args.seeds}")
    
    # Load Data
    raw_data_dict, raw_df = load_data(emb_dir, metadata_path)
    
    if not raw_data_dict:
        logger.error("No valid embeddings found. Exiting.")
        sys.exit(1)
        
    filter_ids = None
    if args.filter_ids_path:
        filter_ids_path = Path(args.filter_ids_path)
        if filter_ids_path.exists():
            logger.info(f"Loading external IDs for Fair Benchmarking from: {filter_ids_path}")
            filter_ids = np.load(filter_ids_path).astype('int64')
        else:
            logger.error(f"External filter IDs file not found: {filter_ids_path}")
            sys.exit(1)
    
    # Apply scientific filters
    filtered_df = filter_catalog(raw_df)
    
    # Align and extract to RAM
    aligned_df, aligned_embeddings = align_data(raw_data_dict, filtered_df, filter_ids)
    
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


    all_gap_rows = []
    easy_targets = set(args.easy_targets)

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
                print(f"    -> Running {mode.upper()} / {probe.upper()}...\n", end="", flush=True)
                
                try:
                    effective_epochs = args.epochs
                    if probe == 'mlp' and target_col in easy_targets:
                        effective_epochs = max(10, int(args.epochs * args.mlp_easy_epoch_factor))

                    metrics_per_seed = []
                    best_epochs = []
                    for seed in args.seeds:
                        preds, true_vals, best_epoch = train_engine(
                            X,
                            y,
                            probe,
                            task_type,
                            output_dim,
                            effective_epochs,
                            args.lr,
                            args.batch_size,
                            target_col,
                            seed=seed,
                            mlp_val_split=args.mlp_val_split,
                            mlp_patience=args.mlp_patience,
                            mlp_min_delta=args.mlp_min_delta,
                            mlp_weight_decay=args.mlp_weight_decay,
                        )
                        seed_metrics = compute_metrics(true_vals, preds, task_type, target_col)
                        metrics_per_seed.append(seed_metrics)
                        best_epochs.append(best_epoch)

                    metrics_mean, metrics_std, avg_best_epoch = aggregate_seed_metrics(metrics_per_seed, best_epochs)
                    
                    # Store Results
                    row = {
                        "Target": target_col, 
                        "Task": task_type, 
                        "Modality": mode,
                        "Probe": probe.upper()
                    }
                    row.update(metrics_mean)
                    
                    # SAVE TO CSV IMMEDIATELY
                    save_row_to_csv(row, csv_path)

                    seed_stats_row = {
                        "Target": target_col,
                        "Task": task_type,
                        "Modality": mode,
                        "Probe": probe.upper(),
                        "N_Seeds": len(metrics_per_seed),
                        "AvgBestEpoch": avg_best_epoch,
                    }
                    for k, v in metrics_mean.items():
                        seed_stats_row[f"{k}_mean"] = v
                    for k, v in metrics_std.items():
                        seed_stats_row[f"{k}_std"] = v
                    save_seed_stats_row_to_csv(seed_stats_row, seed_stats_csv_path)
                    
                    # Add to local list for table
                    target_results.append(row)

                    # Print compact mean ± std using primary metric
                    if task_type == 'regression' and 'R2' in metrics_mean:
                        print(f" Done. [R2: {metrics_mean['R2']:.4f} +/- {metrics_std.get('R2', 0.0):.4f}]")
                    elif task_type == 'classification' and 'F1_score' in metrics_mean:
                        print(f" Done. [F1: {metrics_mean['F1_score']:.4f} +/- {metrics_std.get('F1_score', 0.0):.4f}]")
                    else:
                        print(" Done.")
                    
                    # Save Confusion Matrix
                    if task_type == 'classification' and args.conf_matrix:
                        cm = confusion_matrix(true_vals, preds)
                        plt.figure(figsize=(6, 5))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                    xticklabels=processor.classes_, yticklabels=processor.classes_)
                        plt.title(f"CM: {target_col} ({mode} - {probe.upper()})")
                        plt.tight_layout()
                        safe_target = str(target_col).replace('/', '_').replace(' ', '_')
                        plt.savefig(save_dir / f"cm_{safe_target}_{mode}_{probe}.png")
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

            # Cross-modal transfer gap summary for this target
            gap_rows = compute_transfer_gap_rows(target_results)
            all_gap_rows.extend(gap_rows)
            if gap_rows:
                gap_df = pd.DataFrame(gap_rows)
                gap_cols = [
                    'Target', 'Task', 'Probe', 'PrimaryMetric',
                    'ImageValue', 'SpectraValue', 'Gap_ImageMinusSpectra', 'AbsGap'
                ]
                print(f"--- Cross-Modal Gap for {target_col} ---")
                print(gap_df[gap_cols].to_string(index=False))
                print("-" * 40 + "\n")

    logger.info(f" --> Results saved in: {csv_path}")
    logger.info(f" --> Seed stats saved in: {seed_stats_csv_path}")

    if all_gap_rows:
        gap_df_all = pd.DataFrame(all_gap_rows)
        gap_df_all = gap_df_all.sort_values(['AbsGap', 'Target'], ascending=[False, True])
        gap_df_all.to_csv(gap_csv_path, index=False)
        logger.info(f" --> Cross-modal gap summary saved in: {gap_csv_path}")

if __name__ == "__main__":
    main()