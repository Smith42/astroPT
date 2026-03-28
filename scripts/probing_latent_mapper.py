"""
AstroPT Cross-Modal Latent Mapper & k-NN Evaluator.

This script trains a Multi-Layer Perceptron (MLP) to translate embeddings from a 
source modality (e.g., images) to a target modality (e.g., joint). 
It then evaluates the physical meaning of these mapped embeddings by performing 
a k-Nearest Neighbors (k-NN) search against a gallery of real target embeddings.
Predictions are made using an Inverse Distance Weighted average (or weighted voting)
of the neighbors' physical properties.

Author: Victor Alonso Rodriguez
Date: March 2026
"""

import argparse
import logging
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, 
                             r2_score, mean_squared_error, 
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# Logger Configuration
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-LatentMapper")

warnings.simplefilter('ignore', category=AstropyWarning)
warnings.filterwarnings('ignore', module='astropy.io.fits')

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Targets Definition
DEFAULT_TARGETS = ['Z', 'LOGMSTAR', 'LOGSFR', 'GR', 
                   'flux_detection_total', 'HALPHA_EW', 'HALPHA_FLUX', 'NII_6584_FLUX', 
                   'OIII_5007_FLUX', 'HBETA_FLUX', 'sersic_sersic_vis_radius', 
                   'sersic_sersic_vis_index', 'sersic_sersic_vis_axis_ratio',
                   'has_spiral_arms_yes', 'smoothness', 'gini', 'SPECTYPE', 'data_set_release']

CLASSIFICATION_TARGETS = ['SPECTYPE', 'data_set_release']

CSV_COLUMNS = [
    'Target', 'Task', 'Mapping', 'k_Neighbors',
    'R2', 'RMSE', 'Bias', 'NMAD', 'Outliers',
    'Accuracy', 'Precision', 'Recall', 'F1_score', 'FPR'
]


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="AstroPT Cross-Modal Latent Mapper")
    
    # Data Paths
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to .fits catalog")
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing training weights")
    parser.add_argument("--emb_dir", type=str, required=True, help="Directory containing .npy embedding files")
    parser.add_argument("--save_dir", type=str, required=True, help="Plot Saving Directory")
    parser.add_argument("--overwrite", action="store_true", help="If set, deletes existing CSV files before saving new results.")
    
    # Mapping Configuration
    parser.add_argument("--source", type=str, default="images", choices=['images', 'spectra', 'joint'], help="Source modality to map FROM")
    parser.add_argument("--target", type=str, default="joint", choices=['images', 'spectra', 'joint'], help="Target modality to map TO (Ignored if --all_targets is used)")
    parser.add_argument("--all_targets", action="store_true", help="If set, maps the source to ALL other available modalities sequentially.")
    parser.add_argument("--k_neighbors", type=int, default=5, help="Number of neighbors for k-NN physical retrieval")
    parser.add_argument("--use_mlp", action="store_true", help="If set, trains an MLP to translate embeddings. Otherwise, performs direct Zero-Shot mapping.")
    
    # Training Hyperparams
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs for the mapper")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    return parser.parse_args()


def set_seed(seed: int = 61):
    """Fixes random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Seed set to {seed} for reproducibility.")


class LatentMapperMLP(nn.Module):
    """
    MLP architecture designed to translate a latent vector from one 
    modality space to another modality space.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L2 normalization of the output to match cosine spaces
        out = self.net(x)
        return torch.nn.functional.normalize(out, p=2, dim=1)


# Reusing Downstream Data Loaders
def load_data(emb_dir: Path, metadata_path: Path) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Loads embeddings (.npy) and FITS metadata."""
    logger.info(f"Scanning embeddings directory: {emb_dir}...")
    if not emb_dir.exists(): raise FileNotFoundError(f"Embeddings not found: {emb_dir}")
        
    data_dict = {}
    id_candidates = ['targetid.npy', 'ids.npy', 'target_ids.npy', 'object_ids.npy']
    id_path = next((emb_dir / c for c in id_candidates if (emb_dir / c).exists()), None)
    if not id_path: raise FileNotFoundError("Could not find Target IDs.")
    data_dict['targetid'] = np.load(id_path)

    for mod in ['images', 'spectra', 'joint']:
        candidates = list(emb_dir.glob(f"{mod}*.npy"))
        if candidates:
            best = next((c for c in candidates if c.name == f"{mod}.npy"), candidates[0])
            logger.info(f"Loading {mod} from: {best.name} (memmap)")
            data_dict[mod] = np.load(best, mmap_mode='r')

    try:
        catalog = Table.read(metadata_path)
        df = catalog.to_pandas()
    except Exception as e:
        logger.error(f"Failed to read FITS: {e}")
        sys.exit(1)
    
    for col in df.columns:
        if df[col].dtype == object and isinstance(df[col].iloc[0], bytes):
            try: df[col] = df[col].str.decode('utf-8')
            except: pass
    if 'SURVEY' in df.columns: df['SURVEY'] = df['SURVEY'].str.lower().str.strip()
    return data_dict, df

def filter_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Applies strict spectral quality filtering (SNR > 3 & H-alpha)."""
    initial_len = len(df)
    mask_snr = (df['SNR_SPEC_R'] > 3.0) | (df['SNR_SPEC_Z'] > 3.0) if 'SNR_SPEC_R' in df.columns else pd.Series(True, index=df.index)
    if 'HALPHA_FLUX' in df.columns and 'HALPHA_FLUX_IVAR' in df.columns:
        halpha_err = np.where(df['HALPHA_FLUX_IVAR'] > 0, 1.0 / np.sqrt(df['HALPHA_FLUX_IVAR'].clip(lower=1e-10)), np.inf)
        mask_halpha = df['HALPHA_FLUX'] > (3.0 * halpha_err)
    else: mask_halpha = pd.Series(True, index=df.index)

    filtered_df = df[mask_snr & mask_halpha].copy()
    logger.info(f"Filtered out noisy spectra. Retained {len(filtered_df)} ({len(filtered_df)/initial_len:.1%})")
    return filtered_df

def align_data(embeddings_dict: Dict[str, np.ndarray], catalog_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Aligns filtered metadata with embeddings using TARGETID intersection."""
    emb_ids = embeddings_dict['targetid']
    target_col = 'TARGETID' if 'TARGETID' in catalog_df.columns else 'targetid'
    
    catalog_df[target_col] = catalog_df[target_col].astype('int64')
    catalog_indexed = catalog_df.drop_duplicates(subset=[target_col]).set_index(target_col)
    
    common_ids = np.intersect1d(emb_ids, catalog_indexed.index.values)
    matched_catalog = catalog_indexed.loc[common_ids].reset_index()
    
    id_to_idx = {id_val: i for i, id_val in enumerate(emb_ids)}
    valid_indices = np.array([id_to_idx[uid] for uid in matched_catalog[target_col].values])
    
    aligned_embeddings = {}
    for mod in ['images', 'spectra', 'joint']:
        if mod in embeddings_dict:
            aligned_embeddings[mod] = embeddings_dict[mod][valid_indices]
            
    logger.info(f"Final aligned objects available for mapping: {len(matched_catalog)}")
    return matched_catalog, aligned_embeddings


def train_latent_mapper(
    X_train: np.ndarray, y_train: np.ndarray, 
    X_test: np.ndarray, 
    epochs: int, lr: float, batch_size: int
) -> np.ndarray:
    """
    Trains the MLP to map Source -> Target embeddings.
    Uses CosineEmbeddingLoss to optimize angular similarity.
    """
    logger.info(f"Training Latent Mapper MLP for {epochs} epochs...")
    
    # Input Standard Scaling
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Target embeddings are normalized for cosine similarity
    y_train_tensor = torch.FloatTensor(y_train).to(DEVICE)
    y_train_norm = torch.nn.functional.normalize(y_train_tensor, p=2, dim=1)
    
    train_ds = TensorDataset(torch.FloatTensor(X_train_scaled), y_train_norm.cpu())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = LatentMapperMLP(input_dim, output_dim).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CosineEmbeddingLoss()
    
    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            
            target_ones = torch.ones(xb.size(0)).to(DEVICE)
            loss = criterion(preds, yb, target_ones)
            
            loss.backward()
            optimizer.step()

    # Inference on Test Set
    model.eval()
    test_ds = TensorDataset(torch.FloatTensor(X_test_scaled))
    test_loader = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False)
    
    all_preds = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb[0].to(DEVICE)
            preds = model(xb)
            all_preds.append(preds.cpu().numpy())
            
    logger.info("Latent Mapper Training Complete.")
    return np.concatenate(all_preds)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str, target_name: str) -> Dict[str, float]:
    """Computes downstream metrics (Reused from standard probing)."""
    if task_type == 'regression':
        diff = y_pred - y_true
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        if target_name == 'Z':
            norm_diff = diff / (1 + y_true)
            bias, nmad = np.median(norm_diff), 1.4826 * np.median(np.abs(norm_diff - np.median(norm_diff)))
            outliers = np.mean(np.abs(norm_diff) > 0.15) * 100
        elif target_name == 'LOGMSTAR':
            bias, nmad = np.median(diff), 1.4826 * np.median(np.abs(diff - np.median(diff)))
            outliers = np.mean(np.abs(diff) > 0.25) * 100 
        else:
            bias, nmad = np.median(diff), 1.4826 * np.median(np.abs(diff - np.median(diff)))
            relative_error = np.abs(diff) / (np.abs(y_true) + 1e-5)
            outliers = np.mean(relative_error > 0.30) * 100
            
        return {"R2": r2, "RMSE": rmse, "Bias": bias, "NMAD": nmad, "Outliers": outliers}
    else:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred)
        FP = cm.sum(axis=0) - np.diag(cm)
        TN = cm.sum() - (FP + (cm.sum(axis=1) - np.diag(cm)) + np.diag(cm))
        with np.errstate(divide='ignore', invalid='ignore'):
            fpr_macro = np.mean(np.nan_to_num(FP / (FP + TN)))
        return {"Accuracy": acc, "Precision": precision, "Recall": recall, "F1_score": f1, "FPR": fpr_macro}


def save_row_to_csv(row: Dict[str, Any], csv_path: Path):
    """Appends a single row to the CSV."""
    df = pd.DataFrame([row])
    for col in CSV_COLUMNS:
        if col not in df.columns: df[col] = np.nan
    df = df[CSV_COLUMNS]
    header = not csv_path.is_file()
    df.to_csv(csv_path, mode='a', header=header, index=False)


def main():
    args = parse_args()
    set_seed(61)
    
    emb_dir = Path(args.emb_dir)
    metadata_path = Path(args.metadata_path)
    save_dir = Path(args.save_dir) / "latent_mapper"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"--- LATENT MAPPER INITIATED ---")
    logger.info(f"Source Modality: {args.source.upper()}")
    logger.info(f"Retrieval Method: Weighted k-NN (k={args.k_neighbors})")
    
    # Load & Prepare Data
    raw_data_dict, raw_df = load_data(emb_dir, metadata_path)
    filtered_df = filter_catalog(raw_df)
    aligned_df, aligned_embeddings = align_data(raw_data_dict, filtered_df)
    
    if args.source not in aligned_embeddings:
        logger.error(f"Source modality '{args.source}' not found in embeddings.")
        sys.exit(1)
        
    X_full = aligned_embeddings[args.source]
    
    # Split Indices
    indices = np.arange(len(aligned_df))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Determine target modalities
    if args.all_targets:
        target_modalities = [mod for mod in ['images', 'spectra', 'joint'] if mod != args.source and mod in aligned_embeddings]
        logger.info(f"Batch Mode ON: Will map sequentially to -> {target_modalities}")
    else:
        target_modalities = [args.target]

    # MAIN MAPPING LOOP
    for current_target in target_modalities:
        logger.info(f"\n" + "-"*40)
        logger.info(f"MAPPING FLOW: {args.source.upper()} ---> {current_target.upper()}")
        logger.info("-"*40)
        
        mapping_name = f"{args.source}->{current_target}{' (MLP)' if args.use_mlp else ' (Zero-Shot)'}"
        csv_path = save_dir / f"mapper_{args.source}_to_{current_target}_{'mlp' if args.use_mlp else 'zeroshot'}.csv"
        
        if args.overwrite and csv_path.exists():
            logger.info(f"Overwrite flag is SET. Deleting previous file: {csv_path.name}")
            csv_path.unlink()
        
        Y_full = aligned_embeddings[current_target]
        
        # Train Mapper
        if args.use_mlp:
            logger.info(f"Executing Phase 1: Training Modality Translator ({current_target}) with MLP...")
            y_pred_test_emb = train_latent_mapper(
                X_train=X_full[train_idx], y_train=Y_full[train_idx],
                X_test=X_full[test_idx],
                epochs=args.epochs, lr=args.lr, batch_size=args.batch_size
            )
        else:
            logger.info(f"Executing Phase 1: Direct Zero-Shot Alignment ({args.source.upper()} -> {current_target.upper()})...")
            X_test_tensor = torch.FloatTensor(X_full[test_idx])
            y_pred_test_emb = torch.nn.functional.normalize(X_test_tensor, p=2, dim=1).numpy()
        
        # Fit k-NN on the true target embeddings (Gallery)
        logger.info(f"Executing Phase 2: Fitting k-NN on {current_target.upper()} Gallery (Train Set)...")
        gallery_embeddings = Y_full[train_idx]
        
        # L2 normalize gallery for cosine metric
        gallery_norm = gallery_embeddings / np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
        knn = NearestNeighbors(n_neighbors=args.k_neighbors, metric='cosine')
        knn.fit(gallery_norm)
        
        logger.info("Querying neighbors for fictitious target embeddings...")
        distances, neighbor_indices = knn.kneighbors(y_pred_test_emb)
        
        # Physical Evaluation Loop
        logger.info(f"Executing Phase 3: Physical Evaluation via Retrieved Weighted {current_target.upper()} Neighbors")
        
        for target_col in DEFAULT_TARGETS:
            if target_col not in aligned_df.columns: continue
            
            vals = aligned_df[target_col]
            task_type = 'classification' if target_col in CLASSIFICATION_TARGETS else 'regression'
            
            y_true_test = vals.values[test_idx]
            y_train_gallery = vals.values[train_idx]
            
            if task_type == 'regression':
                valid_test_mask = pd.notna(y_true_test) & (y_true_test > -90)
            else:
                valid_test_mask = pd.notna(y_true_test) & (y_true_test != "") & (y_true_test != "N/A")
                
            if not valid_test_mask.any(): continue
            
            neighbor_props = y_train_gallery[neighbor_indices]
            
            # INVERSE DISTANCE WEIGHTING LOGIC
            weights = 1.0 / (distances + 1e-6)
            
            if task_type == 'regression':
                neighbor_props_float = neighbor_props.astype(float)
                
                valid_mask = ~np.isnan(neighbor_props_float)
                masked_weights = np.where(valid_mask, weights, 0.0)
                sum_weights = np.sum(masked_weights, axis=1)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    weighted_sum = np.nansum(neighbor_props_float * masked_weights, axis=1)
                    y_pred = np.where(sum_weights > 0, weighted_sum / sum_weights, np.nan)
                    
            else:
                # Classification via Weighted Voting
                y_pred_list = []
                for row_idx, row in enumerate(neighbor_props):
                    row_weights = weights[row_idx]
                    class_weights = {}
                    
                    for val, w in zip(row, row_weights):
                        if pd.notna(val) and val != "":
                            class_weights[val] = class_weights.get(val, 0) + w
                            
                    if class_weights:
                        best_class = max(class_weights, key=class_weights.get)
                        y_pred_list.append(best_class)
                    else:
                        y_pred_list.append(None)
                
                y_pred = np.array(y_pred_list)
                
            # Clean final arrays
            final_valid_mask = valid_test_mask & pd.notna(y_pred)
            final_y_true = y_true_test[final_valid_mask]
            final_y_pred = y_pred[final_valid_mask]
            
            if len(final_y_true) < 10: 
                continue
                
            if task_type == 'classification':
                le = LabelEncoder()
                le.fit(y_train_gallery[pd.notna(y_train_gallery) & (y_train_gallery != "")])
                known_mask = np.isin(final_y_true, le.classes_) & np.isin(final_y_pred, le.classes_)
                final_y_true = le.transform(final_y_true[known_mask])
                final_y_pred = le.transform(final_y_pred[known_mask])
                
            # Metrics Calculation
            try:
                metrics = compute_metrics(final_y_true, final_y_pred, task_type, target_col)
                row = {
                    "Target": target_col, 
                    "Task": task_type, 
                    "Mapping": mapping_name,
                    "k_Neighbors": args.k_neighbors
                }
                row.update(metrics)
                save_row_to_csv(row, csv_path)
                
                main_metric = f"R2: {metrics['R2']:.3f}" if task_type == 'regression' else f"Acc: {metrics['Accuracy']:.3f}"
                print(f"  -> {target_col:25} | {main_metric}")
                
            except Exception as e:
                logger.error(f"Failed metrics for {target_col}: {e}")

        logger.info(f"Mapping {args.source}->{current_target} Evaluation complete. Saved to: {csv_path.name}")

    logger.info("All specified mappings have been processed.")

if __name__ == "__main__":
    main()