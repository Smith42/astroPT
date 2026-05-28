"""
AstroPT Cross-Modal Latent Retrieval Evaluator.

This script evaluates how well the model's latent spaces are aligned across modalities
by performing a Zero-Shot k-Nearest Neighbors cross-modal retrieval.

The idea: take an image embedding, find the k closest spectra embeddings in latent space,
and predict physical properties using Inverse Distance Weighting of those neighbors.
If the prediction is good, it means the model has learned a shared representation where
images and spectra that describe the same physical object land near each other.

This is a ZERO-SHOT evaluation. No mapping network is trained — we test whether
the spaces are naturally aligned, which is the gold standard for multimodal foundation models.
It calculates retrieval metrics (Hit@k, MRR) and performs physical property inference 
via neighbor voting/averaging.

Author: Victor Alonso Rodriguez
Date: April 2026
"""

import argparse
import json
import logging
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, 
                             r2_score, mean_squared_error, 
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

# Modality Mapping for AstroPT V4
MODALITY_MAP = {
    'images': 'EuclidImage',
    'spectra': 'DESISpectrum',
    'joint': 'joint',
    'cls': 'cls'
}

# Reverse map for detection
REVERSE_MODALITY_MAP = {v: k for k, v in MODALITY_MAP.items()}

# Logger Configuration
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("AstroPT-LatentRetrieval")

warnings.simplefilter('ignore', category=AstropyWarning)
warnings.filterwarnings('ignore', module='astropy.io.fits')


# Targets Definition
DEFAULT_TARGETS = ['Z', 'LOGMSTAR', 'LOGSFR', 'GR', 
                   'flux_detection_total', 'HALPHA_EW', 'HALPHA_FLUX', 'NII_6584_FLUX', 
                   'OIII_5007_FLUX', 'HBETA_FLUX', 'sersic_sersic_vis_radius', 
                   'sersic_sersic_vis_index', 'sersic_sersic_vis_axis_ratio',
                   'has_spiral_arms_yes', 'smoothness', 'gini', 'SPECTYPE', 'data_set_release']

CLASSIFICATION_TARGETS = ['SPECTYPE', 'data_set_release']

CSV_COLUMNS = [
    'Target', 'Task', 'Mapping', 'k_Neighbors',
    'Hit_at_1', 'Hit_at_5', 'Hit_at_10', 'MRR',
    'R2', 'RMSE', 'Bias', 'NMAD', 'Outliers',
    'Accuracy', 'Precision', 'Recall', 'F1_score', 'FPR'
]


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="AstroPT Cross-Modal Latent Retrieval Evaluator")
    
    # Data Paths
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to .fits catalog")
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing training weights and config.json")
    parser.add_argument("--emb_dir", type=str, required=True, help="Directory containing .npy embedding files")
    parser.add_argument("--save_dir", type=str, default=None, help="Plot Saving Directory (defaults to emb_dir/latent_mapper)")
    parser.add_argument("--save_name", type=str, default="mapper_results.csv", help="Custom name for the output CSV file")
    parser.add_argument("--overwrite", action="store_true", help="If set, deletes existing CSV files before saving new results.")
    
    # Mapping Configuration
    parser.add_argument("--source", type=str, default=None, help="Source modality to map FROM")
    parser.add_argument("--target", type=str, default=None, help="Target modality to map TO")
    parser.add_argument("--all_pairs", action="store_true", help="If set, evaluates all possible (source, target) pairs.")
    parser.add_argument("--k_neighbors", type=int, default=10, help="Number of neighbors for k-NN physical retrieval")
    
    return parser.parse_args()


def set_seed(seed: int = 61):
    """Fixes random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Seed set to {seed} for reproducibility.")


def check_unimodal(weights_dir: Path) -> bool:
    """Reads config.json and returns True if the model is unimodal (should skip)."""
    config_path = weights_dir / "config.json"
    if not config_path.is_file():
        logger.warning(f"config.json not found in {weights_dir}. Assuming multimodal.")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        img_train = config.get("images_train", config.get("img_train", True))
        spec_train = config.get("spectra_train", config.get("spec_train", True))
        
        if not img_train or not spec_train:
            logger.warning(f"Unimodal architecture detected: images_train={img_train}, spectra_train={spec_train}")
            logger.info("Cross-Modal Latent Retrieval requires multimodal training. Exiting cleanly.")
            return True
            
    except Exception as e:
        logger.warning(f"Failed to read config.json: {e}. Assuming multimodal.")
    
    return False


def load_data(emb_dir: Path, metadata_path: Path) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Loads embeddings (.npy) and FITS metadata."""
    logger.info(f"Scanning embeddings directory: {emb_dir}...")
    if not emb_dir.exists(): 
        raise FileNotFoundError(f"Embeddings not found: {emb_dir}")
        
    data_dict = {}
    id_candidates = ['targetid.npy', 'ids.npy', 'target_ids.npy', 'object_ids.npy']
    id_path = next((emb_dir / c for c in id_candidates if (emb_dir / c).exists()), None)
    if not id_path: 
        raise FileNotFoundError("Could not find Target IDs.")
    data_dict['targetid'] = np.load(id_path)

    excluded = set(id_candidates) | {'metadata.npy', 'indices.npy'}
    for f in emb_dir.glob("*.npy"):
        if f.name not in excluded:
            logger.info(f"Detected Modality: {f.stem} (memmap)")
            data_dict[f.stem] = np.load(f, mmap_mode='r')

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
    if 'SURVEY' in df.columns: 
        df['SURVEY'] = df['SURVEY'].str.lower().str.strip()
    return data_dict, df


def filter_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Applies strict spectral quality filtering (SNR > 3 & H-alpha)."""
    initial_len = len(df)
    mask_snr = (df['SNR_SPEC_R'] > 3.0) | (df['SNR_SPEC_Z'] > 3.0) if 'SNR_SPEC_R' in df.columns else pd.Series(True, index=df.index)
    if 'HALPHA_FLUX' in df.columns and 'HALPHA_FLUX_IVAR' in df.columns:
        halpha_err = np.where(df['HALPHA_FLUX_IVAR'] > 0, 1.0 / np.sqrt(df['HALPHA_FLUX_IVAR'].clip(lower=1e-10)), np.inf)
        mask_halpha = df['HALPHA_FLUX'] > (3.0 * halpha_err)
    else: 
        mask_halpha = pd.Series(True, index=df.index)

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
    for mod, arr in embeddings_dict.items():
        if mod == 'targetid': continue
        aligned_embeddings[mod] = arr[valid_indices]
            
    logger.info(f"Final aligned objects available for mapping: {len(matched_catalog)}")
    return matched_catalog, aligned_embeddings


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """L2-normalizes each row to unit norm, preserving hypersphere geometry."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)  # Avoid division by zero
    return X / norms


def compute_retrieval_stats(query_ids: np.ndarray, gallery_ids: np.ndarray, 
                           neighbor_indices: np.ndarray) -> Dict[str, float]:
    """
    Compute retrieval metrics: Hit@1, Hit@5, Hit@10 and MRR.
    """
    hits = {1: 0, 5: 0, 10: 0}
    mrr_sum = 0.0
    n_queries = len(query_ids)
    
    for i, query_id in enumerate(query_ids):
        # IDs of neighbors for this query
        neighbor_gallery_ids = gallery_ids[neighbor_indices[i]]
        
        # Find position of the correct match (same TARGETID)
        matches = np.where(neighbor_gallery_ids == query_id)[0]
        
        if len(matches) > 0:
            rank = matches[0] + 1
            mrr_sum += 1.0 / rank
            for k in hits:
                if rank <= k: hits[k] += 1
    
    return {
        "Hit_at_1": hits[1] / n_queries,
        "Hit_at_5": hits[5] / n_queries,
        "Hit_at_10": hits[10] / n_queries,
        "MRR": mrr_sum / n_queries
    }


def compute_cosine_at_k(query_embeddings: np.ndarray, gallery_embeddings: np.ndarray,
                         neighbor_indices: np.ndarray) -> float:
    """
    Compute mean cosine similarity between each query and its k nearest neighbors.
    
    Higher values indicate tighter cross-modal clustering — the projected embeddings
    land close to their retrieved neighbors in angular space.
    """
    cosines = []
    
    for i in range(len(query_embeddings)):
        q = query_embeddings[i]
        neighbors = gallery_embeddings[neighbor_indices[i]]
        
        # Cosine similarity: dot product of L2-normalized vectors
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        n_norms = neighbors / (np.linalg.norm(neighbors, axis=1, keepdims=True) + 1e-8)
        
        sims = np.dot(n_norms, q_norm)
        cosines.append(np.mean(sims))
    
    return float(np.mean(cosines))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str, target_name: str) -> Dict[str, float]:
    """Computes downstream metrics."""
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
    weights_dir = Path(args.weights_dir)
    metadata_path = Path(args.metadata_path)
    
    # --- UNIMODAL GUARD ---
    if check_unimodal(weights_dir):
        sys.exit(0)
    
    # Save directory
    if args.save_dir:
        save_dir = Path(args.save_dir) / "latent_mapper"
    else:
        save_dir = emb_dir / "latent_mapper"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"--- CROSS-MODAL LATENT RETRIEVAL INITIATED ---")
    logger.info(f"Retrieval Method: Zero-Shot Weighted k-NN (k={args.k_neighbors})")
    logger.info(f"Save Directory: {save_dir}")
    
    # Load & Prepare Data
    raw_data_dict, raw_df = load_data(emb_dir, metadata_path)
    filtered_df = filter_catalog(raw_df)
    aligned_df, aligned_embeddings = align_data(raw_data_dict, filtered_df)
    
    source_name = MODALITY_MAP.get(args.source, args.source)
    if source_name not in aligned_embeddings:
        logger.error(f"Source modality '{args.source}' (internal: {source_name}) not found in embeddings.")
        logger.info(f"Available internal modalities: {list(aligned_embeddings.keys())}")
        sys.exit(1)
        
    X_full = aligned_embeddings[source_name]
    # Get TARGETIDs for MRR computation
    target_col = 'TARGETID' if 'TARGETID' in aligned_df.columns else 'targetid'
    all_ids = aligned_df[target_col].values
    
    # Determine task pairs
    # Determine task pairs with modality translation
    active_modalities = sorted([m for m in aligned_embeddings.keys() if m != 'targetid'])
    
    # Translate requested source/target to actual internal names
    requested_src = MODALITY_MAP.get(args.source, args.source)
    requested_tgt = MODALITY_MAP.get(args.target, args.target)

    if args.all_pairs:
        import itertools
        task_pairs = list(itertools.permutations(active_modalities, 2))
    elif requested_src and requested_tgt:
        task_pairs = [(requested_src, requested_tgt)]
    else:
        # Default: EuclidImage -> DESISpectrum if they exist
        src = next((m for m in active_modalities if 'image' in m.lower()), active_modalities[0])
        tgt = next((m for m in active_modalities if 'spec' in m.lower() and m != src), active_modalities[1] if len(active_modalities) > 1 else src)
        task_pairs = [(src, tgt)]

    # Split Indices
    indices = np.arange(len(aligned_df))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Combined output CSV path (all pairs in the same file)
    csv_path = save_dir / args.save_name
    if args.overwrite and csv_path.exists():
        logger.info(f"Overwrite flag is SET. Deleting previous combined file: {csv_path.name}")
        csv_path.unlink()
        
    # MAIN MAPPING LOOP
    for source_mod, target_mod in task_pairs:
        if source_mod not in aligned_embeddings or target_mod not in aligned_embeddings:
            logger.warning(f"Skipping pair {source_mod}->{target_mod}: Modality missing.")
            continue

        logger.info(f"\n" + "-"*60)
        logger.info(f"ZERO-SHOT RETRIEVAL: {source_mod.upper()} ---> {target_mod.upper()}")
        logger.info("-"*60)
        
        mapping_name = f"{source_mod}->{target_mod}"
        
        
        # --- ZERO-SHOT: L2 normalize source embeddings (no training, no MLP) ---
        logger.info(f"Phase 1: L2 Normalizing {source_mod.upper()} embeddings for Zero-Shot alignment...")
        X_query = l2_normalize(np.array(aligned_embeddings[source_mod][test_idx]))
        
        # --- Fit k-NN on the true target embeddings (Gallery) ---
        logger.info(f"Phase 2: Fitting k-NN on {target_mod.upper()} Gallery (Train Set, {len(train_idx)} objects)...")
        gallery_embeddings = np.array(aligned_embeddings[target_mod][train_idx])
        gallery_norm = l2_normalize(gallery_embeddings)
        
        knn = NearestNeighbors(n_neighbors=max(10, args.k_neighbors), metric='cosine')
        knn.fit(gallery_norm)
        
        logger.info("Querying neighbors for source embeddings in target space...")
        distances, neighbor_indices = knn.kneighbors(X_query)
        
        # --- Cross-Modal Retrieval Metrics ---
        logger.info("Phase 3: Computing Retrieval Metrics (Hit@k, MRR)...")
        
        query_ids = all_ids[test_idx]
        gallery_ids = all_ids[train_idx]
        retrieval_metrics = compute_retrieval_stats(query_ids, gallery_ids, neighbor_indices)
        
        for k, v in retrieval_metrics.items():
            logger.info(f"  -> {k:10}: {v:.4f}")
        
        # --- Physical Evaluation Loop ---
        logger.info(f"Phase 4: Physical Evaluation via Retrieved Weighted {target_mod.upper()} Neighbors")
        
        for target_prop in DEFAULT_TARGETS:
            if target_prop not in aligned_df.columns: 
                continue
            
            vals = aligned_df[target_prop]
            task_type = 'classification' if target_prop in CLASSIFICATION_TARGETS else 'regression'
            
            y_true_test = vals.values[test_idx]
            y_train_gallery = vals.values[train_idx]
            
            if task_type == 'regression':
                valid_test_mask = pd.notna(y_true_test) & (y_true_test > -90)
            else:
                valid_test_mask = pd.notna(y_true_test) & (y_true_test != "") & (y_true_test != "N/A")
                
            if not valid_test_mask.any(): 
                continue
            
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
                metrics = compute_metrics(final_y_true, final_y_pred, task_type, target_prop)
                
                # Add retrieval metrics to the row
                metrics.update(retrieval_metrics)
                
                row = {
                    "Target": target_prop, 
                    "Task": task_type, 
                    "Mapping": mapping_name,
                    "k_Neighbors": args.k_neighbors
                }
                row.update(metrics)
                save_row_to_csv(row, csv_path)
                
                main_metric = f"R2: {metrics['R2']:.3f}" if task_type == 'regression' else f"Acc: {metrics['Accuracy']:.3f}"
                print(f"  -> {target_prop:25} | {main_metric}")
                
            except Exception as e:
                logger.error(f"Failed metrics for {target_prop}: {e}")

        logger.info(f"Mapping {source_mod}->{target_mod} Evaluation complete. Saved to: {csv_path.name}")

    logger.info("All specified mappings have been processed.")

if __name__ == "__main__":
    main()