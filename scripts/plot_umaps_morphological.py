"""
AstroPT Morphological UMAP Plotter (Enhanced Visualization).

Combines UMAP projection with high-quality galaxy thumbnails.
Uses custom channel weighting and Lupton scaling to match individual plots.

Usage:
    python scripts/plot_umaps_morphological.py --emb_dir ... --metadata_path ... --data_dir ...
"""

import argparse
import logging
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import umap
from typing import Tuple, Dict
from astropy.table import Table
from astropy import units as u

# Imports de tu librería
from astropt.euclid_desi_arrow_dataloader import EuclidDESIDatasetArrow

# Logger
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("Morph-UMAP")

# Plotting Style
plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}\usepackage{bm}\usepackage{amsmath}\sisetup{detect-family, separate-uncertainty=true, output-decimal-marker={.}}'
plt.rc('text', usetex=True)
plt.rc('font', family='serif', weight='bold') 

plt.rcParams.update({
    'axes.grid': False,
    'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
    'figure.titlesize': 20, 'figure.titleweight': 'bold',
})

# Mapeo de colores por Survey
DATASET_ORDER = ['main', 'sv3', 'sv1', 'special']
DATASET_COLORS = {'main': '#1f77b4', 'sv3': '#ff7f0e', 'sv1': '#2ca02c', 'special': '#d62728'}

# --- LÓGICA DE VISUALIZACIÓN AVANZADA ---

def make_rgb_lupton(rgb_array: np.ndarray, Q: float = 12.0, stretch: float = 0.5, m: float = 0.0) -> np.ndarray:
    """
    Aplica escalado Lupton (Asinh) a una imagen RGB de 3 canales ya montada.
    Args:
        rgb_array: (3, H, W) float array.
    """
    I = np.mean(rgb_array, axis=0) - m
    I = np.maximum(I, 1e-10)
    
    f_I = np.arcsinh(Q * stretch * I) / Q
    scale_factor = f_I / I
    
    rgb_out = rgb_array * scale_factor[np.newaxis, :, :]
    
    # Normalización final al P99.5 para evitar saturación
    max_rgb = np.percentile(rgb_out, 99.5)
    if max_rgb > 0:
        rgb_out = rgb_out / max_rgb
        
    return np.clip(rgb_out, 0, 1).transpose(1, 2, 0)

def process_raw_to_rgb(vis, h, j, y):
    """
    Convierte bandas raw (VIS, H, J, Y) en un RGB estético usando la lógica
    de ponderación y limpieza de fondo definida por el usuario.
    """
    # 1. Stack inicial
    raw_stack = np.stack([vis, h, j, y], axis=0)

    # 2. Background Subtraction (Mediana espacial)
    bg_val = np.percentile(raw_stack, 50, axis=(1,2), keepdims=True)
    raw_bg = raw_stack - bg_val

    # 3. Normalización por Canal (Clipping)
    raw_norm_list = []
    for c in range(4):
        # Calcular máximo local (P99.5)
        v_max = np.percentile(np.abs(raw_bg[c]), 99.5)
        if v_max <= 0: v_max = 1.0
        
        # Clip y normalizar
        norm_ch = np.clip(raw_bg[c] / v_max, 0, 100)
        raw_norm_list.append(norm_ch)
    
    raw_norm = np.stack(raw_norm_list)

    # 4. Composición de Canales y Pesos
    # Weights [R, G, B] -> [1.2, 1.3, 1.0]
    RGB_WEIGHTS = [1.2, 1.3, 1.0]

    # R = H
    r = raw_norm[1] * RGB_WEIGHTS[0]
    # G = (J + Y) / 2
    g = ((raw_norm[2] + raw_norm[3]) / 2.0) * RGB_WEIGHTS[1]
    # B = VIS
    b = raw_norm[0] * RGB_WEIGHTS[2]

    rgb_input = np.stack([r, g, b], axis=0)

    # 5. Lupton Scaling (Q=12, stretch=0.5)
    return make_rgb_lupton(rgb_input, Q=12.0, stretch=0.5)


# --- LÓGICA DE CARGA DE DATOS (CON ASTROPY FIX) ---
def load_data(emb_dir: str, metadata_path: str) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """
    Carga embeddings .npy y metadatos .fits.
    """
    logger.info(f"Scanning embeddings directory: {emb_dir}...")
    if not os.path.exists(emb_dir):
        raise FileNotFoundError(f"Embeddings directory not found: {emb_dir}")
    
    data_dict = {}
    
    # Load Target IDs
    id_candidates = ['targetid.npy', 'ids.npy', 'target_ids.npy', 'object_ids.npy']
    id_path = None
    for cand in id_candidates:
        p = os.path.join(emb_dir, cand)
        if os.path.exists(p):
            id_path = p
            break
            
    if id_path:
        logger.info(f"Loading Target IDs from {os.path.basename(id_path)}...")
        data_dict['targetid'] = np.load(id_path)
    else:
        raise FileNotFoundError(f"Could not find IDs ({id_candidates}) in {emb_dir}")

    # Load Modalities
    modalities = ['images', 'spectra', 'joint']
    for mod in modalities:
        candidates = glob.glob(os.path.join(emb_dir, f"{mod}*.npy"))
        if candidates:
            best_cand = candidates[0] 
            for c in candidates:
                if os.path.basename(c) == f"{mod}.npy":
                    best_cand = c
                    break
            logger.info(f"Loading {mod} embeddings from {os.path.basename(best_cand)}...")
            data_dict[mod] = np.load(best_cand)
        else:
            logger.warning(f"No embeddings found for modality: {mod}")
            
    catalog = Table.read(metadata_path)
    df = catalog.to_pandas()
    
    # String cleaning
    for col in df.columns:
        if df[col].dtype == object and isinstance(df[col].iloc[0], bytes):
            try: df[col] = df[col].str.decode('utf-8')
            except: pass
            
    if 'SURVEY' in df.columns:
        df['SURVEY'] = df['SURVEY'].str.lower().str.strip()
        
    return data_dict, df

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AstroPT Morphological UMAP")
    parser.add_argument("--emb_dir", type=str, required=True, help="Directory containing .npy files")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to .fits catalog")
    parser.add_argument("--data_dir", type=str, required=True, help="Arrow data root directory")
    
    parser.add_argument("--modality", type=str, default="images", choices=['images', 'spectra', 'joint'])
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--metric", type=str, default='cosine')
    
    parser.add_argument("--out_dir", type=str, default="plots_umap")
    parser.add_argument("--title_suffix", type=str, default="")
    parser.add_argument("--grid_size", type=int, default=100, help="Grid density (e.g. 30x30 images)")
    parser.add_argument("--image_zoom", type=float, default=0.12, help="Thumbnail zoom level")
    
    return parser.parse_args()

def compute_umap(embeddings, n_neighbors, min_dist, metric):
    logger.info(f"Computing UMAP ({n_neighbors}, {min_dist}, {metric})...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, n_components=2, random_state=42, n_jobs=-1)
    return reducer.fit_transform(embeddings)

class DummyRegistry:
    def get_config(self, name): return None

def plot_with_images(coords, df, data_dir, grid_size, zoom, out_dir, suffix):
    # 1. Setup Arrow Dataset
    logger.info("Opening Arrow dataset...")
    ds = EuclidDESIDatasetArrow(arrow_folder_root=data_dir, split='test', modality_registry=DummyRegistry(), transform=None)
    
    logger.info("Building ID map...")
    all_arrow_ids = np.array(ds.ds['targetid'])
    id_to_idx = {tid: i for i, tid in enumerate(all_arrow_ids)}

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(20, 20))
    
    # Sorting for Z-Order based on Survey
    if 'SURVEY' in df.columns:
        df['z_order'] = df['SURVEY'].map({d: i for i, d in enumerate(reversed(DATASET_ORDER))}).fillna(0)
    else:
        df['z_order'] = 0
    df = df.sort_values('z_order')
    coords_sorted = coords[df.index]
    
    # Base Scatter Plot
    if 'SURVEY' in df.columns:
        for ds_name in DATASET_ORDER:
            mask = df['SURVEY'] == ds_name
            if mask.any():
                c = DATASET_COLORS.get(ds_name, 'gray')
                ax.scatter(coords_sorted[mask, 0], coords_sorted[mask, 1], c=c, label=ds_name.upper(), s=5, alpha=0.2, edgecolors='none')
    else:
        ax.scatter(coords_sorted[:, 0], coords_sorted[:, 1], c='blue', s=5, alpha=0.2)

    # 3. Image Grid Placement
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    margin = 0.05 * (x_max - x_min)
    
    x_grid = np.linspace(x_min + margin, x_max - margin, grid_size)
    y_grid = np.linspace(y_min + margin, y_max - margin, grid_size)
    
    logger.info(f"Placing images on {grid_size}x{grid_size} grid...")
    images_placed = 0
    
    for gx in x_grid:
        for gy in y_grid:
            # Nearest Neighbor Search
            dists = np.sum((coords - np.array([gx, gy]))**2, axis=1)
            nearest_idx = np.argmin(dists)
            
            # Retrieve Data
            galaxy_row = df.iloc[nearest_idx]
            target_col = 'TARGETID' if 'TARGETID' in df.columns else 'targetid'
            target_id = galaxy_row[target_col]
            real_x, real_y = coords[nearest_idx]
            
            if target_id not in id_to_idx: continue
            
            arrow_idx = id_to_idx[target_id]
            
            try:
                rec = ds.ds[arrow_idx]
                vis = np.array(rec['image_vis'], dtype=np.float32)
                h = np.array(rec['image_nisp_h'], dtype=np.float32)
                j = np.array(rec['image_nisp_j'], dtype=np.float32)
                y_band = np.array(rec['image_nisp_y'], dtype=np.float32)
                
                # Check empty
                if vis.size == 0 or h.size == 0: continue

                # --- APPLY NEW COLOR LOGIC HERE ---
                rgb_img = process_raw_to_rgb(vis, h, j, y_band)
                
                imagebox = OffsetImage(rgb_img, zoom=zoom)
                ab = AnnotationBbox(imagebox, (real_x, real_y), frameon=False, pad=0.0)
                ax.add_artist(ab)
                images_placed += 1
                
            except Exception as e:
                # logger.debug(f"Failed {target_id}: {e}")
                continue

    logger.info(f"Successfully placed {images_placed} images.")

    # Final Polish
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, markerscale=3)
    ax.set_xticks([]); ax.set_yticks([])
    
    title = r"\textbf{AstroPT Morphological UMAP}"
    if suffix: title += f"\n[{suffix}]"
    ax.set_title(title)
    
    os.makedirs(out_dir, exist_ok=True)
    fname = f"umap_morphological_{suffix.replace(' ', '_')}.png" if suffix else "umap_morphological.png"
    out_path = os.path.join(out_dir, fname)
    logger.info(f"Saving high-res plot to {out_path}...")
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    
    # 1. Cargar Datos
    data_dict, df = load_data(args.emb_dir, args.metadata_path)
    
    # 2. Alinear
    if args.modality not in data_dict:
        raise ValueError(f"Modality '{args.modality}' not found.")
        
    embeddings = data_dict[args.modality]
    ids = data_dict['targetid']
    
    logger.info(f"Aligning {len(embeddings)} embeddings...")
    target_col = 'TARGETID' if 'TARGETID' in df.columns else 'targetid'
    
    df = df[df[target_col].isin(ids)]
    df = df.set_index(target_col)
    df = df.reindex(ids)
    df = df.reset_index()
    
    if len(df) != len(embeddings):
        logger.warning(f"Mismatch: Emb {len(embeddings)} vs DF {len(df)}. Pruning...")
        mask = ~df[target_col].isna()
        df = df[mask]
        embeddings = embeddings[mask]

    # 3. UMAP
    umap_coords = compute_umap(embeddings, args.n_neighbors, args.min_dist, args.metric)
    
    # 4. Plot
    plot_with_images(
        umap_coords, df, args.data_dir, 
        args.grid_size, args.image_zoom, 
        args.out_dir, args.title_suffix
    )

if __name__ == "__main__":
    main()