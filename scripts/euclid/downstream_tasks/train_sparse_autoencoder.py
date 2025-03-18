import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from astropy.io import fits
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import umap

#############################################################################
# 1) CUSTOM TOP-K ACTIVATION
#############################################################################

class TopKActivation(nn.Module):
    """
    A custom activation that keeps only the top-K elements (by value)
    and zeros out the rest. This forces exactly K neurons to be active
    (if K < dimension). If K >= dimension, effectively it's no-op.
    """
    def __init__(self, k: int):
        """
        Parameters
        ----------
        k : int
            Number of neurons to keep 'alive' for each sample.
        """
        super().__init__()
        self.k = k

    def forward(self, x):
        """
        x: shape (batch_size, latent_dim)
        """
        # If k >= latent_dim, no need to zero out anything:
        if self.k >= x.shape[1]:
            return x

        # topk returns (values, indices)
        # We do it along dimension=1 (each row separately).
        top_vals, top_idx = torch.topk(x, k=self.k, dim=1)

        # We'll construct a mask of zeros, except at the top-k indices
        mask = torch.zeros_like(x)
        # put the top values in those positions
        # But to do that easily, we use scatter
        mask.scatter_(1, top_idx, top_vals)

        return mask

#############################################################################
# 2) SPARSE AUTOENCODER WITH TOP-K BOTTLENECK
#############################################################################

class SparseAutoencoderTopK(nn.Module):
    """
    Autoencoder with a top-K activation in the latent layer
    to enforce a maximum number of active neurons.
    """

    def __init__(self, 
                 input_dim: int,
                 latent_dim: int,
                 k_top: int,
                 hidden_dims=None):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of the input embeddings.
        latent_dim : int
            Dimension of the latent (bottleneck) representation.
            Typically >= input_dim if you want an overcomplete representation.
        k_top : int
            How many latent neurons are allowed to remain "active" (top-K).
        hidden_dims : list of ints
            Sizes of hidden layers in encoder/decoder (besides the bottleneck).
            If None or empty, we do a single linear transform to latent,
            then from latent back to input.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = []

        self.latent_dim = latent_dim
        self.k_top = k_top

        # ------ Encoder ------
        # build a feed-forward stack: input -> hidden_dims -> linear to latent
        encoder_layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hd))
            encoder_layers.append(nn.ReLU(inplace=True))
            prev_dim = hd
        # final linear to latent dimension
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        # We do NOT apply ReLU here, because we want the possibility
        # for negative or positive values before top-K.
        self.encoder_dense = nn.Sequential(*encoder_layers)

        # This is our custom top-K activation
        self.topk_activation = TopKActivation(k=k_top)

        # ------ Decoder ------
        # mirror the hidden dims in reverse
        decoder_layers = []
        rev_hidden_dims = hidden_dims[::-1]
        prev_dim = latent_dim
        for hd in rev_hidden_dims:
            decoder_layers.append(nn.Linear(prev_dim, hd))
            decoder_layers.append(nn.ReLU(inplace=True))
            prev_dim = hd

        # final linear to reconstruct input
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """Forward pass up to the top-K activation (latent)."""
        x_dense = self.encoder_dense(x)
        x_topk = self.topk_activation(x_dense)
        return x_topk

    def decode(self, z):
        """Decoder."""
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

#############################################################################
# 3) TRAINING FUNCTION
#############################################################################

def train_sparse_autoencoder_topk(
        zss_subset,
        input_dim,
        latent_dim=128,
        k_top=10,
        hidden_dims=None,
        num_epochs=20,
        batch_size=128,
        lr=1e-3,
        loss='mse',
        device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train an autoencoder with top-K sparsity on the embeddings.

    Parameters
    ----------
    zss_subset : np.ndarray, shape (N, D)
        Embeddings from your anomaly code.
    input_dim : int
        Dimension of embeddings (D).
    latent_dim : int
        Dimension of overcomplete latent representation (>= D recommended).
    k_top : int
        Number of neurons to keep in the top-K activation. 
    hidden_dims : list of int
        Hidden dims for encoder/decoder.
    num_epochs : int
        Training epochs.
    batch_size : int
        Batch size.
    lr : float
        Learning rate.
    device : str
        'cuda' or 'cpu'.

    Returns
    -------
    model : SparseAutoencoderTopK
        The trained model.
    """
    dataset = TensorDataset(torch.from_numpy(zss_subset).float())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SparseAutoencoderTopK(
        input_dim=input_dim,
        latent_dim=latent_dim,
        k_top=k_top,
        hidden_dims=hidden_dims
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if loss == 'mse' :
        criterion = nn.MSELoss()  
    elif loss == 'cosine' :
        def cosine_similarity_loss(x_recon, x_batch):
            x_recon_norm = x_recon / x_recon.norm(dim=1, keepdim=True)
            x_batch_norm = x_batch / x_batch.norm(dim=1, keepdim=True)
            return 1 - torch.mean(torch.sum(x_recon_norm * x_batch_norm, dim=1))

        criterion = cosine_similarity_loss

    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            x_batch = batch[0].to(device)
            optimizer.zero_grad()

            x_recon, _ = model(x_batch)
            loss = criterion(x_recon, x_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Recon Loss: {avg_loss:.4f}")

    return model

#############################################################################
# 4) VISUALIZATION: MOST-ACTIVATED GALAXIES FOR EACH LATENT NEURON
#############################################################################


def asinh_stretch(img, low_cut=1, high_cut=99, asinh_scale=1.0):
    """
    1) Clip image between the low_cut and high_cut percentiles.
    2) Apply arcsinh with a chosen scale factor (asinh_scale).
    3) Normalize result to [0..1].

    Similar to your 'similarity_search' code.
    """
    valid_pixels = img[np.isfinite(img)]
    if valid_pixels.size == 0:
        return np.zeros_like(img)

    vmin = np.percentile(valid_pixels, low_cut)
    vmax = np.percentile(valid_pixels, high_cut)
    clipped = np.clip(img, vmin, vmax)

    arcs = np.arcsinh(asinh_scale * clipped)
    arcs_min, arcs_max = arcs.min(), arcs.max()
    if arcs_max > arcs_min:
        stretched = (arcs - arcs_min) / (arcs_max - arcs_min)
    else:
        stretched = np.zeros_like(arcs)

    return stretched

def visualize_latent_neurons_topk_pdf(
    model,
    zss_subset,
    galaxy_ids,
    image_dir,
    pdf_filename,
    filter_names=["VIS","NIR-H","NIR-J","NIR-Y"],
    top_n=5,
    device='cuda',
    top_var_dims=None,
):
    """
    1) Encodes all embeddings -> gets latent activations z (shape: [N, latent_dim]).
    2) (Optional) Sorts dimensions by variance and keeps only top 'top_var_dims'.
    3) For each selected dimension, find top-N galaxies with highest activation.
    4) Create one PDF page with rows=top_n, columns=len(filter_names)+1 (the last column is RGB).
    5) Skip if all top-N activations in a dimension are zero.

    Parameters
    ----------
    model : nn.Module
        The trained sparse autoencoder with top-K layer.
    zss_subset : (N, D) np.ndarray
        Embedding data for N galaxies.
    galaxy_ids : (N,) np.ndarray
        The IDs of each galaxy. Must be aligned with zss_subset.
    image_dir : str
        Directory with <object_id>.fits files.
    pdf_filename : str
        Path to the output PDF file.
    filter_names : list of str
        Filter names for labeling or for forming the RGB composite.
    top_n : int
        Number of top galaxies to show per dimension.
    device : str
        'cuda', 'mps', or 'cpu'.
    top_var_dims : int or None
        If not None, we only visualize the top 'top_var_dims' latent dimensions
        by variance. E.g., top_var_dims=20 will select the 20 dims with highest variance.
        If None, we visualize all dimensions.
    """
    model.eval()

    # -------------------- 1) Pass data through the autoencoder to get z --------------------
    import torch
    X_torch = torch.from_numpy(zss_subset).float().to(device)
    with torch.no_grad():
        _, z_torch = model(X_torch)
    z = z_torch.cpu().numpy()  # shape: (N, latent_dim)
    N, latent_dim = z.shape
    print(f"Latent code shape = {z.shape}")

    # -------------------- 2) Sort dimensions by variance (optional) --------------------
    var_dim = np.var(z, axis=0)             # shape (latent_dim,)
    idx_sorted_by_var = np.argsort(var_dim)[::-1]  # descending order of variance

    if (top_var_dims is not None) and (top_var_dims < latent_dim):
        idx_sorted_by_var = idx_sorted_by_var[:top_var_dims]
        print(f"Selecting top {top_var_dims} dims by variance out of {latent_dim} total.")
    else:
        print(f"Visualizing ALL {latent_dim} dims (no variance filtering).")

    # For debugging: 
    # let's see how many dims truly have nonzero variance
    print("Dimension variances (descending):")
    for i, d in enumerate(idx_sorted_by_var):
        print(f"  Dim={d}, variance={var_dim[d]:.4f}")

    # -------------------- 3) Create multi-page PDF --------------------
    pdf = PdfPages(pdf_filename)
    print(f"Will store pages in: {pdf_filename}")

    # Loop over dimensions in order of descending variance
    for dim_idx in idx_sorted_by_var:
        activations = z[:, dim_idx]
        # Sort descending by activation
        top_indices = np.argsort(activations)[::-1][:top_n]
        top_vals = activations[top_indices]

        # Skip if all top_n activations are ~0
        if np.allclose(top_vals, 0, atol=1e-7):
            print(f"Dim {dim_idx}: top-{top_n} are all zero, skipping.")
            continue

        # (A) Setup figure: top_n rows, each row => (len(filter_names)+1) columns
        ncols = len(filter_names) + 1
        fig, axes = plt.subplots(top_n, ncols, figsize=(3.5*ncols, 3.5*top_n))
        if top_n == 1:
            axes = np.array([axes])  # shape => (1, ncols)
        fig.suptitle(
            f"Latent Dim {dim_idx} - Var={var_dim[dim_idx]:.4f} - Top {top_n} Activations",
            fontsize=14
        )

        # (B) For each top galaxy, fill one row
        for row_idx, gal_idx in enumerate(top_indices):
            gal_id = galaxy_ids[gal_idx]
            act_val = activations[gal_idx]

            row_axes = axes[row_idx, :] if top_n > 1 else axes[0, :]
            # Label on the left-most subplot
            row_axes[0].set_ylabel(f"ID={gal_id}\nAct={act_val:.3f}",
                                   fontsize=10, rotation=0, labelpad=80)

            fits_file = os.path.join(image_dir, f"{gal_id}.fits")
            if not os.path.isfile(fits_file):
                # If no FITS, mark row
                for ax_ in row_axes:
                    ax_.axis('off')
                row_axes[0].text(0.5, 0.5, f"No file: {gal_id}",
                                 ha='center', va='center', fontsize=8)
                continue

            with fits.open(fits_file) as hdul:
                data_3d = hdul[0].data  # shape: (n_filters, H, W) ?

            if data_3d is None or len(data_3d.shape) < 2:
                # Invalid data
                for ax_ in row_axes:
                    ax_.axis('off')
                row_axes[0].text(0.5, 0.5, "Invalid FITS shape",
                                 ha='center', va='center', fontsize=8)
                continue

            # (C) Display each filter
            for cidx in range(len(filter_names)):
                ax_ = row_axes[cidx]
                ax_.axis('off')

                if data_3d.ndim == 2:
                    # Only one filter
                    channel = data_3d
                elif cidx < data_3d.shape[0]:
                    channel = data_3d[cidx]
                else:
                    ax_.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=8)
                    continue

                stretched = asinh_stretch(
                    channel, low_cut=1, high_cut=99, asinh_scale=1.0
                )
                ax_.imshow(stretched, origin='lower', cmap='gray', vmin=0, vmax=1)
                ax_.set_title(filter_names[cidx], fontsize=8)

            # (D) RGB composite in last column
            ax_rgb = row_axes[-1]
            ax_rgb.axis('off')
            if data_3d.ndim == 3 and data_3d.shape[0] >= 3:
                # Example index selection
                # If 4 filters => R=3, G=2, B=1
                # If 3 filters => R=2, G=1, B=0
                if data_3d.shape[0] == 3:
                    cR, cG, cB = 2, 1, 0
                else:
                    cR, cG, cB = 3, 2, 1

                R_ = asinh_stretch(data_3d[cR], 1, 99, asinh_scale=0.5)
                G_ = asinh_stretch(data_3d[cG], 1, 99, asinh_scale=0.5)
                B_ = asinh_stretch(data_3d[cB], 1, 99, asinh_scale=0.5)

                # Make an (H, W, 3)
                H, W = R_.shape
                rgb = np.zeros((H, W, 3), dtype=np.float32)
                rgb[..., 0] = R_
                rgb[..., 1] = G_
                rgb[..., 2] = B_

                ax_rgb.imshow(rgb, origin='lower')
                ax_rgb.set_title("RGB", fontsize=8)
            else:
                ax_rgb.text(0.5, 0.5, "No RGB", ha='center', va='center', fontsize=8)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()
    print(f"Saved PDF => {pdf_filename}")

    return z


def MAD(arr, median,axis=None):
        """
        Compute the Median Absolute Deviation (MAD) of an array along a given axis.
        
        Parameters
        ----------
        arr : np.ndarray
            Input array.
        axis : int or None, optional
            Axis along which the MAD is computed. The default is to compute the MAD of the flattened array.
        
        Returns
        -------
        mad : np.ndarray
            The MAD of the array.
        """
        mad = np.median(np.abs(arr - median), axis=axis, keepdims=True)
        return mad
    
def flux_to_mag(flux, zeropoint=23.9):
    return zeropoint - 2.5 * np.log10(flux)


if __name__ == "__main__":


    folder_datacubes = '/folder_datacubes/'
    filter_names = ["VIS", "NIR-H", "NIR-J", "NIR-Y"]

    # Load the catalog data

    folder_pdf = '/folder_pdf/'

    catalog_path = "/catalogue_path/EuclidMorphPhysPropSpecZ.fits"
    with fits.open(catalog_path) as hdul:
        catalog_data = hdul[1].data

    # Extract catalog properties
    object_ids = catalog_data['object_id']
    catalog_file_names = catalog_data['VIS_name']
    print("Number of galaxies in the catalog:", len(catalog_file_names))


    parameters = {
        "object_id" : catalog_data['object_id'],
        "sersic_index": catalog_data['sersic_sersic_vis_index'],
        "H_mag": flux_to_mag(catalog_data['flux_h_sersic']),
        "smooth": catalog_data['smooth_or_featured_smooth'],
        "segmentation_area": catalog_data['segmentation_area'],
        "flux_vis_sersic_mag": flux_to_mag(catalog_data['flux_vis_sersic']),
        "ellipticity": catalog_data['ellipticity'],
        "kron_radius": catalog_data['kron_radius'],
        "flux_detection_total_mag": flux_to_mag(catalog_data['flux_detection_total']),
        "flux_vis_4fwhm_aper_mag": flux_to_mag(catalog_data['flux_vis_4fwhm_aper']),
        "spurious_flag": catalog_data['spurious_flag'],
        "det_quality_flag": catalog_data['det_quality_flag'],
        "mumax_minus_mag": catalog_data['mumax_minus_mag'],
        "featured_or_disk": catalog_data['smooth_or_featured_featured_or_disk'],
        "artifact_star_zoom": catalog_data['smooth_or_featured_artifact_star_zoom'],
        "disk_edge_on_yes": catalog_data['disk_edge_on_yes'],
        "disk_edge_on_no": catalog_data['disk_edge_on_no'],
        "bar_strong": catalog_data['bar_strong'],
        "bar_weak": catalog_data['bar_weak'],
        "bar_no": catalog_data['bar_no'],
        "sersic_radius": catalog_data['sersic_sersic_vis_radius'],
        "sersic_axis_ratio": catalog_data['sersic_sersic_vis_axis_ratio'],
        "median_redshift": catalog_data['phz_pp_median_redshift'],
        "median_stellarmass": catalog_data['phz_pp_median_stellarmass'],
        "phys_param_flags": catalog_data['phys_param_flags'],
        "phz_flags": catalog_data['phz_flags'],
        "phz_median": catalog_data['phz_median'],
        "phz_mode_1": catalog_data['phz_mode_1'],
        "phz_mode_2": catalog_data['phz_mode_2'],
        "merging_major_disturbance" : catalog_data['merging_major_disturbance'],
        "merging_merger" : catalog_data['merging_merger'],
        "merging_minor_disturbance" : catalog_data['merging_minor_disturbance'],

    }





    folder_astropt =   "pathtofolder/astropt090M/"




    # Load the test file with galaxy names
    test_file_path = folder_astropt + "train.txt"
    with open(test_file_path, "r") as f:
        test_files = [line.strip() for line in f.readlines()]
    print("Number of galaxies in train.txt", len(test_files))



    # Load embeddings and indices
    zss = np.load(folder_astropt + "zss_64t_mean_train.npy")
    idxs = np.load(folder_astropt + "idxs_64t_mean_train.npy")

    print("Embeddings shape:", zss.shape)
    print("Indices shape:", idxs.shape)

    # Map test file names to catalog indices
    test_to_catalog_idx = {test_name: i for i, test_name in enumerate(catalog_file_names)}

    # Map embeddings to corresponding parameters
    mapped_properties = {key: [] for key in parameters.keys()}

    for test_name in [test_files[idx] for idx in idxs]:
        catalog_idx = test_to_catalog_idx.get(test_name)
        if catalog_idx is not None:
            for key, array in parameters.items():
                mapped_properties[key].append(array[catalog_idx])
        else:
            # If not found, append NaN or skip
            for key in parameters.keys():
                mapped_properties[key].append(np.nan)

    # Convert properties to NumPy arrays
    for key in mapped_properties:
        mapped_properties[key] = np.array(mapped_properties[key])




    # Read from datacubes folder files name which correspond to the ID.fits
    datacubes_files = os.listdir(folder_datacubes)
    datacubes_ids = [int(f.split('.')[0]) for f in datacubes_files if f.endswith('.fits')]

    datacubes_ids = datacubes_ids
    # Select those objects from zss and mapped_properties
    datacubes_ids_set = set(datacubes_ids)
    selected_indices = [i for i, obj_id in enumerate(mapped_properties['object_id']) if obj_id in datacubes_ids_set]

    zss_subset = zss[selected_indices, :]
    mapped_properties_subset = {key: value[selected_indices] for key, value in mapped_properties.items()}





    umap_projection_file = os.path.join(folder_astropt, "umap_projection.npy")

    if os.path.exists(umap_projection_file):
        print('Loading existing UMAP projection')
        zss_2d_subset = np.load(umap_projection_file)
    else:
        print('Computing UMAP')
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        zss_2d_subset = umap_model.fit_transform(zss_subset)
        np.save(umap_projection_file, zss_2d_subset)
        print('UMAP projection finished and saved')


    det_quality_flags = (mapped_properties_subset['det_quality_flag'] == 0)

    # Apply det_quality_flags condition to mapped_properties_subset and zss_2d_subset
    zss_subset = zss_subset[det_quality_flags]
    zss_2d_subset = zss_2d_subset[det_quality_flags]
    mapped_properties_subset = {key: value[det_quality_flags] for key, value in mapped_properties_subset.items()}


    median = np.median(zss_subset, axis=0, keepdims=True)
    mad  = MAD(zss_subset, median, axis=0) + 1e-7
    zss_subset = (zss_subset - median) / mad



    folder_model = '/path/to/model/folder/'
    file_model = 'sparse_autoencoder_model.pth'
    model_path = os.path.join(folder_model, file_model)

    if os.path.exists(model_path):
        print('Loading existing model')
        model = torch.load(model_path)
    else:
        print('Training new model')
        model = train_sparse_autoencoder_topk(
            zss_subset=zss_subset,
            input_dim=zss_subset.shape[1],
            latent_dim=2*zss_subset.shape[1],
            k_top=10,
            hidden_dims=None,
            num_epochs=30,
            batch_size=1000,
            lr=1e-3,
            loss='mse',
            device='mps'
        )
        os.makedirs(folder_model, exist_ok=True)
        torch.save(model, model_path)
        print('Model trained and saved')




    pdf_filename = os.path.join(folder_pdf, "latent_top10_visualization.pdf")
    latent_codes = visualize_latent_neurons_topk_pdf(
        model=model,
        zss_subset=zss_subset,
        galaxy_ids=mapped_properties_subset['object_id'],
        zss_2d_subset = zss_2d_subset,
        image_dir=folder_datacubes,
        pdf_filename=pdf_filename,  # <--- store all pages here
        filter_names=["VIS"],  
        top_n=5,
        device='mps',
        top_var_dims=30

    )



def visualize_latent_neurons_topk_pdf(
    model,
    zss_subset,
    galaxy_ids,
    zss_2d_subset,     # <--- UMAP 2D projection for each galaxy
    image_dir,
    pdf_filename,
    filter_names=["VIS","NIR-H","NIR-J","NIR-Y"],
    top_n=5,
    device='cuda',
    top_var_dims=None,
):
    """
    Creates a multi-page PDF. For each selected latent dimension (based on variance),
    we do TWO pages:
      - Page A: top-N galaxy images in a grid (rows=top_n, cols=len(filter_names)+1).
      - Page B: UMAP scatter with all points in gray, top-N galaxies in red.

    We skip any dimension whose top-N activations are all ~0.

    Parameters
    ----------
    model : nn.Module
        Trained sparse autoencoder with top-K layer.
    zss_subset : (N, D) np.ndarray
        Embedding data for N galaxies (the same used for training).
    galaxy_ids : (N,) np.ndarray
        IDs for each galaxy, same indexing as zss_subset.
    zss_2d_subset : (N, 2) np.ndarray
        UMAP 2D coordinates for each galaxy, aligned with zss_subset.
    image_dir : str
        Directory containing <galaxy_id>.fits files.
    pdf_filename : str
        Output PDF file path.
    filter_names : list of str
        Names for each filter, used in the image mosaic.
    top_n : int
        Number of top galaxies to show per dimension.
    device : str
        'cuda', 'mps', or 'cpu'.
    top_var_dims : int or None
        If not None, only visualize the top 'top_var_dims' dims by variance.
        If None, use all dims.
    """
    model.eval()

    # 1) Encode data to get latent codes
    import torch
    X_torch = torch.from_numpy(zss_subset).float().to(device)
    with torch.no_grad():
        _, z_torch = model(X_torch)
    z = z_torch.cpu().numpy()  # shape: (N, latent_dim)
    N, latent_dim = z.shape
    print(f"Latent code shape = {z.shape}")

    # 2) Sort dimensions by descending variance if requested
    var_dim = np.var(z, axis=0)                  # shape (latent_dim,)
    idx_sorted_by_var = np.argsort(var_dim)[::-1]  # descending

    if (top_var_dims is not None) and (top_var_dims < latent_dim):
        idx_sorted_by_var = idx_sorted_by_var[:top_var_dims]
        print(f"Selecting top {top_var_dims} dims by variance from {latent_dim}.")
    else:
        print(f"Visualizing all {latent_dim} dims (no variance limit).")

    # For debugging, you can see the top dims:
    print("Latent dimension variances (descending) of selected dims:")
    for i, d in enumerate(idx_sorted_by_var):
        print(f"  Dim={d}, var={var_dim[d]:.4f}")

    # 3) Create the PDF
    pdf = PdfPages(pdf_filename)
    print(f"Creating PDF => {pdf_filename}")

    # 4) For each dimension in order of variance
    for dim_idx in idx_sorted_by_var:
        activations = z[:, dim_idx]
        # sort descending
        top_indices = np.argsort(activations)[::-1][:top_n]
        top_vals = activations[top_indices]

        # Skip if top-N are all ~0
        if np.allclose(top_vals, 0, atol=1e-7):
            print(f"Dim {dim_idx}: top-{top_n} are all zero => skipping.")
            continue

        # ========== PAGE A: Image grid for top-N galaxies ==========
        fig, axes = plt.subplots(top_n, len(filter_names),
                                 figsize=(3.5*(len(filter_names)), 3.5*top_n))
        if top_n == 1:
            axes = np.array([axes])  # shape => (1, ncols)

        dim_title = f"Dimension {dim_idx}, Var={var_dim[dim_idx]:.3f}"
        fig.suptitle(f"{dim_title} - Top {top_n} Activations", fontsize=14)

        for row_idx, gal_idx in enumerate(top_indices):
            gal_id = galaxy_ids[gal_idx]
            act_val = activations[gal_idx]

            try :
                row_axes = axes[row_idx, :] if top_n > 1 else axes[0, :]
                row_axes[0].set_ylabel(f"ID={gal_id}\nAct={act_val:.2f}",
                                   fontsize=8, rotation=0, labelpad=60)
            except :
                row_axes = axes[row_idx] if top_n > 1 else axes[0, :]
                row_axes.set_ylabel(f"ID={gal_id}\nAct={act_val:.2f}",
                                   fontsize=8, rotation=0, labelpad=60)


            fits_file = os.path.join(image_dir, f"{gal_id}.fits")
            if not os.path.isfile(fits_file):
                # Mark row as missing
                for ax_ in row_axes:
                    ax_.axis('off')
                row_axes[0].text(0.5, 0.5, f"No FITS: {gal_id}",
                                 ha='center', va='center', fontsize=8)
                continue

            with fits.open(fits_file) as hdul:
                data_3d = hdul[0].data  # shape (n_filters, H, W) or (H, W)

            # If invalid shape, skip
            if data_3d is None or len(data_3d.shape) < 2:
                for ax_ in row_axes:
                    ax_.axis('off')
                row_axes[0].text(0.5, 0.5, "Invalid shape",
                                 ha='center', va='center', fontsize=8)
                continue

            # Show each filter in columns 0..(len(filter_names)-1)
            for cidx, fname in enumerate(filter_names):
                if len(filter_names) > 1 :
                    ax_ = row_axes[cidx]
                else :
                    ax_ = row_axes

                ax_.axis('off')
                if data_3d.ndim == 2:
                    # Single filter
                    channel = data_3d
                elif cidx < data_3d.shape[0]:
                    channel = data_3d[cidx]
                else:
                    ax_.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=8)
                    continue

                stretched = asinh_stretch(channel, low_cut=0.1, high_cut=99, asinh_scale=1.0)
                ax_.imshow(stretched, origin='lower', cmap='gray', vmin=0, vmax=1)
                ax_.set_title(fname, fontsize=8)

        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ========== PAGE B: UMAP scatter highlighting top-N galaxies ==========
        fig2 = plt.figure(figsize=(7,6))
        plt.scatter(zss_2d_subset[0:2000, 0], zss_2d_subset[0:2000, 1],
                    c='gray', s=10, alpha=0.3, label='All galaxies')
        plt.scatter(zss_2d_subset[top_indices, 0], zss_2d_subset[top_indices, 1],
                    c='red', s=40, alpha=1.0, label=f'Top-{top_n} activated')
        plt.title(f"Latent Dim {dim_idx} (Var={var_dim[dim_idx]:.3f})\nTop-{top_n} in red")
        #plt.legend()
        pdf.savefig(fig2)
        plt.close(fig2)

    pdf.close()
    print(f"Done. PDF saved => {pdf_filename}")
    return z
    # That's it. 




    
