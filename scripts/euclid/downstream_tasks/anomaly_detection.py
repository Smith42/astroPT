import numpy as np
import umap
import matplotlib.pyplot as plt
from astropy.io import fits
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import os
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import h5py
from tqdm import tqdm
plt.style.use('dark_background')

 
def min_max_scale(scores):
    """
    Rescale array of scores to the [0, 1] range.
    """
    s_min = np.min(scores)
    s_max = np.max(scores)
    if s_max == s_min:
        # Avoid divide-by-zero if all scores are the same
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)

# Convert fluxes to magnitudes
def flux_to_mag(flux, zeropoint=23.9):
    return zeropoint - 2.5 * np.log10(flux)

def normalize_image(image):
    """
    Normalizes the image data to the range [0, 1].
    """
    min_val = np.nanmin(image)
    max_val = np.nanmax(image)
    if max_val - min_val == 0:
        return np.zeros_like(image)
    return (image - min_val) / (max_val - min_val)

def plot_filters_individually(fits_file, filter_names=['u', 'g', 'r', 'i', 'z']):
    """
    Plots the five SDSS filter images for a single galaxy.
    """
    with fits.open(fits_file) as hdul:
        data = hdul[0].data  # Shape: (num_filters, 64, 64)

    if data.shape[0] < len(filter_names):
        raise ValueError(f"File {fits_file} does not contain enough filters.")

    # Extract SDSS ID from filename
    filename = os.path.basename(fits_file)
    sdss_id = os.path.splitext(filename)[0]

    # Create a figure for the galaxy
    fig, axes = plt.subplots(1, len(filter_names), figsize=(4 * len(filter_names), 4))
    fig.suptitle(f'Galaxy ID: {sdss_id}', fontsize=16)

    if len(filter_names) == 1:
        axes = [axes]

    for idx, filter_name in enumerate(filter_names):
        ax = axes[idx]
        image = data[idx]  # Assuming order matches the filter_names list
        norm_image = normalize_image(image)

        im = ax.imshow(norm_image, origin='lower', cmap='gray')
        ax.set_title(f'Filter: {filter_name}')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show(block=False)

def show_galaxy_images(galaxy_id, image_dir, filter_names):
    """
    Displays the images of a galaxy given its ID.
    """
    fits_file = os.path.join(image_dir, f'{galaxy_id}.fits')
    if not os.path.isfile(fits_file):
        print(f'FITS file for galaxy ID {galaxy_id} not found.')
        return
    plot_filters_individually(fits_file, filter_names)


def compute_combined_scores(embeddings, method_name, param_grid, metric='cosine'):
    """
    For a given method and parameter grid, compute the average anomaly score
    across all parameter combinations.

    Returns
    -------
    avg_scores : np.ndarray of shape (num_samples,)
        The averaged anomaly score across all parameter combos.
    """

    all_scores = []
    num_samples = embeddings.shape[0]

    # Iterate over every combination in the param grid
    for params in param_grid:
        # Run anomaly detection with these specific parameters
        _, scores = run_anomaly_detection(
            method_name=method_name,
            embeddings=embeddings,
            metric=metric,
            top_n=10,  # or any number; we only care about scores, not top indices here
            n_neighbors=params.get('n_neighbors', 20),
            if_n_estimators=params.get('n_estimators', 100),
            if_max_features=params.get('max_features', 20),
            dbscan_eps=params.get('eps', 0.25),
            dbscan_min_samples=params.get('min_samples', 15)
        )
        all_scores.append(scores)

    # Convert to np array shape: (num_combos, num_samples)
    all_scores = np.array(all_scores)
    # Average across the first axis (all parameter combos)
    avg_scores = np.mean(all_scores, axis=0)

    return avg_scores

def run_anomaly_detection(method_name, embeddings, metric='cosine',
                          top_n=10,
                          n_neighbors=20,
                          if_n_estimators=100,
                          if_max_features=20,
                          dbscan_eps=0.25,
                          dbscan_min_samples=15):
    """
    Runs a given anomaly detection method on the embeddings and returns
    the indices of the top anomalies and their scores.

    Parameters
    ----------
    method_name : str
        One of ['LOF', 'IsolationForest', 'OneClassSVM', 'DBSCAN'].
    embeddings : np.ndarray
        Shape: (num_samples, num_features).
    metric : str
        Distance metric for LOF, DBSCAN, etc.
    top_n : int
        Number of top anomalies to return.
    n_neighbors : int
        Hyperparameter for LOF.
    if_n_estimators : int
        Number of estimators for IsolationForest.
    if_max_features : int
        Max features for IsolationForest.
    dbscan_eps : float
        Eps parameter for DBSCAN.
    dbscan_min_samples : int
        Min_samples for DBSCAN.

    Returns
    -------
    top_indices : np.ndarray of shape (top_n,)
        Indices of the top anomalies.
    scores : np.ndarray of shape (num_samples,)
        Anomaly score for each sample. Higher = more anomalous.
    """
    method_name = method_name.lower()

    if method_name == 'lof':
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, 
                                 metric=metric, 
                                 contamination=0.01)
        labels = lof.fit_predict(embeddings)
        scores = -lof.negative_outlier_factor_  # invert sign so higher = more anomalous

    elif method_name == 'isolationforest':
        iso = IsolationForest(
            random_state=42,
            contamination=0.01,
            n_estimators=if_n_estimators,
            max_features=if_max_features
        )
        iso.fit(embeddings)
        scores = -iso.score_samples(embeddings)  # invert sign so higher = more anomalous

    elif method_name == 'oneclasssvm':
        ocsvm = OneClassSVM(gamma='auto')
        ocsvm.fit(embeddings)
        scores = -ocsvm.decision_function(embeddings)  # invert sign so higher = more anomalous

    elif method_name == 'dbscan':
        dbscan = DBSCAN(eps=dbscan_eps, 
                        min_samples=dbscan_min_samples, 
                        metric=metric)
        labels = dbscan.fit_predict(embeddings)
        # DBSCAN doesn't have a continuous score by default;
        # We'll make anomalies=1, inliers=0:
        scores = (labels == -1).astype(float)

    else:
        raise ValueError(f"Unknown method: {method_name}")

    # Get top N anomalies based on scores
    top_indices = np.argsort(scores)[-top_n:]
    return top_indices, scores

def asinh_stretch(img, low_cut=1, high_cut=99, asinh_scale=1.0):
    """
    1) Clip image between the low_cut and high_cut percentiles.
    2) Apply arcsinh with a chosen scale factor (asinh_scale).
    3) Normalize result to [0..1].

    Parameters
    ----------
    img : 2D array
        Image pixel values (can contain NaNs).
    low_cut, high_cut : float
        Percentiles used for clipping (e.g., 1 and 99).
    asinh_scale : float
        Controls how aggressively bright regions are compressed.
        Smaller values => stronger compression.

    Returns
    -------
    stretched : 2D array
        Normalized (0..1) arcsinh-stretched image.
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


# ------------------------------------------------------------------
# 2) Plot SED (same as before, just be sure it works with your df_sed)
# ------------------------------------------------------------------
def plot_sed(ax, galaxy_id, df_sed, do_mag=True, zeropoint=23.9):
    """
    Given an object_id, look up flux or magnitude data in df_sed,
    then plot them with error bars in the provided axis `ax`.

    df_sed is assumed to have columns like:
      object_id, filter1, filter2, ..., filterN, filter1_err, ..., filterN_err
    """
    row = df_sed.loc[df_sed['object_id'] == galaxy_id]
    if row.empty:
        ax.text(0.5, 0.5, f"No SED for {galaxy_id}", ha='center', va='center')
        return

    # Identify filter columns (excluding object_id and any *_err)
    all_cols = df_sed.columns.tolist()
    filter_cols = [c for c in all_cols if c not in ('object_id',) and not c.endswith('_err')]
    error_cols = [f"{c}_err" for c in filter_cols]

    fluxes = row[filter_cols].values.flatten()
    fluxerrs = row[error_cols].values.flatten()

    x = np.arange(len(filter_cols))  # x positions



    if do_mag:
        # Convert flux -> magnitudes
        with np.errstate(divide='ignore', invalid='ignore'):
            mags = zeropoint - 2.5*np.log10(fluxes)
        # Propagate errors in an approximate way
        magerrs = np.zeros_like(mags)
        for i, (fval, ferr) in enumerate(zip(fluxes, fluxerrs)):
            if fval > 0:
                magerrs[i] = (2.5 / np.log(10)) * (ferr / fval)
            else:
                mags[i] = np.nan
                magerrs[i] = np.nan

        ax.errorbar(x, mags, yerr=magerrs, fmt='o', color='tab:red', ecolor='gray')
        ax.set_xticks(x)
        ax.set_xticklabels(filter_cols, rotation=45, ha='right')
        ax.invert_yaxis()  # magnitudes: smaller is brighter
        ax.set_ylabel("Magnitude")
        ax.set_title(f"SED (magnitudes) - ID={galaxy_id}", fontsize=10)
    else:
        # Plot fluxes directly
        ax.errorbar(x, fluxes, yerr=fluxerrs, fmt='o', color='tab:blue', ecolor='gray')
        ax.set_xticks(x)
        ax.set_xticklabels(filter_cols, rotation=45, ha='right')
        ax.set_ylabel("Flux")
        ax.set_title(f"SED (flux) - ID={galaxy_id}", fontsize=10)

def save_anomalies_vis_grid(
    top_indices,
    combined_scores,
    mapped_properties_subset,
    folder_datacubes,
    pdf_filename,
    images_per_row=5,
    images_per_page=15,  # default: 15 images per page (e.g. 3 rows of 5)
    filter_name='VIS'
):
    """
    Create a multipage PDF showing only the VIS-channel images of the anomalies,
    arranged in a grid with a specified number of images per row.
    
    Parameters
    ----------
    top_indices : array-like
        Indices of the anomalies.
    combined_scores : array-like
        Corresponding anomaly scores.
    mapped_properties_subset : dict
        Dictionary (with key 'object_id') used to look up the galaxy ID.
    folder_datacubes : str
        Path to the folder containing the FITS cubes. Each file is assumed to be named "<object_id>.fits".
    pdf_filename : str
        Output PDF filename.
    images_per_row : int, optional
        Number of images to display in a row (default is 5).
    images_per_page : int, optional
        Total number of images per PDF page (default is 15).
    filter_name : str, optional
        Name of the filter to use. (This function assumes that the requested channel is stored
        as the first channel in the FITS file if filter_name=='VIS'.)
    """
    n_total = len(top_indices)
    # Compute how many pages are needed
    n_pages = math.ceil(n_total / images_per_page)
    
    with PdfPages(pdf_filename) as pdf:
        # Process anomalies in chunks corresponding to each page.
        for page in range(n_pages):
            start = page * images_per_page
            end = min(start + images_per_page, n_total)
            n_images_this_page = end - start
            
            # Determine the number of rows needed for this page
            n_rows = math.ceil(n_images_this_page / images_per_row)
            
            # Set figure size (adjust the multiplier as needed)
            fig, axes = plt.subplots(n_rows, images_per_row, figsize=(images_per_row * 4, n_rows * 4))
            
            # In case there is only one row, force axes to be a 1D array
            if n_rows == 1:
                axes = np.atleast_1d(axes)
            else:
                axes = axes.flatten()
            
            # Set a page title
            fig.suptitle(f"Anomalies VIS Grid (Page {page+1} of {n_pages})", fontsize=16)
            
            # Loop over the anomalies in this page
            for ax_idx, overall_idx in enumerate(range(start, end)):
                ax = axes[ax_idx]
                idx = top_indices[overall_idx]
                galaxy_id = mapped_properties_subset['object_id'][idx]
                score_val = combined_scores[idx]
                
                fits_file = os.path.join(folder_datacubes, f"{galaxy_id}.fits")
                
                if not os.path.isfile(fits_file):
                    # Indicate missing FITS file
                    ax.text(0.5, 0.5, f"No FITS for ID={galaxy_id}", 
                            ha='center', va='center', fontsize=10, color='red')
                    ax.set_title(f"ID={galaxy_id}\nScore={score_val:.4f}", fontsize=8)
                    ax.axis('off')
                else:
                    try:
                        with fits.open(fits_file) as hdul:
                            data = hdul[0].data  # expected shape: (num_filters, H, W)
                        
                        # If the FITS cube is 3D, select the VIS channel.
                        # Here we assume that if filter_name is 'VIS', then the VIS image is stored at index 0.
                        if data is None:
                            raise ValueError("Empty FITS data")
                        if data.ndim == 3:
                            # For more robust code, you could search for the correct channel.
                            vis_image = data[0]
                        else:
                            vis_image = data

                        # Apply arcsinh stretching
                        stretched = asinh_stretch(vis_image, low_cut=0.1, high_cut=99.5, asinh_scale=10)
                        
                        # Display the image
                        ax.imshow(stretched, origin='lower', cmap='gray', vmin=0, vmax=1)
                        ax.set_title(f"ID={galaxy_id}\nScore={score_val:.4f}", fontsize=8)
                        ax.axis('off')
                    except Exception as e:
                        ax.text(0.5, 0.5, f"Error loading ID={galaxy_id}\n{e}",
                                ha='center', va='center', fontsize=10, color='red')
                        ax.set_title(f"ID={galaxy_id}\nScore={score_val:.4f}", fontsize=8)
                        ax.axis('off')
            
            # Turn off any unused subplots in the grid
            total_axes = n_rows * images_per_row
            for ax in axes[n_images_this_page:total_axes]:
                ax.axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
            pdf.savefig(fig)
            plt.close(fig)
    
    print(f"Saved VIS anomalies grid to {pdf_filename}")

def save_anomalies_with_images_and_sed_arcsinh(
    top_indices,
    combined_scores,
    mapped_properties_subset,
    zss_2d_subset,
    filter_names,
    df_sed,               # pivoted SED DataFrame
    folder_datacubes,
    pdf_filename,
    top_n
):
    """
    Create a multipage PDF where each page shows:
      - A row of galaxy images in each filter (with arcsinh-based stretching)
      - Below the images: a single subplot for the SED

    The images are clipped [1%..99%] and arcsinh-stretched.

    Parameters
    ----------
    top_indices : array-like
        Indices of anomalies in your subset arrays.
    combined_scores : array-like
        Corresponding anomaly scores (same order).
    mapped_properties_subset : dict
        Dictionary with keys like 'object_id', used to look up object IDs.
    zss_2d_subset : ndarray
        2D UMAP projection of the subset (for optional summary plot).
    filter_names : list
        List of filter names, e.g. ["VIS", "NIR-H", "NIR-J", "NIR-Y"].
    df_sed : DataFrame
        SED table, with columns [object_id, <f1>, <f2>, ..., <f1>_err, ...].
    folder_datacubes : str
        Path to your .fits cubes (filenames like "<object_id>.fits").
    pdf_filename : str
        Output PDF path.
    top_n : int
        How many anomalies to display in the PDF.
    """

    # For the example, we do a short introduction page + a scatter plot page
    # Then one page per anomaly with images in row 1, SED in row 2.

    from matplotlib.gridspec import GridSpec  # for flexible subplot arrangement

    with PdfPages(pdf_filename) as pdf:
        # -- (A) Intro page
        plt.figure(figsize=(10, 4))
        plt.title(f"Top {top_n} Anomalies (Combined Scores)", fontsize=14)
        plt.text(
            0.5, 0.5,
            f"Showing images + SED for the top {top_n} anomalies.\n"
            "Arcsinh-based stretching on images.\n"
            "SED in magnitudes (inverted y-axis).",
            ha="center", va="center", fontsize=12
        )
        plt.axis('off')
        pdf.savefig()
        plt.close()

        # -- (B) 2D scatter (all points vs anomalies)
        plt.figure(figsize=(8, 6))
        plt.scatter(zss_2d_subset[:, 0], zss_2d_subset[:, 1],
                    c='lightgray', alpha=0.5, s=10, label='All points')
        plt.scatter(zss_2d_subset[top_indices, 0], zss_2d_subset[top_indices, 1],
                    c='red', s=30, label='Anomalies')
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.legend()
        plt.title("UMAP embedding + anomalies highlighted")
        pdf.savefig()
        plt.close()

        # -- (C) For each anomaly, make a page with images on top row + SED in bottom row
        for rank, idx_ in enumerate(top_indices, start=1):
            galaxy_id = mapped_properties_subset['object_id'][idx_]
            score_val = combined_scores[idx_]

            # Setup figure: 2 rows, N columns (for the images in row 0),
            # and row 1 (which spans all columns) for the SED.
            n_filters = len(filter_names)
            fig = plt.figure(figsize=(4 * n_filters, 6))
            gs = GridSpec(2, n_filters, height_ratios=[1.2, 1])  # top row a bit taller if you like
            fig.suptitle(f"Rank={rank}, ID={galaxy_id}, Score={score_val:.4f}", fontsize=12)

            # 1) Load .fits
            fits_file = os.path.join(folder_datacubes, f"{galaxy_id}.fits")
            if not os.path.isfile(fits_file):
                # If missing, just note it and move on
                ax = fig.add_subplot(gs[0, :])
                ax.text(0.5, 0.5, f"No FITS file for ID={galaxy_id}", ha='center', va='center', fontsize=12)
                ax.axis('off')

                ax_sed = fig.add_subplot(gs[1, :])
                plot_sed(ax_sed, galaxy_id, df_sed, do_mag=True)
                pdf.savefig()
                plt.close(fig)
                continue

            with fits.open(fits_file) as hdul:
                data = hdul[0].data  # shape: (num_filters, H, W)

            # 2) For each filter, arcsinh-stretch + plot
            for fidx in range(n_filters):
                ax_img = fig.add_subplot(gs[0, fidx])
                if fidx < data.shape[0]:
                    # arcsinh-stretch
                    stretched = asinh_stretch(data[fidx],
                                              low_cut=0.1,
                                              high_cut=99.5,
                                              asinh_scale=10)
                    ax_img.imshow(stretched, origin='lower', cmap='gray', vmin=0, vmax=1)
                    ax_img.set_title(filter_names[fidx], fontsize=10)
                else:
                    ax_img.set_title(f"{filter_names[fidx]} (no data)", fontsize=10)
                ax_img.axis('off')

            # 3) The SED subplot (bottom row spans all columns)
            ax_sed = fig.add_subplot(gs[1, :])
            plot_sed(ax_sed, galaxy_id, df_sed, do_mag=False)

            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)

    print(f"Saved anomalies + arcsinh images + SED plots to {pdf_filename}")



def show_galaxy_images_static_in_subplots(ax_array, galaxy_id, datacubes_dir, filter_names):
    """
    Plot the galaxy images (one filter per subplot) into the provided `ax_array`.
    """
    fits_file = os.path.join(datacubes_dir, f"{galaxy_id}.fits")
    if not os.path.isfile(fits_file):
        for ax in ax_array:
            ax.set_title(f"Missing: {galaxy_id}")
            ax.axis('off')
        return

    with fits.open(fits_file) as hdul:
        data = hdul[0].data  # shape expected: (n_filters, H, W)
    
    for i, filter_name in enumerate(filter_names):
        ax = ax_array[i]
        if i < data.shape[0]:
            image = data[i]
            norm_img = normalize_image(image)
            ax.imshow(norm_img, origin='lower', cmap='gray')
            ax.set_title(f"{filter_name}")
        else:
            ax.set_title("No data")
        ax.axis('off')



def show_galaxy_images_static(galaxy_id, datacubes_dir, filter_names):
    """
    Display the galaxy images in a way suitable for PDF output.
    """
    from astropy.io import fits
    fits_file = os.path.join(datacubes_dir, f'{galaxy_id}.fits')

    if not os.path.isfile(fits_file):
        print(f'FITS file for galaxy ID {galaxy_id} not found.')
        return

    with fits.open(fits_file) as hdul:
        data = hdul[0].data  # Shape: (num_filters, 64, 64)


    fig, axes = plt.subplots(1, len(filter_names), figsize=(4 * len(filter_names), 4))
    fig.suptitle(f'Galaxy ID: {galaxy_id}', fontsize=16)
    if len(filter_names) == 1:
        axes = [axes]

    for idx, filter_name in enumerate(filter_names):
        ax = axes[idx]
        image = data[idx]  
        #min_val = np.nanmin(image)
        #max_val = np.nanmax(image)

        max_val = np.nanmedian(image) + 4*np.nanstd(image)
        min_val = np.nanmedian(image) - 4*np.nanstd(image)

        norm_image = (image - min_val) / (max_val - min_val) if (max_val > min_val) else np.zeros_like(image)
        im = ax.imshow(norm_image, origin='lower', cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Filter: {filter_name}')
        ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])





if __name__ == "__main__":

    folder_datacubes = '/Users/ginesmartinezsolaeche/Galaxies/Euclid/data/datacubes_large/'
    filter_names = ["VIS", "NIR-H", "NIR-J", "NIR-Y"]

    # Load the catalog data

    folder_pdf = '/Users/ginesmartinezsolaeche/Galaxies/anomaly_detection/pdf/'

    catalog_path = "/Users/ginesmartinezsolaeche/Galaxies/anomaly_detection/data/EuclidMorphPhysPropSpecZ.fits"
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



    folder_data = "path/to/embeddings/" 


    #folder_astropt = folder_data + "astroPT090M_sed/"
    #folder_astropt = folder_data + "astropt090M/"
    folder_astropt = folder_data + "astropt090_4channels/"


    file_SED =  folder_data + "euclidQ1_SEDs_dataframe.csv"

    if os.path.exists(file_SED):
        sed_table  = pd.read_csv(file_SED)      
    else :


        # Define the file path
        file_path = folder_data + "euclidQ1_SEDs.hdf5"

        data_list = []
        # Open the HDF5 file
        with h5py.File(file_path, "r") as f:
            # Access the 'euclidQ1' group
            euclidQ1_group = f["euclidQ1"]
            
            # Retrieve all subgroup keys
            subgroup_keys = list(euclidQ1_group.keys())
            
            # Iterate through each subgroup with a progress bar
            for key in tqdm(subgroup_keys, desc="Processing Subgroups"):
                subgroup = euclidQ1_group[key]
                
                # Extract datasets
                filtnames = subgroup['filtnames'][()]
                instrument = subgroup['instrument'][()]
                lambda_ = subgroup['lambda'][()]
                object_id = subgroup['object_id'][()]
                sed = subgroup['sed'][()]
                sederr = subgroup['sederr'][()]
                
                # Decode bytes to strings if necessary
                if isinstance(instrument, bytes):
                    instrument = instrument.decode('utf-8')
                if isinstance(filtnames, bytes):
                    filtnames = filtnames.decode('utf-8')
                
                # Convert byte strings to regular strings in filtnames if it's an array of bytes
                if isinstance(filtnames, (list, tuple, np.ndarray)):
                    filtnames = [name.decode('utf-8') if isinstance(name, bytes) else name for name in filtnames]
                
                # Append the extracted data to the list
                data_list.append({
                    'filtnames': filtnames,
                    'instrument': instrument,
                    'lambda': lambda_,
                    'object_id': object_id,
                    'sed': sed,
                    'sederr': sederr
                })
        # Create the DataFrame
        sed_data = pd.DataFrame(data_list)
        print(sed_data.head())


        sed_data['filtnames'] = sed_data['filtnames'].apply(lambda x: x if isinstance(x, list) else list(x))
        sed_data['sed'] = sed_data['sed'].apply(lambda x: x.tolist() if isinstance(x, (list, tuple, np.ndarray)) else [x])
        sed_data['sederr'] = sed_data['sederr'].apply(lambda x: x.tolist() if isinstance(x, (list, tuple, np.ndarray)) else [x])

        # Step 2: Explode the DataFrame
        df_exploded = sed_data.explode(['filtnames', 'sed', 'sederr'])
        df_exploded = df_exploded.rename(columns={
            'filtnames': 'filter',
            'sed': 'sed',
            'sederr': 'sederr'
        })

        # Step 3: Pivot for 'sed' values
        df_sed = df_exploded.pivot(index='object_id', columns='filter', values='sed').reset_index()

        # Step 4: Pivot for 'sederr' values
        df_sederr = df_exploded.pivot(index='object_id', columns='filter', values='sederr').reset_index()
        df_sederr.columns = [f"{col}_err" if col != 'object_id' else col for col in df_sederr.columns]

        # Step 5: Merge 'sed' and 'sederr' DataFrames
        sed_table = pd.merge(df_sed, df_sederr, on='object_id', how='left')

        # Step 6: Organize the columns
        filter_names = sed_data['filtnames'].iloc[0]  # Assuming consistency
        sed_columns = filter_names
        sederr_columns = [f"{name}_err" for name in filter_names]
        columns_order = ['object_id'] + sed_columns + sederr_columns
        sed_table = sed_table[columns_order]

        # Step 7: Inspect the final DataFrame
        print(sed_table.head())

        sed_table.to_csv(file_SED, index=False)



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

    # Select those objects from zss and mapped_properties
    datacubes_ids_set = set(datacubes_ids)
    selected_indices = [i for i, obj_id in enumerate(mapped_properties['object_id']) if obj_id in datacubes_ids_set]

    zss_subset = zss[selected_indices, :]
    mapped_properties_subset = {key: value[selected_indices] for key, value in mapped_properties.items()}

    # Merge mapped_properties and sed_table on 'object_id'


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




    df_properties = pd.DataFrame(mapped_properties_subset)  
    mapped_properties_subset = pd.merge(
        df_properties,
        sed_table,
        on="object_id",
        how="inner"
    )

    

    filter_cols = [ 'u_megacam', 'g_decam', 'g_hsc', 'r_decam', 'r_megacam',
       'vis', 'i_decam', 'i_panstarrs', 'z_decam', 'z_hsc', 'y', 'j', 'h']


    filter_to_lambda = {
        "u_megacam":   350.,   # nm, approximate
        "g_decam": 480.,
        "g_hsc": 480.,
        "r_decam": 620.,
        "r_megacam": 620.,
        "vis": 725.,
        "i_decam": 750.,
        "i_panstarrs": 750.,
        "z_decam2" :  850.,
        "z_hsc" :  850.,
        'y' : 1000.,
        'j' : 1250.,
        'h' :  1650. }



    cond_euclid1 = (np.isnan(mapped_properties_subset['vis']) == False) & (mapped_properties_subset['vis'] > 2*mapped_properties_subset['vis_err'])
    cond_euclid2 = (np.isnan(mapped_properties_subset['y']) == False) & (mapped_properties_subset['y'] > 2*mapped_properties_subset['y_err'])
    cond_euclid3 = (np.isnan(mapped_properties_subset['j']) == False) & (mapped_properties_subset['j'] > 2*mapped_properties_subset['j_err'])
    cond_euclid4 = (np.isnan(mapped_properties_subset['h']) == False) & (mapped_properties_subset['h'] >2*mapped_properties_subset['h_err'])

    cond_euclid = cond_euclid1 & cond_euclid2 & cond_euclid3 & cond_euclid4 &  (mapped_properties_subset['vis'] > 0)
    det_quality_flags = (mapped_properties_subset['det_quality_flag'] == 0) & cond_euclid

    # Apply det_quality_flags condition to mapped_properties_subset and zss_2d_subset
    zss_subset = zss_subset[det_quality_flags]
    zss_2d_subset = zss_2d_subset[det_quality_flags]
    mapped_properties_subset = {key: value[det_quality_flags] for key, value in mapped_properties_subset.items()}

    mapped_properties_subset = {key: value.reset_index(drop=True) for key, value in mapped_properties_subset.items()}




    #methods = ['DBSCAN', 'LOF', 'IsolationForest', 'OneClassSVM']
    # choose the preferred methods
    methods = [ 'LOF']
    # 1) IsolationForest

    if_param_grid = []
    for n_est in [50, 100, 150, 200]:

        for max_f in [1, 5, 10, 20]:  
            if_param_grid.append({'n_estimators': n_est, 'max_features': max_f})

    # 2) LOF
    lof_param_grid = []
    for n_neighbors in [5, 10, 15, 20, 35, 50]:
        lof_param_grid.append({'n_neighbors': n_neighbors})

    # 3) DBSCAN

    dbscan_param_grid = []
    for min_s in [1, 5, 10, 20, 50]:
         for eps_ in [0.01, 0.05, 0.1]:

            dbscan_param_grid.append({'eps': eps_, 'min_samples': min_s})

    # 4) OneClassSVM - we do nothing (only default)
    ocsvm_param_grid = [ {} ]  # single combination with default parameters

        # -------------------------------------
        # Compute combined (averaged) scores for each method
        # -------------------------------------
    embeddings = zss_subset  # rename for clarity
    if 'DBSCAN' in methods :
        scores_dbscan = compute_combined_scores(embeddings, 'dbscan', dbscan_param_grid, metric='cosine')    
        scores_dbscan_norm = min_max_scale(scores_dbscan)
    else :
        scores_dbscan_norm = np.zeros(det_quality_flags.sum()) 

    if 'IsolationForest' in methods :
        scores_if = compute_combined_scores(embeddings, 'isolationforest', if_param_grid, metric='cosine')
        scores_if_norm = min_max_scale(scores_if)
    else :
        scores_if_norm = np.zeros(det_quality_flags.sum()) 

    if 'LOF' in methods :
        scores_lof = compute_combined_scores(embeddings, 'lof', lof_param_grid, metric='cosine')
        scores_lof_norm = min_max_scale(scores_lof)
    else :
        scores_lof_norm = np.zeros(det_quality_flags.sum()) 

    if 'OneClassSVM' in methods :
        scores_ocsvm = compute_combined_scores(embeddings, 'oneclasssvm', ocsvm_param_grid, metric='cosine')
        scores_ocsvm_norm = min_max_scale(scores_ocsvm)
    else :
        scores_ocsvm_norm = np.zeros(det_quality_flags.sum()) 

    # Suppose we have four arrays of raw scores from each method
    # (averaged over hyperparams or not):
    #   scores_if, scores_lof, scores_dbscan, scores_ocsvm
    # Each array is shape (N,) where N is the number of samples.

    # 1. Normalize each method’s scores to [0,1]:

    combined_scores = (
        scores_if_norm
        + scores_lof_norm
        + scores_dbscan_norm
        + scores_ocsvm_norm
    ) / len(methods)



    # # Decide the threshold for anomalies based on the distribution
    # # For example, we can consider scores in the top 5% as anomalies
    threshold = np.percentile(combined_scores, 99)
    anomalies = combined_scores >= threshold
    num_anomalies = np.sum(anomalies)

    print(f"Number of anomalies based on the threshold: {num_anomalies}")

    # Get the indices of the anomalies
    top_n = num_anomalies
    top_indices = np.argsort(combined_scores)[-top_n:]
    top_indices = top_indices[::-1]  # sort descending

    # Read the CSV file
    csv_path = '/Users/ginesmartinezsolaeche/Galaxies/anomaly_detection/data/q1_discovery_engine_lens_catalog.csv'
    lens_catalog = pd.read_csv(csv_path)

    # Crossmatch with our mapped_properties_subset using 'object_id'
    lens_object_ids = lens_catalog['object_id'].values
    # Count anomalies with 'merging_merger' > 80
    merging_merger_anomalies = np.sum(mapped_properties_subset['merging_merger'][top_indices] > 80)
    print(f"Number of anomalies with 'merging_merger' > 80: {merging_merger_anomalies}")

    # Count anomalies with 'merging_major_disturbance' > 80
    merging_major_disturbance_anomalies = np.sum(mapped_properties_subset['merging_major_disturbance'][top_indices] > 80)
    print(f"Number of anomalies with 'merging_major_disturbance' > 80: {merging_major_disturbance_anomalies}")

    # Count anomalies that are in the lens catalog
    lens_anomalies = np.sum(np.isin(mapped_properties_subset['object_id'][top_indices], lens_object_ids))
    print(f"Number of anomalies in the lens catalog: {lens_anomalies}")

    # Get the scores of the lens anomalies
    lens_anomaly_scores = combined_scores[top_indices][np.isin(mapped_properties_subset['object_id'][top_indices], lens_object_ids)]
    print(f"Scores of the lens anomalies: {lens_anomaly_scores}")




    pdf_name = "top_anomalies_scores_VIS_NISP_LOF_cualitycut.pdf"
    # Just an example call:
    pdf_filename = os.path.join(folder_pdf, pdf_name)
    
    save_anomalies_with_images_and_sed_arcsinh(
        top_indices=top_indices,
        combined_scores=combined_scores,
        mapped_properties_subset=mapped_properties_subset,
        zss_2d_subset=zss_2d_subset,
        filter_names=["VIS", "NIR-H", "NIR-J", "NIR-Y"],  
        df_sed=sed_table,        
        folder_datacubes=folder_datacubes,
        pdf_filename=pdf_filename,
        top_n=len(top_indices))



    save_anomalies_vis_grid(
        top_indices=top_indices,
        combined_scores=combined_scores,
        mapped_properties_subset=mapped_properties_subset,
        folder_datacubes=folder_datacubes,
        pdf_filename=pdf_filename,
        images_per_row=5,
        images_per_page=15
    )
