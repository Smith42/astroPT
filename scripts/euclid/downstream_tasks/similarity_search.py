import numpy as np
import umap
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics.pairwise import cosine_similarity

plt.style.use('dark_background')



def similarity_search(
    query_idx,
    embeddings,
    zss_2d, 
    mapped_properties,
    folder_datacubes,
    filter_names,
    pdf_filename,
    top_n=10,
    select_property='merging_merger',
):
    """
    1) Takes a query galaxy index (query_idx).
    2) Finds the top_n closest galaxies in embedding space (via cosine similarity).
    3) Creates a PDF with:
       - A summary page of neighbors
       - A UMAP scatter plot highlighting the query + neighbors
       - One page per galaxy, showing each filter image (grayscale)
         and a final RGB image (extra column).

    This version uses:
       - Manual percentile-based clipping to avoid saturation in very bright regions.
       - arcsinh stretching (non-linear) to enhance low-surface-brightness features.
    """

    # ----------------------------------------------------------------
    # 1) Validate index, compute cosine similarity
    # ----------------------------------------------------------------
    if query_idx < 0 or query_idx >= len(embeddings):
        raise IndexError(
            f"query_idx {query_idx} is out of bounds "
            f"for embeddings size {len(embeddings)}."
        )

    query_embedding = embeddings[query_idx].reshape(1, -1)
    sims = cosine_similarity(query_embedding, embeddings)[0]  # shape: (N,)

    # ----------------------------------------------------------------
    # 2) Sort galaxies by descending similarity, pick top_n
    # ----------------------------------------------------------------
    top_indices = np.argsort(sims)[::-1]
    nearest_indices = top_indices[:top_n]  # includes the query itself

    query_id = mapped_properties['object_id'][query_idx]
    print(f"Query index = {query_idx}, object_id = {query_id}")
    print("Top matches (index, object_id, similarity):")
    for i in nearest_indices:
        gid = mapped_properties['object_id'][i]
        print(f"   idx={i}, ID={gid}, sim={sims[i]:.4f}")

    # ----------------------------------------------------------------
    # HELPER FUNCTION: Percentile-based arcsinh stretch
    # ----------------------------------------------------------------
    def asinh_stretch(img, low_cut=1, high_cut=99, asinh_scale=1.0):
        """
        1) Clip image between the low_cut and high_cut percentiles.
        2) Apply arcsinh with a chosen scale factor (asinh_scale).
        3) Normalize to 0..1.
        
        The 'asinh_scale' determines how aggressively bright regions are compressed.
        Larger values = less compression (bulges look brighter).
        Smaller values = more compression (bulges less saturated).
        """
        # 1) Ignore NaNs
        valid_pixels = img[np.isfinite(img)]
        if valid_pixels.size == 0:
            # All NaN or empty array
            return np.zeros_like(img)

        # 2) Get percentile cuts
        vmin = np.percentile(valid_pixels, low_cut)
        vmax = np.percentile(valid_pixels, high_cut)
        # 3) Clip
        clipped = np.clip(img, vmin, vmax)

        # 4) arcsinh transform
        #    arcsinh_scale * clipped means controlling the "compression" range.
        #    If arcsinh_scale is small, bulges become less bright relative to faint areas.
        arcs = np.arcsinh(asinh_scale * clipped)

        # 5) Normalize to [0,1]
        arcs_min, arcs_max = arcs.min(), arcs.max()
        if arcs_max > arcs_min:
            stretched = (arcs - arcs_min) / (arcs_max - arcs_min)
        else:
            stretched = np.zeros_like(arcs)

        return stretched

    # ----------------------------------------------------------------
    # 3) Create a PDF with all plots
    # ----------------------------------------------------------------
    with PdfPages(pdf_filename) as pdf:
        # ----------------- (A) SUMMARY PAGE -----------------
        fig = plt.figure(figsize=(10, 4))
        plt.title("Similarity Search Summary", fontsize=16, pad=20)
        lines = [
            f"Query index = {query_idx}",
            f"Query object_id = {query_id}",
            "",
            f"Top {top_n} matches by cosine similarity:"
        ]
        for i in nearest_indices:
            gid = mapped_properties['object_id'][i]
            lines.append(f" idx={i}, ID={gid}, sim={sims[i]:.4f}")

        plt.text(0.5, 0.5, "\n".join(lines),
                 ha='center', va='center', fontsize=12)
        plt.axis('off')
        pdf.savefig(fig)
        plt.close(fig)

        # ----------------- (B) UMAP PLOT -----------------
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(zss_2d[:, 0], zss_2d[:, 1],
                    c='gray', alpha=0.3, s=10, label='All galaxies')
        plt.scatter(zss_2d[nearest_indices, 0], zss_2d[nearest_indices, 1],
                    c='red', s=50, label='Nearest neighbors')
        plt.scatter(zss_2d[query_idx, 0], zss_2d[query_idx, 1],
                    c='yellow', s=100, edgecolor='black', label='Query galaxy')
        plt.legend()
        plt.title(f"Query galaxy (yellow) & top {top_n} neighbors in UMAP space")
        pdf.savefig(fig)
        plt.close(fig)

        # ----------------- (C) ONE PAGE PER GALAXY -----------------
        for i in nearest_indices:
            galaxy_id = mapped_properties['object_id'][i]
            my_property = mapped_properties[select_property][i]
            sim_val = sims[i]

            # We'll create one extra column for the RGB image
            ncols = len(filter_names) + 1  # +1 for the RGB
            fig, axes = plt.subplots(
                1, ncols, figsize=(4 * ncols, 4), squeeze=False
            )
            # axes is shape (1, ncols)

            fig.suptitle(
                f"{select_property}={my_property}, ID={galaxy_id}, sim={sim_val:.4f}",
                fontsize=14
            )

            # Load the FITS file
            fits_path = os.path.join(folder_datacubes, f"{galaxy_id}.fits")
            if not os.path.isfile(fits_path):
                # If no FITS, just put a note
                plt.text(0.5, 0.5, f"No FITS file found for ID={galaxy_id}",
                         ha='center', va='center', fontsize=12)
                plt.axis('off')
                pdf.savefig(fig)
                plt.close(fig)
                continue

            with fits.open(fits_path) as hdul:
                data = hdul[0].data  # shape: (num_filters, H, W)

            # ---------- GRAYSCALE PLOTS FOR EACH FILTER ----------
            for idxf, filter_name in enumerate(filter_names):
                ax = axes[0, idxf]

                if idxf >= data.shape[0]:
                    # The data array doesn't match the # of filters we expect
                    ax.set_title(f"{filter_name}\n[No data in FITS]", fontsize=10)
                    ax.axis('off')
                    continue

                # Apply arcsinh-based stretching
                stretched = asinh_stretch(data[idxf],
                                          low_cut=0.11,   # tweak these percentiles
                                          high_cut=99.9, # for your data
                                          asinh_scale=20)  # smaller => more compression
                #aquimod
                ax.imshow(stretched, origin='lower', cmap='gray', vmin=0, vmax=1)
                ax.set_title(f"Filter: {filter_name}", fontsize=10)
                ax.axis('off')

            # ---------- RGB COMPOSITE ----------
            # Example indices: R = filter[3] + filter[2], G = filter[1], B = filter[0]
            # Adjust if you have fewer or differently-ordered filters
            if (data.shape[0] >= 4) and (len(filter_names) >= 4):
                idx_R1 = 3
                idx_R2 = 2
                idx_G = 1
                idx_B = 0

                # Combine for red
                red_data = data[idx_R1] + data[idx_R2]
                red_stretch = asinh_stretch(red_data, asinh_scale=0.5)
                green_stretch = asinh_stretch(data[idx_G], asinh_scale=0.5)
                blue_stretch = asinh_stretch(data[idx_B], asinh_scale=0.5)

                # Merge into a single (H, W, 3) image
                rgb_image = np.dstack((red_stretch, green_stretch, blue_stretch))

                ax_rgb = axes[0, -1]
                ax_rgb.imshow(rgb_image, origin='lower')
                ax_rgb.set_title("RGB Image", fontsize=10)
                ax_rgb.axis('off')
            else:
                # Not enough filters to form the intended RGB
                ax_rgb = axes[0, -1]
                ax_rgb.text(0.5, 0.5, "Not enough filters for RGB",
                            ha='center', va='center', fontsize=10)
                ax_rgb.axis('off')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\nSimilarity search complete. PDF saved to: {pdf_filename}")







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

def flux_to_mag(flux, zeropoint=23.9):
    """
    Convert flux to magnitude using a given zeropoint.
    """
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
    Plots the five SDSS (or given) filter images for a single galaxy.
    """
    with fits.open(fits_file) as hdul:
        data = hdul[0].data  # Shape: (num_filters, 64, 64)

    if data.shape[0] < len(filter_names):
        raise ValueError(f"File {fits_file} does not contain enough filters.")

    # Extract ID from filename
    filename = os.path.basename(fits_file)
    sdss_id = os.path.splitext(filename)[0]

    # Create a figure
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

def show_galaxy_images_static(galaxy_id, datacubes_dir, filter_names):
    """
    Display the galaxy images in a way suitable for PDF (or screen) output.
    """
    fits_file = os.path.join(datacubes_dir, f'{galaxy_id}.fits')
    if not os.path.isfile(fits_file):
        print(f'FITS file for galaxy ID {galaxy_id} not found.')
        return

    with fits.open(fits_file) as hdul:
        data = hdul[0].data  # shape: (num_filters, 64, 64)

    fig, axes = plt.subplots(1, len(filter_names), figsize=(4 * len(filter_names), 4))
    fig.suptitle(f'Galaxy ID: {galaxy_id}', fontsize=16)
    if len(filter_names) == 1:
        axes = [axes]

    for idx, filter_name in enumerate(filter_names):
        ax = axes[idx]
        image = data[idx]  
        max_val = np.nanmedian(image) + 4*np.nanstd(image)
        min_val = np.nanmedian(image) - 4*np.nanstd(image)
        norm_image = (image - min_val) / (max_val - min_val) if (max_val > min_val) else np.zeros_like(image)
        im = ax.imshow(norm_image, origin='lower', cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Filter: {filter_name}')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# ----------------------------------------------------------------------------------
# Below: Load your data, catalog, embeddings, and run anomaly detection
# (Everything is as you provided.)
# ----------------------------------------------------------------------------------


folder_datacubes = '/Users/ginesmartinezsolaeche/Galaxies/Euclid/data/datacubes_large/'
filter_names = ["VIS", "NIR-Y","NIR-J","NIR-H"]

catalog_path = "/Users/ginesmartinezsolaeche/Galaxies/anomaly_detection/data/EuclidMorphPhysPropSpecZ.fits"
with fits.open(catalog_path) as hdul:
    catalog_data = hdul[1].data

object_ids = catalog_data['object_id']
catalog_file_names = catalog_data['VIS_name']
print("Number of galaxies in the catalog:", len(catalog_file_names))

print("Keys in catalog_data:", catalog_data.columns.names)

parameters = {
    "object_id" : catalog_data['object_id'],
    "Star_prob" : catalog_data['classification.Star_prob'],
    "Gal_prob" : catalog_data['classification.Gal_prob'],
    "QSO_prob" : catalog_data['classification.QSO_prob'],

    'RA' :  catalog_data['right_ascension'],
    'DEC' : catalog_data['declination'],
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

test_file_path = "data/astropt090M/train.txt"
with open(test_file_path, "r") as f:
    test_files = [line.strip() for line in f.readlines()]
print("Number of galaxies in train.txt", len(test_files))

folder_data = "data/astropt090M/"
#folder_data = "astropt090_4channels/"

zss = np.load(folder_data + "zss_64t_mean_train.npy")
idxs = np.load(folder_data + "idxs_64t_mean_train.npy")

print("Embeddings shape:", zss.shape)
print("Indices shape:", idxs.shape)

test_to_catalog_idx = {test_name: i for i, test_name in enumerate(catalog_file_names)}

mapped_properties = {key: [] for key in parameters.keys()}

for test_name in [test_files[idx] for idx in idxs]:
    catalog_idx = test_to_catalog_idx.get(test_name)
    if catalog_idx is not None:
        for key, array in parameters.items():
            mapped_properties[key].append(array[catalog_idx])
    else:
        for key in parameters.keys():
            mapped_properties[key].append(np.nan)

for key in mapped_properties:
    mapped_properties[key] = np.array(mapped_properties[key])

datacubes_files = os.listdir(folder_datacubes)
datacubes_ids = [int(f.split('.')[0]) for f in datacubes_files if f.endswith('.fits')]
datacubes_ids_set = set(datacubes_ids)
selected_indices = [i for i, obj_id in enumerate(mapped_properties['object_id']) if obj_id in datacubes_ids_set]

zss_subset = zss[selected_indices, :]
mapped_properties_subset = {key: value[selected_indices] for key, value in mapped_properties.items()}

umap_projection_file = os.path.join(folder_data, "umap_projection.npy")

if os.path.exists(umap_projection_file):
    print('Loading existing UMAP projection')
    zss_2d_subset = np.load(umap_projection_file)
else:
    print('Computing UMAP')
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    zss_2d_subset = umap_model.fit_transform(zss_subset)
    np.save(umap_projection_file, zss_2d_subset)
    print('UMAP projection finished and saved')

det_quality_flags = mapped_properties_subset['det_quality_flag'] == 0

# Apply det_quality_flags
zss_subset = zss_subset[det_quality_flags]
zss_2d_subset = zss_2d_subset[det_quality_flags]
mapped_properties_subset = {key: value[det_quality_flags] for key, value in mapped_properties_subset.items()}





# ----------------------------------------------------------------------------------
# MERGINGIN_MERGER HISTOGRAM
# Search for indices where 'merging_merger' property is greater than a value 

merger_threshold = 80
n_similar_objects = 100
merging_merger_indices = np.where(mapped_properties_subset['merging_merger'] > merger_threshold)[0]

all_top_merging_merger = []

for query_index in merging_merger_indices :
    print(query_index)
        # Get the top n_similar_objects most similar objects to the current query_index
    top_indices = np.argsort(cosine_similarity(zss_subset[query_index].reshape(1, -1), zss_subset)[0])[::-1][:n_similar_objects+1]
    top_merging_merger = mapped_properties_subset['merging_merger'][top_indices[1:]]
        
    all_top_merging_merger.extend(top_merging_merger)


general_merging_merger = mapped_properties_subset['merging_merger']
    # Select randomly from mapped_properties_subset['merging_merger'] n_similar_objects*len(merging_merger_indices)
random_merging_merger = np.random.choice(mapped_properties_subset['merging_merger'], n_similar_objects * len(merging_merger_indices), replace=False)



plt.figure(figsize=(12, 6))

# Histogram for the combined top 100 most similar objects from each query
plt.hist(all_top_merging_merger, log=True, bins=30, color='red', alpha=1, histtype='step', linewidth=2, label='Top 100 Similar Objects (Combined)')
plt.hist(random_merging_merger,  log=True, bins=30, color='blue', alpha=1, histtype='step', linewidth=2,  label='General Distribution')

plt.xlabel('merging_merger', fontsize=20)
plt.ylabel('Frequency', fontsize=20)


plt.legend(frameon=False, loc='lower left', fontsize=12)

plt.text(0.95, 0.95, 'VIS', transform=plt.gca().transAxes, fontsize=20, verticalalignment='top', horizontalalignment='right')

plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)

plt.tight_layout()
plt.show()


my_id = 2627736497657964330

query_index = np.where(mapped_properties_subset['object_id'] ==  2627736497657964330)[0][0]

folder_pdf = 'pdf/'


pdf_filename = os.path.join(folder_pdf, 'similarity_' + str(mapped_properties_subset['object_id'][query_index]) + '.pdf')




similarity_search(
    query_idx=query_index,
    embeddings=zss_subset,
    zss_2d=zss_2d_subset,
    mapped_properties=mapped_properties_subset,
    folder_datacubes=folder_datacubes,
    filter_names=filter_names,
    pdf_filename=pdf_filename,
    top_n=10,
    select_property='merging_merger'    # change as needed
)



