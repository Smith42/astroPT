import os
import glob
from astropy.io import fits
import numpy as np
from scipy.spatial import cKDTree



image_dir = "/path/to/cutouts/"

datacubes_dir = "/path/to/datacubes/"

file_properties = "/path/to/galaxy_catalogue/EuclidMorphPhysPropSpecZ.fits"

image_size_x = 64 # Cutout size of the datacube
image_size_y = 64

os.makedirs(datacubes_dir, exist_ok=True)

# Define filters
filters = ["VIS", "NIR-H", "NIR-J", "NIR-Y"]

search_radius_arcsec = 1.0
search_radius_deg = search_radius_arcsec / 3600.0

# Read the properties file
with fits.open(file_properties) as hdul:
    data = hdul[1].data
    object_ids = data['object_id']
    ras = data['right_ascension']
    decs = data['declination']

def parse_filename_info(fname):
    # Extract RA/DEC from filename pattern: ...CUTOUT_{RA}_{DEC}.fits
    fname = os.path.basename(fname)
    coord_part = fname.split("CUTOUT_")[1].replace(".fits", "")
    RA_str, DEC_str = coord_part.split("_")
    return float(RA_str), float(DEC_str)

def sph_to_cart(ra_deg, dec_deg):
    # Convert RA, Dec to Cartesian for KD-tree
    rad = np.pi / 180.0
    ra = ra_deg * rad
    dec = dec_deg * rad
    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)
    return x, y, z

# Build dictionaries: filter_dict[filter] = list of (RA, DEC, filename)
filter_dict = {f: [] for f in filters}
for f in filters:
    fpath = os.path.join(image_dir, f)
    fits_files = glob.glob(os.path.join(fpath, f"MOSAIC-{f}*.fits"))
    for ff in fits_files:
        RA, DEC = parse_filename_info(ff)
        filter_dict[f].append((RA, DEC, ff))

# Build KD-trees for each filter
filter_kd = {}
for f in filters:
    if len(filter_dict[f]) == 0:
        filter_kd[f] = None
        continue
    # Convert all RA/DEC for this filter to Cartesian
    coords = np.array([sph_to_cart(r, d) for (r, d, _) in filter_dict[f]])
    tree = cKDTree(coords)
    filter_kd[f] = (tree, filter_dict[f])

def find_closest_image(RA_ref, DEC_ref, f):
    # Uses KD-tree to find the closest image in a given filter
    if filter_kd[f] is None:
        return None
    tree, sources = filter_kd[f]
    x_ref, y_ref, z_ref = sph_to_cart(RA_ref, DEC_ref)
    dist, idx = tree.query([x_ref, y_ref, z_ref], k=1)
    # dist is the Euclidean distance in Cartesian space on the unit sphere
    # Convert this back to an angular distance:
    # dist in 3D ~ 2*sin(d/2), where d is angular distance in radians.
    # For small angles, dist ~ d (radians).
    # Let's approximate: d_radians ~ dist
    d_deg = dist * (180.0/np.pi)
    if d_deg <= search_radius_deg:
        return sources[idx][2]  # filename
    else:
        return None

n_filters = len(filters)

for obj_id, RA_ref, DEC_ref in zip(object_ids, ras, decs):
    # Find the closest VIS image
    vis_file = find_closest_image(RA_ref, DEC_ref, "VIS")
    if vis_file is None:
        # No VIS image found, skip this object
        continue

    # Read VIS image
    with fits.open(vis_file) as vishdu:
        vis_data = vishdu[0].data
        vis_header = vishdu[0].header

    Nx = image_size_x
    Ny = image_size_y
    cube = np.zeros((n_filters,Nx, Ny), dtype=float)

    # Crop VIS data
    ysize, xsize = vis_data.shape
    xstart = (xsize - Nx) // 2
    ystart = (ysize - Ny) // 2
    xend = xstart + Nx
    yend = ystart + Ny
    cube[filters.index("VIS"),:, :] = vis_data[ystart:yend, xstart:xend]

    # For the other filters, find closest match
    for i, f in enumerate(filters):
        if f == "VIS":
            continue
        match_file = find_closest_image(RA_ref, DEC_ref, f)

        if match_file is not None:
            with fits.open(match_file) as hdu:
                fdata = hdu[0].data
 
            yfsize, xfsize = fdata.shape
            xfstart = (xfsize - Nx) // 2
            yfstart = (yfsize - Ny) // 2
            xfend = xfstart + Nx
            yfend = yfstart + Ny
            cube[i, :, :] = fdata[yfstart:yfend, xfstart:xfend]


    # Name the output file using the object_id
    out_fname = f"{obj_id}.fits"

    out_path = os.path.join(datacubes_dir, out_fname)

    primary_hdu = fits.PrimaryHDU(data=cube, header=vis_header)
    # Update header with filter info
    for i, fil in enumerate(filters):
        primary_hdu.header[f'FILTER{i+1}'] = fil
    primary_hdu.header['OBJID'] = obj_id
    primary_hdu.header['RA_OBJ'] = RA_ref
    primary_hdu.header['DEC_OBJ'] = DEC_ref

    hdul = fits.HDUList([primary_hdu])
    hdul.writeto(out_path, overwrite=True)
    print(f"Saved cube for object {obj_id}: {out_path}")
