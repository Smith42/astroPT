import os
import sys
import numpy as np
from astropy.io import fits
from astropy.table import Table

import desispec.io
from desispec import coaddition

from tqdm import tqdm


# Useful function to cross match index
def find_matching_indices(targets, reference_ids):
    """Return indices of `targets` in `reference_ids`."""
    id_to_index = {tid: i for i, tid in enumerate(reference_ids)}
    return np.array([id_to_index[tid] for tid in targets])

# Reading command flags arguments
if len(sys.argv) != 3:
    print("Error: Required flags: TASK_ID and TASK_COUNT")
    sys.exit(1)

task_id = int(sys.argv[1])
task_count = int(sys.argv[2])


# Defining the original paths
catalog_file_dir = "/home/valonso/iac18_aasensio_shared/euclid_q1_desi_dr1"
catalog_file_name = "base_EuclidQ1_DESIDR1.fits"
catalog_file_path = os.path.join(catalog_file_dir, catalog_file_name)


# Desi spectra full sample directories
desi_main_folder = "/home/valonso/iac18_aasensio_shared/desi_dr1_main"
desi_sv3_folder = "/home/valonso/iac18_aasensio_shared/desi_dr1_sv3"


# Preprocess spectra output directory. One for each parallel task
astropt_data = "/home/valonso/iac18_mhuertas_shared/valonso/astroPT/data"
pre_espectra_dir_name = f"desi_dr1_training_spectra_p{task_id}"
pre_espectra_dir_out = os.path.join(astropt_data, pre_espectra_dir_name)

os.makedirs(pre_espectra_dir_out, exist_ok=True) 


# New catalog with preprocess spectra. One for each parallel task
train_catalog_file_name = f"base_EuclidQ1_DESIDR1_TRAIN_p{task_id}.fits"
output_catalog_file = os.path.join(astropt_data, train_catalog_file_name)


# Loading original catalog
catalog_table = Table.read(catalog_file_path, format='fits')
print(f"Loading catalog {catalog_file_name} with {len(catalog_table)} rows.")

catalog_parallel_chunk = np.array_split(catalog_table, task_count)[task_id]
print(f"Task {task_id}: Processing {len(catalog_parallel_chunk)} rows (of {len(catalog_table)} total)")


# List to store successful rows
preprocess_spectra_index = []
preprocess_spectra_path = [] 


# Main loop to process the sample
N = len(catalog_parallel_chunk)
#N=100 # For a testing subsample

for i in tqdm(range(N), desc=f"Processing Task {task_id}"):
    
    # Loading items
    try:
        # Each row is an entry
        entry = catalog_parallel_chunk[i]
        
        # Obtaining required values
        targetid = entry['TARGETID']
        healpix_id = entry['HEALPIX']
        survey = entry['SURVEY'].decode('utf-8').strip()
        program = entry['PROGRAM'].decode('utf-8').strip()
        
        # Creating the name of the DESI spectra file to searching for
        spectrum_filename = f"coadd-{survey.lower()}-{program.lower()}-{healpix_id}.fits"
        
        # Choose subfolder based on survey
        if survey.lower() == "main":
            spectra_folder = desi_main_folder
        elif survey.lower() == "sv3":
            spectra_folder = desi_sv3_folder
        else:
            raise ValueError(f"Survey {survey} not supported.")
        
        # Searching for the spectra inside DESI fits
        spectrum_path = os.path.join(spectra_folder, spectrum_filename)
        
        # Reading DESI fits file for each spectra
        spectra = desispec.io.read_spectra(spectrum_path, 
                                           skip_hdus=['EXP_FIBERMAP', 'SCORES', 'EXTRA_CATALOG', 
                                                      'MASK', 'RESOLUTION'])
        
        # Selecting the target
        selected = spectra.select(targets=[targetid])
        
        # Combining b,r,z cameras data
        combined = coaddition.coadd_cameras(selected)

        reorder_idx = find_matching_indices([targetid], combined.target_ids())

        # Obtaining spectra data values 
        wave = combined.wave["brz"].astype(np.float32)
        flux = combined.flux["brz"][reorder_idx].astype(np.float32)[0]
        ivar = combined.ivar["brz"][reorder_idx].astype(np.float32)[0]
        
        
        # Saving spectra as a new fits
        pre_spectra_out_filename = os.path.join(pre_espectra_dir_out, f"{targetid}.fits")

        # Creating the fits as a table
        col_wave = fits.Column(name='WAVELENGTH', format='E', array=wave)
        col_flux = fits.Column(name='FLUX', format='E', array=flux)
        col_ivar = fits.Column(name='IVAR', format='E', array=ivar)
        coldefs = fits.ColDefs([col_wave, col_flux, col_ivar])
        
        # Creating the HDU
        hdu = fits.BinTableHDU.from_columns(coldefs)
        
        # Adding it to header to cross match
        hdu.header['TARGETID'] = targetid 
        
        # Empty primary HDU
        primary_hdu = fits.PrimaryHDU() 
        
        # Creating the complete fits file
        hdul = fits.HDUList([primary_hdu, hdu])
        hdul.writeto(pre_spectra_out_filename, overwrite=True)
        hdul.close()

        
        # Adding the row to the successful list
        preprocess_spectra_path.append(pre_spectra_out_filename)
        preprocess_spectra_index.append(i)

    # Managing possible errors
    except (FileNotFoundError, ValueError, KeyError, IndexError) as e:
        print(f"[WARNING]: (Task {task_id}) Skipping row {i} (TARGETID {targetid}) due to: {e}")
        continue

# Saving new catalog
print("\nPreprocessing is finished.")
if preprocess_spectra_index:
    
    # New table with new spectra paths
    train_table_rows = catalog_parallel_chunk[preprocess_spectra_index]
    
    # Creating a new table
    train_table_part = Table(train_table_rows)
    
    # Adding path to new catalog
    train_table_part["SPEC_PATH"] = preprocess_spectra_path
    
    # Saving the new catalog table
    train_table_part.write(output_catalog_file, overwrite=True)
    
    print(f"Training Partial catalog saved in {output_catalog_file} with {len(train_table_part)} rows.")
    
else:
    print(f"Task {task_id} processed no rows.")