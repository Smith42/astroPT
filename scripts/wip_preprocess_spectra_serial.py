import os
import glob
import shutil
from astropy.table import Table, vstack
from tqdm import tqdm

print("Initializing parallel fusion")

# Defining the catalog paths
catalog_file_dir = "/home/valonso/iac18_aasensio_shared/euclid_q1_desi_dr1"
train_catalog_file_name = f"base_EuclidQ1_DESIDR1_TRAIN.fits"
output_catalog_file = os.path.join(catalog_file_dir, train_catalog_file_name)

# Wehere partial catalogs and spectra are
astropt_data = "/home/valonso/iac18_mhuertas_shared/valonso/astroPT/data"

# Preprocess spectra output directory
pre_espectra_dir_name = f"desi_dr1_training_spectra"
pre_espectra_dir_out = os.path.join(catalog_file_dir, pre_espectra_dir_name)

os.makedirs(pre_espectra_dir_out, exist_ok=True) 

# Merging of the catalogs
catalog_parts = sorted(glob.glob(os.path.join(astropt_data, "base_EuclidQ1_DESIDR1_TRAIN_p*.fits")))

if not catalog_parts:
    print("[ERROR] Partial catalogs not found. Check again parallel process.")

else:
    print(f"Founded {len(catalog_parts)} partial catalogs to fusion.")
    
    # Loading all tables
    tables = [Table.read(part) for part in catalog_parts]
    
    # Stacking all tables in the same
    final_table = vstack(tables)
    
    print(f"Final table has {len(final_table)} rows.")

    # Moving spectra to same directory
    print(f"Moving individual espectra to {pre_espectra_dir_out}...")
    
    # New routes list
    new_spec_paths = []

    for row in tqdm(final_table, desc="Moving spectra"):
        old_path = row['SPEC_PATH']
        spectrum_filename = os.path.basename(old_path)
        new_path = os.path.join(pre_espectra_dir_out, spectrum_filename)
        
        try:
            # Moving files
            shutil.move(old_path, new_path)
        except FileNotFoundError:
            print(f"[WARNING]: {old_path} not found.")
        except Exception as e:
            print(f"[ERROR] moving {old_path}: {e}")
            
        # Saving the new path
        new_spec_paths.append(new_path)
    
    # Overwriting 'SPEC_PATH' cwith the correct paths
    final_table['SPEC_PATH'] = new_spec_paths
    
    # Saving the final catalog
    final_catalog_path = os.path.join(catalog_file_dir, train_catalog_file_name)
    final_table.write(final_catalog_path, overwrite=True)
    
    print("\n#--- Merging completed ---#")
    print(f"Training catalog saving in: {final_catalog_path}")
    print(f"All spectra moving to: {pre_espectra_dir_out}")
    
    # Removing partial directories
    #print("Removing partial directories...")
    #for i in range(len(catalog_parts)):
    #    try:
    #        os.remove(os.path.join(catalog_file_dir, f"base_EuclidQ1_DESIDR1_TRAIN_p{i}.fits"))
    #        os.rmdir(os.path.join(catalog_file_dir, f"desi_dr1_training_spectra_p{i}"))
    #    except OSError as e:
    #        print(f"[ERROR] Removing partial directory {i} not possible: {e}")