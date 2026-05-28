import os
import glob
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def fix_metadata_instant(folder_path, split_name):
    logging.info(f"Instant repair for: {folder_path} ({split_name})")
    
    arrow_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(folder_path, "*.arrow"))])
    if not arrow_files:
        logging.warning(f"No arrow files in {folder_path}")
        return

    info = {
        "builder_name": "arrow", "config_name": "default",
        "version": {"version_str": "0.0.0", "major": 0, "minor": 0, "patch": 0},
        "splits": {split_name: {"name": split_name, "num_bytes": 0, "num_examples": 0, "dataset_name": "arrow"}},
    }
    
    state = {
        "_data_files": [{"filename": f} for f in arrow_files],
        "_fingerprint": "manual_fix_" + folder_path.replace("/", "_")[-50:],
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": split_name
    }

    with open(os.path.join(folder_path, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    with open(os.path.join(folder_path, "state.json"), "w") as f:
        json.dump(state, f, indent=2)
        
    logging.info(f"Done! Metadata created for {len(arrow_files)} files.")

if __name__ == "__main__":
    base_path = "/home/valonso/iac18_aasensio_shared/euclid_dr1/processed_data_arrow_interpolated_tokenized"
    
    for split in ["train", "test"]:
        folders = sorted(glob.glob(os.path.join(base_path, f"{split}_*")))
        for folder in folders:
            fix_metadata_instant(folder, split)