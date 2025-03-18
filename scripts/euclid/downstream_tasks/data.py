from astropy.io import fits
import numpy as np
import os

class DataLoader:
    def __init__(self, folder_path, embedding_file, idx_file, catalog_path, txt_file):
        self.folder_path = folder_path
        self.embedding_file = embedding_file
        self.idx_file = idx_file
        self.catalog_path = catalog_path
        self.txt_file = txt_file

    def load_data(self, parameter):
        # Load catalog data
        with fits.open(self.catalog_path) as hdul:
            catalog_data = hdul[1].data

        # Load file names
        with open(os.path.join(self.folder_path, self.txt_file), "r") as f:
            file_names = [line.strip() for line in f.readlines()]

        # Load embeddings and indices
        zss = np.load(os.path.join(self.folder_path, self.embedding_file))
        idxs = np.load(os.path.join(self.folder_path, self.idx_file))
        file_names_selected = [file_names[idx] for idx in idxs]

        # Create a mapping from catalog name to index
        catalog_name_to_idx = {name: i for i, name in enumerate(catalog_data['VIS_name'])}

        # Load embeddings and labels
        embeddings, labels = [], []
        for file_name in file_names_selected:
            catalog_idx = catalog_name_to_idx.get(file_name)
            if catalog_idx is None:
                continue  # Skip if file name is not in catalog

            if parameter == 'PhotoZ':
                if not np.isnan(catalog_data['Z'][catalog_idx]):
                    embeddings.append(zss[file_names_selected.index(file_name)])
                    labels.append(catalog_data['Z'][catalog_idx])

            elif parameter == 'logM':
                if (not np.isnan(catalog_data['LOGM'][catalog_idx]) and
                    catalog_data['CHI2'][catalog_idx] < 17 and
                    catalog_data['LOGM'][catalog_idx] > 0 and
                    catalog_data['LOGM_ERR'][catalog_idx] < 0.25):
                    embeddings.append(zss[file_names_selected.index(file_name)])
                    labels.append(catalog_data['LOGM'][catalog_idx])

            elif parameter == 'smooth':
                smooth_value = catalog_data['smooth_or_featured_smooth'][catalog_idx]
                if not np.isnan(smooth_value):
                    label = 1 if smooth_value >= 50 else 0
                    labels.append(label)
                    embeddings.append(zss[file_names_selected.index(file_name)])

        return np.array(embeddings), np.array(labels)

# Function to sample the dataset at different percentages
def get_sampled_data(embeddings, labels, percentage, random_seed=42):
    total_samples = len(embeddings)
    sample_size = int(total_samples * percentage / 100)
    
    # Set the random seed for reproducibility
    np.random.seed(random_seed)
    
    sampled_indices = np.random.choice(total_samples, sample_size, replace=False)
    sampled_embeddings = np.array([embeddings[i] for i in sampled_indices])
    sampled_labels = np.array([labels[i] for i in sampled_indices])
    
    print(f"Sample size for {percentage}%: {len(sampled_labels)}")
    
    return sampled_embeddings, sampled_labels
