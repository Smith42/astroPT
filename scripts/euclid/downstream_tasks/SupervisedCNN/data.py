import numpy as np
import tensorflow as tf
import torch
from astropy.io import fits
import os
from sklearn.model_selection import train_test_split
                       
class GalaxyImageDataset(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=32, image_size=(224, 224), shuffle=True):
        print(f"Initializing dataset with {len(image_paths)} images.")
        
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        #return int(np.floor(len(self.image_paths) / self.batch_size))
        # Ensure at least 1 batch is returned if there are any images
        return max(1, int(np.ceil(len(self.image_paths) / self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.image_paths))
        print(f"Start index: {start_idx}, End index: {end_idx}")
        if start_idx >= len(self.image_paths):  # Edge case for empty batches
            raise IndexError("Index out of range in __getitem__.")    
        batch_images = np.zeros((end_idx - start_idx, *self.image_size, 1), dtype=np.float32)
        batch_labels = np.zeros((end_idx - start_idx,), dtype=np.float32)
    
        for i, idx in enumerate(range(start_idx, end_idx)):
            with fits.open(self.image_paths[idx]) as hdul:
                image_data = hdul[0].data.astype(np.float32)
            image_data = self._process_image(image_data)  # Preprocessing
            batch_images[i] = image_data
            batch_labels[i] = self.labels[idx]
    
        return batch_images, batch_labels
      
    def _process_image(self, image_data):
        """Resize and normalize the FITS image."""
        if len(image_data.shape) == 2:  
            image_data = np.expand_dims(image_data, axis=-1)  
        image_data = tf.image.resize(image_data, self.image_size)
        image_data = image_data / np.max(image_data)  
        return image_data

    @staticmethod
    def get_image_paths_and_labels(txt_file, label_file, parameter, task_type):
        image_paths = []
        labels = []
        valid_samples = 0  # Counter for valid samples
    
        with fits.open(label_file) as hdul:
            label_data = hdul[1].data
            if parameter == 'Smooth':
                label_dict = {name: label for name, label in zip(label_data['VIS_name'], label_data['smooth_or_featured_smooth']) if not np.isnan(label)}
            elif parameter == 'PhotoZ':
                label_dict = {name: label for name, label in zip(label_data['VIS_name'], label_data['Z']) if not np.isnan(label)}
            elif parameter == 'logM':
                label_dict = {name: label for name, label in zip(label_data['VIS_name'], label_data['LOGM']) if (not np.isnan(label) and
                    label_data['CHI2'][list(label_data['VIS_name']).index(name)] < 17 and
                    label > 0 and
                    label_data['LOGM_ERR'][list(label_data['VIS_name']).index(name)] < 0.25)
                }              
            else:
                print('No label for data given, specify parameter')     
            
    
        with open(txt_file, 'r') as f:
            for i, line in enumerate(f):
                image_name = line.strip()
                fits_path = os.path.join('../../../Q1_data/VIS/', image_name)
                # Retrieve the label for this image
                label = label_dict.get(image_name, np.nan)  # Default to np.nan if not found
                
                if not np.isnan(label):  # Only include if the label (Z) is valid
                    if task_type=='classification':
                        binary_label = 1 if label >= 50 else 0
                    else:
                        binary_label = label
                    image_paths.append(fits_path)
                    labels.append(binary_label)
                    valid_samples += 1
                    
                    #if valid_samples <= 5:  # Print a few sample image paths and labels for debugging
                        #print(f"Sample {valid_samples}: {image_name} -> Label: {label}")
    
        print(f"Total valid samples loaded: {valid_samples}")
        #print(labels[:10])
        return image_paths, labels


class GalaxyImageDataset_VIS_NISP(tf.keras.utils.Sequence):
    def __init__(self, vis_paths, nir_y_paths, nir_j_paths, nir_h_paths, labels, batch_size=32, image_size=(224, 224), shuffle=True):
        print(f"Initializing dataset with {len(vis_paths)} images.")
        
        self.vis_paths = vis_paths
        self.nir_y_paths = nir_y_paths
        self.nir_j_paths = nir_j_paths
        self.nir_h_paths = nir_h_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.vis_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.vis_paths) / self.batch_size))  # Use ceil to include all batches

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.vis_paths))
        
        if start_idx >= len(self.vis_paths):  # Avoid accessing beyond data
            raise IndexError("Index out of range")

        batch_images = np.zeros((end_idx - start_idx, *self.image_size, 4), dtype=np.float32)  # 4 channels
        batch_labels = np.zeros((end_idx - start_idx,), dtype=np.float32)
    
        for i, idx in enumerate(range(start_idx, end_idx)):
            vis_image = self._load_and_process_image(self.vis_paths[idx])
            nir_y_image = self._load_and_process_image(self.nir_y_paths[idx])
            nir_j_image = self._load_and_process_image(self.nir_j_paths[idx])
            nir_h_image = self._load_and_process_image(self.nir_h_paths[idx])
            
            # Stack VIS and NIR channels
            batch_images[i] = np.concatenate([vis_image, nir_y_image, nir_j_image, nir_h_image], axis=-1)
            batch_labels[i] = self.labels[idx]    
        if np.isnan(batch_images).any() or np.isinf(batch_images).any():
            print(f"Invalid data detected in batch {index}")
        if np.isnan(batch_labels).any() or np.isinf(batch_labels).any():
            print(f"Invalid labels detected in batch {index}")

        return batch_images, batch_labels

    def _load_and_process_image(self, file_path):
        """Load FITS image, resize, and normalize."""
        with fits.open(file_path) as hdul:
            image_data = hdul[0].data.astype(np.float32)
    
        if np.isnan(image_data).any() or np.isinf(image_data).any():
            print(f"Invalid values detected in file: {file_path}")
    
        if len(image_data.shape) == 2:  
            image_data = np.expand_dims(image_data, axis=-1)  
        image_data = tf.image.resize(image_data, self.image_size)
        max_val = np.max(image_data)
    
        if max_val > 0:
            image_data = image_data / max_val  # Normalize
        else:
            print(f"Warning: Zero max value in file: {file_path}")
            image_data = np.zeros_like(image_data)  # Fallback
    
        return image_data

    @staticmethod
    def get_image_paths_and_labels(all_txt_file, label_file, parameter, task_type):
        vis_paths, nir_y_paths, nir_j_paths, nir_h_paths, labels = [], [], [], [], []
        valid_samples = 0

        with fits.open(label_file) as hdul:
            label_data = hdul[1].data
            if parameter == 'Smooth':
                label_dict = {name: label for name, label in zip(label_data['VIS_name'], label_data['smooth_or_featured_smooth']) if not np.isnan(label)}
            elif parameter == 'PhotoZ':
                label_dict = {name: label for name, label in zip(label_data['VIS_name'], label_data['Z']) if not np.isnan(label)}
            elif parameter == 'logM':
                label_dict = {name: label for name, label in zip(label_data['VIS_name'], label_data['LOGM']) if (not np.isnan(label) and
                    label_data['CHI2'][list(label_data['VIS_name']).index(name)] < 17 and
                    label > 0 and
                    label_data['LOGM_ERR'][list(label_data['VIS_name']).index(name)] < 0.25)
                }              
            else:
                print('No label for data given, specify parameter')          
                   
        with open(all_txt_file, 'r') as f:
            for line in f:
                paths = line.strip().split(',')
                if len(paths) != 4:
                    print(f"Skipping line due to incorrect format: {line}")
                    continue
                
                vis_path, nir_y_path, nir_j_path, nir_h_path = paths
                image_name = os.path.basename(vis_path)  # Extract image name from VIS path
                label = label_dict.get(image_name, np.nan)
             
                if not np.isnan(label):  # Only include if the label (Z) is valid
                    if task_type=='classification':
                        binary_label = 1 if label >= 50 else 0
                    else:
                        binary_label = label
                    vis_paths.append(vis_path)
                    nir_y_paths.append(nir_y_path)
                    nir_j_paths.append(nir_j_path)
                    nir_h_paths.append(nir_h_path)
                    labels.append(binary_label)
                    valid_samples += 1
                    
                    #if valid_samples <= 5:
                        #print(f"Sample {valid_samples}:")
                        #print(f"  VIS: {vis_path}")
                        #print(f"  NIR-Y: {nir_y_path}")
                        #print(f"  NIR-J: {nir_j_path}")
                        #print(f"  NIR-H: {nir_h_path}")
                        #print(f"  Label: {label}")
        
        print(f"Total valid samples loaded: {valid_samples}")
        #print(labels[:10])
        return vis_paths, nir_y_paths, nir_j_paths, nir_h_paths, labels
        
# Function to sample the dataset at different percentages
def get_sampled_data(image_paths, labels, percentage, random_seed=42):
    np.random.seed(random_seed)
    total_samples = len(image_paths)
    sample_size = int(total_samples * percentage / 100)
    sampled_indices = np.random.choice(total_samples, sample_size, replace=False)
    sampled_image_paths = [image_paths[i] for i in sampled_indices]
    sampled_labels = labels[sampled_indices]
    print("sample size", len(sampled_labels))
    return sampled_image_paths, sampled_labels

def get_sampled_data_vis_nisp(vis_paths, nir_y_paths, nir_j_paths, nir_h_paths, labels, percentage, random_seed=42):
    np.random.seed(random_seed)
    total_samples = len(vis_paths)
    sample_size = int(total_samples * percentage / 100)
    sampled_indices = np.random.choice(total_samples, sample_size, replace=False)
    sampled_vis_paths = [vis_paths[i] for i in sampled_indices]
    sampled_nir_y_paths = [nir_y_paths[i] for i in sampled_indices]
    sampled_nir_j_paths = [nir_j_paths[i] for i in sampled_indices]
    sampled_nir_h_paths = [nir_h_paths[i] for i in sampled_indices]
    sampled_labels = [labels[i] for i in sampled_indices]
    print("Sample size:", len(sampled_labels))
    print(f"Total samples: {total_samples}, Sample size: {sample_size}, Percentage: {percentage}")
    if sample_size == 0:
        raise ValueError("Sample size is 0. Check percentage or dataset size.")

    return sampled_vis_paths, sampled_nir_y_paths, sampled_nir_j_paths, sampled_nir_h_paths, sampled_labels



# Function to create train-test splits and datasets
def create_train_test_for_percentage(image_paths, labels, percentage=100):
    sampled_image_paths, sampled_labels = get_sampled_data(image_paths, labels, percentage, random_seed=42)
    
    # Now split into train and test (80-20 split)
    train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(
        sampled_image_paths, sampled_labels, test_size=0.2, random_state=42
    )
    
    # Initialize datasets
    # For small datasets, reducing the batch_size dynamically based on the dataset size.
    train_dataset = GalaxyImageDataset(train_image_paths, train_labels, batch_size=min(32, len(train_image_paths)))
    test_dataset = GalaxyImageDataset(test_image_paths, test_labels, batch_size=min(32, len(test_image_paths)))

    
    return train_dataset, test_dataset

def create_train_test_for_percentage_vis_nisp(vis_paths, nir_y_paths, nir_j_paths, nir_h_paths, labels, percentage=100):
    sampled_vis_paths, sampled_nir_y_paths, sampled_nir_j_paths, sampled_nir_h_paths, sampled_labels = get_sampled_data_vis_nisp(
        vis_paths, nir_y_paths, nir_j_paths, nir_h_paths, labels, percentage, random_seed=42
    )
    (
        train_vis_paths, test_vis_paths,
        train_nir_y_paths, test_nir_y_paths,
        train_nir_j_paths, test_nir_j_paths,
        train_nir_h_paths, test_nir_h_paths,
        train_labels, test_labels
    ) = train_test_split(
        sampled_vis_paths, sampled_nir_y_paths, sampled_nir_j_paths, sampled_nir_h_paths, sampled_labels,
        test_size=0.2, random_state=42
    )

    train_dataset = GalaxyImageDataset_VIS_NISP(
        train_vis_paths, train_nir_y_paths, train_nir_j_paths, train_nir_h_paths, train_labels,
        batch_size=min(32, len(train_vis_paths))
    )
    test_dataset = GalaxyImageDataset_VIS_NISP(
        test_vis_paths, test_nir_y_paths, test_nir_j_paths, test_nir_h_paths, test_labels,
        batch_size=min(32, len(train_vis_paths))
    )
    return train_dataset, test_dataset
     
