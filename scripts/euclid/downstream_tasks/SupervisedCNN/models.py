import numpy as np
import tensorflow as tf
import torch
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import seaborn as sns
from data import create_train_test_for_percentage
from data import create_train_test_for_percentage_vis_nisp

# Disable GPU (for debugging or running on CPU)
#tf.config.set_visible_devices([], 'GPU')


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)


def create_model(image_height, image_width, channels=1):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, channels)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')  
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model
        
def create_model_binary_classification(image_height, image_width, channels=1):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, channels)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model_on_subset(image_paths, labels, n_mc_runs, parameter, modality, percentage=100):
    # Prepare the dataset
    train_dataset, test_dataset = create_train_test_for_percentage(image_paths, labels, percentage)
    # Print progress about dataset sizes
    print(f"Initialized training dataset with {len(train_dataset)} images.")
    print(f"Initialized testing dataset with {len(test_dataset)} images.")


    # Create model (CNN)
    if parameter == 'Smooth':
        model = create_model_binary_classification(image_height=224, image_width=224,channels=1)
    else:
        model = create_model(image_height=224, image_width=224,channels=1)        
    print("model created")
    
    # Add a progress bar callback
    progbar = tf.keras.utils.Progbar(len(train_dataset))

    # Save the model during training
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(f'{parameter}_Classification_Supervised_{modality}_{percentage}%_{n_mc_runs}.keras', save_best_only=True)
    print("checkpoint writtten")

    # Training the model and monitoring the progress
    history = model.fit(train_dataset, validation_data=test_dataset, epochs=10, callbacks=[checkpoint_cb])

    return model, test_dataset  # Include test_dataset for evaluation later
    

def train_model_on_subset_vis_nisp(vis, nir_y, nir_j, nir_h, labels, n_mc_runs, parameter, modality, percentage=100):
    train_dataset, test_dataset = create_train_test_for_percentage_vis_nisp(vis, nir_y, nir_j, nir_h, labels, percentage)
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")
    
    if parameter == 'Smooth':
        model = create_model_binary_classification(image_height=224, image_width=224, channels=4)
    else:
        model = create_model(image_height=224, image_width=224, channels=4)      
    print("model created") 

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        f'{parameter}_Classification_Supervised_{modality}_{percentage}%_{n_mc_runs}.keras', save_best_only=True
    )
    model.fit(train_dataset, validation_data=test_dataset, epochs=10, callbacks=[checkpoint_cb])

    return model, test_dataset
