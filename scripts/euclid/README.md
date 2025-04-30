# Downstream Task Scripts  

This repository provides exemplary scripts associated with **AstroPT models** trained on Euclid data.  
The models are hosted on **[Hugging Face](https://huggingface.co/collections/msiudek/astropt-euclid-67d061928ac0a447265ac8b8)**.  

## Available Models  
We provide three AstroPT models trained on different input data:  
- **VIS** – trained on VIS images,  
- **VIS_NISP** – trained on VIS and NISP (NIR H, J, and Y) images,  
- **VIS_NISP_SED** – trained on VIS, NISP images, and Spectral Energy Distributions (SEDs).  

These models are accompanied by corresponding **datasets** containing FITS images and SEDs.  
Additionally, the `metadata` directory includes **physical and morphological properties** of Euclid galaxies, complemented with:  
- **Spectroscopic redshifts (spec-z)**, from DESI EDR (*[DESI Collaboration et al. 2024](https://ui.adsabs.harvard.edu/abs/2024AJ....168...58D/abstract)*),
- **Stellar masses** from the DESI EDR VAC of physical properties (*[Siudek et al. 2024](https://ui.adsabs.harvard.edu/abs/2024A%26A...691A.308S/abstract)*).

## Repository Structure  
This repository includes:  
- **Plots**: Additional figures used in *Siudek et al. 2025* (Figures **2, 10, 12, 14**)  
- **Visualizations**: UMAP plots from *Siudek et al. 2025* (Figures **6, 7, 8, 9, C.2**)  
- **Downstream Tasks**: Scripts for:  
  - **Photo-z prediction** , 
  - **Stellar mass estimation** , 
  - **Morphological classification**,  
  - **Anomaly & similarity search** . 

## Downstream Task Scripts  

### 1. **Anomaly Detection**  
- **Script**: `anomaly_detection.py`  
- **Methods**: Utilizes **Isolation Forest** and **Local Outlier Factor** algorithms.  

### 2. **Similarity Search**  
- **Script**: `similarity_search.py`  
- **Functionality**: Finds the top `n` closest galaxies in the embedding space to a given query galaxy.  

### 3. **Photo-z Prediction**  
- **Notebook**: `train_photo_z.ipynb`  
- **Modality Options**: `VIS`, `VIS_NISP`, or `VIS_NISP_SED`  
- **Models Available**:  
  - Linear model ,
  - MLP model.

### 4. **Stellar Mass Estimation**  
- **Notebook**: `train_logM.ipynb`  
- **Method**: Regression task using the same approach as photo-z prediction.  

### 5. **Morphological Classification**  
- **Notebook**: `train_Smooth.ipynb`  
- **Task**: Classification of early vs. late-type galaxies.  

### 6. **Euclid NNPZ Performance Evaluation**  
- **Script**: `Euclid_NNPZ_accuracy.py`  
- **Functionality**: Computes Euclid NNPZ photo-z and stellar mass performance . 

### 7. **Supervised Training for Photo-z & Stellar Mass Prediction**  
- **Folder**: `SupervisedCNN/`  
- **Training Parameters**:  
  - `PhotoZ` → Photo-z prediction,  
  - `logM` → Stellar mass prediction, 
  - `Smooth` → Early vs. late-type galaxy classification.  
- **Task Types**:  
  - **Regression** (`task_type = "regression"`)  
  - **Classification** (`task_type = "classification"`)  
- **Input Modalities**:  
  - **VIS** (`modality = 'VIS'`), 
  - **VIS_NISP** (`modality = 'VIS_NISP'`).  



