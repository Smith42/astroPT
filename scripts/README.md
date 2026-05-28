# AstroPT Scripts Directory

This directory contains the launchers and support tools for the AstroPT model. The codebase has been modularized to separate core training logic from data preparation and science analysis.

## Core Scripts
- **[train.py](train.py)**: The primary entry point for training. Uses the refactored `Trainer` class and supports YAML configuration files.
- **[eval.py](eval.py)**: Standalone evaluation script. Loads checkpoints and calculates zero-shot reconstruction metrics or exports raw predictions.

## Subdirectories

### 🛠️ [preprocessing/](preprocessing/)
Tools for data cleaning, interpolation, normalization, and converting datasets to the Arrow format.
*Includes: dataset filters, interpolation scripts, and normalization calculators.*

### 🔍 [inference/](inference/)
Scripts for using trained models to extract embeddings or generate predictions from new data.
*Includes: embedding extraction for AstroPT, AstroCLIP, and ResNet baselines.*

### 🎯 [probing/](probing/)
Downstream task evaluation using linear probes or MLPs to validate the scientific representations learned by the model.

### 📊 [baselines/](baselines/)
Scripts for training reference models (e.g., supervised ResNets or Spectra-only models) used for scientific comparison.

### 🧪 [experimental/](experimental/)
New models or training techniques currently under development (e.g., Spectral Diffusion).

### 🏛️ [legacy/](legacy/)
Archive of old monolithic scripts, AION-specific trainers, and code that is no longer part of the main production pipeline.

---
**Note**: For science-heavy analysis, plotting, and anomaly detection, please refer to the top-level `analysis/` directory.
