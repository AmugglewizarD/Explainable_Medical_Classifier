# config.py
"""
Central configuration for dataset paths, model names, hyperparameters.

*** KAGGLE OPTIMIZED VERSION ***
- DATA_ROOT is now a dictionary mapping modalities to their Kaggle input paths.
- All output paths (ROOT, SAVE_PATH, RESULTS_DIR) point to /kaggle/working/
- Optimized BATCH_SIZE, K_FOLDS, and NUM_WORKERS.
"""
from pathlib import Path
import torch
import os

# ROOT is the writable directory in Kaggle
ROOT = Path("/kaggle/working/")

# Data
# --- KAGGLE CHANGE ---
# This is the most important change.
# You MUST update these paths to match the dataset "slugs" you add to your notebook.
# The structure *inside* these paths should be:
# /kaggle/input/your-xray-slug/train_labels.csv
# /kaggle/input/your-xray-slug/test_labels.csv
# /kaggle/input/your-xray-slug/images/00000013_005.png (etc.)

DATA_ROOT = {
    # Example: /kaggle/input/nih-chest-xray-dataset
    "XRAY": Path("/kaggle/input/your-xray-dataset-name/"), 
    
    # Example: /kaggle/input/pcam-histopathology
    "HISTOPATHOLOGY": Path("/kaggle/input/your-histo-dataset-name/"), 
    
    # Example: /kaggle/input/mri-head-scans
    "MRI": Path("/kaggle/input/your-mri-dataset-name/") 
}

# All output files go to /kaggle/working/
CLASS_MAP_JSON = ROOT / "data" / "class_map.json"

# OPTIMIZED: Increased batch size (AMP helps)
BATCH_SIZE = 32 
# OPTIMIZED: Kaggle notebooks typically have 4 vCPUs. 
# Set this to 0 or 2 if you get dataloader errors.
NUM_WORKERS = 4 

# --- 2D Configuration ---
IMG_SIZE_2D = 224

# --- 3D Configuration ---
IMG_SIZE_3D = (96, 96, 96) # (Depth, Height, Width)

# --- Multimodal Configuration ---
MODALITY_CONFIG = {
    "XRAY": {"dim": 2, "channels": 3, "size": IMG_SIZE_2D, "model_type": "2D"},
    "HISTOPATHOLOGY": {"dim": 2, "channels": 3, "size": IMG_SIZE_2D, "model_type": "2D"},
    "MRI": {"dim": 3, "channels": 1, "size": IMG_SIZE_3D, "model_type": "3D"},
}

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-4
EPOCHS = 10

# OPTIMIZED: Reduced K-Folds for faster CV
K_FOLDS = 3 

# --- Paths ---
SAVE_PATH = ROOT / "models" / "weights"
RESULTS_DIR = ROOT / "results" 

# FIXED (Fix 1): Changed to a directory path for save_pretrained
MODEL_2D_CHECKPOINT = SAVE_PATH / "vit2d_checkpoint"   
MODEL_3D_CHECKPOINT = SAVE_PATH / "best_model_3d.pth"  

# Model
VIT_2D_PRETRAINED = "google/vit-base-patch16-224-in21k"
VIT_3D_PRETRAINED = "monai/vitautoenc" # Not used, we train from scratch

# Explainability
LIME_SAMPLES = 1000
SHAP_NSAMPLES = 100 

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
