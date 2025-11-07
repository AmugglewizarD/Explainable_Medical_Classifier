# config.py
"""
Central configuration for dataset paths, model names, hyperparameters.

*** OPTIMIZED VERSION ***
- Increased BATCH_SIZE (requires VRAM, but faster with AMP)
- Increased NUM_WORKERS (faster data loading)
- Decreased K_FOLDS (significant time save on CV)
"""
from pathlib import Path
import torch

ROOT = Path(__file__).parent

# Data
DATA_ROOT = ROOT / "data" / "raw"
CLASS_MAP_JSON = ROOT / "data" / "class_map.json"

# OPTIMIZED: Increased batch size. AMP helps reduce memory,
# allowing for larger batches, which trains faster.
# If you get "Out of Memory" errors, reduce this to 24 or 16.
BATCH_SIZE = 32 

# OPTIMIZED: Increased num_workers.
# This uses more CPU cores to load data, preventing
# the GPU from waiting.
NUM_WORKERS = 8 # Adjust based on your CPU cores (e.g., os.cpu_count())

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

# OPTIMIZED: Reduced K-Folds.
# This is the biggest time-saver for the K-Fold process.
# 5-fold CV is robust but time-consuming. 3-fold is a common
# compromise for faster iteration without sacrificing *too much*
# validation robustness.
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
