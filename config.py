# config.py
"""
Central configuration for dataset paths, model names, hyperparameters.
"""
from pathlib import Path
import torch

ROOT = Path(__file__).parent

# Data
DATA_ROOT = ROOT / "data" / "raw"
BATCH_SIZE = 32
NUM_WORKERS = 4

# --- 2D Configuration ---
IMG_SIZE_2D = 224  # ViT 2D input size

# --- 3D Configuration ---
# MONAI ViT standard input size
IMG_SIZE_3D = (96, 96, 96) # (Depth, Height, Width)

# --- Multimodal Configuration ---
MODALITY_CONFIG = {
    "XRAY": {"dim": 2, "channels": 3, "size": IMG_SIZE_2D, "model_type": "2D"},
    "CT": {"dim": 3, "channels": 1, "size": IMG_SIZE_3D, "model_type": "3D"}, 
    "MRI": {"dim": 3, "channels": 1, "size": IMG_SIZE_3D, "model_type": "3D"},
}
DEFAULT_MODALITY = "XRAY" 

# Model
VIT_2D_PRETRAINED = "google/vit-base-patch16-224-in21k"  # 2D ViT
VIT_3D_PRETRAINED = "monai/vitautoenc" # MONAI ViT (conceptual path)
NUM_LABELS = 2  # Assuming binary classification

# Explainability
LIME_SAMPLES = 1000
SHAP_BACKGROUND_SAMPLES = 50

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
