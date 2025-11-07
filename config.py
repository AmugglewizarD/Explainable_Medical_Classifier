from pathlib import Path

IMG_SIZE_2D = 224
BATCH_SIZE = 16
NUM_WORKERS = 2

DATA_ROOT = Path("data")
CLASS_MAP_JSON = "class_map.json"

# Kaggle default paths (override in notebook/Colab)
KAGGLE_PATHS = {
    "XRAY": Path("/kaggle/input/data"),
    "SKIN": Path("/kaggle/input/skin-cancer-mnist-ham10000"),
    "MRI": Path("/kaggle/input/brain-mri-images-for-brain-tumor-detection"),
}
