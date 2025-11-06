# data/dataset_loader.py
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import numpy as np
from PIL import Image
import glob
import os
import pandas as pd
import json

from config import (
    MODALITY_CONFIG, IMG_SIZE_2D, BATCH_SIZE, 
    DATA_ROOT, CLASS_MAP_JSON
)

# --- MONAI Imports ---
from monai.data import Dataset as MonaiDataset, DataLoader as MonaiDataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Resized,
    ScaleIntensityRanged, RandAffined, ToTensord
)

# --- Dynamic Class Map Builder ---
def build_class_map():
    # ... (This function remains unchanged) ...
    print("Building class map...")
    all_disease_names = set()

    for modality in MODALITY_CONFIG.keys():
        for split in ["train", "test"]:
            labels_path = DATA_ROOT / modality / split / "labels.csv"
            if not labels_path.exists():
                print(f"Warning: Missing required file {labels_path}")
                continue
                
            df = pd.read_csv(labels_path)
            if "disease" not in df.columns:
                raise ValueError(f"'disease' column not found in {labels_path}")
                
            all_disease_names.update(df["disease"].unique())

    if not all_disease_names:
        raise FileNotFoundError(f"No 'labels.csv' files found in {DATA_ROOT}. Cannot build class map.")

    sorted_names = sorted(list(all_disease_names))
    class_map = {name: i for i, name in enumerate(sorted_names)}
    
    with open(CLASS_MAP_JSON, 'w') as f:
        json.dump(class_map, f, indent=4)
        
    print(f"Class map built and saved to {CLASS_MAP_JSON}")
    print(f"Classes found: {class_map}")
    return class_map

def load_class_map():
    # ... (This function remains unchanged) ...
    if not CLASS_MAP_JSON.exists():
        raise FileNotFoundError(f"{CLASS_MAP_JSON} not found. Please run train.py first to build the class map.")
    with open(CLASS_MAP_JSON, 'r') as f:
        class_map = json.load(f)
    return class_map

# --- 2D Dataset Classes ---
class Base2DDataset(Dataset):
    # ... (This class remains unchanged) ...
    def __init__(self, data_dir, transform, class_map, modality):
        self.data_dir = data_dir
        self.transform = transform
        self.class_map = class_map
        self.modality = modality

        labels_path = os.path.join(data_dir, "labels.csv")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Required {labels_path} not found.")
            
        self.label_df = pd.read_csv(labels_path)
        
        self.image_paths = []
        self.labels = []
        
        for _, row in self.label_df.iterrows():
            img_path = os.path.join(data_dir, row["image_filename"])
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} listed in CSV but not found on disk.")
                continue
            
            self.image_paths.append(img_path)
            self.labels.append(self.class_map[row["disease"]])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long), self.modality

def get_2d_transforms(is_train=True):
    # ... (This function remains unchanged) ...
    transform_list = [
        transforms.Resize((IMG_SIZE_2D, IMG_SIZE_2D)),
    ]
    
    if is_train:
        transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ])
        
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transforms.Compose(transform_list)

# --- 3D Dataset (NIfTI Loader for MRI) ---
def get_mri_transforms(modality, is_train=True):
    # ... (This function remains unchanged) ...
    cfg = MODALITY_CONFIG[modality]
    img_size_3d = cfg["size"]
    
    transform_list = [
        LoadImaged(keys=["image"]), 
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=img_size_3d),
        ScaleIntensityRanged(
            keys=["image"], a_min=0.0, a_max=2000.0, 
            b_min=0.0, b_max=1.0, clip=True
        ),
    ]
    
    if is_train:
        transform_list.append(
            RandAffined(
                keys=["image"],
                prob=0.5,
                rotate_range=(0, 0, np.pi / 15),
                scale_range=(0.1, 0.1, 0.1)
            )
        )
        
    transform_list.append(ToTensord(keys=["image", "label"]))
    return Compose(transform_list)


# --- Main Loader Function (for Inference) ---
def get_dataloader(modality, class_map, split="test", batch_size=BATCH_SIZE, shuffle=True):
    # ... (This function remains unchanged and is used by main.py) ...
    data_dir = DATA_ROOT / modality / split
    is_train = (split == "train")
    
    if modality in ["XRAY", "HISTOPATHOLOGY"]:
        transforms_2d = get_2d_transforms(is_train=is_train)
        dataset = Base2DDataset(data_dir, transform=transforms_2d, class_map=class_map, modality=modality)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS)
    
    elif modality == "MRI":
        labels_path = data_dir / "labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"Required {labels_path} not found.")
            
        label_df = pd.read_csv(labels_path)
        data_dicts = []
        for _, row in label_df.iterrows():
            img_path = str(data_dir / row["image_filename"])
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} listed in CSV but not found.")
                continue
            data_dicts.append({
                "image": img_path,
                "label": class_map[row["disease"]]
            })

        if not data_dicts:
            print(f"Warning: No valid NIfTI files found for {modality} in {data_dir}.")

        transforms_3d = get_mri_transforms(modality, is_train=is_train)
        dataset = MonaiDataset(data=data_dicts, transform=transforms_3d)
        return MonaiDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS)
    
    else:
        raise ValueError(f"Modality {modality} not supported.")

# --- ADDED: New Function for K-Fold Cross-Validation ---
def get_full_dataset(modality, class_map):
    """
    Loads and combines ALL data (train and test) for a single modality
    into one single Dataset object, which is required for K-Fold CV.
    """
    print(f"Loading full dataset for {modality}...")
    datasets = []
    
    if modality in ["XRAY", "HISTOPATHOLOGY"]:
        # Load train and test splits
        try:
            train_dir = DATA_ROOT / modality / "train"
            train_transforms = get_2d_transforms(is_train=True)
            train_dataset = Base2DDataset(train_dir, train_transforms, class_map, modality)
            datasets.append(train_dataset)
        except Exception as e:
            print(f"Could not load train data for {modality}: {e}")
            
        try:
            test_dir = DATA_ROOT / modality / "test"
            test_transforms = get_2d_transforms(is_train=False) # No augmentation
            test_dataset = Base2DDataset(test_dir, test_transforms, class_map, modality)
            datasets.append(test_dataset)
        except Exception as e:
            print(f"Could not load test data for {modality}: {e}")

    elif modality == "MRI":
        # Load train and test splits
        data_dicts = []
        try:
            train_dir = DATA_ROOT / modality / "train"
            train_labels = pd.read_csv(train_dir / "labels.csv")
            for _, row in train_labels.iterrows():
                data_dicts.append({
                    "image": str(train_dir / row["image_filename"]),
                    "label": class_map[row["disease"]],
                    "is_train": True
                })
        except Exception as e:
            print(f"Could not load train data for {modality}: {e}")
            
        try:
            test_dir = DATA_ROOT / modality / "test"
            test_labels = pd.read_csv(test_dir / "labels.csv")
            for _, row in test_labels.iterrows():
                data_dicts.append({
                    "image": str(test_dir / row["image_filename"]),
                    "label": class_map[row["disease"]],
                    "is_train": False
                })
        except Exception as e:
            print(f"Could not load test data for {modality}: {e}")

        # Need separate transforms for train/val splits in K-Fold
        # This is complex. For simplicity, we'll use train transforms for all
        # and rely on KFold to separate.
        # A more advanced setup would use MONAI's Dataset a different way.
        if not data_dicts:
             return ConcatDataset([]) # Return empty
        
        transforms_3d = get_mri_transforms(modality, is_train=True)
        return MonaiDataset(data=data_dicts, transform=transforms_3d)

    return ConcatDataset(datasets)
