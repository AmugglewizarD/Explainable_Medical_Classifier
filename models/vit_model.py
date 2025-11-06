# models/vit_model.py
"""
Model wrappers for Vision Transformers (2D and 3D) and a central Multimodal router.
"""

from transformers import ViTForImageClassification, AutoFeatureExtractor
import torch
import torch.nn.functional as F
import numpy as np
from config import (
    VIT_2D_PRETRAINED, NUM_LABELS, DEVICE, 
    MODALITY_CONFIG, IMG_SIZE_3D,
    MODEL_2D_CHECKPOINT, MODEL_3D_CHECKPOINT
)
from PIL import Image
import os

from monai.networks.nets import ViT
from monai.utils import ensure_tuple

# --- 2D Model Wrapper (HuggingFace) ---
class ViT2DWrapper:
    def __init__(self, num_labels, load_from_scratch=False):
        if load_from_scratch:
            # Initialize for training
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(VIT_2D_PRETRAINED)
            self.model = ViTForImageClassification.from_pretrained(
                VIT_2D_PRETRAINED, 
                num_labels=num_labels,
                ignore_mismatched_sizes=True # Allow re-sizing classifier head
            )
        else:
            # Initialize for inference (load fine-tuned model)
            if not MODEL_2D_CHECKPOINT.exists():
                raise FileNotFoundError(f"Trained model not found at {MODEL_2D_CHECKPOINT}. Please run train.py first.")
            print(f"Loading trained 2D model from {MODEL_2D_CHECKPOINT}...")
            # We must load the feature extractor from the same path
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_2D_CHECKPOINT)
            self.model = ViTForImageClassification.from_pretrained(MODEL_2D_CHECKPOINT)
            
        self.model.to(DEVICE)
        self.num_labels = num_labels

    def preprocess(self, pil_images):
        inputs = self.feature_extractor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)
        return pixel_values

    @torch.no_grad()
    def predict(self, pil_images):
        self.model.eval()
        if not pil_images:
            return np.array([]).reshape(0, self.num_labels)
        pixel_values = self.preprocess(pil_images)
        outputs = self.model(pixel_values)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs
    
    def save_checkpoint(self, path):
        """Saves the fine-tuned model and feature extractor."""
        print(f"Saving 2D model checkpoint to {path}")
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.feature_extractor.save_pretrained(path)

# --- 3D Model Wrapper (MONAI) ---
class ViT3DWrapper:
    """Real 3D Vision Transformer wrapper using MONAI."""
    def __init__(self, num_labels, load_from_scratch=False):
        cfg_3d = MODALITY_CONFIG["MRI"] # Use MRI config as default
        in_channels = cfg_3d["channels"]
        img_size_3d = ensure_tuple(cfg_3d["size"])

        self.model = ViT(
            in_channels=in_channels, img_size=img_size_3d, patch_size=(16, 16, 16),
            hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12,
            classification=True, num_classes=num_labels, dropout_rate=0.1,
        ).to(DEVICE)
        
        if not load_from_scratch:
            # Initialize for inference
            if not MODEL_3D_CHECKPOINT.exists():
                raise FileNotFoundError(f"Trained model not found at {MODEL_3D_CHECKPOINT}. Please run train.py first.")
            print(f"Loading trained 3D model from {MODEL_3D_CHECKPOINT}...")
            self.load_checkpoint(MODEL_3D_CHECKPOINT)
            
        self.num_labels = num_labels

    def preprocess(self, volume_tensors):
        volume_batch = torch.stack(volume_tensors).to(DEVICE)
        return volume_batch # Shape (B, C, D, H, W)

    @torch.no_grad()
    def predict(self, volume_tensors):
        self.model.eval()
        if not volume_tensors:
            return np.array([]).reshape(0, self.num_labels)
        
        volume_batch = self.preprocess(volume_tensors)
        logits, _ = self.model(volume_batch)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def save_checkpoint(self, path):
        """Saves the fine-tuned 3D model weights."""
        print(f"Saving 3D model checkpoint to {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        """Loads fine-tuned 3D model weights."""
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))

# --- Multimodal Router ---
class MultimodalViTWrapper:
    """Routes inference to the correct specialized model based on modality."""
    def __init__(self, num_labels, load_from_scratch=False):
        self.models = {
            "2D": ViT2DWrapper(num_labels, load_from_scratch),
            "3D": ViT3DWrapper(num_labels, load_from_scratch)
        }
        if load_from_scratch:
            print("Wrapper initialized in TRAINING mode (from scratch weights).")
        else:
            print("Wrapper initialized in INFERENCE mode (loading fine-tuned weights).")

    def predict(self, input_data, modality_type):
        model_type = MODALITY_CONFIG.get(modality_type, {}).get("model_type")
        if not model_type or model_type not in self.models:
            raise ValueError(f"Unsupported modality: {modality_type}")
        return self.models[model_type].predict(input_data)
    
    def get_model(self, modality_type):
        model_type = MODALITY_CONFIG.get(modality_type, {}).get("model_type")
        return self.models[model_type].model

    def get_wrapper(self, modality_type):
        model_type = MODALITY_CONFIG.get(modality_type, {}).get("model_type")
        return self.models[model_type]
