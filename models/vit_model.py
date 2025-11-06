# models/vit_model.py
"""
Model wrappers for Vision Transformers (2D and 3D) and a central Multimodal router.
"""

from transformers import ViTForImageClassification, AutoFeatureExtractor
import torch
import torch.nn.functional as F
import numpy as np
from config import VIT_2D_PRETRAINED, VIT_3D_PRETRAINED, NUM_LABELS, DEVICE, MODALITY_CONFIG
from PIL import Image

# --- 2D Model Wrapper (Original, renamed) ---
class ViT2DWrapper:
    # ... (content of original ViTWrapper, renamed)
    def __init__(self, pretrained_name=VIT_2D_PRETRAINED, num_labels=NUM_LABELS):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_name)
        self.model = ViTForImageClassification.from_pretrained(pretrained_name, num_labels=num_labels)
        self.model.to(DEVICE)
        self.model.eval()

    def preprocess(self, pil_images):
        inputs = self.feature_extractor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)
        return pixel_values

    @torch.no_grad()
    def predict(self, pil_images):
        pixel_values = self.preprocess(pil_images)
        outputs = self.model(pixel_values)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def save(self, path):
        self.model.save_pretrained(path)
        self.feature_extractor.save_pretrained(path)

    def load(self, path):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(path)
        self.model = ViTForImageClassification.from_pretrained(path)
        self.model.to(DEVICE)
        self.model.eval()

# --- 3D Model Wrapper (Conceptual for CT/MRI) ---
class ViT3DWrapper:
    """Conceptual class for a 3D model, for demonstration purposes."""
    def __init__(self, pretrained_name=VIT_3D_PRETRAINED, num_labels=NUM_LABELS):
        print(f"Loading conceptual 3D model from: {pretrained_name}")
        # MOCK MODEL: Simple linear layer to process a flattened 3D volume
        input_features = MODALITY_CONFIG["CT"]["depth"] * MODALITY_CONFIG["CT"]["size"] * MODALITY_CONFIG["CT"]["size"]
        self.model = torch.nn.Linear(input_features, num_labels).to(DEVICE)
        self.model.eval()
        self.num_labels = num_labels

    def preprocess(self, volume_tensors):
        """Accepts a list of 3D torch tensors and flattens them for prediction."""
        # Volumes are expected as (D, H, W) -> flatten to 1D
        flat_volumes = torch.stack(volume_tensors).flatten(start_dim=1).float().to(DEVICE)
        return flat_volumes

    @torch.no_grad()
    def predict(self, volume_tensors):
        """Returns softmax probabilities for given list of 3D volume tensors."""
        if not volume_tensors:
            return np.array([]).reshape(0, self.num_labels)
        
        flat_volumes = self.preprocess(volume_tensors)
        
        if flat_volumes.shape[1] != self.model.in_features:
             # Handle mismatch due to mock data shapes
             return np.zeros((len(volume_tensors), self.num_labels)) 

        logits = self.model(flat_volumes)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

# --- Multimodal Router ---
class MultimodalViTWrapper:
    """Routes inference to the correct specialized model based on modality."""
    def __init__(self):
        self.models = {
            "2D": ViT2DWrapper(),
            "3D": ViT3DWrapper()
        }

    def predict(self, input_data, modality_type):
        """
        Predicts based on the modality_type.
        input_data can be a list of PIL images (2D) or list of Tensors (3D).
        """
        model_type = MODALITY_CONFIG.get(modality_type, {}).get("model_type")
        if not model_type or model_type not in self.models:
            raise ValueError(f"Unsupported or misconfigured modality: {modality_type}")

        return self.models[model_type].predict(input_data)
