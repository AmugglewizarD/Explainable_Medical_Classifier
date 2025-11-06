# main.py
"""
Inference and Explanation script.

This script loads the models trained by 'train.py' to perform classification
and generate explanations (LIME for 2D, Grad-CAM for 3D).

*** YOU MUST RUN 'train.py' BEFORE RUNNING THIS SCRIPT ***
"""

import numpy as np
from PIL import Image
import torch
import json

# Import data loader *only* for helper functions and class map
from data.dataset_loader import get_dataloader, load_class_map, get_2d_transforms
from models.vit_model import MultimodalViTWrapper
from explainability.lime_explainer import LIME2DExplainer
from explainability.gradcam_explainer import GradCAM3DExplainer
from utils.visualization import show_image_with_mask, show_3d_explanation, show_metrics
from utils.metrics import evaluate_model
from config import MODALITY_CONFIG, DEVICE, IMG_SIZE_2D

def demo_modality(modality_type, multimodal_model, class_map, class_names_list, sample_count=4):
    print(f"\n========================================================")
    print(f"DEMO: {modality_type} (Dataset: {modality_type})")
    print(f"========================================================")

    # 1. Load Data (using 'test' split)
    try:
        dl = get_dataloader(modality_type, class_map, split="test", batch_size=sample_count, shuffle=False)
        batch = next(iter(dl))
    except Exception as e:
        print(f"Could not load data for {modality_type}: {e}")
        print("Please ensure your 'test' data and 'labels.csv' are set up.")
        return

    # 2. Prepare Inputs based on modality
    if modality_type in ["XRAY", "HISTOPATHOLOGY"]:
        input_tensors, labels = batch[0], batch[1]
        
        # Denormalize for LIME visualization
        denorm_transform = get_2d_transforms(is_train=False) # Get basic transform
        # We need to manually create denormalized PIL-like images
        vis_input_list = []
        pil_input_list = []
        for img_tensor in input_tensors:
            # Create PIL for model input (from normalized tensor)
            pil_img = transforms.ToPILImage()( (img_tensor * 0.5) + 0.5 ) # Denorm to [0,1]
            pil_input_list.append(pil_img)
            
            # Create numpy for LIME input
            vis_input_list.append(np.array(pil_img))
            
        model_input = pil_input_list
        explainer_input_np = vis_input_list[0] # First sample for LIME
        
    elif modality_type == "MRI":
        input_tensors, labels = batch["image"], batch["label"]
        model_input = input_tensors # List of (C, D, H, W) tensors
        explainer_input_tensor = model_input[0].unsqueeze(0).to(DEVICE) # (1, C, D, H, W)
        
    else:
        raise ValueError("Modality not supported in demo.")
        
    # 3. Prediction and Evaluation
    probs = multimodal_model.predict(model_input, modality_type)
    preds = probs.argmax(axis=1)
    
    true_labels_np = labels.cpu().numpy().flatten()[:sample_count]
    report = evaluate_model(true_labels_np, preds, labels=list(range(len(class_names_list))))
    show_metrics(report, title=f"Classification Report for {modality_type}")
    
    # 4. Explainability (Routing)
    pred_index = preds[0]
    pred_name = class_names_list[pred_index]
    
    print(f"\n--- Explanation for first sample (Pred: {pred_index} - '{pred_name}') ---")

    if modality_type in ["XRAY", "HISTOPATHOLOGY"]:
        # 2D Explainer (LIME)
        def predict_wrapper_2d(images_np):
            pil_list = [Image.fromarray(x.astype('uint8')) for x in images_np]
            return multimodal_model.predict(pil_list, modality_type)
            
        lime = LIME2DExplainer(predict_wrapper_2d)
        explanation, temp, mask = lime.explain(explainer_input_np, label=pred_index, num_features=10)
        show_image_with_mask(explainer_input_np, mask, title=f"LIME 2D Mask for {modality_type} (Pred: '{pred_name}')")
        
    elif modality_type == "MRI":
        # 3D Explainer (Grad-CAM)
        raw_3d_model = multimodal_model.get_model(modality_type)
        target_layer = "model.transformer.blocks[-1].norm1" 
        
        gradcam = GradCAM3DExplainer(raw_3d_model, target_layer)
        heatmap = gradcam.explain(explainer_input_tensor, label=pred_index)
        
        show_3d_explanation(
            explainer_input_tensor.squeeze(0), # (C, D, H, W)
            heatmap[0], # (1, D, H, W)
            title=f"3D Grad-CAM for {modality_type} (Pred: '{pred_name}')"
        )

def main_demo():
    print("--- Running Inference & Explanation ---")
    
    # 1. Load the dynamic class map created by train.py
    try:
        class_map = load_class_map()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        
    num_labels = len(class_map)
    # Create inverted map for printing names
    class_names_list = {v: k for k, v in class_map.items()}

    # 2. Initialize Wrapper in INFERENCE mode
    # This will automatically load from MODEL_2D_CHECKPOINT and MODEL_3D_CHECKPOINT
    try:
        multimodal_vit = MultimodalViTWrapper(num_labels=num_labels, load_from_scratch=False)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Could not load trained models. Did you run train.py successfully?")
        return

    # Run 2D Demo (XRAY)
    demo_modality(
        modality_type="XRAY", 
        multimodal_model=multimodal_vit,
        class_map=class_map,
        class_names_list=class_names_list,
        sample_count=4
    )
    
    # Run 2D Demo (HISTOPATHOLOGY)
    demo_modality(
        modality_type="HISTOPATHOLOGY", 
        multimodal_model=multimodal_vit,
        class_map=class_map,
        class_names_list=class_names_list,
        sample_count=4
    )
    
    # Run 3D Demo (MRI)
    demo_modality(
        modality_type="MRI", 
        multimodal_model=multimodal_vit,
        class_map=class_map,
        class_names_list=class_names_list,
        sample_count=2
    )

if __name__ == "__main__":
    main_demo()
