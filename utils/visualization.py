# utils/visualization.py
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import numpy as np
import torch
import shap # ADDED for SHAP plotting
import os
from config import RESULTS_DIR

def show_image(image_np, title="Image"):
    # ... (function unchanged) ...
    plt.imshow(image_np)
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_image_with_mask(image_np, mask, title="Explanation Mask", save_path=None):
    """Displays an image with a LIME-style mask overlay (2D)."""
    # Use mark_boundaries to draw lines around superpixels
    boundary_img = mark_boundaries(image_np, mask, color=(1, 0, 0), outline_color=(1, 1, 0))
    plt.imshow(boundary_img)
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def show_3d_explanation(volume_tensor, heatmap_tensor, title="3D Grad-CAM Explanation", save_path=None):
    """
    Visualizes a 3D Grad-CAM heatmap overlaid on a central slice 
    of the original 3D volume.
    """
    # ... (function mostly unchanged, added save_path) ...
    volume_np = volume_tensor.cpu().numpy().squeeze() 
    heatmap_np = heatmap_tensor.cpu().numpy().squeeze() 

    center_slice_idx = volume_np.shape[0] // 2
    
    volume_slice = volume_np[center_slice_idx, :, :]
    heatmap_slice = heatmap_np[center_slice_idx, :, :]

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(volume_slice, cmap='gray')
    plt.title(f"Original Volume (Slice {center_slice_idx})")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(volume_slice, cmap='gray')
    plt.imshow(heatmap_slice, cmap='jet', alpha=0.5) 
    plt.title(title)
    plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def show_metrics(report, title="Model Evaluation"):
    # ... (function unchanged) ...
    print(f"\n--- {title} ---\n")
    print(report)

# --- ADDED: New Function for SHAP (Point 4) ---
def show_shap_explanation(shap_values, image_np, class_names, title="SHAP Explanation", save_path=None):
    """
    Saves a SHAP image plot.
    """
    print("Generating SHAP plot...")
    # Reshape for shap.image_plot
    # shap_values shape: (classes, height, width, channels)
    # image_np shape: (height, width, channels)
    
    # Ensure shap_values is in the correct format (list of arrays)
    if not isinstance(shap_values, list):
         shap_values = [shap_values[i] for i in range(shap_values.shape[0])]

    shap.image_plot(
        shap_values,
        image_np.reshape(1, *image_np.shape), # Add batch dim
        labels=np.array([f"Class {i} ({class_names[i]})" for i in range(len(class_names))]),
        show=False
    )
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
