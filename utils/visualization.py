# utils/visualization.py
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import numpy as np

def show_image(image_np, title="Image"):
    """Displays a single 2D image."""
    plt.imshow(image_np)
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_image_with_mask(image_np, mask, title="Explanation Mask"):
    """Displays an image with a LIME-style mask overlay (2D)."""
    # Use mark_boundaries to draw lines around superpixels
    boundary_img = mark_boundaries(image_np, mask, color=(1, 0, 0), outline_color=(1, 1, 0))
    plt.imshow(boundary_img)
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_3d_explanation(volume_np, explainer_output, title="3D Explanation (Conceptual)"):
    """Conceptual function to visualize 3D explanations."""
    print(f"--- {title} ---")
    print("Conceptual Visualization: In a real environment, this would show axial, sagittal, and coronal planes with highlighted voxels.")
    
    # Simple demonstration: Show one central slice
    if volume_np.ndim == 3: # Assuming (D, H, W) for grayscale 3D
        center_slice_idx = volume_np.shape[0] // 2
        center_slice = volume_np[center_slice_idx, :, :]
        
        # Convert to RGB if single channel for matplotlib display
        if center_slice.ndim == 2:
            center_slice_rgb = np.stack([center_slice] * 3, axis=-1)
        else:
             center_slice_rgb = center_slice
        
        plt.figure(figsize=(6, 6))
        plt.imshow(center_slice_rgb, cmap='gray') # Use grayscale for medical scans
        plt.title(f"{title}\nCentral Slice ({center_slice_idx})")
        plt.axis('off')
        plt.show()
    else:
        print("Input is not a 3D volume and cannot be visualized as such.")
    
def show_metrics(report, title="Model Evaluation"):
    """Displays a classification report."""
    print(f"\n--- {title} ---\n")
    print(report)
