"""
sharpening.py
--------------
Image Sharpening and Edge Enhancement module for the Image Processing Toolkit.

Features:
- Sharpening using custom convolution kernel
- Edge enhancement using Laplacian filter
- Saves output images automatically
- Visual comparison between Original, Sharpened, and Edge Enhanced images

Author: Gagan
Repository: ImageProcessingToolkit
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------------
# Function: Sharpen Image
# -------------------------------
def sharpen_image(img):
    """
    Apply sharpening filter to enhance image edges.
    """
    # Define sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

# -------------------------------
# Function: Edge Enhancement
# -------------------------------
def enhance_edges(img):
    """
    Use Laplacian operator to detect and enhance edges.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_64F)
    edges = np.uint8(np.absolute(edges))

    # Convert to RGB for blending
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    enhanced = cv2.addWeighted(img, 0.8, edges_colored, 0.2, 0)

    return enhanced

# -------------------------------
# Function: Display and Save Outputs
# -------------------------------
def display_and_save_results(original, sharpened, edge_enhanced, output_dir="output"):
    """
    Display and save the original, sharpened, and edge-enhanced images.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save images
    cv2.imwrite(os.path.join(output_dir, "sharpened.png"),
                cv2.cvtColor(sharpened, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "edge_enhanced.png"),
                cv2.cvtColor(edge_enhanced, cv2.COLOR_RGB2BGR))

    # Display all images
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(original)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(sharpened)
    ax[1].set_title("Sharpened Image")
    ax[1].axis("off")

    ax[2].imshow(edge_enhanced)
    ax[2].set_title("Edge Enhanced Image")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()

# -------------------------------
# Main Program Entry Point
# -------------------------------
if __name__ == "__main__":
    # Load image
    img = cv2.imread("lenna.png")
    if img is None:
        print("❌ Error: 'lenna.png' not found. Please place it in this folder.")
        exit()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply filters
    sharpened = sharpen_image(img)
    edge_enhanced = enhance_edges(img)

    # Display & Save
    display_and_save_results(img, sharpened, edge_enhanced)
