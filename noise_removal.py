import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def add_noise(img, mode):
    """Add different types of noise to the image."""
    noisy = random_noise(img, mode=mode)
    noisy = np.array(255 * noisy, dtype=np.uint8)
    return noisy

def apply_filters(noisy):
    """Apply different filters to restore noisy images."""
    mean = cv2.blur(noisy, (5, 5))
    median = cv2.medianBlur(noisy, 5)
    gaussian = cv2.GaussianBlur(noisy, (5, 5), 0)
    bilateral = cv2.bilateralFilter(noisy, 9, 75, 75)
    return {"Mean": mean, "Median": median, "Gaussian": gaussian, "Bilateral": bilateral}

def evaluate(original, filtered):
    """Compute PSNR and SSIM metrics."""
    psnr = peak_signal_noise_ratio(original, filtered)
    ssim = structural_similarity(original, filtered, channel_axis=2)
    return psnr, ssim

if __name__ == "__main__":
    # Load the image
    img = cv2.imread("lenna.png")
    if img is None:
        print("❌ Error: Image file not found. Please place 'lenna.png' in this folder.")
        exit()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Add noise
    noisy = add_noise(img, 's&p')

    # Apply filters
    filters = apply_filters(noisy)

    # Evaluate and print metrics
    for name, f_img in filters.items():
        psnr, ssim = evaluate(img, f_img)
        print(f"{name} Filter -> PSNR: {psnr:.2f}, SSIM: {ssim:.3f}")

    # Display
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); plt.imshow(noisy); plt.title("Noisy Image"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(filters['Median']); plt.title("Restored (Median)"); plt.axis("off")
    plt.show()
