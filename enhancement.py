import cv2
import numpy as np
import matplotlib.pyplot as plt

def enhance_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error: Image not found. Please check your file path.")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y_channel, Cr, Cb = cv2.split(ycrcb)

    # Global Histogram Equalization
    equalized_y = cv2.equalizeHist(y_channel)
    equalized_img = cv2.merge((equalized_y, Cr, Cb))
    equalized_img = cv2.cvtColor(equalized_img, cv2.COLOR_YCrCb2RGB)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_y = clahe.apply(y_channel)
    clahe_img = cv2.merge((clahe_y, Cr, Cb))
    clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_YCrCb2RGB)

    # Show results
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(img); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(equalized_img); ax[1].set_title("Histogram Equalized"); ax[1].axis("off")
    ax[2].imshow(clahe_img); ax[2].set_title("CLAHE Enhanced"); ax[2].axis("off")
    plt.show()

if __name__ == "__main__":
    enhance_image("lenna.png")
