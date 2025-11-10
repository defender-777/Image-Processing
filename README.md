# 🧠 Image Processing Projects

This repository contains foundational **Image Processing** projects implemented using **OpenCV**, **NumPy**, and **Matplotlib**.

---

## 📸 Projects Included

### 1️⃣ Image Noise Removal and Restoration
- Adds Gaussian, Salt & Pepper, and Speckle noise to an image.
- Restores the image using Mean, Median, Gaussian, and Bilateral filters.
- Evaluates performance using **PSNR** and **SSIM** metrics.

### 2️⃣ Image Enhancement and Histogram Equalization
- Improves image contrast and brightness dynamically.
- Uses both **Global Histogram Equalization** and **CLAHE** (Contrast Limited Adaptive Histogram Equalization).
- Works on color images using YCrCb color space transformations.

### 3️⃣ Image Sharpening and Edge Enhancement 
Performs image sharpening and edge enhancement using OpenCV.
Outputs:
- `sharpened.png`
- `edge_enhanced.png`
Features:
- Sharpening with convolution kernel
- Edge detection with Laplacian filter
- Side-by-side visualization of results

---

## ⚙️ Requirements
Install dependencies:
```bash
pip install -r requirements.txt
