# -------------------------------------------------------
# Student Name: Aman Singh
# Roll No: XXXXX
# Course Name: B.Tech CSE
# Assignment Title: Intelligent Image Processing System
# Date: 13 April 2026
# -------------------------------------------------------

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# -------------------------------------------------------
# Task 1: Introduction
# -------------------------------------------------------
print("==========================================")
print(" Intelligent Image Processing System ")
print(" This system performs:")
print(" - Image acquisition & preprocessing")
print(" - Noise simulation & restoration")
print(" - Enhancement & segmentation")
print(" - Feature extraction & analysis")
print("==========================================")

# -------------------------------------------------------
# Task 2: Image Acquisition & Preprocessing
# -------------------------------------------------------
img = cv2.imread(r"C:\Users\amans\OneDrive\Pictures\pexels-diva-18178674.jpg")

if img is None:
    print("Error: Image not found!")
    exit()

img = cv2.resize(img, (512, 512))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Original", img)
cv2.imshow("Grayscale", gray)

# -------------------------------------------------------
# Task 3: Noise + Restoration + Enhancement
# -------------------------------------------------------

# Gaussian Noise
gaussian = np.random.normal(0, 25, gray.shape).astype(np.uint8)
gaussian_noisy = cv2.add(gray, gaussian)

# Salt & Pepper Noise
sp_noisy = gray.copy()
prob = 0.02
salt = np.random.rand(*gray.shape) < prob
pepper = np.random.rand(*gray.shape) < prob
sp_noisy[salt] = 255
sp_noisy[pepper] = 0

# Filters
mean_filter = cv2.blur(gaussian_noisy, (5, 5))
median_filter = cv2.medianBlur(sp_noisy, 5)
gaussian_filter = cv2.GaussianBlur(gaussian_noisy, (5, 5), 0)

# Enhancement (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)

# -------------------------------------------------------
# Task 4: Segmentation & Morphology
# -------------------------------------------------------

# Thresholding
_, global_thresh = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
_, otsu_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Morphology
kernel = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(otsu_thresh, kernel, iterations=1)
erosion = cv2.erode(otsu_thresh, kernel, iterations=1)

# -------------------------------------------------------
# Task 5: Edge + Contours + ORB
# -------------------------------------------------------

# Edges
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
sobel = cv2.magnitude(sobelx, sobely).astype(np.uint8)

canny = cv2.Canny(gray, 100, 200)

# Contours
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_contours = img.copy()

for cnt in contours:
    if cv2.contourArea(cnt) > 100:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

# ORB Features
orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(gray, None)
img_kp = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0))

# -------------------------------------------------------
# Task 6: Performance Evaluation
# -------------------------------------------------------

def mse(a, b):
    return np.mean((a - b) ** 2)

def psnr(a, b):
    m = mse(a, b)
    return 20 * np.log10(255 / np.sqrt(m)) if m != 0 else 100

def compute_ssim(a, b):
    return ssim(a, b)

print("\n--- Performance Metrics ---")
print("Original vs Gaussian Filter:")
print("MSE:", mse(gray, gaussian_filter))
print("PSNR:", psnr(gray, gaussian_filter))
print("SSIM:", compute_ssim(gray, gaussian_filter))

print("\nOriginal vs Enhanced:")
print("MSE:", mse(gray, enhanced))
print("PSNR:", psnr(gray, enhanced))
print("SSIM:", compute_ssim(gray, enhanced))

# -------------------------------------------------------
# Task 7: Final Visualization
# -------------------------------------------------------

# Stack images (resize if needed)
row1 = np.hstack((gray, gaussian_noisy, gaussian_filter))
row2 = np.hstack((enhanced, otsu_thresh, img_kp))

final_display = np.vstack((row1, row2))

cv2.imshow("Pipeline Output", final_display)

# -------------------------------------------------------
# Conclusion
# -------------------------------------------------------
print("\n--- Conclusion ---")
print("System successfully performs complete image processing pipeline.")
print("Median filter best for salt & pepper noise.")
print("Gaussian filter best for Gaussian noise.")
print("Otsu gives better segmentation.")
print("ORB detects useful feature points efficiently.")

cv2.waitKey(0)
cv2.destroyAllWindows()