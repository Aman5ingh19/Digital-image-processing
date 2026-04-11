import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the image
image_path = r"C:\Users\amans\OneDrive\Pictures\pexels-diva-18178674.jpg"
img = cv2.imread(image_path)

# Check if image loaded properly
if img is None:
    print("Error: Image not found or path is incorrect")
    exit()

# Convert BGR to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Step 2: Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert to float32 (important for Harris)
gray_float = np.float32(gray)

# Step 3: Apply Harris Corner Detector
dst = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)

# Dilate result to mark corners clearly
dst = cv2.dilate(dst, None)

# Copy image for marking corners
corner_img = img_rgb.copy()

# Threshold for detecting strong corners
corner_img[dst > 0.01 * dst.max()] = [255, 0, 0]  # mark in RED (RGB)

# Step 4: Display using subplots
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

# Grayscale Image
plt.subplot(1, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")

# Harris Corner Detection
plt.subplot(1, 3, 3)
plt.imshow(corner_img)
plt.title("Harris Corners")
plt.axis("off")

plt.tight_layout()
plt.show()