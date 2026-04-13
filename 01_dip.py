# -----------------------------------------------
# Name: Aman Singh
# Roll No: XXXXX
# Course: B.Tech CSE
# Unit: Image Processing
# Assignment Title: Document Scanner System
# -----------------------------------------------

import cv2
import numpy as np

# -----------------------------------------------
# Task 1: Introduction
# -----------------------------------------------
print("======================================")
print(" Welcome to Document Scanner System ")
print(" This system performs:")
print(" - Image acquisition")
print(" - Resolution analysis")
print(" - Gray level quantization")
print("======================================")

# -----------------------------------------------
# Task 2: Image Acquisition
# -----------------------------------------------

# Load image (change path if needed)
img = cv2.imread(r"C:\Users\amans\OneDrive\Pictures\pexels-diva-18178674.jpg")

if img is None:
    print("Error: Image not found!")
    exit()

# Resize to 512x512
img_resized = cv2.resize(img, (512, 512))

# Convert to grayscale
gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# Display images
cv2.imshow("Original Image", img_resized)
cv2.imshow("Grayscale Image", gray)

# -----------------------------------------------
# Task 3: Image Sampling (Resolution Analysis)
# -----------------------------------------------

# Down-sampling
high_res = gray  # 512x512
medium_res = cv2.resize(gray, (256, 256))
low_res = cv2.resize(gray, (128, 128))

# Upscale back for comparison
medium_up = cv2.resize(medium_res, (512, 512))
low_up = cv2.resize(low_res, (512, 512))

cv2.imshow("High Resolution (512x512)", high_res)
cv2.imshow("Medium Resolution (256->512)", medium_up)
cv2.imshow("Low Resolution (128->512)", low_up)

# -----------------------------------------------
# Task 4: Image Quantization
# -----------------------------------------------

# 256 levels (8-bit) → original
quant_256 = gray

# 16 levels (4-bit)
quant_16 = (gray // 16) * 16

# 4 levels (2-bit)
quant_4 = (gray // 64) * 64

cv2.imshow("256 Gray Levels", quant_256)
cv2.imshow("16 Gray Levels", quant_16)
cv2.imshow("4 Gray Levels", quant_4)

# -----------------------------------------------
# Wait & Close
# -----------------------------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()