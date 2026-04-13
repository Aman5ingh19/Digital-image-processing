import cv2
import numpy as np

# -----------------------------------------------
# Task 1: Image Compression (RLE)
# -----------------------------------------------

# Load grayscale image
img = cv2.imread(r"C:\Users\amans\OneDrive\Pictures\pexels-diva-18178674.jpg", 0)

if img is None:
    print("Error: Image not found!")
    exit()

# Flatten image
pixels = img.flatten()

# Run-Length Encoding
rle = []
count = 1

for i in range(1, len(pixels)):
    if pixels[i] == pixels[i - 1]:
        count += 1
    else:
        rle.append((pixels[i - 1], count))
        count = 1

rle.append((pixels[-1], count))

# Compression ratio
original_size = len(pixels)
compressed_size = len(rle) * 2  # (value, count)

compression_ratio = original_size / compressed_size
savings = (1 - (compressed_size / original_size)) * 100

print("Compression Ratio:", compression_ratio)
print("Storage Savings (%):", savings)

# -----------------------------------------------
# Task 2: Image Segmentation
# -----------------------------------------------

# Global Thresholding
_, global_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Otsu Thresholding
_, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow("Global Threshold", global_thresh)
cv2.imshow("Otsu Threshold", otsu_thresh)

# -----------------------------------------------
# Task 3: Morphological Processing
# -----------------------------------------------

kernel = np.ones((5, 5), np.uint8)

# On Otsu result (better input)
dilation = cv2.dilate(otsu_thresh, kernel, iterations=1)
erosion = cv2.erode(otsu_thresh, kernel, iterations=1)

cv2.imshow("Dilation", dilation)
cv2.imshow("Erosion", erosion)

# -----------------------------------------------
# Task 4: Analysis and Interpretation
# -----------------------------------------------

print("\n--- Analysis ---")
print("1. Otsu thresholding gives better segmentation than global threshold.")
print("2. Dilation expands regions (fills gaps).")
print("3. Erosion removes noise and shrinks regions.")
print("4. Morphological operations improve region clarity.")

print("\n--- Clinical Relevance ---")
print("• Segmentation helps identify abnormal regions (tumor, fracture).")
print("• Morphology refines boundaries for better diagnosis.")
print("• Compression reduces storage for medical imaging systems.")

# -----------------------------------------------
# End
# -----------------------------------------------
cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()