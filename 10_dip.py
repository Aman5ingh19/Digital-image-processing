import cv2
import numpy as np

# Read image
img = cv2.imread(r"C:\Users\amans\OneDrive\Pictures\pexels-diva-18178674.jpg")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Compute Mean
mean_val = np.mean(gray)

# Compute Variance
variance_val = np.var(gray)

print("Mean:", mean_val)
print("Variance:", variance_val)