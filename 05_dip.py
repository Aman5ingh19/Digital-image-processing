import cv2
import numpy as np

# Read image (grayscale)
img = cv2.imread(r"C:\Users\amans\OneDrive\Pictures\pexels-diva-18178674.jpg", 0)

# -------------------------------
# Add Gaussian Noise
# -------------------------------
mean = 0
sigma = 25
gaussian = np.random.normal(mean, sigma, img.shape).astype('uint8')
gaussian_noise = cv2.add(img, gaussian)

# -------------------------------
# Add Salt & Pepper Noise
# -------------------------------
sp_noise = img.copy()
prob = 0.02

# Salt noise
salt = np.random.rand(*img.shape) < prob
sp_noise[salt] = 255

# Pepper noise
pepper = np.random.rand(*img.shape) < prob
sp_noise[pepper] = 0

# -------------------------------
# Mean Filter
# -------------------------------
mean_filtered = cv2.blur(gaussian_noise, (5, 5))

# -------------------------------
# Median Filter
# -------------------------------
median_filtered = cv2.medianBlur(sp_noise, 5)

# -------------------------------
# Display Results
# -------------------------------
cv2.imshow("Original", img)
cv2.imshow("Gaussian Noise", gaussian_noise)
cv2.imshow("Mean Filtered", mean_filtered)

cv2.imshow("Salt & Pepper Noise", sp_noise)
cv2.imshow("Median Filtered", median_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()