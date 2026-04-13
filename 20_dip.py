import cv2
import numpy as np
from scipy.signal import wiener

# Read image
img = cv2.imread(r"C:\Users\amans\OneDrive\Pictures\pexels-diva-18178674.jpg", 0)

# -------------------------------
# Add Periodic Noise (sinusoidal)
# -------------------------------
rows, cols = img.shape
x = np.arange(cols)
y = np.arange(rows)
xv, yv = np.meshgrid(x, y)

periodic_noise = 30 * np.sin(2 * np.pi * xv / 20)
noisy_img = img + periodic_noise
noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

# -------------------------------
# Inverse Filtering (Simple FFT)
# -------------------------------
f = np.fft.fft2(noisy_img)
fshift = np.fft.fftshift(f)

# Simple inverse (no degradation model, conceptual)
inv_fshift = np.fft.ifftshift(fshift)
inverse_img = np.fft.ifft2(inv_fshift)
inverse_img = np.abs(inverse_img)
inverse_img = np.clip(inverse_img, 0, 255).astype(np.uint8)

# -------------------------------
# Wiener Filtering
# -------------------------------
wiener_img = wiener(noisy_img, (5, 5))

# -------------------------------
# Display Results
# -------------------------------
cv2.imshow("Original", img)
cv2.imshow("Periodic Noise", noisy_img)
cv2.imshow("Inverse Filter", inverse_img)
cv2.imshow("Wiener Filter", wiener_img.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()