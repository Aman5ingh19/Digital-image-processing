import cv2
import numpy as np

# -----------------------------------------------
# Task 1: Image Selection and Preprocessing
# -----------------------------------------------

# Load image
img = cv2.imread(r"C:\Users\amans\OneDrive\Pictures\pexels-diva-18178674.jpg")

if img is None:
    print("Error: Image not found!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Original Image", img)
cv2.imshow("Grayscale Image", gray)

# -----------------------------------------------
# Task 2: Noise Modeling
# -----------------------------------------------

# Gaussian Noise
mean = 0
sigma = 25
gaussian_noise = np.random.normal(mean, sigma, gray.shape).astype(np.uint8)
gaussian_noisy = cv2.add(gray, gaussian_noise)

# Salt & Pepper Noise
sp_noisy = gray.copy()
prob = 0.02

salt = np.random.rand(*gray.shape) < prob
pepper = np.random.rand(*gray.shape) < prob

sp_noisy[salt] = 255
sp_noisy[pepper] = 0

cv2.imshow("Gaussian Noise", gaussian_noisy)
cv2.imshow("Salt & Pepper Noise", sp_noisy)

# -----------------------------------------------
# Task 3: Image Restoration
# -----------------------------------------------

# Mean Filter
mean_gaussian = cv2.blur(gaussian_noisy, (5, 5))
mean_sp = cv2.blur(sp_noisy, (5, 5))

# Median Filter
median_gaussian = cv2.medianBlur(gaussian_noisy, 5)
median_sp = cv2.medianBlur(sp_noisy, 5)

# Gaussian Filter
gauss_gaussian = cv2.GaussianBlur(gaussian_noisy, (5, 5), 0)
gauss_sp = cv2.GaussianBlur(sp_noisy, (5, 5), 0)

cv2.imshow("Mean Filter (Gaussian Noise)", mean_gaussian)
cv2.imshow("Median Filter (Gaussian Noise)", median_gaussian)
cv2.imshow("Gaussian Filter (Gaussian Noise)", gauss_gaussian)

cv2.imshow("Mean Filter (SP Noise)", mean_sp)
cv2.imshow("Median Filter (SP Noise)", median_sp)
cv2.imshow("Gaussian Filter (SP Noise)", gauss_sp)

# -----------------------------------------------
# Task 4: Performance Evaluation
# -----------------------------------------------

def mse(original, restored):
    return np.mean((original - restored) ** 2)

def psnr(original, restored):
    m = mse(original, restored)
    if m == 0:
        return 100
    return 20 * np.log10(255 / np.sqrt(m))

# Compute metrics for Gaussian noise
print("\n--- Gaussian Noise ---")
print("Mean Filter -> MSE:", mse(gray, mean_gaussian), "PSNR:", psnr(gray, mean_gaussian))
print("Median Filter -> MSE:", mse(gray, median_gaussian), "PSNR:", psnr(gray, median_gaussian))
print("Gaussian Filter -> MSE:", mse(gray, gauss_gaussian), "PSNR:", psnr(gray, gauss_gaussian))

# Compute metrics for Salt & Pepper noise
print("\n--- Salt & Pepper Noise ---")
print("Mean Filter -> MSE:", mse(gray, mean_sp), "PSNR:", psnr(gray, mean_sp))
print("Median Filter -> MSE:", mse(gray, median_sp), "PSNR:", psnr(gray, median_sp))
print("Gaussian Filter -> MSE:", mse(gray, gauss_sp), "PSNR:", psnr(gray, gauss_sp))

# -----------------------------------------------
# Task 5: Analytical Discussion
# -----------------------------------------------

print("\n--- Analysis ---")
print("1. Median filter performs best for Salt & Pepper noise.")
print("2. Gaussian filter performs best for Gaussian noise.")
print("3. Mean filter blurs image and loses details.")
print("4. Median filter preserves edges better.")
print("5. Gaussian filter balances smoothing and detail preservation.")

# -----------------------------------------------
# End
# -----------------------------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()