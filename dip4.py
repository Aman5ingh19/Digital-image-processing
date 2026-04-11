import cv2
import matplotlib.pyplot as plt

# Step 1: Read image
image_path = r"C:\Users\amans\OneDrive\Pictures\pexels-diva-18178674.jpg"
img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found")
    exit()

# Convert BGR → RGB (for matplotlib)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Step 2: Apply Saliency Detection (Spectral Residual method)
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
(success, saliencyMap) = saliency.computeSaliency(img)

# Convert saliency map to 0–255
saliencyMap = (saliencyMap * 255).astype("uint8")

# Step 3: Threshold for better visualization (optional)
_, threshMap = cv2.threshold(saliencyMap, 0, 255,
                             cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# ---- Subplots ----
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')

# Saliency Map
plt.subplot(1, 3, 2)
plt.imshow(saliencyMap, cmap='gray')
plt.title("Saliency Map")
plt.axis('off')

# Thresholded Map
plt.subplot(1, 3, 3)
plt.imshow(threshMap, cmap='gray')
plt.title("Thresholded Saliency")
plt.axis('off')

plt.tight_layout()
plt.show()