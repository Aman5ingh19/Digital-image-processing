import cv2
import matplotlib.pyplot as plt

# Read image
image_path = r"C:\Users\amans\OneDrive\Pictures\pexels-diva-18178674.jpg"
img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found")
    exit()

# Convert to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# SIFT detector
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw keypoints
sift_img = cv2.drawKeypoints(
    img_rgb, keypoints, None, (0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Subplots
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sift_img)
plt.title("SIFT Keypoints")
plt.axis('off')

plt.tight_layout()
plt.show()