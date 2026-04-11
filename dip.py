import cv2
import matplotlib.pyplot as plt

# Step 1: Read the image
image_path = r"C:\Users\amans\OneDrive\Pictures\pexels-diva-18178674.jpg"
img = cv2.imread(image_path)

# Check if image loaded properly
if img is None:
    print("Error: Image not found or path is incorrect")
    exit()

# Step 2: Convert BGR to RGB (for correct display in matplotlib)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Step 3: Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 4: Apply Canny Edge Detection
edges = cv2.Canny(gray, 100, 200)

# Step 5: Display all images using subplots
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

# Canny Edge Detection
plt.subplot(1, 3, 3)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis("off")

# Adjust layout and show
plt.tight_layout()
plt.show()