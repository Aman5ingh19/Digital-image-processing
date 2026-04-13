import cv2
import numpy as np

img_path = r"C:\Users\goswa\Downloads\Fig0305(a)(DFT_no_log).tif"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Check if image loaded
if img is None:
    print("Image not loaded")
    exit()

# Convert to float
img_float = img.astype(np.float32)

# Log transformation
log_img = np.log1p(img_float)   # log(1 + img)

# Normalize to 0–255
log_img = cv2.normalize(log_img, None, 0, 255, cv2.NORM_MINMAX)

# Convert back to uint8
log_img = np.uint8(log_img)

# Display
cv2.imshow("Original Image", img)
cv2.imshow("Log Transformed Image", log_img)
cv2.waitKey(0)
cv2.destroyAllWindows()