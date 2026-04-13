import cv2
import numpy as np

# -----------------------------------------------
# Load Image
# -----------------------------------------------
img = cv2.imread(r"C:\Users\amans\OneDrive\Pictures\pexels-diva-18178674.jpg")

if img is None:
    print("Error: Image not found!")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -----------------------------------------------
# Task 1: Edge Detection
# -----------------------------------------------

# Sobel Operator
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)
sobel = np.uint8(np.clip(sobel, 0, 255))

# Canny Edge Detector
canny = cv2.Canny(gray, 100, 200)

cv2.imshow("Sobel Edge", sobel)
cv2.imshow("Canny Edge", canny)

# -----------------------------------------------
# Task 2: Object Representation
# -----------------------------------------------

# Find contours
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_contours = img.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if area > 100:  # ignore small noise
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

        print("Object Area:", area, "Perimeter:", perimeter)

cv2.imshow("Bounding Boxes", img_contours)

# -----------------------------------------------
# Task 3: Feature Extraction (ORB)
# -----------------------------------------------

orb = cv2.ORB_create()

keypoints, descriptors = orb.detectAndCompute(gray, None)

img_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))

cv2.imshow("ORB Keypoints", img_keypoints)

print("Number of Keypoints:", len(keypoints))

# -----------------------------------------------
# Task 4: Comparative Analysis
# -----------------------------------------------

print("\n--- Analysis ---")
print("1. Canny gives sharper and cleaner edges than Sobel.")
print("2. Sobel detects gradients but includes noise.")
print("3. ORB detects keypoints efficiently and is fast.")
print("4. Feature extraction helps track objects in traffic monitoring.")

print("\n--- Traffic Monitoring Use ---")
print("• Detect vehicles using contours and edges.")
print("• Track movement using keypoints.")
print("• Analyze traffic density and flow.")
print("• Useful in surveillance and smart city systems.")

# -----------------------------------------------
# End
# -----------------------------------------------
cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()