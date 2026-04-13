import cv2

img = cv2.imread(r"C:\Users\amans\OneDrive\Pictures\pexels-diva-18178674.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Discontinuity (Edge Detection)
edges = cv2.Canny(gray, 100, 200)

# Similarity (Thresholding)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("Original", img)
cv2.imshow("Edges", edges)
cv2.imshow("Threshold", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()