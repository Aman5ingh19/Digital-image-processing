import cv2

img = cv2.imread(r"C:\Users\amans\OneDrive\Pictures\pexels-diva-18178674.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert to binary
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    print("Object Area:", area)

    cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)

cv2.imshow("Objects", img)
cv2.waitKey(0)
cv2.destroyAllWindows()