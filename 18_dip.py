import cv2

# Read image
img = cv2.imread(r"C:\Users\amans\OneDrive\Pictures\pexels-diva-18178674.jpg", 0)

# Flatten image to 1D
pixels = img.flatten()

# Run-Length Encoding
encoded = []
count = 1

for i in range(1, len(pixels)):
    if pixels[i] == pixels[i - 1]:
        count += 1
    else:
        encoded.append((pixels[i - 1], count))
        count = 1

encoded.append((pixels[-1], count))

print("RLC Output (first 20):", encoded[:20])