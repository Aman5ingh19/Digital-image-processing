import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import defaultdict

# Step 1: Read image
image_path = r"C:\Users\amans\OneDrive\Pictures\pexels-diva-18178674.jpg"
img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found")
    exit()

# Convert to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Step 2: Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 3: Count pixel frequencies
freq = defaultdict(int)
for pixel in gray.flatten():
    freq[pixel] += 1

# ---------------- Huffman Coding ---------------- #

# Node class
class Node:
    def __init__(self, freq, pixel, left=None, right=None):
        self.freq = freq
        self.pixel = pixel
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

# Build Huffman Tree
heap = [Node(freq[p], p) for p in freq]
heapq.heapify(heap)

while len(heap) > 1:
    left = heapq.heappop(heap)
    right = heapq.heappop(heap)
    merged = Node(left.freq + right.freq, None, left, right)
    heapq.heappush(heap, merged)

# Generate codes
huffman_codes = {}

def generate_codes(node, current_code=""):
    if node is None:
        return
    if node.pixel is not None:
        huffman_codes[node.pixel] = current_code
        return
    generate_codes(node.left, current_code + "0")
    generate_codes(node.right, current_code + "1")

root = heap[0]
generate_codes(root)

# Encode image (bit length calculation)
encoded_bits = sum(len(huffman_codes[p]) for p in gray.flatten())

# Original size (8 bits per pixel)
original_bits = gray.size * 8

compression_ratio = original_bits / encoded_bits

# ---------------- Subplots ---------------- #

plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')

# Grayscale Image
plt.subplot(1, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')

# Histogram (Pixel Frequency)
plt.subplot(1, 3, 3)
plt.hist(gray.flatten(), bins=256, range=[0,256])
plt.title("Pixel Frequency Histogram")

plt.tight_layout()
plt.show()

# Print compression details
print("Original size (bits):", original_bits)
print("Compressed size (bits):", encoded_bits)
print("Compression Ratio:", round(compression_ratio, 2))