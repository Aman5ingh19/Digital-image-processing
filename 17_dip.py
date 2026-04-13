import cv2
import numpy as np

# -------------------------------
# STEP 1: Read Image
# -------------------------------
img = cv2.imread(r"C:\Users\amans\OneDrive\Pictures\pexels-diva-18178674.jpg", 0)

# Convert image to 1D pixel stream
pixels = img.flatten()

# -------------------------------
# GOLUMB CODING (Conceptual)
# -------------------------------
def golomb_encode(numbers, m):
    encoded = []

    for n in numbers[:100]:  # limit for demo (Golomb works on sequences)
        q = n // m   # quotient
        r = n % m    # remainder

        # Unary coding for quotient
        unary = '1' * q + '0'

        # Binary coding for remainder
        b = int(np.ceil(np.log2(m)))
        binary = format(r, f'0{b}b')

        encoded.append(unary + binary)

    return encoded


golomb_output = golomb_encode(pixels, m=4)
print("Golomb Coding Output (first 10):")
print(golomb_output[:10])


# -------------------------------
# LZW CODING
# -------------------------------
def lzw_compress(data):
    dictionary = {i: i for i in range(256)}
    w = ""
    result = []
    code = 256

    # Convert pixel values to string for processing
    data = ''.join([chr(p) for p in data[:1000]])  # limit size

    for c in data:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            dictionary[wc] = code
            code += 1
            w = c

    if w:
        result.append(dictionary[w])

    return result


lzw_output = lzw_compress(pixels)

print("\nLZW Output (first 20):")
print(lzw_output[:20])