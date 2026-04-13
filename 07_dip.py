import cv2
import os

# Create folder path where image will be saved
save_path = r"C:\Users\Public\Webcam_Images"

# Create folder if it does not exist
os.makedirs(save_path, exist_ok=True)

# Access laptop webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not access the webcam")
    exit()

print("Press 's' to save image and 'q' to quit")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Failed to capture image")
        break

    cv2.imshow("Webcam Image Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save image when 's' is pressed
    if key == ord('s'):
        file_name = os.path.join(save_path, "captured_image.jpg")
        cv2.imwrite(file_name, frame)
        print(f"Image saved at: {file_name}")

    # Exit when 'q' is pressed
    elif key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()