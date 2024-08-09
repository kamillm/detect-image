import cv2
import os
import numpy as np

# Define paths
input_image_path = "60.jpg"
output_dir = "illustrations_output"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def is_valid_contour(contour, image_width, image_height):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    area = cv2.contourArea(contour)

    # Filter based on typical size and shape of illustrations
    if 0.5 < aspect_ratio < 2.0 and area > (image_width * image_height) * 0.01:
        return True
    return False

# Load and preprocess image
image = cv2.imread(input_image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
edged_image = cv2.Canny(blurred_image, 50, 150)

# Find contours
contours, _ = cv2.findContours(edged_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter and extract contours
illustration_count = 1
for contour in contours:
    if is_valid_contour(contour, image.shape[1], image.shape[0]):
        x, y, w, h = cv2.boundingRect(contour)
        illustration = image[y:y+h, x:x+w]
        illustration_path = os.path.join(output_dir, f"illustration_{illustration_count}.jpg")
        cv2.imwrite(illustration_path, illustration)
        illustration_count += 1

print(f"Illustration extraction completed. Check the {output_dir} directory for results.")