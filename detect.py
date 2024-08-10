import cv2
import os
import numpy as np

# Define the input image path
problem_image_path = "60.jpg"  # Replace with the correct path to your image
output_dir = "output"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


def is_image_contour(contour, image_width, image_height):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    area = cv2.contourArea(contour)

    # Filter contours based on aspect ratio and area
    if 0.5 < aspect_ratio < 2.0 and area > (image_width * image_height) * 0.005:
        return True
    return False


def group_contours(contours, max_distance=50):
    centers = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))

    clusters = []
    for center in centers:
        placed = False
        for cluster in clusters:
            if np.linalg.norm(np.array(cluster[-1]) - np.array(center)) < max_distance:
                cluster.append(center)
                placed = True
                break
        if not placed:
            clusters.append([center])

    grouped_contours = []
    for cluster in clusters:
        group_contour = np.vstack(
            [contours[centers.index(center)] for center in cluster]
        )
        grouped_contours.append(group_contour)

    return grouped_contours


# Process the single image
problem_image = cv2.imread(problem_image_path)
if problem_image is None:
    print(f"Failed to load {problem_image_path}")
else:
    gray_problem_image = cv2.cvtColor(problem_image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_problem_image, (5, 5), 0)
    edged_image = cv2.Canny(blurred_image, 50, 150)

    contours, _ = cv2.findContours(
        edged_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    filtered_contours = [
        contour
        for contour in contours
        if is_image_contour(contour, problem_image.shape[1], problem_image.shape[0])
    ]
    grouped_contours = group_contours(filtered_contours)

    if grouped_contours:
        # Merge all grouped contours into one bounding box
        all_x, all_y, all_w, all_h = [], [], [], []
        for group in grouped_contours:
            x, y, w, h = cv2.boundingRect(group)
            all_x.append(x)
            all_y.append(y)
            all_w.append(w)
            all_h.append(h)

        x_min = min(all_x)
        y_min = min(all_y)
        x_max = max([x + w for x, w in zip(all_x, all_w)])
        y_max = max([y + h for y, h in zip(all_y, all_h)])

        # Extract and save the combined group image
        group_image = problem_image[y_min:y_max, x_min:x_max]
        group_image_name = os.path.splitext(os.path.basename(problem_image_path))[0]
        group_image_path = os.path.join(output_dir, f"group_combined_{group_image_name}.jpg")
        cv2.imwrite(group_image_path, group_image)
        print(f"Saved combined group image to {group_image_path}")

print("Group object detection and extraction completed. Check the output directory for results.")
