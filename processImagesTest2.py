import cv2
import numpy as np

# Thresholds for blue color in HSV
lower_blue = np.array([8, 8, 110])  # Lower HSV threshold for blue
upper_blue = np.array([65, 50, 260])  # Upper HSV threshold for blue

def filter_image(image):
    """
    Processes the input image to filter by blue color and keep areas larger than a threshold.
    """
    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Apply color segmentation for blue
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Apply morphological opening to clean up noise
    kernel = np.ones((3, 3), np.uint8)
    opened_mask = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=1)

    # crop image
    # Define crop coordinates
    start_y, end_y = 5, 240
    start_x, end_x = 15, 290
    # Create a black image of the same size as the opened_mask
    cropped_img = np.zeros_like(opened_mask)
    # Copy the pixels within the crop to their positions in the black image
    cropped_img[start_y:end_y, start_x:end_x] = opened_mask[start_y:end_y, start_x:end_x]

    # Filter areas larger than 1500 pixels
    contours, _ = cv2.findContours(cropped_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(cropped_img)

    for cnt in contours:
        if cv2.contourArea(cnt) >= 1500:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return filtered_mask


def check_boxes(image_path):
    # Load the image
    image = cv2.imread(image_path)
    original = image.copy()  # Keep a copy for displaying results

    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Mask for the blue area (tune these values to match the blue color)
    lower_blue = np.array([90, 70, 100])  # Lower bound for blue in HSV
    upper_blue = np.array([130, 120, 230])  # Upper bound for blue in HSV
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Create a horizontal rectangular kernel for dilation
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))  # Width=20, Height=1

    # Apply dilation with the horizontal kernel
    blue_mask = cv2.dilate(blue_mask, horizontal_kernel, iterations=1)

    # Find contours for the blue area
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(blue_contours) == 0:
        print("Error: No blue area detected")
        return

    # Assuming the largest blue contour is the area of interest
    blue_area = max(blue_contours, key=cv2.contourArea)

     # Get the bounding box of the blue area
    x, y, w, h = cv2.boundingRect(blue_area)
    blue_rect = {"x_min": x, "x_max": x + w, "y_min": y, "y_max": y + h}

    # Draw the blue area contour and bounding box on the original image
    cv2.drawContours(original, [blue_area], -1, (255, 0, 0), 2)
    cv2.rectangle(original, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Create a binary mask for boxes by thresholding
    box_mask = filter_image(image)

    # Find contours for the boxes
    box_contours, _ = cv2.findContours(box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if each box is outside the blue area based on the bounding rectangle
    for contour in box_contours:
        box_x, box_y, box_w, box_h = cv2.boundingRect(contour)

        box_rect = {
            "x_min": box_x,
            "x_max": box_x + box_w,
            "y_min": box_y,
            "y_max": box_y + box_h,
        }

        # Check if the box cuts the left, right, or bottom segments of the blue area
        is_outside = (
            box_rect["x_min"] < blue_rect["x_min"]
            or box_rect["x_max"] > blue_rect["x_max"]
            or box_rect["y_max"] > blue_rect["y_max"]
        )

        # Draw the box on the image
        color = (0, 255, 0) if not is_outside else (0, 0, 255)
        cv2.rectangle(
            original, (box_rect["x_min"], box_rect["y_min"]),
            (box_rect["x_max"], box_rect["y_max"]),
            color, 2
        )

        # Display a message if the box is outside
        if is_outside:
            cv2.putText(original, "Cartons dehors !", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Result", original)
            cv2.imshow("Blue Mask", blue_mask)
            cv2.imshow("Box Mask", box_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

    # If all boxes are inside
    cv2.putText(original, "Ok cartons dedans", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the results
    cv2.imshow("Result", original)
    cv2.imshow("Blue Mask", blue_mask)
    cv2.imshow("Box Mask", box_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test the function
root = "iot_camera_image_detection/test_data/test/"
image_paths = ["Camera 1_line_PBimgN5.jpg","Camera 1_line_PBimgN6.jpg", "Camera 1_line_PBimgN7.jpg", "Camera 1_line_PBimgN8.jpg", "Camera 1_line_PBimgN9.jpg", "Camera 1_line_PBimgN1.jpg", "Camera 1_line_PBimgN2.jpg", "Camera 1_line_PBimgN3.jpg", "Camera 1_line_PBimgN4.jpg"] 
            #    "Camera 1_line_PBimgN4.jpg","Camera 1_2PBimgN3.jpg", "Camera 1_imgN3.jpg","Camera 1_imgN5.jpg", "Camera 1_imgN6.jpg", "Camera 1_imgN7.jpg", "Camera 1_imgN1.jpg" ]

for path in image_paths:
    check_boxes(root+path)
