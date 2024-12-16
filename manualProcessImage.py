import cv2
import numpy as np

# Global Variables
image_paths = ["iot_camera_image_detection/test_data/Camera 1_line_imgN3.jpg", "iot_camera_image_detection/test_data/Camera 1_imgN4.jpg", "iot_camera_image_detection/test_data/Camera 1_imgN6.jpg", "iot_camera_image_detection/test_data/Camera 1_imgN7.jpg"]
color_modes = ["BGR", "HSV", "GRAY"]  # Supported color modes
current_mode_index = 0  # Current color mode index
edge_detection_enabled = False  # Flag for edge detection
start_point = None  # Starting point for mouse drag
end_point = None  # Ending point for mouse drag
is_drawing = False  # Mouse drag flag
roi_selected = False  # Flag for region selection


def draw_rectangle(event, x, y, flags, param):
    """
    Mouse callback function for area selection.
    """
    global start_point, end_point, is_drawing, roi_selected

    if event == cv2.EVENT_LBUTTONDOWN:  # Start drawing
        start_point = (x, y)
        end_point = None
        is_drawing = True
        roi_selected = False

    elif event == cv2.EVENT_MOUSEMOVE and is_drawing:  # Update rectangle
        end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:  # Finalize rectangle
        end_point = (x, y)
        is_drawing = False
        roi_selected = True

def get_rectangle_coordinates():
    """
    Get the coordinates of the rectangle area and print them.
    """
    if roi_selected and start_point and end_point:
        x1, y1 = start_point
        x2, y2 = end_point
        start_x, end_x = min(x1, x2), max(x1, x2)
        start_y, end_y = min(y1, y2), max(y1, y2)
        print(f"Rectangle Coordinates - Start X: {start_x}, End X: {end_x}, Start Y: {start_y}, End Y: {end_y}")


def area_analyse(image):
    """
    Analyze the colors in the selected area.
    """
    if roi_selected and start_point and end_point:
        x1, y1 = start_point
        x2, y2 = end_point
        roi = image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

        if roi.size > 0:
            # Calculate the mean color of the region
            mean_color = cv2.mean(roi)[:3]
            return mean_color


def change_color_mode(image):
    """
    Change the image color mode.
    """
    global current_mode_index
    current_mode_index = (current_mode_index + 1) % len(color_modes)
    mode = color_modes[current_mode_index]

    if mode == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif mode == "GRAY":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image  # Default to BGR


def edge_detect(image):
    """
    Apply edge detection.
    """
    return cv2.Canny(image, 50, 150)


def process_image(image_path):
    """
    Main loop to handle image processing, mouse interactions, and key events.
    """
    global edge_detection_enabled

    # Load and display the image
    image = cv2.imread(image_path)
    original_image = image.copy()
    processed_image = image.copy()
    cv2.namedWindow("Image Analysis")
    cv2.setMouseCallback("Image Analysis", draw_rectangle)

    while True:
        display_image = processed_image.copy()

        # Draw the rectangle if dragging
        if start_point and end_point:
            cv2.rectangle(display_image, start_point, end_point, (0, 255, 0), 2)

        # If edge detection is enabled, apply it
        if edge_detection_enabled:
            edges = edge_detect(cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY))
            display_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Show the image
        cv2.imshow("Image Analysis", display_image)

        

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Quit
            break
        elif key == ord("c"):  # Change color mode
            processed_image = change_color_mode(original_image)
        elif key == ord("e"):  # Toggle edge detection
            edge_detection_enabled = not edge_detection_enabled
        elif key == ord("p"): # Print colors in selected area
            mean_color = area_analyse(processed_image)
            if mean_color:
                print(f"Selected Area - Mean Color: {mean_color}")
        elif key == ord("k"):  # Print rectangle coordinates
            get_rectangle_coordinates()
    cv2.destroyAllWindows()


# Main Loop
for path in image_paths:
    process_image(path)
