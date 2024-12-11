import cv2
import numpy as np

# Load images
image_paths = ["iot_camera_image_detection/test_data/frameN3.jpg", "iot_camera_image_detection/test_data/frameN4.jpg"]

# Global variables for mouse interaction
start_point = None
end_point = None
is_drawing = False
selected_roi = None

def mouse_callback(event, x, y, flags, param):
    global start_point, end_point, is_drawing, selected_roi
    
    if event == cv2.EVENT_LBUTTONDOWN:  # Start drawing
        is_drawing = True
        start_point = (x, y)
        end_point = None

    elif event == cv2.EVENT_MOUSEMOVE:  # Update rectangle while dragging
        if is_drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:  # Finish drawing
        is_drawing = False
        end_point = (x, y)

        if start_point and end_point:  # Ensure valid points
            x1, y1 = start_point
            x2, y2 = end_point
            x1, x2 = sorted([x1, x2])  # Ensure top-left to bottom-right
            y1, y2 = sorted([y1, y2])
            selected_roi = (x1, y1, x2, y2)

def process_image_with_mouse(image):
    global start_point, end_point, selected_roi

    # Load the image
    clone = image.copy()

    # Set up mouse callback
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", mouse_callback)

    while True:
        # Display the image with the current rectangle if drawing
        display_image = clone.copy()
        if start_point and end_point:
            cv2.rectangle(display_image, start_point, end_point, (0, 255, 0), 2)

        cv2.imshow("Select ROI", display_image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):  # Quit the current image
            break

        if key == ord("c") and selected_roi:  # Confirm ROI and calculate pixel values
            x1, y1, x2, y2 = selected_roi

            # Extract the selected region
            roi = image[y1:y2, x1:x2]

            # Compute the mean pixel values
            mean_values = cv2.mean(roi)[:3]  # Ignore alpha if present
            print(f"Mean BGR values in selected region: {mean_values}")

            # Highlight the region in red
            cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 0, 255), 2)

def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    original = image.copy()

    # Step 1: Improve contrast
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Step 2: Detect the blue area
#     # Mean BGR values in selected region: (94.41758241758242, 52.142857142857146, 22.98901098901099)
# Mean BGR values in selected region: (83.84848484848484, 64.13636363636364, 41.96969696969697)
# Mean BGR values in selected region: (108.47441860465116, 85.14883720930233, 65.5953488372093)
# Mean BGR values in selected region: (79.2967032967033, 46.956043956043956, 27.681318681318682)
    # lower_blue = np.array([120, 130, 130])  # Adjust for "bland" blue
    # upper_blue = np.array([145, 162, 175])
    # lower_blue = np.array([72, 40, 20])  # Adjust for "bland" blue
    # upper_blue = np.array([120, 100, 90])

    lower_blue = np.array([10, 20, 130])  # Adjust for "bland" blue
    upper_blue = np.array([70, 70, 190])
    mask_blue = cv2.inRange(image, lower_blue, upper_blue)

    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_area_contour = max(contours, key=cv2.contourArea) if contours else None

    if blue_area_contour is not None:
        cv2.drawContours(original, [blue_area_contour], -1, (255, 0, 0), 2)  # Blue contour for blue area

    # Step 3: Detect the boxes using edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Detect rectangles based on contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4 and cv2.contourArea(cnt) > 500:
            # Detect rectangle
            x, y, w, h = cv2.boundingRect(approx)
            box_center = (x + w // 2, y + h // 2)

            # Check if the box is inside the blue area
            inside_blue = cv2.pointPolygonTest(blue_area_contour, box_center, False) >= 0 if blue_area_contour else False

            color = (0, 255, 0) if inside_blue else (0, 0, 255)  # Green if inside, Red if outside
            cv2.rectangle(original, (x, y), (x + w, y + h), color, 2)

    # Step 4: Show the results
    cv2.imshow(f"Processed Image - {image_path}", original)
    cv2.imshow(f"Contrast image", image)
    cv2.imshow("Blue Area Mask", mask_blue)
    cv2.imshow("Edge Detection", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def area_analyse():
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    original = image.copy()
    process_image_with_mouse(original)

    # Step 1: Improve contrast
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    process_image_with_mouse(image)

# Process each image
for path in image_paths:
    # process_image(path)
    area_analyse()
    

cv2.destroyAllWindows()