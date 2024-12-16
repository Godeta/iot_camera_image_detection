import cv2
import numpy as np

# Global Variables
root = "iot_camera_image_detection/test_data/"
image_paths = ["Camera 1_PBimgN3.jpg", "Camera 1_imgN3.jpg","Camera 1_imgN5.jpg", "Camera 1_imgN6.jpg", "Camera 1_imgN7.jpg", "Camera 1_imgN1.jpg" ]
lower_blue = np.array([15, 10, 110])  # Lower HSV threshold for blue
upper_blue = np.array([65, 30, 260])  # Upper HSV threshold for blue


def preprocess_image(image):
    """Preprocess the image by converting it to HSV and applying histogram equalization."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    processed_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)


def color_segmentation(image):
    """Apply color segmentation to detect blue areas."""
    mask = cv2.inRange(image, lower_blue, upper_blue)
    return mask


def area_segmentation(mask):
    """Find the largest contour in the mask (blue area)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv2.contourArea)
    return None


def apply_erosion(mask, kernel_size=3, iterations=1):
    """Apply erosion to clean up the mask."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(mask, kernel, iterations=iterations)


def detect_edges(image):
    """Detect edges using Canny edge detection."""
    return cv2.Canny(image, 50, 150)

def opening(mask, kernel_size=3, iterations=1):
    """
    Reinforce the white areas in the binary mask by applying morphological opening.
    This helps to remove noise while maintaining the shapes.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return opened_mask


def labeling(mask):
    """
    Label connected regions in the binary mask and return the labeled image.
    Each distinct region is assigned a unique label.
    """
    num_labels, labeled_image = cv2.connectedComponents(mask)
    return num_labels, labeled_image


def filter_by_area(mask, min_size):
    """
    Keep only areas in the binary mask that are larger than the given size.
    Smaller regions are removed.
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)

    for cnt in contours:
        if cv2.contourArea(cnt) >= min_size:
            # Draw the large contours on a blank mask
            cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return filtered_mask


def area_size(mask):
    """
    Return the size (area in pixels) of the white regions in the binary mask.
    """
    return np.sum(mask == 255)

def biggest_area_dimensions(mask):
    """
    Find the dimensions (width and height) of the bounding rectangle of the largest contour in the binary mask.
    Returns the width and height of the largest area.
    """
    # Find all contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0, 0  # Return 0,0 if no contours are found

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the largest contour
    x, y, width, height = cv2.boundingRect(largest_contour)

    return width, height


def close_and_fill(segmentation_mask, edges, min_white_area=5000):
    """
    Combine segmentation and edges.
    - Close gaps in edges and fill white regions in the segmentation mask with large areas.
    """
    # Close gaps in edges
    kernel = np.ones((5, 5), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Combine with segmentation mask
    combined = cv2.bitwise_or(closed_edges, segmentation_mask)

    # Find contours in the segmentation mask
    contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_white_area:
            # Fill large areas in the segmentation mask
            cv2.drawContours(combined, [cnt], -1, 255, thickness=cv2.FILLED)

    return combined


def find_and_analyze_contours(edges, blue_contour, original):
    """
    Find contours of boxes and return an image with contours drawn.
    """
    result_image = original.copy()
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4 and cv2.contourArea(cnt) > 500:  # Box detection
            x, y, w, h = cv2.boundingRect(approx)
            box_center = (x + w // 2, y + h // 2)

            # Check if the box is inside the blue area
            # inside_blue = cv2.pointPolygonTest(blue_contour, box_center, False) >= 0 if blue_contour else False
            inside_blue = cv2.pointPolygonTest(blue_contour, box_center, False) >= 0 if blue_contour is not None and len(blue_contour) > 0 else False


            color = (0, 255, 0) if inside_blue else (0, 0, 255)  # Green if inside, Red if outside
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
    return result_image


def process_image(image_path):
    """Process a single image through the pipeline."""
    # Step 1: Load and preprocess the image
    image = cv2.imread(image_path)
    original = image.copy()
    hsv_image = preprocess_image(image)

    # Step 2: Apply color segmentation
    mask_blue = color_segmentation(hsv_image)

    # Step 3: Find the largest blue area contour
    blue_contour = area_segmentation(mask_blue)

    # Step 4: Apply erosion to clean up the mask
    eroded_mask = apply_erosion(mask_blue)

    # Step 5: Detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = detect_edges(gray)

    # Step 6: Close gaps and fill white areas
    filled_mask = close_and_fill(eroded_mask, edges)

    # Step 7: Find and analyze contours
    result_image = find_and_analyze_contours(edges, blue_contour, image)

        # Apply opening to clean the mask
    opened_mask = opening(mask_blue)

    # Define crop coordinates
    start_y, end_y = 5, 240
    start_x, end_x = 15, 290
    # Create a black image of the same size as the opened_mask
    cropped_img = np.zeros_like(opened_mask)
    # Copy the pixels within the crop to their positions in the black image
    cropped_img[start_y:end_y, start_x:end_x] = opened_mask[start_y:end_y, start_x:end_x]

    # Label connected regions
    num_labels, labeled_image = labeling(opened_mask)

    # Filter by area size
    filtered_mask = filter_by_area(cropped_img, min_size=1500)

    # Get the size of a specific region
    # region_size = area_size(filtered_mask)
    largest_width, largest_height = biggest_area_dimensions(filtered_mask)
    print(f"Largest Area - Width: {largest_width}, Height: {largest_height}, hauteur estimée : {largest_height/2} cm")

    print(f"Number of regions: {num_labels}")
    # print(f"Region size: {region_size} pixels")

    # Display results
    # cv2.imshow("Original Image", original)
    # cv2.imshow("HSV Image", hsv_image)

    cv2.imshow("Blue Area Mask", mask_blue)
    cv2.imshow("Eroded Mask", eroded_mask)
    cv2.imshow("Filled Mask", filled_mask)
    cv2.imshow("Final Result", result_image)
    cv2.imshow("Opened", opened_mask)
    # cv2.imshow("labeled", labeled_image)
    cv2.imshow("filtered mask", filtered_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Main Loop
for path in image_paths:
    process_image(root+path)
