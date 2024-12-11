import cv2
import numpy as np

# Load the first and second images
image1_path = "iot_camera_image_detection/test_data/cardboardOK.png"  # Cardboard inside the area
image2_path = "iot_camera_image_detection/test_data/carboardNOK.png"  # Cardboard cutting the black line

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Function to detect black lines and analyze gaps
def analyze_line_continuity(image):
    # Convert to grayscale and apply thresholding to detect black regions
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the black line is the largest contour by area
    if contours:
        black_line_contour = max(contours, key=cv2.contourArea)

        # Draw the detected black line contour
        cv2.drawContours(image, [black_line_contour], -1, (0, 255, 0), 3)

        # Create a mask for the black line
        mask = np.zeros_like(thresh)
        cv2.drawContours(mask, [black_line_contour], -1, 255, -1)

        # Check for gaps (regions in the mask with no black line due to obstruction)
        gaps = cv2.bitwise_and(thresh, cv2.bitwise_not(mask))

        # Count non-zero pixels in the gaps
        gap_pixels = cv2.countNonZero(gaps)
        print(gap_pixels)
        # Threshold to determine if there is an obstruction
        if gap_pixels > 10000:  # Adjust this value based on the size of the line
            return "Erreur mauvais placement"
        else:
            return "Correct placement"
    else:
        return "Black line not detected"

# Analyze the two images
result1 = analyze_line_continuity(image1)
# print(result1)
result2 = analyze_line_continuity(image2)

# Show results
print(result1, result2)
