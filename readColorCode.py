import cv2
import numpy as np
import time  # Import the time module for adding a pause

# Open the default camera
cam = cv2.VideoCapture(1)
# rstp_url = 'http://192.168.143.239:81/stream'
# cam = cv2.VideoCapture(rstp_url)

# Define HSV ranges for colors
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([30, 255, 255])

# Define HSV ranges for other colors
color_ranges = {
    "Red1": ([0, 120, 70], [10, 255, 255]),
    "Red2": ([170, 120, 70], [180, 255, 255]),
    "Blue": ([100, 150, 70], [140, 255, 255]),
    "Green": ([40, 100, 70], [80, 255, 255]),
    "Orange": ([10, 100, 70], [20, 255, 255]),
    "Purple": ([140, 100, 70], [170, 255, 255]),
}

# Function to detect squares with uniform color
def detect_uniform_squares(mask, hsv_frame):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) > 500:  # Detect significant squares
            x, y, w, h = cv2.boundingRect(approx)
            square_region = hsv_frame[y:y+h, x:x+w]
            mean, stddev = cv2.meanStdDev(square_region)
            if np.mean(stddev) < 10:  # Check for uniformity (low standard deviation)
                squares.append((x, y, w, h))
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)
    return squares

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect yellow squares
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    yellow_squares = detect_uniform_squares(yellow_mask, hsv)

    # Check if two yellow squares are detected
    if len(yellow_squares) >= 2:
        yellow_squares = sorted(yellow_squares, key=lambda sq: sq[0])  # Sort by x-coordinate
        first_square = yellow_squares[0]
        second_square = yellow_squares[1]

        # Define the bounding region between the two yellow squares
        x_min = min(first_square[0], second_square[0])
        x_max = max(first_square[0] + first_square[2], second_square[0] + second_square[2])
        y_min = min(first_square[1], second_square[1])
        y_max = max(first_square[1] + first_square[3], second_square[1] + second_square[3])

         # Draw the rectangle with red borders on the frame
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
        region = frame[y_min:y_max, x_min:x_max]
        hsv_region = hsv[y_min:y_max, x_min:x_max]
        detected_colors = []

        # Detect squares of other colors in the region
        for color_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv_region, np.array(lower), np.array(upper))
            squares = detect_uniform_squares(mask, hsv_region)
            for sq in squares:
                # Draw the detected squares with red borders
                cv2.rectangle(frame, (sq[0] + x_min, sq[1] + y_min), (sq[0] + sq[2] + x_min, sq[1] + sq[3] + y_min), (255, 255,0), 2)

                # Check if the square size matches the yellow squares
                if 0.8 * first_square[2] <= sq[2] <= 1.2 * first_square[2]:  # Approximate size match
                    detected_colors.append(color_name)
                    break  # Only count each color once

        if detected_colors:
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
            print(f"Colors detected between the two yellow squares: {', '.join(detected_colors)}")
            
            # time.sleep(5)

    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()
