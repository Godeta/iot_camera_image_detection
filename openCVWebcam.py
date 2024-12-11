import cv2
import numpy as np

# Open the default camera
cam = cv2.VideoCapture(1)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

# Function to detect squares
def detect_squares(mask, color_name):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) > 500:  # Detect squares with significant area
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)
            if color_name == "Red":
                cv2.imshow("Red Square Detected", frame)
            elif color_name == "Blue":
                cv2.imshow("Blue Square Detected", frame)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for red and blue
    red_lower1 = np.array([0, 120, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 70])
    red_upper2 = np.array([180, 255, 255])
    blue_lower = np.array([100, 150, 70])
    blue_upper = np.array([140, 255, 255])

    # Create masks for red and blue
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.add(red_mask1, red_mask2)  # Combine red masks
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # Detect and highlight squares
    detect_squares(red_mask, "Red")
    detect_squares(blue_mask, "Blue")

    # Write the frame to the output file
    out.write(frame)

    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()
