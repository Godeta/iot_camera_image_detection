import cv2
import numpy as np

# Paths to the video files
video_nok_path = "iot_camera_image_detection/test_data/cardboardOK.mp4"
video_ok_path = "iot_camera_image_detection/test_data/cardboardNOK.mp4"


# Function to process a frame and analyze line continuity
def process_frame(frame):
    # Convert to grayscale and apply thresholding to detect black regions
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the black line is the largest contour by area
    if contours:
        black_line_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(black_line_contour)
        print(area)

        # Create a mask for the black line
        mask = np.zeros_like(thresh)
        cv2.drawContours(mask, [black_line_contour], -1, 255, -1)

        # Check for gaps (regions in the mask with no black line due to obstruction)
        gaps = cv2.bitwise_and(thresh, cv2.bitwise_not(mask))
        # Count non-zero pixels in the gaps
        gap_pixels = cv2.countNonZero(gaps)
        print(gap_pixels)

        # Draw the contour in red if the area is too large (indicating obstruction),
        # otherwise draw in green.
        if gap_pixels > 8000:  # Adjust threshold as needed for your specific video
            cv2.drawContours(frame, [black_line_contour], -1, (0, 0, 255), 3)  # Red for "too big"
        else:
            cv2.drawContours(frame, [black_line_contour], -1, (0, 255, 0), 3)  # Green for "ok"

    return frame

def process_frame_with_segmentation(frame):
    """
    Process the frame to detect black lines as closed regions using segmentation.
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV range for black color
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])  # Adjust upper limit for black detection

    # Create a binary mask for the black color
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # Find contours from the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours to analyze closed regions
    for contour in contours:
        # Approximate the contour to reduce noise
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Calculate area and perimeter of the contour
        area = cv2.contourArea(approx)
        convex = cv2.isContourConvex(approx)
        perimeter = cv2.arcLength(approx, True)
        print("a : " + str(area) + " p : " + str(perimeter) + " c ? " + str(convex))
        # Heuristic: Detect valid closed black regions based on area and shape
        if area > 1000 and convex:  # Adjust area threshold
            # Draw in green for valid closed regions
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
        else:
            # Draw in red for incomplete regions or small areas
            cv2.drawContours(frame, [approx], -1, (0, 0, 255), 3)

    return frame


# Test the updated process_frame_with_segmentation function on video analysis
def play_video_with_segmentation(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the current frame with segmentation
        processed_frame = process_frame_with_segmentation(frame)

        # Show the processed frame
        cv2.imshow("Video Analysis with Segmentation", processed_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()


# Test with carboard videos (assuming files are uploaded)
print("Analyzing carboardOK.mp4 with segmentation...")
play_video_with_segmentation(video_ok_path)

print("Analyzing carboardNOK.mp4 with segmentation...")
play_video_with_segmentation(video_nok_path)

cv2.destroyAllWindows()



# Function to play and process a video
def play_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the current frame
        processed_frame = process_frame(frame)

        # Show the processed frame
        cv2.imshow("Video Analysis", processed_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()


# # Analyze the two videos
# print("Analyzing video: carboardOK.mp4")
# play_video(video_ok_path)

# print("Analyzing video: carboardNOK.mp4")
# play_video(video_nok_path)

# cv2.destroyAllWindows()
