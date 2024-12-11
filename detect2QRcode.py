import cv2

# File path for the video to process
video_ok_path = "2_qr_code_tests/OK.mp4"

# RTSP URL for live video stream (replace with your stream URL)
# 65, 8
rstp_url = 'http://192.168.43.65:81/stream'

# Initialize a global counter for saved frames
saved_frame_counter = 1

# Function to process a single frame and detect QR codes
def process_frame(frame):
    """
    Process a frame to detect QR codes and handle the logic for displaying
    frames with two QR codes.

    Args:
        frame (numpy.ndarray): The current video frame.

    Returns:
        numpy.ndarray: The processed frame with QR code detection.
    """

    global saved_frame_counter  # Explicitly declare the global variable
    # Initialize the QR code detector
    qcd = cv2.QRCodeDetector()

    # Detect and decode QR codes in the frame
    ret_qr, decoded_info, points, _ = qcd.detectAndDecodeMulti(frame)

    if ret_qr:
        detected_count = 0  # Counter for detected QR codes
        for s, p in zip(decoded_info, points):
            if s:  # Only count valid QR codes
                print(s)
                detected_count += 1
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            frame = cv2.polylines(frame, [p.astype(int)], True, color, 4)
        
        # If two QR codes are detected, print a message and display in a new window
        if detected_count == 2:
            print("Ready to take picture")
            cv2.imshow("Two QR Codes Detected", frame)

    # Check for user input to save the frame as an image
    key = cv2.waitKey(1) & 0xFF
    if key == ord('t'):
        # Save the current frame as a JPG image
        filename = f"imageN{saved_frame_counter}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Frame saved as: {filename}")
        saved_frame_counter += 1

    return frame

# Function to play and process a video
def play_video(video_path):
    """
    Play a video file and process each frame using the process_frame function.

    Args:
        video_path (str): Path to the video file.
    """
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
    cv2.destroyAllWindows()

# Function to process a live video stream
def play_live(rtsp_url):
    """
    Process a live video stream from an RTSP URL and detect QR codes.

    Args:
        rtsp_url (str): URL of the live video stream.
    """
    cam = cv2.VideoCapture(rtsp_url)

    if not cam.isOpened():
        print(f"Error opening live stream: {rtsp_url}")
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to read frame from live stream.")
            break

        # Process the current frame
        processed_frame = process_frame(frame)

        # Show the processed frame
        cv2.imshow("Live Video Analysis", processed_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    # Run the video analysis on the specified video
    # play_video(video_ok_path)
    play_live(rstp_url)
