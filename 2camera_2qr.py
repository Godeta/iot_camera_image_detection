import cv2
import threading

# RTSP URLs for live video streams
rtsp_url_1 = 'http://192.168.43.65:81/stream'
rtsp_url_2 = 'http://192.168.43.8:81/stream'

# Initialize a global counter for saved frames
saved_frame_counter = 1

# Function to process a single frame and detect QR codes
def process_frame(frame, window_name):
    """
    Process a frame to detect QR codes and handle the logic for displaying
    frames with two QR codes.

    Args:
        frame (numpy.ndarray): The current video frame.
        window_name (str): Name of the display window

    Returns:
        numpy.ndarray: The processed frame with QR code detection.
    """
    global saved_frame_counter
    
    # Initialize the QR code detector
    qcd = cv2.QRCodeDetector()

    # Detect and decode QR codes in the frame
    ret_qr, decoded_info, points, _ = qcd.detectAndDecodeMulti(frame)

    if ret_qr:
        detected_count = 0
        for s, p in zip(decoded_info, points):
            if s:
                print(f"{window_name}: {s}")
                detected_count += 1
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            frame = cv2.polylines(frame, [p.astype(int)], True, color, 4)
        
        if detected_count == 2:
            print(f"{window_name}: Ready to take picture")
            cv2.imshow(f"{window_name} - Two QR Codes", frame)

    # Check for user input to save the frame
    key = cv2.waitKey(1) & 0xFF
    if key == ord('t'):
        filename = f"{window_name}_line_PBimgN{saved_frame_counter}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Frame saved as: {filename}")
        saved_frame_counter += 1

    return frame

def process_camera_stream(rtsp_url, window_name):
    """
    Process a live video stream from an RTSP URL and detect QR codes.

    Args:
        rtsp_url (str): URL of the live video stream.
        window_name (str): Name of the display window
    """
    cam = cv2.VideoCapture(rtsp_url)

    if not cam.isOpened():
        print(f"Error opening live stream: {rtsp_url}")
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            print(f"Failed to read frame from {window_name}")
            break

        # Process the current frame
        processed_frame = process_frame(frame, window_name)

        # Show the processed frame
        cv2.imshow(window_name, processed_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    # Create threads for each camera stream
    thread1 = threading.Thread(target=process_camera_stream, 
                             args=(rtsp_url_1, "Camera 1"))
    thread2 = threading.Thread(target=process_camera_stream, 
                             args=(rtsp_url_2, "Camera 2"))

    # Start both threads
    thread1.start()
    thread2.start()

    # Wait for both threads to complete
    thread1.join()
    thread2.join()