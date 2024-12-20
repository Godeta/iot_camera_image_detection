# OpenCV Python program to detect cars in video frame
# import libraries of python OpenCV
import cv2
import face_recognition

# capture frames from a video
# cap = cv2.VideoCapture( r'C:\Users\vchan\OneDrive\Desktop\Untitled Folder\people_presenting.mp4',0)
rstp_url = 'http://192.168.43.65:81/stream'
cap = cv2.VideoCapture(rstp_url)

# Initialize variables
face_locations = []

while True:
    # Grab a single frame of video
    ret, frame = cap.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)

    # Display the results
    for top, right, bottom, left in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)
    
    # Wait for Enter key to stop
    if cv2.waitKey(25) == 13:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
