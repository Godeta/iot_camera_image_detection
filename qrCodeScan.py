import cv2

camera_id = 1
delay = 1
window_name = 'OpenCV QR Code'

qcd = cv2.QRCodeDetector()
# cam = cv2.VideoCapture(camera_id)
rstp_url = 'http://192.168.43.8:81/stream'
cam = cv2.VideoCapture(rstp_url)

while True:
    ret, frame = cam.read()

    if ret:
        ret_qr, decoded_info, points, _ = qcd.detectAndDecodeMulti(frame)
        if ret_qr:
            for s, p in zip(decoded_info, points):
                if s:
                    print(s)
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                frame = cv2.polylines(frame, [p.astype(int)], True, color, 8)
        cv2.imshow(window_name, frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cv2.destroyWindow(window_name)