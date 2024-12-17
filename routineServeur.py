import os
import shutil
import time
import cv2

from processImagesTest2 import biggest_area_dimensions, filter_image, check_boxes

# processImagesTest2.py

# Paths for input and processed folders
input_folder = "photoCam"
processed_folder = "processedCam"

# Ensure processed folder exists
os.makedirs(processed_folder, exist_ok=True)

def process_checkQrCode(image_path):
    """Simulated function to process QrCode images."""
    print(f"Processing QrCode image: {image_path}")
    image = cv2.imread(image_path)
    if image is not None:
        # Initialize the QR code detector
        qcd = cv2.QRCodeDetector()

        # Detect and decode QR codes in the frame
        ret_qr, decoded_info, points, _ = qcd.detectAndDecodeMulti(image)

        if ret_qr:
            for s, p in zip(decoded_info, points):
                if s:
                    print(f"qr code value : {s}")
                # frame = cv2.polylines(image, [p.astype(int)], True, color, 4)
    else:
        print(f"Failed to read QrCode image: {image_path}")

def monitor_folder():
    """Continuously monitor the input folder for new files."""
    print("Monitoring folder for new files...")

    # Set of already processed files to avoid reprocessing
    processed_files = set()

    while True:
        # List all files in the input folder
        for filename in os.listdir(input_folder):
            file_path = os.path.join(input_folder, filename)

            # Skip directories or already processed files
            if not os.path.isfile(file_path) or filename in processed_files:
                continue

            # Move file to the processed folder
            new_path = os.path.join(processed_folder, filename)
            shutil.move(file_path, new_path)
            print(f"Moved {filename} to {processed_folder}")

            # Apply the appropriate process based on filename
            if filename.startswith("QrCode"):
                process_checkQrCode(new_path)
            elif filename.startswith("boxes"):
                check_boxes(new_path)
            else:
                print(f"Unknown file type: {filename}. Skipping processing.")

            # Add to processed files to avoid reprocessing
            processed_files.add(filename)

        # Wait for a short time before checking again
        time.sleep(1)

if __name__ == "__main__":
    monitor_folder()
