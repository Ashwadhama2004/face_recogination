import cv2
import numpy as np
import datetime

# Global variables
is_recording = False
brightness_value = 1.0  # Set initial brightness value here
recording_path = None
face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def apply_portrait_mode(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(frame)
    faces = face_cap.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"Faces detected: {len(faces)}")
    for (x, y, w, h) in faces:
        mask[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
    blurred = cv2.GaussianBlur(frame, (21, 21), 30)
    portrait_frame = np.where(mask > 0, frame, blurred)
    return portrait_frame

def toggle_recording():
    global is_recording, recording_path, out
    if not is_recording:
        is_recording = True
        recording_path = f'recording_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(recording_path, fourcc, 20.0, (640, 480))
        print(f"Started recording: {recording_path}")
    else:
        is_recording = False
        if 'out' in globals():
            out.release()
            print(f"Stopped recording: {recording_path}")

def capture_photo(frame):
    frame = cv2.convertScaleAbs(frame, alpha=brightness_value, beta=0)
    photo_path = f'photo_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    cv2.imwrite(photo_path, frame)
    print(f"Photo captured: {photo_path}")

def update_brightness(value):
    global brightness_value
    brightness_value = value / 100.0
    print(f"Brightness updated: {brightness_value}")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Check if the video capture is initialized correctly
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

portrait_mode = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturing video frame.")
            continue

        # Adjust brightness
        frame = cv2.convertScaleAbs(frame, alpha=brightness_value, beta=0)

        # Apply portrait mode if toggled
        if portrait_mode:
            frame = apply_portrait_mode(frame)

        # Display the frame
        cv2.imshow('Video', frame)

        # Check for user input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Press 'q' to quit
            break
        elif key == ord('r'):  # Press 'r' to toggle recording
            toggle_recording()
        elif key == ord('c'):  # Press 'c' to capture photo
            capture_photo(frame)
        elif key == ord('p'):  # Press 'p' to toggle portrait mode
            portrait_mode = not portrait_mode
            print(f"Portrait mode {'enabled' if portrait_mode else 'disabled'}")
        elif key == ord('+'):  # Press '+' to increase brightness
            update_brightness(min(brightness_value * 100 + 10, 200))  # Increase by 10, max 200
        elif key == ord('-'):  # Press '-' to decrease brightness
            update_brightness(max(brightness_value * 100 - 10, 0))  # Decrease by 10, min 0

except KeyboardInterrupt:
    print("Video stream stopped.")

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()