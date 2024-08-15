import cv2
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import datetime

# Global variables
is_recording = False
brightness_value = 1.0
recording_path = None

# Load the cascade classifier
face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def apply_portrait_mode(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(frame)
    faces = face_cap.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"Faces detected: {len(faces)}")  # Debugging
    for (x, y, w, h) in faces:
        mask[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
    blurred = cv2.GaussianBlur(frame, (21, 21), 30)
    portrait_frame = np.where(mask > 0, frame, blurred)
    return portrait_frame

def toggle_recording(change):
    global is_recording, recording_path
    if change['new']:
        is_recording = True
        recording_path = f'recording_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        global out
        out = cv2.VideoWriter(recording_path, fourcc, 20.0, (640, 480))
        print(f"Started recording: {recording_path}")  # Debugging
    else:
        is_recording = False
        if 'out' in globals():
            out.release()
            print(f"Stopped recording: {recording_path}")  # Debugging

def capture_photo(b):
    ret, frame = cap.read()
    if ret:
        frame = cv2.convertScaleAbs(frame, alpha=brightness_value, beta=0)
        photo_path = f'photo_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        cv2.imwrite(photo_path, frame)
        print(f"Photo captured: {photo_path}")  # Debugging
    else:
        print("Failed to capture photo.")  # Debugging

def update_brightness(change):
    global brightness_value
    brightness_value = change['new']
    print(f"Brightness updated: {brightness_value}")  # Debugging

# Initialize video capture
cap = cv2.VideoCapture(0)

# Check if the video capture is initialized correctly
if not cap.isOpened():
    print("Error: Could not open video capture.")
else:
    print("Video capture opened successfully.")

# Create UI elements
brightness_slider = widgets.FloatSlider(value=1.0, min=0.5, max=2.0, step=0.1, description='Brightness')
record_button = widgets.ToggleButton(value=False, description='Record', button_style='success')
capture_button = widgets.Button(description='Capture Photo', button_style='info')
portrait_mode_toggle = widgets.ToggleButton(value=False, description='Portrait Mode', button_style='info')

# Set event handlers
brightness_slider.observe(update_brightness, names='value')
record_button.observe(toggle_recording, names='value')
capture_button.on_click(capture_photo)

# Display UI elements
display(brightness_slider, record_button, capture_button, portrait_mode_toggle)

# Initialize Matplotlib figure
fig, ax = plt.subplots()
plt.ion()
plt.show()

# Main loop to display video frames
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturing video frame.")
            continue

        # Adjust brightness
        frame = cv2.convertScaleAbs(frame, alpha=brightness_value, beta=0)

        # Apply portrait mode if toggled
        if portrait_mode_toggle.value:
            frame = apply_portrait_mode(frame)

        # Convert to RGB for Matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display with Matplotlib
        ax.clear()
        ax.imshow(frame_rgb)
        ax.axis('off')
        plt.draw()
        plt.pause(0.01)

        # Record video if enabled
        if is_recording:
            if 'out' in globals():
                out.write(frame)

except KeyboardInterrupt:
    print("Video stream stopped.")

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
