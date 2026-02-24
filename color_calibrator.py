import cv2
import numpy as np
import time
import os

# ==============================
# Configuration for Calibrator
# ==============================
CAMERA_INDEX = 1 # IMPORTANT: Use the same camera index as your main tracking script
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

CALIBRATION_FILE = "opponent_color_calibration.npz"

# Dummy function for trackbar callback (does nothing as values are read directly)
def nothing(x):
    pass

# ==============================
# Main Color Calibrator Logic
# ==============================

print("Starting Opponent Color Calibrator...")
print("Adjust the HSV trackbars to isolate the opponent bot.")
print(f"Press 's' to save the current HSV values to '{CALIBRATION_FILE}'.")
print("Press 'ESC' to exit without saving.")

# Create a window for trackbars
cv2.namedWindow("Color Calibration Trackbars")
cv2.resizeWindow("Color Calibration Trackbars", 600, 300) # Give it some reasonable size

# Create trackbars for HSV minimum and maximum values
cv2.createTrackbar("H_min", "Color Calibration Trackbars", 0, 179, nothing)
cv2.createTrackbar("H_max", "Color Calibration Trackbars", 179, 179, nothing)
cv2.createTrackbar("S_min", "Color Calibration Trackbars", 0, 255, nothing)
cv2.createTrackbar("S_max", "Color Calibration Trackbars", 255, 255, nothing)
cv2.createTrackbar("V_min", "Color Calibration Trackbars", 0, 255, nothing)
cv2.createTrackbar("V_max", "Color Calibration Trackbars", 255, 255, nothing)

# Load previous calibration values if they exist, and set trackbars
if os.path.exists(CALIBRATION_FILE):
    try:
        npzfile = np.load(CALIBRATION_FILE)
        lower_hsv_prev = npzfile['lower_hsv']
        upper_hsv_prev = npzfile['upper_hsv']

        cv2.setTrackbarPos("H_min", "Color Calibration Trackbars", lower_hsv_prev[0])
        cv2.setTrackbarPos("S_min", "Color Calibration Trackbars", lower_hsv_prev[1])
        cv2.setTrackbarPos("V_min", "Color Calibration Trackbars", lower_hsv_prev[2])
        cv2.setTrackbarPos("H_max", "Color Calibration Trackbars", upper_hsv_prev[0])
        cv2.setTrackbarPos("S_max", "Color Calibration Trackbars", upper_hsv_prev[1])
        cv2.setTrackbarPos("V_max", "Color Calibration Trackbars", upper_hsv_prev[2])
        print(f"Loaded previous HSV values from {CALIBRATION_FILE}.")
    except Exception as e:
        print(f"Error loading previous calibration: {e}. Starting with default values.")
else:
    # Set initial values based on your bluish-green bot from before
    cv2.setTrackbarPos("H_min", "Color Calibration Trackbars", 70)
    cv2.setTrackbarPos("H_max", "Color Calibration Trackbars", 100)
    cv2.setTrackbarPos("S_min", "Color Calibration Trackbars", 100)
    cv2.setTrackbarPos("S_max", "Color Calibration Trackbars", 255)
    cv2.setTrackbarPos("V_min", "Color Calibration Trackbars", 100)
    cv2.setTrackbarPos("V_max", "Color Calibration Trackbars", 255)
    print("No previous calibration found. Starting with default bluish-green values.")


cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"Error: Could not open video stream from camera index {CAMERA_INDEX} using CAP_DSHOW backend.")
    print("Please check if the camera is connected and the index is correct.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Skipping this frame...")
        # Importantly, we don't break here, we continue to try and read the next frame.
        # This allows the program to recover if the camera only briefly drops a frame.
        if cv2.waitKey(1) == 27: # Still allow ESC to exit if no frames are coming.
            break
        continue # Skip the rest of the loop for this iteration

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current positions of trackbars
    h_min = cv2.getTrackbarPos("H_min", "Color Calibration Trackbars")
    h_max = cv2.getTrackbarPos("H_max", "Color Calibration Trackbars")
    s_min = cv2.getTrackbarPos("S_min", "Color Calibration Trackbars")
    s_max = cv2.getTrackbarPos("S_max", "Color Calibration Trackbars")
    v_min = cv2.getTrackbarPos("V_min", "Color Calibration Trackbars")
    v_max = cv2.getTrackbarPos("V_max", "Color Calibration Trackbars")

    # Create NumPy arrays for lower and upper HSV bounds
    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])

    # Create the HSV mask
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Apply morphological operations for cleaning up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Display frames
    cv2.imshow("Original Feed", frame)
    cv2.imshow("Masked Opponent", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Save the current HSV values
        np.savez(CALIBRATION_FILE, lower_hsv=lower_hsv, upper_hsv=upper_hsv)
        print(f"HSV values saved to '{CALIBRATION_FILE}':")
        print(f"  Lower: {lower_hsv}")
        print(f"  Upper: {upper_hsv}")
    elif key == 27: # ESC key
        break

cap.release()
cv2.destroyAllWindows()
print("Color calibration application closed.")