import cv2
import numpy as np
import time
import os

# ==============================
# Configuration for Calibrator
# ==============================
# --- INPUT SOURCE CONFIGURATION ---
USE_IMAGE_FILE_INPUT_CALIB = True # <<< TOGGLE: True to use a static image file for calibration
IMAGE_INPUT_FILE_CALIB = "video_input/kraken2.png" # <<< Specify your image file here
# For example: "C:/Projects/0-demos-and-testing/robo-vision/video_output/snapshot_opponent.png"

# If not using image, then specify video or camera:
USE_VIDEO_FILE_INPUT_CALIB = False # <<< TOGGLE: True to use a video file for calibration
VIDEO_INPUT_FILE_CALIB = "video_output/raw_2026-03-02_15-30-00.mp4" # <<< Specify a raw video file for calibration

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
cv2.resizeWindow("Color Calibration Trackbars", 600, 300)

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
    cv2.setTrackbarPos("H_min", "Color Calibration Trackbars", 70)
    cv2.setTrackbarPos("H_max", "Color Calibration Trackbars", 100)
    cv2.setTrackbarPos("S_min", "Color Calibration Trackbars", 100)
    cv2.setTrackbarPos("S_max", "Color Calibration Trackbars", 255)
    cv2.setTrackbarPos("V_min", "Color Calibration Trackbars", 100)
    cv2.setTrackbarPos("V_max", "Color Calibration Trackbars", 255)
    print("No previous calibration found. Starting with default bluish-green values.")


cap = None # Initialize cap to None
input_is_image = False
fixed_frame = None # To store the image if using image file input
calib_fps = 30.0 # Default FPS for live camera/video, used for waitKey

if USE_IMAGE_FILE_INPUT_CALIB:
    fixed_frame = cv2.imread(IMAGE_INPUT_FILE_CALIB)
    if fixed_frame is None:
        print(f"Error: Could not load image file '{IMAGE_INPUT_FILE_CALIB}' for calibration. Falling back to camera/video.")
        USE_IMAGE_FILE_INPUT_CALIB = False # Disable image input if failed
    else:
        print(f"Image file '{IMAGE_INPUT_FILE_CALIB}' loaded for calibration.")
        input_is_image = True
        # Resize image to FRAME_WIDTH/HEIGHT if needed
        fixed_frame = cv2.resize(fixed_frame, (FRAME_WIDTH, FRAME_HEIGHT))

if not input_is_image: # Proceed with video or camera if image loading failed or not selected
    if USE_VIDEO_FILE_INPUT_CALIB:
        cap = cv2.VideoCapture(VIDEO_INPUT_FILE_CALIB)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{VIDEO_INPUT_FILE_CALIB}' for calibration. Falling back to camera.")
            USE_VIDEO_FILE_INPUT_CALIB = False # Disable video input if failed
        else:
            calib_fps = cap.get(cv2.CAP_PROP_FPS) # Get FPS from video for playback speed
            print(f"Video file '{VIDEO_INPUT_FILE_CALIB}' opened for calibration at {calib_fps:.2f} FPS.")
    
    if not USE_VIDEO_FILE_INPUT_CALIB: # Fallback to live camera
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"FATAL Error: Could not open video stream from camera index {CAMERA_INDEX} using CAP_DSHOW backend.")
            print("Please check if the camera is connected and the index is correct. Exiting.")
            exit()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        print(f"Live camera opened for calibration.")

while True:
    if input_is_image:
        frame = fixed_frame
        ret = True
        wait_time_loop = 1 # <<< FIX: Set to 1 for continuous updates with image input
    else:
        ret, frame = cap.read()
        if not ret:
            if USE_VIDEO_FILE_INPUT_CALIB:
                print("End of video file or failed to grab frame. Looping video for calibration.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video if using file
                continue # Try to read again
            else:
                print("Failed to grab frame. Exiting...")
                break
        wait_time_loop = max(1, int(1000 / calib_fps)) # Match playback speed to source FPS


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("H_min", "Color Calibration Trackbars")
    h_max = cv2.getTrackbarPos("H_max", "Color Calibration Trackbars")
    s_min = cv2.getTrackbarPos("S_min", "Color Calibration Trackbars")
    s_max = cv2.getTrackbarPos("S_max", "Color Calibration Trackbars")
    v_min = cv2.getTrackbarPos("V_min", "Color Calibration Trackbars")
    v_max = cv2.getTrackbarPos("V_max", "Color Calibration Trackbars")

    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    cv2.imshow("Original Feed", frame)
    cv2.imshow("Masked Opponent", mask)

    key = cv2.waitKey(wait_time_loop) & 0xFF # <<< FIX: Use the new wait_time_loop

    if key == ord('s'):
        np.savez(CALIBRATION_FILE, lower_hsv=lower_hsv, upper_hsv=upper_hsv)
        print(f"HSV values saved to '{CALIBRATION_FILE}':")
        print(f"  Lower: {lower_hsv}")
        print(f"  Upper: {upper_hsv}")
        # Removed the extra print for image, as it's now continuously updating
    elif key == 27: # ESC key
        break

if cap is not None:
    cap.release()
cv2.destroyAllWindows()
print("Color calibration application closed.")