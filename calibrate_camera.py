import cv2
import numpy as np
import glob
import os
import time # Import time for delays

# Define the chessboard pattern dimensions (number of inner corners)
CHECKERBOARD = (7, 7) # IMPORTANT: Changed to (7, 7) for an 8x8 square chessboard
# Size of one square in the chessboard, e.g., in millimeters. This is for 3D reconstruction.
# If you only care about angles and pixel distances, the exact size isn't strictly necessary for intrinsic params.
# But good practice to define for full pose estimation.
SQUARE_SIZE_MM = 20.0 # Make sure to accurately measure your chessboard squares

# Termination criteria for the iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points: (0,0,0), (1,0,0), (2,0,0) ... (CHECKERBOARD[0]-1, CHECKERBOARD[1]-1, 0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM

# Arrays to store object points and image points from all calibration images
objpoints = [] # 3D point in real world space
imgpoints = [] # 2D points in image plane

# --- Image Capture for Calibration ---
# Make sure a 'calibration_images' directory exists
if not os.path.exists("calibration_images"):
    os.makedirs("calibration_images")

# Use the same camera index as your external webcam for calibration
calib_cap = cv2.VideoCapture(1) # IMPORTANT: Use CAMERA_INDEX = 1 here for your external webcam
if not calib_cap.isOpened():
    print(f"Error: Could not open video stream for calibration from camera index 1.")
    exit()

calib_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Use the same resolution as your main program
calib_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n--- Camera Calibration Image Capture ---")
print("Instructions:")
print("1. Display a printed chessboard pattern to the camera.")
print("2. Move the chessboard to various positions and orientations within the camera's view.")
print("3. Ensure the chessboard fills a significant part of the frame and is slightly angled.")
print("4. Press 's' to save a good calibration image (aim for 15-20 diverse images).")
print("5. Press 'ESC' to finish capturing and start calibration.")

img_count = 0
while True:
    ret, frame = calib_cap.read()
    if not ret:
        print("Failed to grab frame during calibration capture. Exiting.")
        break

    display_frame = frame.copy()
    cv2.putText(display_frame, f"Saved Images: {img_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Checkerboard: {CHECKERBOARD[0]}x{CHECKERBOARD[1]} inner corners", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display_frame, "Press 's' to save, 'ESC' to calibrate", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow('Calibration Capture', display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        filename = f"calibration_images/calib_img_{img_count:03d}.png"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        img_count += 1
        time.sleep(0.1) # Small delay to avoid multiple captures for one press
    elif key == 27: # ESC key
        break

calib_cap.release()
cv2.destroyAllWindows()

# --- Process Captured Images for Calibration ---
print("\n--- Processing Images and Calibrating Camera ---")
images = glob.glob('calibration_images/*.png')

if not images:
    print("No calibration images found in 'calibration_images' folder. Calibration aborted.")
    exit()

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    # The (CHECKERBOARD[0], CHECKERBOARD[1]) order is important here
    ret, corners = cv2.findChessboardCorners(gray, (CHECKERBOARD[0], CHECKERBOARD[1]), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        # Refine corner locations
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners (optional, for visual feedback)
        img = cv2.drawChessboardCorners(img, (CHECKERBOARD[0], CHECKERBOARD[1]), corners2, ret)
        cv2.imshow('Calibration Image', img)
        cv2.waitKey(500) # Display for 0.5 seconds to see detection
    else:
        print(f"Warning: Chessboard corners not found in {fname}")

cv2.destroyAllWindows()

# --- Calibrate the camera ---
if len(objpoints) >= 6: # Need a minimum of typically 6-10 valid images for good calibration
    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        print("\nCamera calibrated successfully!")
        print("Camera matrix (mtx):\n", mtx)
        print("Distortion coefficients (dist):\n", dist)

        # Save the camera matrix and distortion coefficients
        np.savez("camera_calibration.npz", mtx=mtx, dist=dist)
        print("\nCalibration parameters saved to 'camera_calibration.npz'")

        # Calculate re-projection error to check accuracy
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        print( "Total re-projection error: {:.4f} pixels".format(mean_error/len(objpoints)) )
    else:
        print("Camera calibration failed.")
else:
    print(f"Not enough successful chessboard detections ({len(objpoints)}/{len(images)}) for reliable calibration. Need at least 6 good images.")