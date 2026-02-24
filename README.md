# Autonomous Robot Tracking & Targeting System

This project provides a Python-based computer vision solution for detecting and tracking two robots (your own and an opponent) from an overhead camera view. It differentiates your robot using ArUco markers and the opponent using color-based contour detection. The system calculates your robot's orientation and the relative angle and distance to the opponent. Currently logging autonomous commands, but in future sending those commands via RF to the robot.

## Features

*   **Robust Camera Initialization:** Utilizes `cv2.CAP_DSHOW` for fast camera startup on Windows.
*   **ArUco Marker Detection & Pose Estimation:** Tracks your robot's position and orientation using one or more ArUco markers.
    *   Supports multiple markers for improved robustness against occlusion and more stable pose estimation (using the midpoint and vector between markers).
*   **Opponent Bot Color Tracking:** Detects the opponent robot using HSV color thresholding and contour detection.
*   **Interactive Color Calibration:** A dedicated script (`color_calibrator.py`) with live trackbars for easily tuning HSV values for new opponent bots.
*   **Camera Calibration Utility:** A dedicated script (`calibrate_camera.py`) to calibrate your camera for accurate pose estimation.
*   **Relative Angle & Distance Calculation:** Computes the angle and distance from your robot's forward direction to the opponent bot.
*   **Autonomous Command Logging:** Logs `FORWARD`, `LEFT`, or `RIGHT` commands based on the relative angle to the opponent, with a customizable angle margin and logging interval.
*   **Real-time Visual Feedback:** Displays bounding boxes, directional lines, and tracking information in a live video feed.
*   **Frame Drop Resilience:** Robust error handling to skip invalid frames from the camera, preventing crashes.

## Prerequisites

Before you begin, ensure you have Python installed (preferably Python 3.x).

## Installation

**Install Required Python Libraries:**
Open your terminal or command prompt and run:
```bash
pip install opencv-python numpy
```
For ArUco marker support (which might be in the contrib package depending on your OpenCV version), it's often safer to install:
```bash
pip install opencv-contrib-python
```

## Setup & Calibration

This project requires two calibration steps: Camera Calibration and Opponent Color Calibration.

### Step 1: Camera Calibration (`calibrate_camera.py`)

Camera calibration is essential for accurate ArUco marker pose (position and orientation) estimation. You'll need a printed chessboard pattern for this.

1.  **Prepare a Chessboard:**
    *   Print a checkerboard pattern on a flat, rigid surface. You can find printable openCV camera calibration patterns online. I used a Chessboard I already had.
2.  **Edit `calibrate_camera.py`:**
    *   Open `calibrate_camera.py` and adjust the following constants:
        *   `CHECKERBOARD = (7, 7)`: This should be `(number_of_inner_corners_horizontally, number_of_inner_corners_vertically)`. For a standard 8x8 square chessboard, this is `(7, 7)`.
        *   `SQUARE_SIZE_MM = 24.0`: **Measure the actual side length of one square** on your printed chessboard in millimeters. This is important for accurate 3D pose estimation.
        *   `calib_cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)`: Ensure `CAMERA_INDEX` (the `1` in this example) matches your external webcam's index.
3.  **Create `calibration_images` Folder:**
    *   In the same directory as your scripts, create a new folder named `calibration_images`.
4.  **Run Calibration Image Capture:**
    ```bash
    python calibrate_camera.py
    ```
    *   A live camera feed will appear.
    *   Hold the chessboard pattern in front of the camera. Move it to various positions and orientations within the camera's view. Ensure the entire chessboard is visible and slightly angled.
    *   Press the `s` key to save a good calibration image. Aim for 15-20 diverse images.
    *   Press the `ESC` key when you are done capturing images.
5.  **Calibration Process:**
    *   The script will then automatically process the captured images. If it successfully finds chessboard corners, it will display the image with the corners drawn.
    *   If calibration is successful, it will save the camera matrix and distortion coefficients to `camera_calibration.npz` and print a re-projection error. A total re-projection error below 1.0 is generally considered good.

### Step 2: Opponent Bot Color Calibration (`color_calibrator.py`)

This step allows you to quickly adjust and save the HSV color range for detecting the opponent robot.

1.  **Edit `color_calibrator.py`:**
    *   `CAMERA_INDEX = 1`: Ensure this matches your external webcam's index.
2.  **Run the Color Calibrator:**
    ```bash
    python color_calibrator.py
    ```
    *   Three windows will appear: "Original Feed", "Masked Opponent", and "Color Calibration Trackbars".
3.  **Place Opponent Bot:**
    *   Place the opponent robot you wish to track in the camera's view. Ideally, try to isolate it initially to make calibration easier.
4.  **Adjust Sliders:**
    *   Use the `H_min`, `H_max`, `S_min`, `S_max`, `V_min`, `V_max` sliders in the "Color Calibration Trackbars" window.
    *   **Goal:** In the "Masked Opponent" window, your opponent robot should appear as a solid white blob, and everything else in the image should be completely black.
    *   **Tip:** Start by broadly adjusting `H_min` and `H_max` to isolate the primary color. Then use `S_min` to remove dull/gray areas (like the floor). Finally, use `V_min`/`V_max` to fine-tune and remove shadows or bright reflections.
5.  **Save Calibration:**
    *   Once you are satisfied with the mask, press the `s` key on your keyboard. This will save the `lower_hsv` and `upper_hsv` arrays to `opponent_color_calibration.npz`.
6.  **Exit:**
    *   Press the `ESC` key to close the calibrator.

## Running the Main Tracking System (`main.py`)

After completing both calibration steps, you can run the main tracking system.

1.  **Edit `main.py`:**
    *   `CAMERA_INDEX = 1`: Confirm this matches your external webcam's index.
    *   `OUR_BOT_ARUCO_IDS = [1, 3]`: This list should contain the IDs of **all** ArUco markers placed on your robot. (Your current setup uses ID 3 on the left and ID 1 on the right).
    *   `ARUCO_MARKER_SIZE_MM = 100`: **Measure the actual physical side length of your ArUco markers** in millimeters. This is crucial for accurate pose estimation.
    *   `FORWARD_ANGLE_MARGIN_DEG = 5`: Adjust this value. If the `relative_angle_deg` to the opponent is within `+/- 5` degrees, the command will be `FORWARD`.
    *   `LOG_INTERVAL_SECONDS = 2`: Adjust how frequently autonomous commands are printed to the console.
2.  **Place Robots:**
    *   Place your robot (with ArUco markers) and the opponent robot in the camera's view.
3.  **Run the Main Script:**
    ```bash
    python main.py
    ```
    *   A live video feed will appear, displaying:
        *   Purple bounding boxes around your robot's ArUco markers.
        *   A combined purple bounding box around your robot (if multiple markers are detected).
        *   A yellow line extending from each of your markers indicating their forward direction.
        *   A blue bounding box around the opponent robot.
        *   A yellow line connecting the center of your robot to the center of the opponent robot.
        *   Text showing the relative angle and distance (in pixels) to the opponent, positioned near your robot.
        *   FPS counter.
    *   Autonomous commands (`FORWARD`, `LEFT`, `RIGHT`) will be printed to your console at the specified interval.
4.  **Exit:**
    *   Press the `ESC` key to close the application.

## Important Notes & Troubleshooting

*   **Camera Index:** If `CAMERA_INDEX = 1` doesn't work, try `0`, `2`, etc.
*   **Slow Camera Startup:** If your external webcam is still slow to start, ensure its drivers are updated from the manufacturer's website. You can also try different USB ports. The `cv2.CAP_DSHOW` backend in the code is critical for good performance on Windows.
*   **"Failed to grab frame. Skipping this frame..."**: These messages indicate your camera is intermittently dropping frames. While the code handles it gracefully, frequent occurrences might suggest hardware (USB cable, port, camera) or driver issues.
*   **ArUco Marker `ARUCO_MARKER_SIZE_MM`:** Inaccurate measurement here will lead to incorrect 3D pose estimation and potentially skewed angle/distance calculations.
*   **Opponent Color Tracking:** The color detection is highly sensitive to lighting. If your environment changes significantly, you'll need to re-run `color_calibrator.py`.
*   **Robot Orientation:** The current code assumes your robot's forward direction is 90 degrees clockwise from the vector pointing from marker ID 3 (left) to marker ID 1 (right). If your robot's physical orientation differs, you might need to adjust the `our_orientation_rad` calculation in `main.py` (e.g., change `-(math.pi / 2)` to `+(math.pi / 2)` or a different offset).
*   **Smoothing Alpha (`SMOOTHING_ALPHA`):** Adjust this value (0.0 to 1.0) to control how much smoothing is applied to detected positions/orientations. Higher values (closer to 1.0) mean less smoothing (more reactive), lower values mean more smoothing (more stable but delayed response).