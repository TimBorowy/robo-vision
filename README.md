# Autonomous Robot Tracking & Targeting System

This project provides a Python-based computer vision solution for detecting and tracking two robots (your own and an opponent) from an overhead camera view. The system implements a robust fallback hierarchy for your robot (ArUco markers prioritized over color detection) and uses size-based heuristics to track the opponent. It calculates your robot's orientation and the relative angle and distance to the opponent to generate autonomous RC commands.

## Features

*   **Arena Masking:** Defines a rigid playable polygon. Movement and colors outside this defined area are completely ignored, keeping processing focused.
*   **Sensor Priority Tracking (Our Bot):** 
    *   *Priority 1:* Tracks your robot's position and orientation using rigid ArUco marker geometry.
    *   *Priority 2:* Seamlessly falls back to HSV color-blob tracking for position if ArUco markers are lost, while utilizing "Heading Memory" to remember the last known direction.
*   **Opponent Targeting:** Detects the opponent by dynamically identifying the largest moving object in the arena (excluding our own robot), preventing the system from getting distracted by small debris or shadows. Dedicated HSV color tracking can also be toggled.
*   **Relative Angle & Distance Calculation:** Computes the angle and distance from your robot's forward vector to the opponent bot.
*   **Autonomous Command Logging & RF Control:** Translates angles into `FORWARD`, `LEFT`, or `RIGHT` commands and broadcasts them over an ELRS serial module using the CRSF protocol.
*   **Real-time Visual Feedback:** Displays bounding boxes, directional lines, and tracking information in a live video feed.

## Prerequisites

Ensure you have Python 3.x installed.

## Installation

Open your terminal or command prompt and run:
```bash
pip install opencv-python numpy
pip install opencv-contrib-python
pip install pyserial
```

## Setup & Calibration

### Step 1: Camera Calibration (`calibrate_camera.py`)

Camera calibration is essential for accurate ArUco marker pose estimation. 

1.  **Prepare a Chessboard:** Print a standard OpenCV checkerboard pattern.
2.  **Edit `calibrate_camera.py`:**
    *   `CHECKERBOARD = (7, 7)`: Number of inner corners.
    *   `SQUARE_SIZE_MM`: Measure the actual physical side length of one square in millimeters.
3.  **Run:** `python calibrate_camera.py`
    *   Press `s` to capture ~15-20 images at different angles.
    *   Press `ESC` to process and save `camera_calibration.npz`.

### Step 2: Set the Arena Bounds (`main.py`)

Ensure the camera ignores the crowd and off-table areas.
1. In `main.py`, locate the `ARENA_POLYGON_POINTS` array. 
2. Adjust the (x, y) coordinate pairs to draw an exact box around your playable area. *Note: Ensure your coordinates are mapped to the correct resolution (e.g., 1280x720).*

### Step 3: (Optional) Opponent Color Calibration (`calibrate_color.py`)

If you want to use dedicated color tracking for the opponent instead of "Largest Object" tracking:
1. Run `python calibrate_color.py`
2. Adjust the HSV sliders until the opponent robot is isolated in solid white on the mask window.
3. Press `s` to save `opponent_color_calibration.npz`.

## Running the Tracking System

1.  **Edit Constants in `main.py`:**
    *   `CAMERA_INDEX`: Set to your overhead webcam index.
    *   `LEFT_MARKER_ID` / `RIGHT_MARKER_ID`: Ensure these match the physical ArUco tags on your robot.
    *   `ARUCO_MARKER_SIZE_MM`: Measure your printed tags carefully.
    *   `ELRS_SERIAL_PORT`: Set to the correct COM port (e.g., `COM3`) for your radio transmitter.
2.  **Run:**
    ```bash
    python main.py
    ```

## Important Notes & Troubleshooting

*   **ArUco Position vs Color Offset:** The origin of the targeting vector will prioritize the ArUco markers. If your markers are dropped due to extreme motion blur, you may see the targeting box temporarily jump to the center of your color-blob. This is normal fallback behavior. 
*   **Opponent Targeting the Wrong Object:** Ensure `MIN_CONTOUR_AREA` is large enough so that random shadows are ignored. If the red box is jumping to a referee's hand, ensure the `ARENA_POLYGON_POINTS` are dialed in to exclude the edges of the table.
*   **"Failed to grab frame" warnings:** These indicate dropped frames from the USB bus. The code handles this gracefully, but consider reducing webcam resolution or trying a different USB controller if this happens frequently.
```