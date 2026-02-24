import cv2
import numpy as np
import math
import time
import os

# ==============================
# Configuration
# ==============================

CAMERA_INDEX = 1 # IMPORTANT: Set this to your external webcam index
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

MIN_CONTOUR_AREA = 1000 # Minimum pixel area for a contour to be considered a bot
SMOOTHING_ALPHA = 0.4 # Alpha for exponential moving average smoothing (0.0 - 1.0, higher means less smoothing)

# IMPORTANT: List of ArUco IDs used for OUR BOT.
# ID 3 is on the left side, ID 1 on the right side.
OUR_BOT_ARUCO_IDS = [1, 3]

ARUCO_MARKER_SIZE_MM = 100 # IMPORTANT: The actual physical side length of your ArUco marker in millimeters

# Angle margin for considering "FORWARD" when logging autonomous commands
FORWARD_ANGLE_MARGIN_DEG = 5

# Logging interval for autonomous control decisions
LOG_INTERVAL_SECONDS = 2

# Global variables for camera calibration results
camera_matrix = None
dist_coeffs = None

# ==============================
# Helpers
# ==============================

def smooth(old, new, alpha):
  """Applies exponential moving average for smoothing values."""
  if old is None:
    return new
  return alpha * new + (1 - alpha) * old

def compute_distance_pixels(p1, p2):
  """Computes the distance (pixels) between two points."""
  dx = p2[0] - p1[0]
  dy = p2[1] - p1[1]
  distance = math.sqrt(dx * dx + dy * dy)
  return distance

def contour_center(contour):
  """Calculates the centroid of a contour."""
  M = cv2.moments(contour)
  if M["m00"] == 0:
    return None
  cx = int(M["m10"] / M["m00"])
  cy = int(M["m01"] / M["m00"])
  return (cx, cy)

def load_camera_params(filename="camera_calibration.npz"):
    """Loads camera intrinsic matrix and distortion coefficients."""
    global camera_matrix, dist_coeffs
    if not os.path.exists(filename):
        print(f"Error: Camera calibration file '{filename}' not found.")
        print("Please run 'calibrate_camera.py' first to generate it.")
        print("Using dummy values for now. ArUco pose estimation will be inaccurate!")
        camera_matrix = np.array([[FRAME_WIDTH, 0, FRAME_WIDTH/2], [0, FRAME_WIDTH, FRAME_HEIGHT/2], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        return False
    try:
        npzfile = np.load(filename)
        camera_matrix = npzfile['mtx']
        dist_coeffs = npzfile['dist']
        print(f"Loaded camera calibration from {filename}")
        return True
    except Exception as e:
        print(f"Error loading camera calibration: {e}")
        print("Using dummy values for now. ArUco pose estimation will be inaccurate!")
        camera_matrix = np.array([[FRAME_WIDTH, 0, FRAME_WIDTH/2], [0, FRAME_WIDTH, FRAME_HEIGHT/2], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        return False

# ==============================
# Main Application Logic
# ==============================

print("Starting application...")
calibration_loaded = load_camera_params() # Load calibration at startup

start_time_profiling = time.time()
# ==============================
# Camera Setup
# ==============================
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"Error: Could not open video stream from camera index {CAMERA_INDEX} using CAP_DSHOW backend.")
    print("Please check if the camera is connected and the index is correct.")
    print("If issues persist, try removing 'cv2.CAP_DSHOW' or trying other backends like 'cv2.CAP_MSMF'.")
    exit()
print(f"Camera opened in {time.time() - start_time_profiling:.2f} seconds.")

start_time_profiling = time.time()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
print(f"Camera properties set in {time.time() - start_time_profiling:.2f} seconds.")


start_time_profiling = time.time()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
print(f"ArUco detector initialized in {time.time() - start_time_profiling:.2f} seconds.")

our_center_smoothed = None
our_orientation_smoothed_rad = None # Smoothed orientation in radians
other_center_smoothed = None

last_time = time.time() # For FPS calculation
last_log_time = time.time() # For logging autonomous commands

print("Starting robot tracking prototype. Press 'ESC' to exit.")

while True:
  ret, frame = cap.read()
  if not ret:
    print("Failed to grab frame. Exiting...")
    break

  display = frame.copy()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # ---------------------------------
  # Detect our bot (ArUco Markers)
  # ---------------------------------
  corners, ids, rejected = aruco_detector.detectMarkers(gray)

  # Variables to hold detected marker data for our bot
  marker1_data = {'center': None, 'orientation_rad': None, 'bbox': None} # ID 1 (Right)
  marker3_data = {'center': None, 'orientation_rad': None, 'bbox': None} # ID 3 (Left)
  
  all_our_bboxes = [] # To keep track of all our bot's marker bounding boxes for combined bbox

  if ids is not None and calibration_loaded:
    for i, marker_id in enumerate(ids):
      if marker_id in OUR_BOT_ARUCO_IDS:
        pts = corners[i][0]
        x_min = int(np.min(pts[:, 0]))
        x_max = int(np.max(pts[:, 0]))
        y_min = int(np.min(pts[:, 1]))
        y_max = int(np.max(pts[:, 1]))

        current_bbox = (x_min, y_min, x_max, y_max)
        all_our_bboxes.append(current_bbox) # Store individual marker bbox

        current_center = (int(pts[:, 0].mean()), int(pts[:, 1].mean()))
        
        cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (162, 0, 255), 2) # Purple box for each of our bot's markers

        # --- Pose Estimation for individual ArUco Marker ---
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners[i]], ARUCO_MARKER_SIZE_MM, camera_matrix, dist_coeffs
        )

        current_orientation_rad = None
        if rvecs is not None and len(rvecs) > 0:
            rvec = rvecs[0][0] # Rotation vector
            tvec = tvecs[0][0] # Translation vector

            # Project a forward vector for the marker itself
            object_points = np.array([[0, 0, 0], [0, ARUCO_MARKER_SIZE_MM / 2, 0]], dtype=np.float32).reshape(-1, 1, 3)
            img_pts, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)

            p_center_fwd = tuple(img_pts[0][0].astype(int))
            p_forward_end = tuple(img_pts[1][0].astype(int))

            cv2.line(display, p_center_fwd, p_forward_end, (0, 255, 255), 2) # YELLOW line for forward direction of *each* marker

            dx_forward = p_forward_end[0] - p_center_fwd[0]
            dy_forward = p_forward_end[1] - p_center_fwd[1]
            current_orientation_rad = math.atan2(dy_forward, dx_forward)
        
        # Store data based on marker ID
        if marker_id == 1: # Right marker
            marker1_data = {'center': current_center, 'orientation_rad': current_orientation_rad, 'bbox': current_bbox}
        elif marker_id == 3: # Left marker
            marker3_data = {'center': current_center, 'orientation_rad': current_orientation_rad, 'bbox': current_bbox}


  # --- Determine overall bot pose based on detected markers ---
  our_center = None
  our_orientation_rad = None
  our_bbox = None

  if marker1_data['center'] is not None and marker3_data['center'] is not None:
    # Both markers found: Calculate center as midpoint, orientation from vector between markers
    our_center = (
        (marker1_data['center'][0] + marker3_data['center'][0]) // 2,
        (marker1_data['center'][1] + marker3_data['center'][1]) // 2
    )

    # Calculate vector from left marker (ID 3) to right marker (ID 1)
    dx_lr = marker1_data['center'][0] - marker3_data['center'][0]
    dy_lr = marker1_data['center'][1] - marker3_data['center'][1]

    # Deriving forward orientation:
    # The vector (dx_lr, dy_lr) points from left (ID 3) to right (ID 1).
    # If the robot is generally facing "up" in the image, (dx_lr, dy_lr) will be roughly (positive, near_zero).
    # To get a "forward" vector pointing "up" (-Y direction), we need to rotate (dx_lr, dy_lr) 90 degrees clockwise.
    # A 90-degree clockwise rotation of (x, y) results in (y, -x) in image coordinates (Y-down).
    # So, forward_dx = dy_lr, forward_dy = -dx_lr
    our_orientation_rad = math.atan2(-dx_lr, dy_lr) # Swapped dx and dy components and negated new dy (old dx)

    # Combine bounding boxes
    min_x = min(marker1_data['bbox'][0], marker3_data['bbox'][0])
    min_y = min(marker1_data['bbox'][1], marker3_data['bbox'][1])
    max_x = max(marker1_data['bbox'][2], marker3_data['bbox'][2])
    max_y = max(marker1_data['bbox'][3], marker3_data['bbox'][3])
    our_bbox = (min_x, min_y, max_x, max_y)

  elif marker1_data['center'] is not None:
    # Only ID 1 found: use its center and orientation
    our_center = marker1_data['center']
    our_orientation_rad = marker1_data['orientation_rad']
    our_bbox = marker1_data['bbox']
  elif marker3_data['center'] is not None:
    # Only ID 3 found: use its center and orientation
    our_center = marker3_data['center']
    our_orientation_rad = marker3_data['orientation_rad']
    our_bbox = marker3_data['bbox']

  # ---------------------------------
  # Detect other bot (Color-based contour)
  # Tuned for the bluish-green bot in the provided image
  # ---------------------------------
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  lower_hsv = np.array([70, 100, 100])
  upper_hsv = np.array([100, 255, 255])

  mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

  kernel = np.ones((5, 5), np.uint8)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  largest_contour = None
  largest_area = 0

  for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < MIN_CONTOUR_AREA:
      continue

    cxcy = contour_center(cnt)
    if cxcy is None:
      continue

    is_overlap_with_our_bot = False
    if our_bbox is not None:
        x_min_our, y_min_our, x_max_our, y_max_our = our_bbox
        if x_min_our - 20 <= cxcy[0] <= x_max_our + 20 and y_min_our - 20 <= cxcy[1] <= y_max_our + 20:
            is_overlap_with_our_bot = True
    if is_overlap_with_our_bot:
        continue

    if area > largest_area:
      largest_area = area
      largest_contour = cnt

  other_bbox = None
  other_center = None

  if largest_contour is not None:
    x, y, w, h = cv2.boundingRect(largest_contour)
    other_bbox = (x, y, x + w, y + h)
    other_center = contour_center(largest_contour)

    cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2) # Blue box for other bot

  # ---------------------------------
  # Smoothing bot positions and orientation
  # ---------------------------------
  if our_center is not None:
    our_center_smoothed = smooth(
      our_center_smoothed,
      np.array(our_center),
      SMOOTHING_ALPHA
    )
  if our_orientation_rad is not None:
      if our_orientation_smoothed_rad is None:
          our_orientation_smoothed_rad = our_orientation_rad
      else:
          # Circular smoothing for angles
          old_complex = math.cos(our_orientation_smoothed_rad) + 1j * math.sin(our_orientation_smoothed_rad)
          new_complex = math.cos(our_orientation_rad) + 1j * math.sin(our_orientation_rad)
          smoothed_complex = SMOOTHING_ALPHA * new_complex + (1 - SMOOTHING_ALPHA) * old_complex
          our_orientation_smoothed_rad = math.atan2(smoothed_complex.imag, smoothed_complex.real)


  if other_center is not None:
    other_center_smoothed = smooth(
      other_center_smoothed,
      np.array(other_center),
      SMOOTHING_ALPHA
    )

  # ---------------------------------
  # Calculate and display Relative Angle + Distance
  # ---------------------------------
  if our_center_smoothed is not None and other_center_smoothed is not None and our_orientation_smoothed_rad is not None:

    # Draw line between our bot and the other bot
    p1_our_smoothed = tuple(our_center_smoothed.astype(int))
    p2_other_smoothed = tuple(other_center_smoothed.astype(int))
    cv2.line(display, p1_our_smoothed, p2_other_smoothed, (0, 255, 255), 2) # Yellow line

    # Calculate the angle of the vector from our bot's center to the other bot's center
    dx_target = other_center_smoothed[0] - our_center_smoothed[0]
    dy_target = other_center_smoothed[1] - our_center_smoothed[1]
    angle_to_target_rad = math.atan2(dy_target, dx_target)

    # Calculate the relative angle: angle from our bot's *forward direction* to the target bot
    relative_angle_rad = angle_to_target_rad - our_orientation_smoothed_rad

    # Normalize the relative angle to be within -pi to pi radians (-180 to 180 degrees)
    while relative_angle_rad > math.pi:
        relative_angle_rad -= 2 * math.pi
    while relative_angle_rad < -math.pi:
        relative_angle_rad += 2 * math.pi

    relative_angle_deg = math.degrees(relative_angle_rad)
    distance_px = compute_distance_pixels(our_center_smoothed, other_center_smoothed)

    text = f"Rel A: {relative_angle_deg:.1f}°  D: {distance_px:.0f}px"

    # Display text near our bot's combined bounding box
    if our_bbox is not None:
      x_min, y_min, _, _ = our_bbox
      cv2.putText(
        display,
        text,
        (x_min, y_min - 10), # Position text just above our bot's bbox
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (162, 0, 255), # Purple text
        2
      )

    # ---------------------------------
    # Autonomous Control Logging (FIXED LEFT/RIGHT LOGIC)
    # ---------------------------------
    current_time_log = time.time()
    if current_time_log - last_log_time >= LOG_INTERVAL_SECONDS:
        direction = ""
        if abs(relative_angle_deg) <= FORWARD_ANGLE_MARGIN_DEG:
            direction = "FORWARD"
        elif relative_angle_deg < -FORWARD_ANGLE_MARGIN_DEG: # Target is to our LEFT (negative angle in this system)
            direction = "LEFT"
        else: # relative_angle_deg > FORWARD_ANGLE_MARGIN_DEG (Target is to our RIGHT (positive angle in this system))
            direction = "RIGHT"

        print(f"[{time.strftime('%H:%M:%S')}] Autonomous Command: {direction} (Rel A: {relative_angle_deg:.1f}°, D: {distance_px:.0f}px)")
        last_log_time = current_time_log


  # ---------------------------------
  # FPS Display
  # ---------------------------------
  current_time = time.time()
  fps = 1.0 / (current_time - last_time)
  last_time = current_time

  cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Yellow text

  # Display the resulting frame
  cv2.imshow("Robot Tracking Prototype", display)
  # Optional: Display the mask for opponent detection tuning
  # cv2.imshow("Mask", mask)

  # Exit on 'ESC' key press
  if cv2.waitKey(1) == 27:
    break

# ==============================
# Cleanup
# ==============================
cap.release()
cv2.destroyAllWindows()
print("Application closed.")