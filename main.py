import cv2
import numpy as np
import math
import time
import os
from datetime import datetime
from robot_controller import RobotController

# ==============================
# Configuration
# ==============================

# --- INPUT SOURCE CONFIGURATION ---
USE_VIDEO_FILE_INPUT = True # <<< TOGGLE: True to use a video file, False for live camera
VIDEO_INPUT_FILE = "video_input/kraken-vs-knackwurst-stream.mp4" # <<< Specify your recorded raw video file here
                                                            # (e.g., from your 'video_output' folder)

CAMERA_INDEX = 1 # IMPORTANT: Set this to your external webcam index (only used if USE_VIDEO_FILE_INPUT is False)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

MIN_CONTOUR_AREA = 500 # Adjusted minimum pixel area for a contour (may need tuning)
SMOOTHING_ALPHA = 0.4 # Alpha for exponential moving average smoothing (0.0 - 1.0, higher means less smoothing)

RIGHT_MARKER_ID = 101
LEFT_MARKER_ID = 102

OUR_BOT_ARUCO_IDS = [RIGHT_MARKER_ID, LEFT_MARKER_ID] # ID 101 is on the left side, ID 102 on the right side.
ARUCO_MARKER_SIZE_MM = 50 # IMPORTANT: The actual physical side length of your ArUco marker in millimeters

# Angle margin for considering "FORWARD" when logging autonomous commands
FORWARD_ANGLE_MARGIN_DEG = 5

# Logging interval for autonomous control decisions
LOG_INTERVAL_SECONDS = 2

# File names for calibration data
CAMERA_CALIBRATION_FILE = "camera_calibration.npz"
OPPONENT_COLOR_CALIBRATION_FILE = "opponent_color_calibration.npz"

# ==============================
# Opponent Detection Configuration
# ==============================
ENABLE_BG_SUBTRACTION = True # <<< TOGGLE: True for background subtraction, False for HSV color detection

# Background Subtractor parameters (tune if needed)
BG_SUBTRACTOR_HISTORY = 500
BG_SUBTRACTOR_VAR_THRESHOLD = 16
BG_SUBTRACTOR_DETECT_SHADOWS = True

# ==============================
# Our Bot Color Tracking Configuration
# (For redundant position tracking)
# ==============================
ENABLE_OUR_BOT_COLOR_TRACKING = True # <<< TOGGLE: True to enable color tracking for our bot

# IMPORTANT: Tune these HSV values for your bot's purple/hotpink color.
# You can use 'color_calibrator.py' temporarily, or tune directly here
# by uncommenting mask display.
OUR_BOT_COLOR_LOWER_HSV = np.array([50, 0, 81]) # Example for purple/magenta
OUR_BOT_COLOR_UPPER_HSV = np.array([179, 22, 193]) # Example for purple/magenta

# ==============================
# Configuration for Video Recording
# ==============================
ENABLE_VIDEO_RECORDING = True # <<< TOGGLE THIS TO ENABLE/DISABLE VIDEO RECORDING
RECORDING_FPS = 30.0         # Target FPS for recorded videos (will be overwritten if using video file input)
RECORDING_CODEC = 'mp4v'     # Codec: 'mp4v' for .mp4 (Windows/Linux), 'MJPG' for .avi (more universal but larger)
VIDEO_OUTPUT_DIR = "video_output" # Directory to save recorded videos

# ==============================
# Configuration for Control
# = ============================
# SET THIS TO True TO ENABLE SERIAL COMMUNICATION WITH THE ROBOT
ENABLE_SERIAL_COMMS_DEFAULT = False # <<< Default toggle state for live camera
ENABLE_SERIAL_COMMS = ENABLE_SERIAL_COMMS_DEFAULT # This will be set to False if using video file input

ELRS_SERIAL_PORT = 'COMX' # <<< IMPORTANT: CHANGE THIS TO YOUR ACTUAL SERIAL PORT
ELRS_BAUDRATE = 420000 # Standard ELRS serial baudrate

# RC Channel mapping (adjust based on your robot's setup and OpenTX/EdgeTX config)
CHANNEL_THROTTLE = 0   # Corresponds to Channel 1 on your radio
CHANNEL_STEERING = 1   # Corresponds to Channel 2
CHANNEL_MODE_SWITCH = 4 # Corresponds to Channel 5 (Aux1) - for Manual/Autonomous
CHANNEL_KILL_SWITCH = 7 # Corresponds to Channel 8 (Aux4) - for Emergency Stop

# Default RC values (neutral for 1000-2000 PWM range)
RC_MIN = 1000
RC_CENTER = 1500
RC_MAX = 2000

# Autonomous control parameters (adjust these for your robot's speed/turn responsiveness)
AUTONOMOUS_SPEED_FORWARD = 1600 # Example forward speed
AUTONOMOUS_SPEED_STOP = 1500    # Stop speed
AUTONOMOUS_TURN_LEFT = 1200     # Example left turn value (turn left)
AUTONOMOUS_TURN_RIGHT = 1800    # Example right turn value (turn right)
AUTONOMOUS_TURN_STRAIGHT = 1500 # Straight steering


# Global variables for calibration results
camera_matrix = None
dist_coeffs = None
opponent_lower_hsv = None
opponent_upper_hsv = None # Only used if ENABLE_BG_SUBTRACTION is False

robot_controller = None # Instance of the RobotController class
raw_video_writer = None
processed_video_writer = None
bg_subtractor = None # Background subtractor object

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

def load_camera_params(filename=CAMERA_CALIBRATION_FILE):
    """Loads camera intrinsic matrix and distortion coefficients."""
    global camera_matrix, dist_coeffs
    if not os.path.exists(filename):
        print(f"Error: Camera calibration file '{filename}' not found.")
        print("Please run 'calibrate_camera.py' first to generate it.")
        print("ArUco pose estimation will be inaccurate without calibration!")
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
        print("ArUco pose estimation will be inaccurate without calibration!")
        camera_matrix = np.array([[FRAME_WIDTH, 0, FRAME_WIDTH/2], [0, FRAME_WIDTH, FRAME_HEIGHT/2], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        return False

# This function is only called if ENABLE_BG_SUBTRACTION is False
def load_opponent_color_params(filename=OPPONENT_COLOR_CALIBRATION_FILE):
    """Loads opponent bot's HSV color range."""
    global opponent_lower_hsv, opponent_upper_hsv
    if not os.path.exists(filename):
        print(f"Error: Opponent color calibration file '{filename}' not found.")
        print("Please run 'color_calibrator.py' first to set and save color ranges.")
        print("Using default bluish-green values for now.")
        opponent_lower_hsv = np.array([70, 100, 100])
        opponent_upper_hsv = np.array([100, 255, 255])
        return False
    try:
        npzfile = np.load(filename)
        opponent_lower_hsv = npzfile['lower_hsv']
        opponent_upper_hsv = npzfile['upper_hsv']
        print(f"Loaded opponent color calibration from {filename}")
        return True
    except Exception as e:
        print(f"Error loading opponent color calibration: {e}")
        print("Using default bluish-green values for now.")
        opponent_lower_hsv = np.array([70, 100, 100])
        opponent_upper_hsv = np.array([100, 255, 255])
        return False

# ==============================
# Main Application Logic
# ==============================

print("Starting application...")
calibration_loaded = load_camera_params() # Load camera calibration at startup

if not ENABLE_BG_SUBTRACTION:
    color_calibration_loaded = load_opponent_color_params() # Load opponent color calibration (if not using BG sub)
else:
    print("Opponent detection using Background Subtraction. HSV color calibration not loaded.")
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=BG_SUBTRACTOR_HISTORY,
        varThreshold=BG_SUBTRACTOR_VAR_THRESHOLD,
        detectShadows=BG_SUBTRACTOR_DETECT_SHADOWS
    )
    print("Background Subtractor (MOG2) initialized.")

# --- Adjust serial comms based on input source ---
if USE_VIDEO_FILE_INPUT:
    ENABLE_SERIAL_COMMS = False
    print("WARNING: Video file input detected. Disabling serial communications to robot.")

# Initialize the RobotController
robot_controller = RobotController(
    serial_port=ELRS_SERIAL_PORT,
    baudrate=ELRS_BAUDRATE,
    channel_throttle=CHANNEL_THROTTLE,
    channel_steering=CHANNEL_STEERING,
    channel_mode_switch=CHANNEL_MODE_SWITCH,
    channel_kill_switch=CHANNEL_KILL_SWITCH,
    rc_min=RC_MIN,
    rc_center=RC_CENTER,
    rc_max=RC_MAX,
    autonomous_speed_forward=AUTONOMOUS_SPEED_FORWARD,
    autonomous_speed_stop=AUTONOMOUS_SPEED_STOP,
    autonomous_turn_left=AUTONOMOUS_TURN_LEFT,
    autonomous_turn_right=AUTONOMOUS_TURN_RIGHT,
    autonomous_turn_straight=AUTONOMOUS_TURN_STRAIGHT,
    enable_comms=ENABLE_SERIAL_COMMS # Pass the toggle here
)


start_time_profiling = time.time()
# ==============================
# Camera/Video Input Setup
# ==============================
if USE_VIDEO_FILE_INPUT:
    cap = cv2.VideoCapture(VIDEO_INPUT_FILE)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{VIDEO_INPUT_FILE}'.")
        exit()
    # Get actual FPS from video file
    RECORDING_FPS = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video file '{VIDEO_INPUT_FILE}' opened at {RECORDING_FPS:.2f} FPS.")
    # No cap.set for width/height needed for video files, they are read as-is.
else:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Could not open video stream from camera index {CAMERA_INDEX} using CAP_DSHOW backend.")
        print("Please check if the camera is connected and the index is correct.")
        print("If issues persist, try removing 'cv2.CAP_DSHOW' or trying other backends like 'cv2.CAP_MSMF'.")
        exit()
    print(f"Camera opened in {time.time() - start_time_profiling:.2f} seconds.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    print(f"Camera properties set in {time.time() - start_time_profiling:.2f} seconds.")


start_time_profiling = time.time()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250) # Updated dictionary as per previous discussion
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
print(f"ArUco detector initialized in {time.time() - start_time_profiling:.2f} seconds.")

# ==============================
# Video Recording Setup
# ==============================
if ENABLE_VIDEO_RECORDING:
    if not os.path.exists(VIDEO_OUTPUT_DIR):
        os.makedirs(VIDEO_OUTPUT_DIR)
        print(f"Created video output directory: {VIDEO_OUTPUT_DIR}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    raw_video_filename = os.path.join(VIDEO_OUTPUT_DIR, f"raw_{timestamp}.mp4")
    processed_video_filename = os.path.join(VIDEO_OUTPUT_DIR, f"processed_{timestamp}.mp4")
    
    current_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    current_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*RECORDING_CODEC) # Define the codec

    raw_video_writer = cv2.VideoWriter(raw_video_filename, fourcc, RECORDING_FPS, (current_frame_width, current_frame_height))
    processed_video_writer = cv2.VideoWriter(processed_video_filename, fourcc, RECORDING_FPS, (current_frame_width, current_frame_height))

    if not raw_video_writer.isOpened():
        print(f"Error: Could not open raw video writer for {raw_video_filename}")
        ENABLE_VIDEO_RECORDING = False # Disable recording if writer fails
    if not processed_video_writer.isOpened():
        print(f"Error: Could not open processed video writer for {processed_video_filename}")
        ENABLE_VIDEO_RECORDING = False # Disable recording if writer fails
    
    if ENABLE_VIDEO_RECORDING:
        print(f"Video recording enabled. Raw video: {raw_video_filename}, Processed video: {processed_video_filename}")
    else:
        print("Video recording disabled due to errors during writer initialization.")


our_center_smoothed = None
our_orientation_smoothed_rad = None # Smoothed orientation in radians
other_center_smoothed = None

last_time = time.time() # For FPS calculation
last_log_time = time.time() # For logging autonomous commands

print("Starting robot tracking prototype. Press 'ESC' to exit.")

try: # Use a try-finally block for graceful shutdown
    while True:
      ret, frame = cap.read()
      if not ret:
        print("End of video file or failed to grab frame. Exiting...")
        break # Break the loop if no frame is read

      # --- RECORD RAW FOOTAGE ---
      if ENABLE_VIDEO_RECORDING and raw_video_writer.isOpened():
          raw_video_writer.write(frame)

      display = frame.copy()
      
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


      # ----------------------------------------------------
      # OUR BOT DETECTION (ARUCO + COLOR REDUNDANCY)
      # ----------------------------------------------------
      our_aruco_center = None
      our_aruco_orientation_rad = None
      our_aruco_bbox = None
      
      # Detect ArUco markers for our bot
      corners, ids, rejected = aruco_detector.detectMarkers(gray)

      marker_right_data = {'center': None, 'orientation_rad': None, 'bbox': None} # ID 102 (Right)
      marker_left_data = {'center': None, 'orientation_rad': None, 'bbox': None} # ID 101 (Left)
      
      # Process ArUco markers
      if ids is not None and calibration_loaded:
        for i, marker_id in enumerate(ids):
          if marker_id in OUR_BOT_ARUCO_IDS:
            pts = corners[i][0]
            x_min, y_min = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))
            x_max, y_max = int(np.max(pts[:, 0])), int(np.max(pts[:, 1]))

            current_bbox = (x_min, y_min, x_max, y_max)
            
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (162, 0, 255), 2) # Purple box for ArUco markers

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[i]], ARUCO_MARKER_SIZE_MM, camera_matrix, dist_coeffs
            )

            current_orientation_rad = None
            if rvecs is not None and len(rvecs) > 0:
                rvec = rvecs[0][0]
                tvec = tvecs[0][0]

                object_points = np.array([[0, 0, 0], [0, ARUCO_MARKER_SIZE_MM / 2, 0]], dtype=np.float32).reshape(-1, 1, 3)
                img_pts, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)

                p_center_fwd = tuple(img_pts[0][0].astype(int))
                p_forward_end = tuple(img_pts[1][0].astype(int))

                cv2.line(display, p_center_fwd, p_forward_end, (0, 255, 255), 2) # YELLOW line for forward direction of *each* marker

                dx_forward = p_forward_end[0] - p_center_fwd[0]
                dy_forward = p_forward_end[1] - p_center_fwd[1]
                current_orientation_rad = math.atan2(dy_forward, dx_forward)
            
            if marker_id == 102: # Right marker
                marker_right_data = {'center': current_center, 'orientation_rad': current_orientation_rad, 'bbox': current_bbox}
            elif marker_id == 101: # Left marker
                marker_left_data = {'center': current_center, 'orientation_rad': current_orientation_rad, 'bbox': current_bbox}

      # Combine ArUco data
      if marker_right_data['center'] is not None and marker_left_data['center'] is not None:
        our_aruco_center = (
            (marker_right_data['center'][0] + marker_left_data['center'][0]) // 2,
            (marker_right_data['center'][1] + marker_left_data['center'][1]) // 2
        )
        dx_lr = marker_right_data['center'][0] - marker_left_data['center'][0]
        dy_lr = marker_right_data['center'][1] - marker_left_data['center'][1]
        our_aruco_orientation_rad = math.atan2(-dx_lr, dy_lr) # 90-deg CW from left-to-right vector
        # Combined ArUco bbox
        min_x = min(marker_right_data['bbox'][0], marker_left_data['bbox'][0])
        min_y = min(marker_right_data['bbox'][1], marker_left_data['bbox'][1])
        max_x = max(marker_right_data['bbox'][2], marker_left_data['bbox'][2])
        max_y = max(marker_right_data['bbox'][3], marker_left_data['bbox'][3])
        our_aruco_bbox = (min_x, min_y, max_x, max_y)
      elif marker_right_data['center'] is not None:
        our_aruco_center = marker_right_data['center']
        our_aruco_orientation_rad = marker_right_data['orientation_rad']
        our_aruco_bbox = marker_right_data['bbox']
      elif marker_left_data['center'] is not None:
        our_aruco_center = marker_left_data['center']
        our_aruco_orientation_rad = marker_left_data['orientation_rad']
        our_aruco_bbox = marker_left_data['bbox']

      # --- Our Bot Color Tracking (for redundant position) ---
      our_color_center = None
      our_color_bbox = None
      if ENABLE_OUR_BOT_COLOR_TRACKING:
          our_bot_mask = cv2.inRange(hsv_frame, OUR_BOT_COLOR_LOWER_HSV, OUR_BOT_COLOR_UPPER_HSV)
          kernel_color = np.ones((5, 5), np.uint8)
          our_bot_mask = cv2.morphologyEx(our_bot_mask, cv2.MORPH_CLOSE, kernel_color)
          our_bot_mask = cv2.morphologyEx(our_bot_mask, cv2.MORPH_OPEN, kernel_color)

          our_bot_contours, _ = cv2.findContours(our_bot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
          
          largest_our_bot_contour = None
          largest_our_bot_area = 0
          for cnt in our_bot_contours:
              area = cv2.contourArea(cnt)
              if area > largest_our_bot_area:
                  largest_our_bot_area = area
                  largest_our_bot_contour = cnt
          
          if largest_our_bot_contour is not None and largest_our_bot_area > MIN_CONTOUR_AREA:
              x, y, w, h = cv2.boundingRect(largest_our_bot_contour)
              our_color_bbox = (x, y, x + w, y + h)
              our_color_center = contour_center(largest_our_bot_contour)
              cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 1) # Yellow border for color tracking

      # --- Combine Our Bot Position & Orientation ---
      our_center = None
      our_orientation_rad = None
      our_bbox_combined = None # Overall bounding box for our bot

      if our_color_center is not None:
          our_center = our_color_center # Prioritize color for position if found
          our_bbox_combined = our_color_bbox
      elif our_aruco_center is not None:
          our_center = our_aruco_center # Fallback to ArUco for position
          our_bbox_combined = our_aruco_bbox

      # Orientation *always* comes from ArUco (color blobs don't give orientation)
      # If ArUco is present, use its orientation. If not, smoothing will hold last known.
      our_orientation_rad = our_aruco_orientation_rad

      # ----------------------------------------------------
      # OPPONENT BOT DETECTION (BACKGROUND SUBTRACTION or HSV)
      # ----------------------------------------------------
      other_bbox = None
      other_center = None

      opponent_mask_display = None # For optional display

      if ENABLE_BG_SUBTRACTION:
          # Apply background subtractor
          fg_mask = bg_subtractor.apply(frame)
          # Apply morphological operations to clean up foreground mask
          kernel_fg = np.ones((5, 5), np.uint8)
          fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_fg)
          fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_fg)
          opponent_mask_display = fg_mask # Store for display

          contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      else:
          # Fallback to HSV color detection if BG subtraction is disabled
          mask_hsv_opponent = cv2.inRange(hsv_frame, opponent_lower_hsv, opponent_upper_hsv)
          kernel_hsv = np.ones((5, 5), np.uint8)
          mask_hsv_opponent = cv2.morphologyEx(mask_hsv_opponent, cv2.MORPH_CLOSE, kernel_hsv)
          mask_hsv_opponent = cv2.morphologyEx(mask_hsv_opponent, cv2.MORPH_OPEN, kernel_hsv)
          opponent_mask_display = mask_hsv_opponent # Store for display
          
          contours, _ = cv2.findContours(mask_hsv_opponent, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      
      largest_contour = None
      largest_area = 0

      for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
          continue

        cxcy = contour_center(cnt)
        if cxcy is None:
          continue

        # Skip contour if it significantly overlaps with our bot's detected area
        is_overlap_with_our_bot = False
        if our_bbox_combined is not None:
            x_min_our, y_min_our, x_max_our, y_max_our = our_bbox_combined
            if x_min_our - 20 <= cxcy[0] <= x_max_our + 20 and y_min_our - 20 <= cxcy[1] <= y_max_our + 20:
                is_overlap_with_our_bot = True
        if is_overlap_with_our_bot:
            continue

        if area > largest_area:
          largest_area = area
          largest_contour = cnt

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
      if our_orientation_rad is not None: # Only smooth ArUco-derived orientation
          if our_orientation_smoothed_rad is None:
              our_orientation_smoothed_rad = our_orientation_rad
          else:
              old_complex = math.cos(our_orientation_smoothed_rad) + 1j * math.sin(our_orientation_smoothed_rad)
              new_complex = math.cos(our_orientation_rad) + 1j * math.sin(our_orientation_rad)
              smoothed_complex = SMOOTHING_ALPHA * new_complex + (1 - SMOOTHING_ALPHA) * old_complex
              our_orientation_smoothed_rad = math.atan2(smoothed_complex.imag, smoothed_complex.real)


      if other_center is not None:
        other_center_smoothed = smooth(
          np.array(other_center_smoothed) if other_center_smoothed is not None else None,
          np.array(other_center),
          SMOOTHING_ALPHA
        )
        
      # Draw smoothed combined bounding box for our bot (using green)
      if our_center_smoothed is not None and our_bbox_combined is not None:
          bbox_width = our_bbox_combined[2] - our_bbox_combined[0]
          bbox_height = our_bbox_combined[3] - our_bbox_combined[1]
          smooth_x_min = int(our_center_smoothed[0] - bbox_width / 2)
          smooth_y_min = int(our_center_smoothed[1] - bbox_height / 2)
          smooth_x_max = int(our_center_smoothed[0] + bbox_width / 2)
          smooth_y_max = int(our_center_smoothed[1] + bbox_height / 2)
          cv2.rectangle(display, (smooth_x_min, smooth_y_min), (smooth_x_max, smooth_y_max), (0, 255, 0), 2) # Green box


      # ---------------------------------
      # Calculate and display Relative Angle + Distance
      # And Autonomous Control Logic
      # ---------------------------------
      # Default commands to stop and manual mode for RC failsafe if vision is lost
      current_throttle_rc = AUTONOMOUS_SPEED_STOP
      current_steering_rc = AUTONOMOUS_TURN_STRAIGHT
      current_mode_switch_rc = RC_MAX # Assume Autonomous by default in code
      current_kill_switch_rc = RC_MAX # Assume Kill Switch OFF (RC_MAX)

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

        # Display text near our bot's combined bounding box (using the smoothed green box for reference)
        if our_center_smoothed is not None and our_bbox_combined is not None:
          # Calculate approximate text position based on smoothed center
          bbox_width = our_bbox_combined[2] - our_bbox_combined[0]
          bbox_height = our_bbox_combined[3] - our_bbox_combined[1]
          text_x = int(our_center_smoothed[0] - bbox_width / 2)
          text_y = int(our_center_smoothed[1] - bbox_height / 2 - 10) # 10 pixels above top of box
          cv2.putText(
            display,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0), # Green text
            2
          )

        # ---------------------------------
        # Autonomous Control Decision
        # ---------------------------------
        direction = ""
        if abs(relative_angle_deg) <= FORWARD_ANGLE_MARGIN_DEG:
            direction = "FORWARD"
            current_throttle_rc = AUTONOMOUS_SPEED_FORWARD
            current_steering_rc = AUTONOMOUS_TURN_STRAIGHT
        elif relative_angle_deg < -FORWARD_ANGLE_MARGIN_DEG: # Target is to our LEFT (negative angle)
            direction = "LEFT"
            current_throttle_rc = AUTONOMOUS_SPEED_FORWARD # Can be adjusted to stop or slow down for turns
            current_steering_rc = AUTONOMOUS_TURN_LEFT
        else: # relative_angle_deg > FORWARD_ANGLE_MARGIN_DEG (Target is to our RIGHT (positive angle))
            direction = "RIGHT"
            current_throttle_rc = AUTONOMOUS_SPEED_FORWARD # Can be adjusted to stop or slow down for turns
            current_steering_rc = AUTONOMOUS_TURN_RIGHT
        
        # Always set mode switch to autonomous when vision is active and tracking
        current_mode_switch_rc = RC_MAX

        current_time_log = time.time()
        if current_time_log - last_log_time >= LOG_INTERVAL_SECONDS:
            print(f"[{time.strftime('%H:%M:%S')}] Autonomous Command: {direction} (Rel A: {relative_angle_deg:.1f}°, D: {distance_px:.0f}px)")
            last_log_time = current_time_log
      
      else: # Vision lost or not enough data for autonomous decision
          current_throttle_rc = AUTONOMOUS_SPEED_STOP
          current_steering_rc = AUTONOMOUS_TURN_STRAIGHT
          current_mode_switch_rc = RC_MIN # Force Manual Mode if autonomous vision is lost
          if time.time() - last_log_time >= LOG_INTERVAL_SECONDS:
              print(f"[{time.strftime('%H:%M:%S')}] Autonomous Vision Lost - Sending Neutral/Manual Commands.")
              last_log_time = time.time()


      # Send RC commands using the RobotController instance
      robot_controller.send_commands(current_throttle_rc, current_steering_rc, current_mode_switch_rc, current_kill_switch_rc)


      # ---------------------------------
      # FPS Display
      # ---------------------------------
      # If playing from video file, attempt to match playback speed to recorded FPS
      if USE_VIDEO_FILE_INPUT:
          wait_time = max(1, int(1000 / RECORDING_FPS)) # Wait in ms per frame
          if cv2.waitKey(wait_time) == 27:
              break
      else: # Live camera
          current_time = time.time()
          fps = 1.0 / (current_time - last_time)
          last_time = current_time
          cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Yellow text
          if cv2.waitKey(1) == 27:
              break


      # --- RECORD PROCESSED FOOTAGE ---
      if ENABLE_VIDEO_RECORDING and processed_video_writer.isOpened():
          processed_video_writer.write(display)

      # Display the resulting frame
      cv2.imshow("Robot Tracking Prototype", display)
      # Optional: Display opponent mask for debugging
      # if opponent_mask_display is not None:
      #     cv2.imshow("Opponent Mask", opponent_mask_display)
      # Optional: Display our bot's color mask for debugging
      # if ENABLE_OUR_BOT_COLOR_TRACKING and 'our_bot_mask' in locals():
      #     cv2.imshow("Our Bot Color Mask", our_bot_mask)


finally: # This block always executes, even if an error occurs or loop breaks
    print("Application closing. Sending neutral commands to robot.")
    robot_controller.send_commands(RC_CENTER, RC_CENTER, RC_MIN, RC_MAX)
    robot_controller.close_serial() # Close the serial port

    # ==============================
    # Cleanup
    # ==============================
    cap.release()
    cv2.destroyAllWindows()
    
    if ENABLE_VIDEO_RECORDING:
        if raw_video_writer is not None and raw_video_writer.isOpened():
            raw_video_writer.release()
            print("Raw video writer released.")
        if processed_video_writer is not None and processed_video_writer.isOpened():
            processed_video_writer.release()
            print("Processed video writer released.")
    
    print("Application closed.")