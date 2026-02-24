import cv2
import numpy as np
import math
import time

# ==============================
# Configuration
# ==============================

CAMERA_INDEX = 1 # Often 0 for default webcam, try 1, 2, etc. if it doesn't work.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

MIN_CONTOUR_AREA = 2000 # Minimum pixel area for a contour to be considered a bot
SMOOTHING_ALPHA = 0.4 # Alpha for exponential moving average smoothing (0.0 - 1.0, higher means less smoothing)

OUR_BOT_ARUCO_ID = 1 # IMPORTANT: Set this to the specific ID of the ArUco marker on your bot.
                     # You can generate and print ArUco markers online (e.g., Aruco marker generator).

# ==============================
# Helpers
# ==============================

def smooth(old, new, alpha):
  """Applies exponential moving average for smoothing values."""
  if old is None:
    return new
  return alpha * new + (1 - alpha) * old

def compute_angle_and_distance(p1, p2):
  """Computes the angle (degrees) and distance (pixels) between two points."""
  dx = p2[0] - p1[0]
  dy = p2[1] - p1[1]
  distance = math.sqrt(dx * dx + dy * dy)
  angle = math.degrees(math.atan2(dy, dx))
  return angle, distance

def contour_center(contour):
  """Calculates the centroid of a contour."""
  M = cv2.moments(contour)
  if M["m00"] == 0:
    return None
  cx = int(M["m10"] / M["m00"])
  cy = int(M["m01"] / M["m00"])
  return (cx, cy)

# ==============================
# Camera Setup
# ==============================

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open video stream from camera index {CAMERA_INDEX}.")
    print("Please check if the camera is connected and the index is correct.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# ArUco setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Smoothed positions for our bot and the other bot
our_center_smoothed = None
other_center_smoothed = None

last_time = time.time() # For FPS calculation

# ==============================
# Main Loop
# ==============================

print("Starting robot tracking prototype. Press 'ESC' to exit.")

while True:
  ret, frame = cap.read()
  if not ret:
    print("Failed to grab frame. Exiting...")
    break

  display = frame.copy()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # ---------------------------------
  # Detect our bot (ArUco Marker)
  # ---------------------------------
  corners, ids, rejected = aruco_detector.detectMarkers(gray)

  our_bbox = None
  our_center = None

  if ids is not None:
    # Look for our specific ArUco marker ID
    our_marker_index = -1
    for i, marker_id in enumerate(ids):
      if marker_id == OUR_BOT_ARUCO_ID:
        our_marker_index = i
        break

    if our_marker_index != -1:
      # Found our bot's marker
      pts = corners[our_marker_index][0]
      x_min = int(np.min(pts[:, 0]))
      x_max = int(np.max(pts[:, 0]))
      y_min = int(np.min(pts[:, 1]))
      y_max = int(np.max(pts[:, 1]))

      our_bbox = (x_min, y_min, x_max, y_max)
      our_center = (int(pts[:, 0].mean()), int(pts[:, 1].mean())) # Center of the marker

      cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) # Green box for our bot

  # ---------------------------------
  # Detect other bot (Color-based contour)
  # This assumes the opponent bot has a distinct color/darkness
  # compared to the arena floor and our bot (after ArUco removal).
  # You WILL need to tune 'lower' and 'upper' HSV values for your specific setup.
  # ---------------------------------
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  # Example range for very dark objects on a potentially lighter floor:
  # Adjust these values based on the actual color of the opponent bot and arena.
  # Use a tool like https://www.peko.ws/hsv-color-picker/ or OpenCV's trackbars to find appropriate values.
  lower_hsv = np.array([0, 0, 0])      # Example: for detecting black/dark objects
  upper_hsv = np.array([180, 255, 80]) # Example: adjust the 'V' (Value/Brightness) to isolate dark areas

  mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

  # Morphological operations to clean up the mask
  kernel = np.ones((5, 5), np.uint8)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Fills small holes
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Removes small noise

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

    # Skip contour if it significantly overlaps with our bot's detected area
    # This helps prevent detecting our own bot as the opponent
    if our_bbox is not None:
      x_min, y_min, x_max, y_max = our_bbox
      # Check if the contour's bounding box center is inside our bot's bounding box
      # or if a significant portion overlaps (this is a simple check)
      if x_min - 20 <= cxcy[0] <= x_max + 20 and y_min - 20 <= cxcy[1] <= y_max + 20: # Added a small buffer
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
  # Smoothing bot positions
  # ---------------------------------
  if our_center is not None:
    our_center_smoothed = smooth(
      our_center_smoothed,
      np.array(our_center),
      SMOOTHING_ALPHA
    )

  if other_center is not None:
    other_center_smoothed = smooth(
      other_center_smoothed,
      np.array(other_center),
      SMOOTHING_ALPHA
    )

  # ---------------------------------
  # Calculate and display Angle + Distance
  # ---------------------------------
  if our_center_smoothed is not None and other_center_smoothed is not None:

    angle, distance = compute_angle_and_distance(
      our_center_smoothed,
      other_center_smoothed
    )

    # Draw line between bots
    p1 = tuple(our_center_smoothed.astype(int))
    p2 = tuple(other_center_smoothed.astype(int))
    cv2.line(display, p1, p2, (0, 255, 255), 2) # Yellow line

    text = f"A: {angle:.1f}  D: {distance:.0f}px"

    # Display text near our bot's bounding box
    if our_bbox is not None:
      x_min, y_min, _, _ = our_bbox
      cv2.putText(
        display,
        text,
        (x_min, y_min - 10), # Position text just above our bot's bbox
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0), # Green text
        2
      )

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

  # Exit on 'ESC' key press
  if cv2.waitKey(1) == 27:
    break

# ==============================
# Cleanup
# ==============================
cap.release()
cv2.destroyAllWindows()
print("Application closed.")