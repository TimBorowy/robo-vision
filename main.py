# main.py
import cv2
import numpy as np
import math
import time
import os
from datetime import datetime

from robot_controller import RobotController
from arena_tracker import ArenaTracker
from kraken_tracker import KrakenTracker
from opponent_tracker import OpponentTracker

# ==============================
# Global Configuration
# ==============================

# --- INPUT SOURCE CONFIGURATION ---
USE_VIDEO_FILE_INPUT = True # <<< TOGGLE: True to use a video file, False for live camera
VIDEO_INPUT_FILE = "video_input/kraken-vs-knackwurst-stream-720.mp4" # <<< Specify your recorded raw video file here

CAMERA_INDEX = 1 # IMPORTANT: Set this to your external webcam index (only used if USE_VIDEO_FILE_INPUT is False)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

MIN_CONTOUR_AREA = 500 # Minimum pixel area for a contour to be considered a bot

# --- SMOOTHING ---
SMOOTHING_ALPHA = 0.4 # Alpha for exponential moving average smoothing (0.0 - 1.0, higher means less smoothing)

# --- OUR BOT (KRAKEN) CONFIGURATION ---
# IMPORTANT: These IDs should be 101 for Left, 102 for Right based on your previous description
LEFT_MARKER_ID = 101
RIGHT_MARKER_ID = 102
# OUR_BOT_ARUCO_IDS is now used internally by KrakenTracker, no need to define here anymore
OUR_BOT_ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_250
ARUCO_MARKER_SIZE_MM = 50 # IMPORTANT: The actual physical side length of your ArUco marker in millimeters

ENABLE_OUR_BOT_COLOR_TRACKING = True # <<< TOGGLE: True to enable color tracking for our bot
OUR_BOT_COLOR_LOWER_HSV = np.array([45, 0, 82]) # Example for purple/magenta - TUNE THIS!
OUR_BOT_COLOR_UPPER_HSV = np.array([136, 35, 203]) # Example for purple/magenta - TUNE THIS!

# Minimum percentage of overlap for a candidate opponent contour to be considered "our bot"
# (i.e., if a potential opponent's bounding box overlaps our bot's bounding box by this much, it's our bot)
OUR_BOT_OVERLAP_THRESHOLD_PERCENT = 0.5 # 0.5 means 50% overlap

# --- OPPONENT BOT CONFIGURATION ---
ENABLE_OPPONENT_COLOR_TRACKING = False # <<< TOGGLE: True to use HSV for opponent, False to use closest moving object
# If ENABLE_OPPONENT_COLOR_TRACKING is True, you need to tune these HSV values.
# Use 'calibrate_color.py' for this.
OPPONENT_COLOR_LOWER_HSV = np.array([70, 100, 100])
OPPONENT_COLOR_UPPER_HSV = np.array([100, 255, 255])

# --- ARENA & BACKGROUND SUBTRACTION ---
# For now, default arena polygon is full frame.
# You can define a custom polygon here:[(x1,y1), (x2,y2), ...]
# Example: ARENA_POLYGON_POINTS =[(100,100), (FRAME_WIDTH-100, 100), (FRAME_WIDTH-100, FRAME_HEIGHT-100), (100, FRAME_HEIGHT-100)]
ARENA_POLYGON_POINTS =[(340,60), (FRAME_WIDTH-370, 90), (FRAME_WIDTH, FRAME_HEIGHT-150), (FRAME_WIDTH, FRAME_HEIGHT), (0, FRAME_HEIGHT), (0, FRAME_HEIGHT-200)] # Set to None for full frame, or provide list of points
BG_SUBTRACTOR_HISTORY = 300
BG_SUBTRACTOR_VAR_THRESHOLD = 16
BG_SUBTRACTOR_DETECT_SHADOWS = True

# --- AUTONOMOUS CONTROL LOGIC ---
FORWARD_ANGLE_MARGIN_DEG = 5
LOG_INTERVAL_SECONDS = 2

# --- VIDEO RECORDING ---
ENABLE_VIDEO_RECORDING = True
RECORDING_FPS = 30.0         # Will be overwritten if using video file input
RECORDING_CODEC = 'mp4v'
VIDEO_OUTPUT_DIR = "video_output"

# --- FILE NAMES FOR CALIBRATION ---
CAMERA_CALIBRATION_FILE = "camera_calibration.npz"
OPPONENT_COLOR_CALIBRATION_FILE = "opponent_color_calibration.npz" # Only used if ENABLE_OPPONENT_COLOR_TRACKING is True

# --- ROBOT CONTROLLER CONFIGURATION ---
# Note: ELRS_SERIAL_PORT will be passed to RobotController instance
ELRS_SERIAL_PORT = 'COMX' # <<< IMPORTANT: CHANGE THIS TO YOUR ACTUAL SERIAL PORT
ENABLE_SERIAL_COMMS_DEFAULT = True # <<< Default toggle state for live camera (will be overridden by video input)


# ==============================
# Global Variables / Objects
# ==============================
camera_matrix = None
dist_coeffs = None
raw_video_writer = None
processed_video_writer = None

# ==============================
# Helpers
# ==============================

def load_camera_params(filename=CAMERA_CALIBRATION_FILE):
    global camera_matrix, dist_coeffs
    if not os.path.exists(filename):
        print(f"Error: Camera calibration file '{filename}' not found.")
        print("Please run 'calibrate_camera.py' first to generate it.")
        print("ArUco pose estimation will be inaccurate without calibration!")
        camera_matrix = np.array([[FRAME_WIDTH, 0, FRAME_WIDTH/2],[0, FRAME_WIDTH, FRAME_HEIGHT/2], [0, 0, 1]], dtype=np.float32)
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
        camera_matrix = np.array([[FRAME_WIDTH, 0, FRAME_WIDTH/2],[0, FRAME_WIDTH, FRAME_HEIGHT/2], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        return False

# This function is only called if ENABLE_OPPONENT_COLOR_TRACKING is True
def load_opponent_color_params(filename=OPPONENT_COLOR_CALIBRATION_FILE):
    lower_hsv, upper_hsv = None, None
    if not os.path.exists(filename):
        print(f"Error: Opponent color calibration file '{filename}' not found.")
        print("Please run 'calibrate_color.py' first to set and save color ranges.")
        print("Using default bluish-green values for opponent color tracking (might be incorrect).")
        lower_hsv = np.array([70, 100, 100])
        upper_hsv = np.array([100, 255, 255])
        return False, lower_hsv, upper_hsv
    try:
        npzfile = np.load(filename)
        lower_hsv = npzfile['lower_hsv']
        upper_hsv = npzfile['upper_hsv']
        print(f"Loaded opponent color calibration from {filename}")
        return True, lower_hsv, upper_hsv
    except Exception as e:
        print(f"Error loading opponent color calibration: {e}")
        print("Using default bluish-green values for opponent color tracking (might be incorrect).")
        lower_hsv = np.array([70, 100, 100])
        upper_hsv = np.array([100, 255, 255])
        return False, lower_hsv, upper_hsv

# ==============================
# Main Application Logic
# ==============================

def main():
    global camera_matrix, dist_coeffs, raw_video_writer, processed_video_writer

    print("Starting application...")
    
    # Load calibrations
    calibration_loaded = load_camera_params()
    
    opponent_hsv_lower, opponent_hsv_upper = None, None
    if ENABLE_OPPONENT_COLOR_TRACKING:
        _, opponent_hsv_lower, opponent_hsv_upper = load_opponent_color_params()

    # --- Adjust serial comms based on input source ---
    current_enable_serial_comms = ENABLE_SERIAL_COMMS_DEFAULT
    if USE_VIDEO_FILE_INPUT:
        current_enable_serial_comms = False
        print("WARNING: Video file input detected. Disabling serial communications to robot.")

    # Initialize RobotController
    robot_controller = RobotController(
        serial_port=ELRS_SERIAL_PORT,
        enable_comms=current_enable_serial_comms
    )

    start_time_profiling = time.time()
    # ==============================
    # Camera/Video Input Setup
    # ==============================
    cap = None
    if USE_VIDEO_FILE_INPUT:
        cap = cv2.VideoCapture(VIDEO_INPUT_FILE)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{VIDEO_INPUT_FILE}'. Exiting.")
            exit()
        global RECORDING_FPS
        RECORDING_FPS = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video file '{VIDEO_INPUT_FILE}' opened at {RECORDING_FPS:.2f} FPS.")
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"FATAL Error: Could not open video stream from camera index {CAMERA_INDEX} using CAP_DSHOW backend. Exiting.")
            exit()
        print(f"Camera opened in {time.time() - start_time_profiling:.2f} seconds.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        print(f"Camera properties set in {time.time() - start_time_profiling:.2f} seconds.")

    # ==============================
    # Initialize Tracking Subsystems
    # ==============================
    arena_tracker = ArenaTracker(
        FRAME_WIDTH, FRAME_HEIGHT, MIN_CONTOUR_AREA,
        BG_SUBTRACTOR_HISTORY, BG_SUBTRACTOR_VAR_THRESHOLD, BG_SUBTRACTOR_DETECT_SHADOWS
    )
    if ARENA_POLYGON_POINTS:
        # Note: If using ARENA_POLYGON_POINTS with a video file, ensure the points are scaled
        # correctly if the video's resolution is different from FRAME_WIDTH/HEIGHT.
        # For simplicity, assuming FRAME_WIDTH/HEIGHT matches the video resolution for custom polygon.
        arena_tracker.set_arena_polygon(ARENA_POLYGON_POINTS)

    kraken_tracker = KrakenTracker(
        OUR_BOT_ARUCO_DICT_TYPE, ARUCO_MARKER_SIZE_MM, 
        LEFT_MARKER_ID, RIGHT_MARKER_ID, # Pass specific marker IDs
        OUR_BOT_COLOR_LOWER_HSV, OUR_BOT_COLOR_UPPER_HSV,
        camera_matrix, dist_coeffs, SMOOTHING_ALPHA,
        ENABLE_OUR_BOT_COLOR_TRACKING, MIN_CONTOUR_AREA
    )

    opponent_tracker = OpponentTracker(
        ENABLE_OPPONENT_COLOR_TRACKING, opponent_hsv_lower, opponent_hsv_upper,
        SMOOTHING_ALPHA, MIN_CONTOUR_AREA
    )

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
        
        fourcc = cv2.VideoWriter_fourcc(*RECORDING_CODEC)

        raw_video_writer = cv2.VideoWriter(raw_video_filename, fourcc, RECORDING_FPS, (current_frame_width, current_frame_height))
        processed_video_writer = cv2.VideoWriter(processed_video_filename, fourcc, RECORDING_FPS, (current_frame_width, current_frame_height))

        if not raw_video_writer.isOpened():
            print(f"Error: Could not open raw video writer for {raw_video_filename}")
            raw_video_writer = None # Mark as failed
        if not processed_video_writer.isOpened():
            print(f"Error: Could not open processed video writer for {processed_video_filename}")
            processed_video_writer = None # Mark as failed
        
        if raw_video_writer or processed_video_writer:
            print(f"Video recording enabled. Raw: {raw_video_writer.getBackendName() if raw_video_writer else 'N/A'}, Processed: {processed_video_writer.getBackendName() if processed_video_writer else 'N/A'}")
        else:
            print("Video recording disabled due to errors during writer initialization.")


    last_time = time.time()
    last_log_time = time.time()

    print("Starting robot tracking prototype. Press 'ESC' to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video file or failed to grab frame. Exiting...")
                break

            if raw_video_writer and raw_video_writer.isOpened():
                raw_video_writer.write(frame)

            display_frame = frame.copy()
            
            # Pre-process frame for different detectors
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:,:,2]
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # <<< NEW: Fetch the arena mask from ArenaTracker so we can use it elsewhere
            arena_mask = arena_tracker.get_arena_mask(frame.shape)

            # 1. Update Kraken Tracker (our bot) - Pass arena_mask
            kraken_center_smoothed, kraken_orientation_smoothed_rad, kraken_bbox_current = \
                kraken_tracker.update(frame, hsv_frame, gray_frame, arena_mask)

            # 2. Update Arena Tracker (all other moving objects)
            all_moving_objects = arena_tracker.detect_moving_objects(frame, kraken_bbox_current, OUR_BOT_OVERLAP_THRESHOLD_PERCENT)
            
            # 3. Update Opponent Tracker (the specific opponent we target) - Pass arena_mask
            opponent_center_smoothed, opponent_bbox_current = \
                opponent_tracker.update(frame, hsv_frame, all_moving_objects, kraken_center_smoothed, arena_mask)
            
            # ====================================================
            # Calculate and display Relative Angle + Distance
            # And Autonomous Control Logic
            # ====================================================
            current_throttle_rc = robot_controller.AUTONOMOUS_SPEED_STOP
            current_steering_rc = robot_controller.AUTONOMOUS_TURN_STRAIGHT
            current_mode_switch_rc = robot_controller.RC_MAX # Assume Autonomous by default in code
            current_kill_switch_rc = robot_controller.RC_MAX # Assume Kill Switch OFF

            if kraken_center_smoothed is not None and opponent_center_smoothed is not None and kraken_orientation_smoothed_rad is not None:
                try:
                    p1_kraken = (int(round(float(kraken_center_smoothed[0]))), int(round(float(kraken_center_smoothed[1]))))
                    p2_opponent = (int(round(float(opponent_center_smoothed[0]))), int(round(float(opponent_center_smoothed[1]))))
                    
                    if abs(p1_kraken[0]) < 16384 and abs(p1_kraken[1]) < 16384 and abs(p2_opponent[0]) < 16384 and abs(p2_opponent[1]) < 16384:
                        cv2.line(display_frame, p1_kraken, p2_opponent, (0, 255, 255), 2) # Yellow line
                except Exception:
                    pass # Failsafe against drawing crashes

                dx_target = opponent_center_smoothed[0] - kraken_center_smoothed[0]
                dy_target = opponent_center_smoothed[1] - kraken_center_smoothed[1]
                angle_to_target_rad = math.atan2(dy_target, dx_target)

                relative_angle_rad = angle_to_target_rad - kraken_orientation_smoothed_rad
                while relative_angle_rad > math.pi:
                    relative_angle_rad -= 2 * math.pi
                while relative_angle_rad < -math.pi:
                    relative_angle_rad += 2 * math.pi

                relative_angle_deg = math.degrees(relative_angle_rad)
                distance_px = math.sqrt(dx_target * dx_target + dy_target * dy_target)

                text = f"Rel A: {relative_angle_deg:.1f}°  D: {distance_px:.0f}px"

                if kraken_center_smoothed is not None and kraken_bbox_current is not None:
                    bbox_width = kraken_bbox_current[2] - kraken_bbox_current[0]
                    bbox_height = kraken_bbox_current[3] - kraken_bbox_current[1]
                    text_x = int(kraken_center_smoothed[0] - bbox_width / 2)
                    text_y = int(kraken_center_smoothed[1] - bbox_height / 2 - 10)
                    cv2.putText(display_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                direction = ""
                if abs(relative_angle_deg) <= FORWARD_ANGLE_MARGIN_DEG:
                    direction = "FORWARD"
                    current_throttle_rc = robot_controller.AUTONOMOUS_SPEED_FORWARD
                    current_steering_rc = robot_controller.AUTONOMOUS_TURN_STRAIGHT
                elif relative_angle_deg < -FORWARD_ANGLE_MARGIN_DEG:
                    direction = "LEFT"
                    current_throttle_rc = robot_controller.AUTONOMOUS_SPEED_FORWARD
                    current_steering_rc = robot_controller.AUTONOMOUS_TURN_LEFT
                else:
                    direction = "RIGHT"
                    current_throttle_rc = robot_controller.AUTONOMOUS_SPEED_FORWARD
                    current_steering_rc = robot_controller.AUTONOMOUS_TURN_RIGHT
                
                current_mode_switch_rc = robot_controller.RC_MAX

                current_time_log = time.time()
                if current_time_log - last_log_time >= LOG_INTERVAL_SECONDS:
                    print(f"[{time.strftime('%H:%M:%S')}] Autonomous Command: {direction} (Rel A: {relative_angle_deg:.1f}°, D: {distance_px:.0f}px)")
                    last_log_time = current_time_log
            
            else:
                current_throttle_rc = robot_controller.AUTONOMOUS_SPEED_STOP
                current_steering_rc = robot_controller.AUTONOMOUS_TURN_STRAIGHT
                current_mode_switch_rc = robot_controller.RC_MIN # Force Manual Mode if vision lost
                if time.time() - last_log_time >= LOG_INTERVAL_SECONDS:
                    print(f"[{time.strftime('%H:%M:%S')}] Autonomous Vision Lost - Sending Neutral/Manual Commands.")
                    last_log_time = time.time()

            robot_controller.send_commands(current_throttle_rc, current_steering_rc, current_mode_switch_rc, current_kill_switch_rc)

            # ====================================================
            # Draw All Visual Elements
            # ====================================================
            arena_tracker.draw_arena(display_frame) # Draws arena polygon
            arena_tracker.draw_moving_objects(display_frame) # Draws green-yellow for all moving objects
            kraken_tracker.draw(display_frame) # Draws Kraken's smoothed green box & yellow heading line
            opponent_tracker.draw(display_frame) # Draws opponent's blue box


            # FPS Display
            if USE_VIDEO_FILE_INPUT:
                wait_time_ms = max(1, int(1000 / RECORDING_FPS))
                if cv2.waitKey(wait_time_ms) == 27:
                    break
            else:
                current_time = time.time()
                fps = 1.0 / (current_time - last_time)
                last_time = current_time
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if cv2.waitKey(1) == 27:
                    break

            if processed_video_writer and processed_video_writer.isOpened():
                processed_video_writer.write(display_frame)

            cv2.imshow("Robot Tracking Prototype", display_frame)


    finally:
        print("Application closing. Sending neutral commands to robot.")
        # Ensure final neutral commands are sent using instance attributes
        robot_controller.send_commands(robot_controller.RC_CENTER, robot_controller.RC_CENTER, robot_controller.RC_MIN, robot_controller.RC_MAX)
        robot_controller.close_serial()

        cap.release()
        cv2.destroyAllWindows()
        
        if raw_video_writer and raw_video_writer.isOpened():
            raw_video_writer.release()
            print("Raw video writer released.")
        if processed_video_writer and processed_video_writer.isOpened():
            processed_video_writer.release()
            print("Processed video writer released.")
        
        print("Application closed.")

if __name__ == "__main__":
    main()