# main.py
import cv2
import numpy as np
import os
import time
from arena_tracker import ArenaTracker
from kraken_tracker import KrakenTracker
from opponent_tracker import OpponentTracker

# ==============================
# GLOBAL CONFIGURATION
# ==============================

# --- INPUT SOURCE ---
USE_VIDEO_FILE = False
VIDEO_PATH = "video_input/kraken-vs-knackwurst-stream-720.mp4"
CAMERA_INDEX = 1
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# --- KRAKEN (OUR BOT) ---
# Tag IDs for Top and Bottom faces
KRAKEN_TAGS_TOP = [101, 102]      # 101: Left, 102: Right
KRAKEN_TAGS_BOTTOM = [103, 104]   # 103: Left, 104: Right
TAG_SIZE_MM = 50

# --- TRACKING ---
ARENA_POINTS = [(340,60), (FRAME_WIDTH-370, 90), (FRAME_WIDTH, FRAME_HEIGHT-150), (FRAME_WIDTH, FRAME_HEIGHT), (0, FRAME_HEIGHT), (0, FRAME_HEIGHT-200)]
YOLO_MODEL_PATH = "yolov8n.pt" # Update to 'robot_model.pt' after training
YOLO_CONFIDENCE = 0.4
SMOOTHING_ALPHA = 0.4

CAMERA_CALIB_FILE = "camera_calibration.npz"

def load_camera_params():
    if os.path.exists(CAMERA_CALIB_FILE):
        data = np.load(CAMERA_CALIB_FILE)
        return data['mtx'], data['dist']
    print("Warning: Calibration file not found. Accuracy will be degraded.")
    return np.eye(3), np.zeros(5)

def main():
    mtx, dist = load_camera_params()

    # Using CAP_DSHOW for fast initialization on Windows
    if USE_VIDEO_FILE:
        cap = cv2.VideoCapture(VIDEO_PATH)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    arena = ArenaTracker(FRAME_WIDTH, FRAME_HEIGHT)
    arena.set_arena_polygon(ARENA_POINTS)

    # Initialize KrakenTracker with new ID lists
    kraken = KrakenTracker(
        mtx, dist,
        tags_top=KRAKEN_TAGS_TOP,
        tags_bottom=KRAKEN_TAGS_BOTTOM,
        tag_size_mm=TAG_SIZE_MM,
        smoothing_alpha=SMOOTHING_ALPHA
    )

    opponent = OpponentTracker(model_path=YOLO_MODEL_PATH, confidence=YOLO_CONFIDENCE)

    print("Phase 2 Vision System Active. Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        start_time = time.time()

        # 1. Pre-process
        mask = arena.get_arena_mask(frame.shape)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

        # 2. Track Kraken (Handles 4 tags + Inverted states)
        k_center, k_head, k_bbox, is_inverted = kraken.update(gray)

        # 3. Track Opponent
        o_center, o_bbox = opponent.update(masked_frame, k_bbox)

        # 4. Visualization
        display_frame = frame.copy()
        arena.draw_arena(display_frame)
        kraken.draw(display_frame)
        opponent.draw(display_frame)

        # UI Overlay
        if k_center is not None:
            status = "INVERTED" if is_inverted else "NORMAL"
            color = (0, 165, 255) if is_inverted else (255, 0, 255)
            cv2.putText(display_frame, f"KRAKEN: {status}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if k_center is not None and o_center is not None:
            cv2.line(display_frame, tuple(k_center.astype(int)), tuple(o_center.astype(int)), (0, 255, 255), 2)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Phase 2 - AI Tracking", display_frame)
        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()