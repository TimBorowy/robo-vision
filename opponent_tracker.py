# opponent_tracker.py
from ultralytics import YOLO
import cv2
import numpy as np
import torch

class OpponentTracker:
    """
    Tracks the opponent using YOLOv8.
    Visualizes all potential robot candidates in white and the filtered target in red.
    """
    def __init__(self, model_path='robot_model.pt', confidence=0.45):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"OpponentTracker: Initializing YOLOv8 on {self.device}...")

        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.confidence = confidence
        self.current_opponent_bbox = None
        self.smoothed_center = None
        self.alpha = 0.4

        # DEBUG: Store all raw detections to draw white boxes
        self.all_detections = []

    def update(self, frame, kraken_bbox):
        """Runs YOLO inference and filters out Kraken."""
        results = self.model(frame, stream=False, verbose=False, conf=self.confidence, device=self.device)

        self.all_detections = []
        valid_opponents = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # b = [x1, y1, x2, y2]
                b = box.xyxy[0].cpu().numpy().astype(int)

                # Save every raw detection for the white box visualization
                self.all_detections.append(b)

                # --- IMPROVED SUBTRACTION LOGIC ---
                if self._is_kraken(b, kraken_bbox):
                    continue # Discard this box, it's our robot

                area = (b[2] - b[0]) * (b[3] - b[1])
                center = (int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2))
                valid_opponents.append({'bbox': b, 'area': area, 'center': center})

        if valid_opponents:
            # Sort by area: the largest valid object is our opponent
            valid_opponents.sort(key=lambda x: x['area'], reverse=True)
            best = valid_opponents[0]
            self.current_opponent_bbox = best['bbox']

            new_center = np.array(best['center'])
            if self.smoothed_center is None:
                self.smoothed_center = new_center
            else:
                self.smoothed_center = self.alpha * new_center + (1 - self.alpha) * self.smoothed_center
        else:
            self.current_opponent_bbox = None

        return self.smoothed_center, self.current_opponent_bbox

    def _is_kraken(self, yolo_bbox, kraken_bbox):
        """
        Aggressive overlap check. Returns True if yolo_bbox is Kraken.
        """
        if kraken_bbox is None:
            return False

        # Calculate intersection coordinates
        x1 = max(yolo_bbox[0], kraken_bbox[0])
        y1 = max(yolo_bbox[1], kraken_bbox[1])
        x2 = min(yolo_bbox[2], kraken_bbox[2])
        y2 = min(yolo_bbox[3], kraken_bbox[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area <= 0:
            return False

        yolo_area = (yolo_bbox[2] - yolo_bbox[0]) * (yolo_bbox[3] - yolo_bbox[1])
        kraken_area = (kraken_bbox[2] - kraken_bbox[0]) * (kraken_bbox[3] - kraken_bbox[1])

        # Check 1: Intersection relative to YOLO box (Handles Kraken inside a big YOLO box)
        overlap_yolo = inter_area / float(yolo_area)
        # Check 2: Intersection relative to Kraken box (Handles Marker box inside a big YOLO box)
        overlap_kraken = inter_area / float(kraken_area)

        # If either box covers 25% of the other, we consider it a match.
        # This 0.25 is aggressive to prevent "jumps".
        return (overlap_yolo > 0.25 or overlap_kraken > 0.25)

    def draw(self, display_frame):
        # 1. Draw ALL raw YOLO detections in White (DEBUG)
        for b in self.all_detections:
            cv2.rectangle(display_frame, (b[0], b[1]), (b[2], b[3]), (255, 255, 255), 1)
            cv2.putText(display_frame, "robot?", (b[0], b[3] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 2. Draw the filtered Opponent in Red (TARGET)
        if self.current_opponent_bbox is not None:
            b = self.current_opponent_bbox
            # Thick red box
            cv2.rectangle(display_frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
            # Semi-transparent label background
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (b[0], b[1] - 25), (b[0] + 100, b[1]), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
            cv2.putText(display_frame, "OPPONENT", (b[0] + 5, b[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)