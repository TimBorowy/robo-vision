# opponent_tracker.py
from ultralytics import YOLO
import cv2
import numpy as np
import torch

class OpponentTracker:
    """
    Tracks the opponent using YOLOv8 optimized for GPU.
    Filters out 'Kraken' by checking for overlap with AprilTag bounding box.
    """
    def __init__(self, model_path='yolov8n.pt', confidence=0.45):
        # Automatically select GPU (cuda) if available, otherwise CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"OpponentTracker: Initializing YOLOv8 on {self.device}...")

        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.confidence = confidence
        self.current_opponent_bbox = None
        self.smoothed_center = None
        self.alpha = 0.4 # Smoothing factor

    def update(self, frame, kraken_bbox):
        """Runs YOLO inference and identifies the opponent."""
        # Use half precision on GPU for faster inference
        results = self.model(frame, stream=False, verbose=False, conf=self.confidence, device=self.device)

        potential_opponents = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates [x1, y1, x2, y2]
                b = box.xyxy[0].cpu().numpy().astype(int)

                # Check if this box is actually our robot (Kraken)
                if self._is_kraken(b, kraken_bbox):
                    continue

                area = (b[2] - b[0]) * (b[3] - b[1])
                center = (int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2))
                potential_opponents.append({'bbox': b, 'area': area, 'center': center})

        if potential_opponents:
            # Heuristic: The largest remaining detected robot is the opponent
            potential_opponents.sort(key=lambda x: x['area'], reverse=True)
            best = potential_opponents[0]
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
        """Intersection over Union (IoU) check to filter out self-detections."""
        if kraken_bbox is None:
            return False

        x1 = max(yolo_bbox[0], kraken_bbox[0])
        y1 = max(yolo_bbox[1], kraken_bbox[1])
        x2 = min(yolo_bbox[2], kraken_bbox[2])
        y2 = min(yolo_bbox[3], kraken_bbox[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area <= 0: return False

        yolo_area = (yolo_bbox[2] - yolo_bbox[0]) * (yolo_bbox[3] - yolo_bbox[1])
        # If more than 40% of the YOLO detection is inside Kraken's marker zone, ignore it
        return (inter_area / float(yolo_area)) > 0.4

    def draw(self, display_frame):
        if self.current_opponent_bbox is not None:
            b = self.current_opponent_bbox
            cv2.rectangle(display_frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cv2.putText(display_frame, "OPPONENT", (b[0], b[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)