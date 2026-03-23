# kraken_tracker.py
import cv2
import numpy as np
import math
from pupil_apriltags import Detector

class KrakenTracker:
    """
    Tracks 'Kraken' using a Hybrid approach:
    - Primary: AprilTags (Family 36h11)
    - Fallback: ArUco Markers (For legacy test data compatibility)
    """
    def __init__(self, camera_matrix, dist_coeffs, tags_top, tags_bottom, tag_size_mm=50, smoothing_alpha=0.4):
        # 1. AprilTag Detector
        self.at_detector = Detector(families='tag36h11', nthreads=1, quad_decimate=1.0)

        # 2. ArUco Detector (Legacy Support)
        # Using the 4x4_250 dictionary as per your original Phase 1 code
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        self.camera_params = [camera_matrix[0,0], camera_matrix[1,1], camera_matrix[0,2], camera_matrix[1,2]]
        self.tag_size = tag_size_mm
        self.smoothing_alpha = smoothing_alpha

        self.tags_top = tags_top       # [LeftID, RightID]
        self.tags_bottom = tags_bottom # [LeftID, RightID]

        self.smoothed_center = None
        self.smoothed_orientation_rad = None
        self.current_bbox = None
        self.is_inverted = False

    def update(self, gray_frame):
        """Detects Kraken using either AprilTags or ArUco markers."""
        all_pts = []
        pts_left = []
        pts_right = []
        visible_top = 0
        visible_bottom = 0

        # --- 1. Try AprilTag Detection ---
        at_results = self.at_detector.detect(gray_frame, estimate_tag_pose=True, camera_params=self.camera_params, tag_size=self.tag_size)

        for tag in at_results:
            all_pts.append(tag.center)
            if tag.tag_id in self.tags_top:
                visible_top += 1
                if tag.tag_id == self.tags_top[0]: pts_left.append(tag.center)
                else: pts_right.append(tag.center)
            elif tag.tag_id in self.tags_bottom:
                visible_bottom += 1
                if tag.tag_id == self.tags_bottom[0]: pts_left.append(tag.center)
                else: pts_right.append(tag.center)

        # --- 2. Fallback to ArUco Detection (If no AprilTags found) ---
        if not all_pts:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray_frame)
            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    # ArUco markers only used for 'Top' face in this fallback
                    if marker_id in self.tags_top:
                        center = np.mean(corners[i][0], axis=0)
                        all_pts.append(center)
                        visible_top += 1
                        if marker_id == self.tags_top[0]: pts_left.append(center)
                        else: pts_right.append(center)

        if not all_pts:
            self.current_bbox = None # Crucial: Let the system know Kraken is lost
            return None, None, None, self.is_inverted

        # --- 3. Resolve Pose ---
        self.is_inverted = (visible_bottom > visible_top)
        raw_center = np.mean(all_pts, axis=0)

        raw_orientation = None
        if pts_left and pts_right:
            l_pt, r_pt = pts_left[0], pts_right[0]
            angle_lr = math.atan2(r_pt[1] - l_pt[1], r_pt[0] - l_pt[0])
            raw_orientation = angle_lr - (math.pi / 2)

        # Apply Smoothing
        self.smoothed_center = self._smooth_vec(self.smoothed_center, raw_center)
        if raw_orientation is not None:
            self.smoothed_orientation_rad = self._smooth_angle(self.smoothed_orientation_rad, raw_orientation)

        # Generate Exclusion Bounding Box (Slightly larger to ensure YOLO ignores it)
        all_pts_np = np.array(all_pts)
        x1, y1 = np.min(all_pts_np, axis=0) - 40
        x2, y2 = np.max(all_pts_np, axis=0) + 40
        self.current_bbox = (int(x1), int(y1), int(x2), int(y2))

        return self.smoothed_center, self.smoothed_orientation_rad, self.current_bbox, self.is_inverted

    def _smooth_vec(self, old, new):
        if old is None: return new
        return self.smoothing_alpha * new + (1 - self.smoothing_alpha) * old

    def _smooth_angle(self, old, new):
        if old is None: return new
        return math.atan2(
            self.smoothing_alpha * math.sin(new) + (1 - self.smoothing_alpha) * math.sin(old),
            self.smoothing_alpha * math.cos(new) + (1 - self.smoothing_alpha) * math.cos(old)
        )

    def draw(self, display_frame):
        if self.smoothed_center is not None and self.current_bbox is not None:
            c = tuple(self.smoothed_center.astype(int))
            color = (0, 165, 255) if self.is_inverted else (255, 0, 255)
            # Draw Kraken's "Inclusion Zone" in purple/orange
            cv2.rectangle(display_frame, (self.current_bbox[0], self.current_bbox[1]),
                          (self.current_bbox[2], self.current_bbox[3]), color, 1, cv2.LINE_AA)
            cv2.circle(display_frame, c, 6, color, -1)

            if self.smoothed_orientation_rad is not None:
                length = 70
                end_x = int(c[0] + length * math.cos(self.smoothed_orientation_rad))
                end_y = int(c[1] + length * math.sin(self.smoothed_orientation_rad))
                cv2.line(display_frame, c, (end_x, end_y), (0, 255, 0), 3)