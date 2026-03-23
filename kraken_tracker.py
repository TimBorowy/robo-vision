# kraken_tracker.py
import cv2
import numpy as np
import math
from pupil_apriltags import Detector

class KrakenTracker:
    """
    Tracks 'Kraken' using 4 possible AprilTags.
    Supports face-flipping (Top vs Bottom).
    """
    def __init__(self, camera_matrix, dist_coeffs, tags_top, tags_bottom, tag_size_mm=50, smoothing_alpha=0.4):
        self.detector = Detector(families='tag36h11', nthreads=1, quad_decimate=1.0)
        self.camera_params = [camera_matrix[0,0], camera_matrix[1,1], camera_matrix[0,2], camera_matrix[1,2]]
        self.tag_size = tag_size_mm
        self.smoothing_alpha = smoothing_alpha

        # Configurable IDs from main.py
        self.tags_top = tags_top       # [LeftID, RightID]
        self.tags_bottom = tags_bottom # [LeftID, RightID]

        self.smoothed_center = None
        self.smoothed_orientation_rad = None
        self.current_bbox = None
        self.is_inverted = False

    def update(self, gray_frame):
        tags = self.detector.detect(gray_frame, estimate_tag_pose=True, camera_params=self.camera_params, tag_size=self.tag_size)

        pts_left = []
        pts_right = []
        all_pts = []
        visible_top = 0
        visible_bottom = 0

        for tag in tags:
            all_pts.append(tag.center)
            if tag.tag_id in self.tags_top:
                visible_top += 1
                if tag.tag_id == self.tags_top[0]: pts_left.append(tag.center) # Left
                else: pts_right.append(tag.center) # Right
            elif tag.tag_id in self.tags_bottom:
                visible_bottom += 1
                if tag.tag_id == self.tags_bottom[0]: pts_left.append(tag.center) # Left
                else: pts_right.append(tag.center) # Right

        if not all_pts:
            return None, None, None, self.is_inverted

        # Determine if we are inverted based on majority vote of visible tags
        if visible_bottom > visible_top:
            self.is_inverted = True
        elif visible_top > visible_bottom:
            self.is_inverted = False

        raw_center = np.mean(all_pts, axis=0)

        # Calculate heading: Vector from Left Tag to Right Tag
        raw_orientation = None
        if pts_left and pts_right:
            l_pt = pts_left[0]
            r_pt = pts_right[0]
            # Heading is 90 deg clockwise from the L->R vector
            # But if inverted, the robot's "Forward" is relative to the mechanical build
            angle_lr = math.atan2(r_pt[1] - l_pt[1], r_pt[0] - l_pt[0])
            raw_orientation = angle_lr - (math.pi / 2)

        self.smoothed_center = self._smooth_vec(self.smoothed_center, raw_center)
        if raw_orientation is not None:
            self.smoothed_orientation_rad = self._smooth_angle(self.smoothed_orientation_rad, raw_orientation)

        all_pts_np = np.array(all_pts)
        x1, y1 = np.min(all_pts_np, axis=0) - 30
        x2, y2 = np.max(all_pts_np, axis=0) + 30
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
        if self.smoothed_center is not None:
            c = tuple(self.smoothed_center.astype(int))
            color = (0, 165, 255) if self.is_inverted else (255, 0, 255)
            cv2.circle(display_frame, c, 6, color, -1)

            if self.current_bbox:
                cv2.rectangle(display_frame, (self.current_bbox[0], self.current_bbox[1]),
                              (self.current_bbox[2], self.current_bbox[3]), color, 2)

            if self.smoothed_orientation_rad is not None:
                length = 70
                end_x = int(c[0] + length * math.cos(self.smoothed_orientation_rad))
                end_y = int(c[1] + length * math.sin(self.smoothed_orientation_rad))
                cv2.line(display_frame, c, (end_x, end_y), (0, 255, 0), 3)