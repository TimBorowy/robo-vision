# kraken_tracker.py
import cv2
import numpy as np
import math

class KrakenTracker:
    """
    Tracks our robot ("Kraken").
    Utilizes strict prioritization:
    - Priority 1: ArUco Markers (Highly accurate position and orientation)
    - Priority 2: HSV Color Blob (Fallback position, relies on last known ArUco orientation)
    """
    def __init__(self, aruco_dict_type, aruco_marker_size_mm,
                 left_marker_id, right_marker_id,
                 our_bot_color_lower_hsv, our_bot_color_upper_hsv,
                 camera_matrix, dist_coeffs, smoothing_alpha,
                 enable_color_tracking=True, min_contour_area=500):
        
        self.aruco_detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(aruco_dict_type),
            cv2.aruco.DetectorParameters()
        )
        self.aruco_marker_size_mm = aruco_marker_size_mm
        self.left_marker_id = left_marker_id
        self.right_marker_id = right_marker_id
        self.our_bot_aruco_ids = [left_marker_id, right_marker_id]
        
        self.our_bot_color_lower_hsv = our_bot_color_lower_hsv
        self.our_bot_color_upper_hsv = our_bot_color_upper_hsv
        self.enable_color_tracking = enable_color_tracking
        self.min_contour_area = min_contour_area
        
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.smoothing_alpha = smoothing_alpha

        self._individual_marker_draw_data =[] 

        self.current_center = None
        self.current_orientation_rad = None
        self.current_bbox_combined = None

        self.smoothed_center = None
        self.smoothed_orientation_rad = None
        
        # Memory variable used when ArUco tracking is temporarily lost
        self._last_known_orientation_rad = None 

        print(f"KrakenTracker: Initialized with ArUco dict type {aruco_dict_type}.")

    def update(self, frame, hsv_frame, gray_frame, arena_mask=None):
        """
        Updates Kraken's position and orientation based on ArUco and/or color detection.
        Returns (smoothed_center, smoothed_orientation_rad, current_bbox_combined)
        """
        aruco_center = None
        aruco_orientation_rad = None
        aruco_bbox = None

        marker_right = {'center': None, 'orientation_rad': None, 'bbox': None}
        marker_left = {'center': None, 'orientation_rad': None, 'bbox': None}
        
        self._individual_marker_draw_data =[]

        # --- 1. Detect ArUco Markers ---
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray_frame)
        
        if ids is not None and self.camera_matrix is not None:
            for i, marker_id in enumerate(ids):
                if marker_id in self.our_bot_aruco_ids:
                    pts = corners[i][0]
                    x_min, y_min = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))
                    x_max, y_max = int(np.max(pts[:, 0])), int(np.max(pts[:, 1]))

                    bbox = (x_min, y_min, x_max, y_max)
                    center = (int(pts[:, 0].mean()), int(pts[:, 1].mean()))

                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners[i]], self.aruco_marker_size_mm, self.camera_matrix, self.dist_coeffs
                    )

                    orientation_rad = None
                    if rvecs is not None and len(rvecs) > 0:
                        rvec, tvec = rvecs[0][0], tvecs[0][0]

                        object_points = np.array([[0, 0, 0],[0, self.aruco_marker_size_mm / 2, 0]], dtype=np.float32).reshape(-1, 1, 3)
                        img_pts, _ = cv2.projectPoints(object_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)

                        try:
                            pt1_flat = img_pts[0].ravel()
                            pt2_flat = img_pts[1].ravel()
                            x1, y1 = float(pt1_flat[0]), float(pt1_flat[1])
                            x2, y2 = float(pt2_flat[0]), float(pt2_flat[1])

                            # Ignore bad projections
                            if not (math.isnan(x1) or math.isnan(y1) or math.isnan(x2) or math.isnan(y2)):
                                if abs(x1) < 16384 and abs(y1) < 16384 and abs(x2) < 16384 and abs(y2) < 16384:
                                    self._individual_marker_draw_data.append({
                                        'center_fwd': (int(round(x1)), int(round(y1))),
                                        'end_fwd': (int(round(x2)), int(round(y2))),
                                        'bbox': bbox,
                                        'id': marker_id 
                                    })
                                    orientation_rad = math.atan2(y2 - y1, x2 - x1)
                        except (ValueError, TypeError, OverflowError):
                            pass
                    
                    if marker_id == self.right_marker_id:
                        marker_right = {'center': center, 'orientation_rad': orientation_rad, 'bbox': bbox}
                    elif marker_id == self.left_marker_id:
                        marker_left = {'center': center, 'orientation_rad': orientation_rad, 'bbox': bbox}

        # Determine overall bot pose from ArUco data
        if marker_right['center'] is not None and marker_left['center'] is not None:
            # Both markers visible: Center is midpoint, heading is right-angle to connecting vector
            aruco_center = (
                (marker_right['center'][0] + marker_left['center'][0]) // 2,
                (marker_right['center'][1] + marker_left['center'][1]) // 2
            )
            dx_lr = marker_right['center'][0] - marker_left['center'][0]
            dy_lr = marker_right['center'][1] - marker_left['center'][1]
            aruco_orientation_rad = math.atan2(-dx_lr, dy_lr)
            
            min_x = min(marker_right['bbox'][0], marker_left['bbox'][0])
            min_y = min(marker_right['bbox'][1], marker_left['bbox'][1])
            max_x = max(marker_right['bbox'][2], marker_left['bbox'][2])
            max_y = max(marker_right['bbox'][3], marker_left['bbox'][3])
            aruco_bbox = (min_x, min_y, max_x, max_y)

        elif marker_right['center'] is not None:
            aruco_center = marker_right['center']
            aruco_orientation_rad = marker_right['orientation_rad']
            aruco_bbox = marker_right['bbox']

        elif marker_left['center'] is not None:
            aruco_center = marker_left['center']
            aruco_orientation_rad = marker_left['orientation_rad']
            aruco_bbox = marker_left['bbox']

        # --- 2. Detect Color Blob (Fallback) ---
        color_center = None
        color_bbox = None
        
        if self.enable_color_tracking:
            color_mask = cv2.inRange(hsv_frame, self.our_bot_color_lower_hsv, self.our_bot_color_upper_hsv)
            
            if arena_mask is not None:
                color_mask = cv2.bitwise_and(color_mask, color_mask, mask=arena_mask)

            kernel = np.ones((5, 5), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            largest_contour = None
            largest_area = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > largest_area:
                    largest_area = area
                    largest_contour = cnt
            
            if largest_contour is not None and largest_area > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(largest_contour)
                color_bbox = (x, y, x + w, y + h)
                color_center = self._contour_center(largest_contour)

        # --- 3. Resolve Priorities ---
        self.current_center = None
        self.current_orientation_rad = None
        self.current_bbox_combined = None

        # Priority 1: ArUco Tracking
        if aruco_center is not None:
            self.current_center = aruco_center 
            self.current_bbox_combined = aruco_bbox
            self.current_orientation_rad = aruco_orientation_rad
            
            if self.current_orientation_rad is not None:
                self._last_known_orientation_rad = self.current_orientation_rad

        # Priority 2: Color Blob Fallback + Heading Memory
        elif color_center is not None:
            self.current_center = color_center
            self.current_bbox_combined = color_bbox
            self.current_orientation_rad = self._last_known_orientation_rad

        # --- 4. Apply Smoothing ---
        if self.current_center is not None:
            self.smoothed_center = self._smooth(self.smoothed_center, np.array(self.current_center), self.smoothing_alpha)
        
        if self.current_orientation_rad is not None:
            if self.smoothed_orientation_rad is None:
                self.smoothed_orientation_rad = self.current_orientation_rad
            else:
                # Smooth angular orientation securely using complex numbers
                old_complex = math.cos(self.smoothed_orientation_rad) + 1j * math.sin(self.smoothed_orientation_rad)
                new_complex = math.cos(self.current_orientation_rad) + 1j * math.sin(self.current_orientation_rad)
                smoothed_complex = self.smoothing_alpha * new_complex + (1 - self.smoothing_alpha) * old_complex
                self.smoothed_orientation_rad = math.atan2(smoothed_complex.imag, smoothed_complex.real)
        
        return (self.smoothed_center, self.smoothed_orientation_rad, self.current_bbox_combined)

    def draw(self, display_frame):
        for marker_data in self._individual_marker_draw_data:
            x1, y1, x2, y2 = marker_data['bbox']
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (162, 0, 255), 2) # Purple box
            cv2.putText(display_frame, f"ID: {marker_data['id']}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (162, 0, 255), 1)
            
            try:
                cv2.line(display_frame, marker_data['center_fwd'], marker_data['end_fwd'], (0, 255, 255), 2)
            except Exception:
                pass

        if self.smoothed_center is not None and self.current_bbox_combined is not None:
            bbox_w = self.current_bbox_combined[2] - self.current_bbox_combined[0]
            bbox_h = self.current_bbox_combined[3] - self.current_bbox_combined[1]
            smooth_x1 = int(self.smoothed_center[0] - bbox_w / 2)
            smooth_y1 = int(self.smoothed_center[1] - bbox_h / 2)
            smooth_x2 = int(self.smoothed_center[0] + bbox_w / 2)
            smooth_y2 = int(self.smoothed_center[1] + bbox_h / 2)
            cv2.rectangle(display_frame, (smooth_x1, smooth_y1), (smooth_x2, smooth_y2), (200, 60, 225), 2) # Pink box

            if self.smoothed_orientation_rad is not None:
                line_len = 50 
                try:
                    end_x = int(round(float(self.smoothed_center[0] + line_len * math.cos(self.smoothed_orientation_rad))))
                    end_y = int(round(float(self.smoothed_center[1] + line_len * math.sin(self.smoothed_orientation_rad))))
                    p_center = (int(round(float(self.smoothed_center[0]))), int(round(float(self.smoothed_center[1]))))
                    
                    if abs(end_x) < 16384 and abs(end_y) < 16384 and abs(p_center[0]) < 16384 and abs(p_center[1]) < 16384:
                        cv2.line(display_frame, p_center, (end_x, end_y), (0, 255, 0), 2)
                except Exception:
                    pass

    def _smooth(self, old, new, alpha):
        if old is None:
            return new
        return alpha * new + (1 - alpha) * old

    def _contour_center(self, contour):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)