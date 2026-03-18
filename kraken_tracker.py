# kraken_tracker.py
# (Only the update method needs changing, but here is the whole file context)
import cv2
import numpy as np
import math
import time

class KrakenTracker:
    # ... __init__ stays exactly the same ...
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
        self.our_bot_aruco_ids =[left_marker_id, right_marker_id]
        
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

        print(f"KrakenTracker: Initialized with ArUco dict type {aruco_dict_type}.")

    def update(self, frame, hsv_frame, gray_frame, arena_mask=None): # <<< ADDED arena_mask
        """
        Updates Kraken's position and orientation based on ArUco and/or color detection.
        Returns (smoothed_center, smoothed_orientation_rad, current_bbox_combined)
        """
        our_aruco_center = None
        our_aruco_orientation_rad = None
        our_aruco_bbox = None

        marker_right_data = {'center': None, 'orientation_rad': None, 'bbox': None}
        marker_left_data = {'center': None, 'orientation_rad': None, 'bbox': None}
        
        self._individual_marker_draw_data =[]

        # 1. ArUco Detection
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray_frame)
        
        if ids is not None and self.camera_matrix is not None:
            for i, marker_id in enumerate(ids):
                if marker_id in self.our_bot_aruco_ids:
                    pts = corners[i][0]
                    x_min, y_min = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))
                    x_max, y_max = int(np.max(pts[:, 0])), int(np.max(pts[:, 1]))

                    current_bbox = (x_min, y_min, x_max, y_max)
                    current_center = (int(pts[:, 0].mean()), int(pts[:, 1].mean()))

                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers([corners[i]], self.aruco_marker_size_mm, self.camera_matrix, self.dist_coeffs)

                    current_orientation_rad = None
                    if rvecs is not None and len(rvecs) > 0:
                        rvec = rvecs[0][0]
                        tvec = tvecs[0][0]

                        object_points = np.array([[0, 0, 0],[0, self.aruco_marker_size_mm / 2, 0]], dtype=np.float32).reshape(-1, 1, 3)
                        img_pts, _ = cv2.projectPoints(object_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)

                        pt1_flat = img_pts[0].ravel()
                        pt2_flat = img_pts[1].ravel()
                        
                        try:
                            x1, y1 = float(pt1_flat[0]), float(pt1_flat[1])
                            x2, y2 = float(pt2_flat[0]), float(pt2_flat[1])

                            if not (math.isnan(x1) or math.isnan(y1) or math.isnan(x2) or math.isnan(y2)):
                                if abs(x1) < 16384 and abs(y1) < 16384 and abs(x2) < 16384 and abs(y2) < 16384:
                                    
                                    p_center_fwd = (int(round(x1)), int(round(y1)))
                                    p_forward_end = (int(round(x2)), int(round(y2)))
                                    
                                    self._individual_marker_draw_data.append({
                                        'center_fwd': p_center_fwd,
                                        'end_fwd': p_forward_end,
                                        'bbox': current_bbox,
                                        'id': marker_id 
                                    })

                                    dx_forward = x2 - x1
                                    dy_forward = y2 - y1
                                    current_orientation_rad = math.atan2(dy_forward, dx_forward)
                        except (ValueError, TypeError, OverflowError):
                            pass
                    
                    if marker_id == self.right_marker_id:
                        marker_right_data = {'center': current_center, 'orientation_rad': current_orientation_rad, 'bbox': current_bbox}
                    elif marker_id == self.left_marker_id:
                        marker_left_data = {'center': current_center, 'orientation_rad': current_orientation_rad, 'bbox': current_bbox}

        # Combine ArUco data for overall bot pose
        if marker_right_data['center'] is not None and marker_left_data['center'] is not None:
            our_aruco_center = (
                (marker_right_data['center'][0] + marker_left_data['center'][0]) // 2,
                (marker_right_data['center'][1] + marker_left_data['center'][1]) // 2
            )
            dx_lr = marker_right_data['center'][0] - marker_left_data['center'][0]
            dy_lr = marker_right_data['center'][1] - marker_left_data['center'][1]
            our_aruco_orientation_rad = math.atan2(-dx_lr, dy_lr)
            
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

        # 2. Our Bot Color Tracking
        our_color_center = None
        our_color_bbox = None
        if self.enable_color_tracking:
            our_bot_mask = cv2.inRange(hsv_frame, self.our_bot_color_lower_hsv, self.our_bot_color_upper_hsv)
            
            # <<< NEW: Apply arena mask to our bot's color tracking as well
            if arena_mask is not None:
                our_bot_mask = cv2.bitwise_and(our_bot_mask, our_bot_mask, mask=arena_mask)

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
            
            if largest_our_bot_contour is not None and largest_our_bot_area > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(largest_our_bot_contour)
                our_color_bbox = (x, y, x + w, y + h)
                our_color_center = self._contour_center(largest_our_bot_contour)

        # 3. Combine Position & Orientation (Current Frame)
        self.current_center = None
        self.current_orientation_rad = None
        self.current_bbox_combined = None

        if our_color_center is not None:
            self.current_center = our_color_center
            self.current_bbox_combined = our_color_bbox
        elif our_aruco_center is not None:
            self.current_center = our_aruco_center 
            self.current_bbox_combined = our_aruco_bbox
        
        self.current_orientation_rad = our_aruco_orientation_rad

        # 4. Apply Smoothing
        self.smoothed_center = self._smooth(self.smoothed_center, np.array(self.current_center), self.smoothing_alpha) if self.current_center is not None else self.smoothed_center
        
        if self.current_orientation_rad is not None:
            if self.smoothed_orientation_rad is None:
                self.smoothed_orientation_rad = self.current_orientation_rad
            else:
                old_complex = math.cos(self.smoothed_orientation_rad) + 1j * math.sin(self.smoothed_orientation_rad)
                new_complex = math.cos(self.current_orientation_rad) + 1j * math.sin(self.current_orientation_rad)
                smoothed_complex = self.smoothing_alpha * new_complex + (1 - self.smoothing_alpha) * old_complex
                self.smoothed_orientation_rad = math.atan2(smoothed_complex.imag, smoothed_complex.real)
        
        return (self.smoothed_center, self.smoothed_orientation_rad, self.current_bbox_combined)

    def draw(self, display_frame):
        # ... drawing method stays completely the same ...
        for marker_data in self._individual_marker_draw_data:
            x_min, y_min, x_max, y_max = marker_data['bbox']
            cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (162, 0, 255), 2) # Purple box
            cv2.putText(display_frame, f"ID: {marker_data['id']}", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (162, 0, 255), 1)
            
            try:
                cv2.line(display_frame, marker_data['center_fwd'], marker_data['end_fwd'], (0, 255, 255), 2)
            except Exception:
                pass

        if self.smoothed_center is not None and self.current_bbox_combined is not None:
            bbox_width = self.current_bbox_combined[2] - self.current_bbox_combined[0]
            bbox_height = self.current_bbox_combined[3] - self.current_bbox_combined[1]
            smooth_x_min = int(self.smoothed_center[0] - bbox_width / 2)
            smooth_y_min = int(self.smoothed_center[1] - bbox_height / 2)
            smooth_x_max = int(self.smoothed_center[0] + bbox_width / 2)
            smooth_y_max = int(self.smoothed_center[1] + bbox_height / 2)
            cv2.rectangle(display_frame, (smooth_x_min, smooth_y_min), (smooth_x_max, smooth_y_max), (200, 60, 225), 2) # Pink box

            if self.smoothed_orientation_rad is not None:
                line_length = 50 
                try:
                    end_x = int(round(float(self.smoothed_center[0] + line_length * math.cos(self.smoothed_orientation_rad))))
                    end_y = int(round(float(self.smoothed_center[1] + line_length * math.sin(self.smoothed_orientation_rad))))
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