# kraken_tracker.py
import cv2
import numpy as np
import math
import time

class KrakenTracker:
    """
    Tracks our robot ("Kraken") using ArUco markers for orientation and redundant position,
    and optionally HSV color for high-speed position.
    """
    def __init__(self, aruco_dict_type, aruco_marker_size_mm, our_bot_aruco_ids,
                 our_bot_color_lower_hsv, our_bot_color_upper_hsv,
                 camera_matrix, dist_coeffs, smoothing_alpha,
                 enable_color_tracking=True, min_contour_area=500):
        
        self.aruco_detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(aruco_dict_type),
            cv2.aruco.DetectorParameters()
        )
        self.aruco_marker_size_mm = aruco_marker_size_mm
        self.our_bot_aruco_ids = our_bot_aruco_ids
        
        self.our_bot_color_lower_hsv = our_bot_color_lower_hsv
        self.our_bot_color_upper_hsv = our_bot_color_upper_hsv
        self.enable_color_tracking = enable_color_tracking
        self.min_contour_area = min_contour_area # Used for color tracking
        
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.smoothing_alpha = smoothing_alpha

        # Store individual marker data for drawing
        self._individual_marker_draw_data = [] # List of {'center_fwd', 'end_fwd', 'bbox'}

        self.current_center = None
        self.current_orientation_rad = None
        self.current_bbox_combined = None

        self.smoothed_center = None
        self.smoothed_orientation_rad = None

        print(f"KrakenTracker: Initialized with ArUco dict type {aruco_dict_type}.")

    def update(self, frame, hsv_frame, gray_frame):
        """
        Updates Kraken's position and orientation based on ArUco and/or color detection.
        Returns (smoothed_center, smoothed_orientation_rad, current_bbox_combined)
        """
        our_aruco_center = None
        our_aruco_orientation_rad = None
        our_aruco_bbox = None # Bbox encompassing all detected ArUco markers

        marker_right_data = {'center': None, 'orientation_rad': None, 'bbox': None} # ID 102 (Right)
        marker_left_data = {'center': None, 'orientation_rad': None, 'bbox': None} # ID 101 (Left)
        
        self._individual_marker_draw_data = [] # Reset draw data for this frame

        # 1. ArUco Detection
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray_frame)
        
        if ids is not None and self.camera_matrix is not None:
            # print(f"KrakenTracker: Detected ArUco IDs: {ids.flatten()}") # Debug print
            for i, marker_id in enumerate(ids):
                if marker_id in self.our_bot_aruco_ids:
                    pts = corners[i][0]
                    x_min, y_min = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))
                    x_max, y_max = int(np.max(pts[:, 0])), int(np.max(pts[:, 1]))

                    current_bbox = (x_min, y_min, x_max, y_max)
                    current_center = (int(pts[:, 0].mean()), int(pts[:, 1].mean()))

                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners[i]], self.aruco_marker_size_mm, self.camera_matrix, self.dist_coeffs
                    )

                    current_orientation_rad = None
                    if rvecs is not None and len(rvecs) > 0:
                        rvec = rvecs[0][0]
                        tvec = tvecs[0][0]

                        # Project a forward vector for the marker itself
                        object_points = np.array([[0, 0, 0], [0, self.aruco_marker_size_mm / 2, 0]], dtype=np.float32).reshape(-1, 1, 3)
                        img_pts, _ = cv2.projectPoints(object_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)

                        p_center_fwd = tuple(img_pts[0][0].astype(int))
                        p_forward_end = tuple(img_pts[1][0].astype(int))
                        
                        # Store for drawing later
                        self._individual_marker_draw_data.append({
                            'center_fwd': p_center_fwd,
                            'end_fwd': p_forward_end,
                            'bbox': current_bbox
                        })

                        dx_forward = img_pts[1][0,0] - img_pts[0][0,0]
                        dy_forward = img_pts[1][0,1] - img_pts[0][0,1]
                        current_orientation_rad = math.atan2(dy_forward, dx_forward)
                    
                    if marker_id == 102: # Right marker
                        marker_right_data = {'center': current_center, 'orientation_rad': current_orientation_rad, 'bbox': current_bbox}
                    elif marker_id == 101: # Left marker
                        marker_left_data = {'center': current_center, 'orientation_rad': current_orientation_rad, 'bbox': current_bbox}
        # else:
            # print("KrakenTracker: No ArUco IDs detected for our bot.") # Debug print

        # Combine ArUco data for overall bot pose
        if marker_right_data['center'] is not None and marker_left_data['center'] is not None:
            our_aruco_center = (
                (marker_right_data['center'][0] + marker_left_data['center'][0]) // 2,
                (marker_right_data['center'][1] + marker_left_data['center'][1]) // 2
            )
            dx_lr = marker_right_data['center'][0] - marker_left_data['center'][0]
            dy_lr = marker_right_data['center'][1] - marker_left_data['center'][1]
            our_aruco_orientation_rad = math.atan2(-dx_lr, dy_lr) # 90-deg CW from left-to-right vector
            
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

        # 2. Our Bot Color Tracking (for redundant position)
        our_color_center = None
        our_color_bbox = None
        if self.enable_color_tracking:
            our_bot_mask = cv2.inRange(hsv_frame, self.our_bot_color_lower_hsv, self.our_bot_color_upper_hsv)
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
        self.current_bbox_combined = None # This will be the bbox for the green smoothed box

        if our_color_center is not None:
            self.current_center = our_color_center # Prioritize color for position if found
            self.current_bbox_combined = our_color_bbox
        elif our_aruco_center is not None:
            self.current_center = our_aruco_center # Fallback to ArUco for position
            self.current_bbox_combined = our_aruco_bbox
        
        # Orientation *always* comes from ArUco (color blobs don't give orientation)
        self.current_orientation_rad = our_aruco_orientation_rad # Will be None if no ArUco detected

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
        """Draws our robot's elements on the display frame."""
        # Draw individual ArUco markers and their forward lines
        for marker_data in self._individual_marker_draw_data:
            x_min, y_min, x_max, y_max = marker_data['bbox']
            cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (162, 0, 255), 2) # Purple box
            cv2.line(display_frame, marker_data['center_fwd'], marker_data['end_fwd'], (0, 255, 255), 2) # Yellow line

        # Draw smoothed combined bounding box for our bot (using Pink)
        if self.smoothed_center is not None and self.current_bbox_combined is not None:
            bbox_width = self.current_bbox_combined[2] - self.current_bbox_combined[0]
            bbox_height = self.current_bbox_combined[3] - self.current_bbox_combined[1]
            smooth_x_min = int(self.smoothed_center[0] - bbox_width / 2)
            smooth_y_min = int(self.smoothed_center[1] - bbox_height / 2)
            smooth_x_max = int(self.smoothed_center[0] + bbox_width / 2)
            smooth_y_max = int(self.smoothed_center[1] + bbox_height / 2)
            cv2.rectangle(display_frame, (smooth_x_min, smooth_y_min), (smooth_x_max, smooth_y_max), (200, 60, 225), 2) # Pink box

            # Draw the smoothed orientation line (from smoothed center)
            if self.smoothed_orientation_rad is not None:
                line_length = 50 # pixels
                end_x = int(self.smoothed_center[0] + line_length * math.cos(self.smoothed_orientation_rad))
                end_y = int(self.smoothed_center[1] + line_length * math.sin(self.smoothed_orientation_rad))
                cv2.line(display_frame, tuple(self.smoothed_center.astype(int)), (end_x, end_y), (0, 255, 0), 2) # Green line for overall bot heading


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