# arena_tracker.py
import cv2
import numpy as np
import time

class ArenaTracker:
    """
    Manages the arena polygon, background subtraction, and detection of all moving objects.
    """
    def __init__(self, frame_width, frame_height, min_contour_area,
                 bg_subtractor_history=500, bg_subtractor_var_threshold=16, bg_subtractor_detect_shadows=True):
        
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.min_contour_area = min_contour_area
        
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=bg_subtractor_history,
            varThreshold=bg_subtractor_var_threshold,
            detectShadows=bg_subtractor_detect_shadows
        )
        print("ArenaTracker: Background Subtractor (MOG2) initialized.")

        # Default arena polygon: full frame (can be set interactively later)
        self.arena_polygon = np.array([
            [[0, 0]], [[frame_width, 0]], [[frame_width, frame_height]], [[0, frame_height]]
        ], dtype=np.int32)
        print("ArenaTracker: Default arena polygon set to full frame.")

        self.moving_objects = [] # Stores (center, bbox) for all detected moving objects

    def set_arena_polygon(self, polygon_points):
        """
        Sets the arena polygon. `polygon_points` should be a list of (x, y) tuples.
        Example: [(100,100), (FRAME_WIDTH-100, 100), (FRAME_WIDTH-100, FRAME_HEIGHT-100), (100, FRAME_HEIGHT-100)]
        """
        if len(polygon_points) < 3:
            print("ArenaTracker: Warning: Polygon must have at least 3 points. Using default full frame.")
            return

        self.arena_polygon = np.array([[[p[0], p[1]]] for p in polygon_points], dtype=np.int32)
        print(f"ArenaTracker: Arena polygon updated to {polygon_points}.")

    def detect_moving_objects(self, frame, our_bot_bbox_combined=None, our_bot_overlap_threshold=0.5):
        """
        Detects all moving objects within the arena polygon, excluding our bot.
        Returns a list of (center_point, bbox) for each detected object.
        """
        fg_mask = self.bg_subtractor.apply(frame)
        
        kernel_fg = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_fg)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_fg)

        # Apply arena mask to fg_mask
        arena_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(arena_mask, [self.arena_polygon], 255)
        fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=arena_mask)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.moving_objects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue

            cxcy = self.contour_center(cnt)
            if cxcy is None:
                continue
            
            x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
            current_bbox = (x_cnt, y_cnt, x_cnt + w_cnt, y_cnt + h_cnt)

            # --- Exclude our bot using overlap check ---
            is_overlap_with_our_bot = False
            if our_bot_bbox_combined is not None:
                x1_our, y1_our, x2_our, y2_our = our_bot_bbox_combined
                x1_other, y1_other, x2_other, y2_other = current_bbox

                xi1 = max(x1_our, x1_other)
                yi1 = max(y1_our, y1_other)
                xi2 = min(x2_our, x2_other)
                yi2 = min(y2_our, y2_other)

                inter_width = max(0, xi2 - xi1)
                inter_height = max(0, yi2 - yi1)
                intersection_area = inter_width * inter_height

                our_bot_combined_area = (x2_our - x1_our) * (y2_our - y1_our)
                candidate_area = w_cnt * h_cnt

                if intersection_area > 0:
                    if (candidate_area > 0 and (intersection_area / candidate_area) > our_bot_overlap_threshold) or \
                       (our_bot_combined_area > 0 and (intersection_area / our_bot_combined_area) > our_bot_overlap_threshold):
                        is_overlap_with_our_bot = True

            if is_overlap_with_our_bot:
                # Uncomment for debugging filtered-out objects
                # cv2.rectangle(frame, (x_cnt, y_cnt), (x_cnt + w_cnt, y_cnt + h_cnt), (0, 0, 255), 1) # Red for filtered
                continue 
            
            self.moving_objects.append({'center': cxcy, 'bbox': current_bbox, 'area': area})
        
        # Sort by area descending to prioritize larger objects (e.g., actual robots over small debris)
        self.moving_objects.sort(key=lambda x: x['area'], reverse=True)
        return self.moving_objects

    def draw_arena(self, display_frame):
        """Draws the arena polygon on the display frame."""
        cv2.polylines(display_frame, [self.arena_polygon], True, (100, 100, 255), 4) # Redish polygon

    def draw_moving_objects(self, display_frame):
        """Draws white squares around all detected moving objects."""
        for obj in self.moving_objects:
            x, y, w, h = obj['bbox']
            # White color (BGR: B=255, G=255, R=255)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 255), 2) # White box

    def contour_center(self, contour):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
        return (cx, cy)