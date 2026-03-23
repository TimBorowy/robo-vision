# arena_tracker.py
import cv2
import numpy as np

class ArenaTracker:
    """
    Manages the arena polygon, background subtraction, and detection of moving objects.
    Applies masks to ignore movement outside the designated arena area.
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

        # Default arena polygon: full frame
        self.arena_polygon = np.array([
            [[0, 0]], [[frame_width, 0]], [[frame_width, frame_height]], [[0, frame_height]]
        ], dtype=np.int32)
        print("ArenaTracker: Default arena polygon set to full frame.")

        self.moving_objects =[] # Stores dicts: {'center': (x,y), 'bbox': (x1,y1,x2,y2), 'area': float}
        self.cached_arena_mask = None

    def set_arena_polygon(self, polygon_points):
        """
        Sets the bounds of the arena. Movement outside this polygon is ignored.
        `polygon_points`: List of (x, y) tuples.
        """
        if len(polygon_points) < 3:
            print("ArenaTracker: Warning: Polygon must have at least 3 points. Using default full frame.")
            return

        self.arena_polygon = np.array([[[p[0], p[1]]] for p in polygon_points], dtype=np.int32)
        self.cached_arena_mask = None # Invalidate the cache
        print(f"ArenaTracker: Arena polygon updated to {polygon_points}.")

    def get_arena_mask(self, frame_shape):
        """Generates or retrieves the cached binary mask of the arena area."""
        if self.cached_arena_mask is None or self.cached_arena_mask.shape != frame_shape[:2]:
            self.cached_arena_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
            cv2.fillPoly(self.cached_arena_mask, [self.arena_polygon], 255)
        return self.cached_arena_mask

    def detect_moving_objects(self, frame, kraken_bbox=None, kraken_center_smoothed=None, 
                              overlap_threshold=0.5, center_buffer_px=30):
        """
        Detects all moving objects within the arena polygon, explicitly excluding our bot.
        Returns a list of dictionaries containing center, bounding box, and area, sorted by area descending.
        """
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Cleanup noise
        kernel_fg = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_fg)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_fg)

        # Restrict detection to the arena polygon
        arena_mask = self.get_arena_mask(frame.shape)
        fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=arena_mask)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.moving_objects =[]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue

            center = self.contour_center(cnt)
            if center is None:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            obj_bbox = (x, y, x + w, y + h)

            # --- Exclude our bot to prevent self-targeting ---
            is_our_bot = False
            x1_obj, y1_obj, x2_obj, y2_obj = obj_bbox

            # Check 1: Bounding Box Overlap (Used when markers/color are actively tracking)
            if kraken_bbox is not None:
                x1_krak, y1_krak, x2_krak, y2_krak = kraken_bbox

                inter_x1 = max(x1_krak, x1_obj)
                inter_y1 = max(y1_krak, y1_obj)
                inter_x2 = min(x2_krak, x2_obj)
                inter_y2 = min(y2_krak, y2_obj)

                inter_width = max(0, inter_x2 - inter_x1)
                inter_height = max(0, inter_y2 - inter_y1)
                intersection_area = inter_width * inter_height

                kraken_area = (x2_krak - x1_krak) * (y2_krak - y1_krak)
                obj_area = w * h

                if intersection_area > 0:
                    if (obj_area > 0 and (intersection_area / obj_area) > overlap_threshold) or \
                       (kraken_area > 0 and (intersection_area / kraken_area) > overlap_threshold):
                        is_our_bot = True

            # Check 2: Smoothed Center Inclusion (Used as fallback if markers flicker for a frame)
            if not is_our_bot and kraken_center_smoothed is not None:
                kx, ky = kraken_center_smoothed
                if (x1_obj - center_buffer_px <= kx <= x2_obj + center_buffer_px) and \
                   (y1_obj - center_buffer_px <= ky <= y2_obj + center_buffer_px):
                    is_our_bot = True

            if is_our_bot:
                continue 
            
            self.moving_objects.append({'center': center, 'bbox': obj_bbox, 'area': area})
        
        # Sort descending so the largest moving objects are first in the list
        self.moving_objects.sort(key=lambda x: x['area'], reverse=True)
        return self.moving_objects

    def draw_arena(self, display_frame):
        """Draws the arena polygon."""
        cv2.polylines(display_frame,[self.arena_polygon], True, (100, 100, 255), 4)

    def draw_moving_objects(self, display_frame):
        """Draws white bounding boxes around all detected generic moving objects."""
        for obj in self.moving_objects:
            x1, y1, x2, y2 = obj['bbox']
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    def contour_center(self, contour):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)