# opponent_tracker.py
import cv2
import numpy as np
import math

class OpponentTracker:
    """
    Identifies the opponent bot from a list of moving objects.
    Can use HSV color tracking or simply target the closest moving object.
    """
    def __init__(self, enable_hsv_tracking=False, hsv_lower=None, hsv_upper=None, smoothing_alpha=0.4, min_contour_area=500):
        self.enable_hsv_tracking = enable_hsv_tracking
        self.hsv_lower = hsv_lower
        self.hsv_upper = hsv_upper
        self.smoothing_alpha = smoothing_alpha
        self.min_contour_area = min_contour_area

        self.current_center = None
        self.current_bbox = None
        self.smoothed_center = None
        
        print("OpponentTracker: Initialized.")
        if self.enable_hsv_tracking:
            print("OpponentTracker: Using HSV color tracking.")
        else:
            print("OpponentTracker: Targeting closest moving object.")

    def update(self, frame, hsv_frame, all_moving_objects, kraken_center_smoothed):
        """
        Identifies the opponent bot.
        `all_moving_objects` is a list of {'center': (x,y), 'bbox': (x1,y1,x2,y2), 'area': area}
        from ArenaTracker.
        Returns (smoothed_center, current_bbox)
        """
        potential_opponents = []
        
        if self.enable_hsv_tracking and self.hsv_lower is not None and self.hsv_upper is not None:
            # Filter objects by HSV color first
            opponent_mask = cv2.inRange(hsv_frame, self.hsv_lower, self.hsv_upper)
            kernel_color = np.ones((5, 5), np.uint8)
            opponent_mask = cv2.morphologyEx(opponent_mask, cv2.MORPH_CLOSE, kernel_color)
            opponent_mask = cv2.morphologyEx(opponent_mask, cv2.MORPH_OPEN, kernel_color)

            opponent_contours, _ = cv2.findContours(opponent_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in opponent_contours:
                area = cv2.contourArea(cnt)
                if area < self.min_contour_area:
                    continue
                cxcy = self._contour_center(cnt)
                if cxcy is not None:
                    x,y,w,h = cv2.boundingRect(cnt)
                    potential_opponents.append({'center': cxcy, 'bbox': (x,y,x+w,y+h), 'area': area})
            
            # Prioritize the largest color-matching object
            if potential_opponents:
                potential_opponents.sort(key=lambda x: x['area'], reverse=True)
                self.current_center = potential_opponents[0]['center']
                self.current_bbox = potential_opponents[0]['bbox']
            else:
                self.current_center = None
                self.current_bbox = None

        else: # Target closest moving object from ArenaTracker's list
            if kraken_center_smoothed is not None:
                closest_dist_sq = float('inf')
                closest_obj = None
                for obj in all_moving_objects:
                    dist_sq = (obj['center'][0] - kraken_center_smoothed[0])**2 + \
                              (obj['center'][1] - kraken_center_smoothed[1])**2
                    if dist_sq < closest_dist_sq:
                        closest_dist_sq = dist_sq
                        closest_obj = obj
                
                if closest_obj:
                    self.current_center = closest_obj['center']
                    self.current_bbox = closest_obj['bbox']
                else:
                    self.current_center = None
                    self.current_bbox = None
            else:
                # If Kraken's position is unknown, just pick the largest moving object as opponent (rumble style)
                if all_moving_objects:
                    largest_obj = all_moving_objects[0] # Already sorted by area in ArenaTracker
                    self.current_center = largest_obj['center']
                    self.current_bbox = largest_obj['bbox']
                else:
                    self.current_center = None
                    self.current_bbox = None

        # Apply smoothing
        self.smoothed_center = self._smooth(self.smoothed_center, np.array(self.current_center), self.smoothing_alpha) if self.current_center is not None else self.smoothed_center
        
        return (self.smoothed_center, self.current_bbox)

    def draw(self, display_frame):
        """Draws the opponent bot's elements on the display frame."""
        if self.current_bbox is not None: # Draw blue box around current detected opponent
            x, y, w, h = self.current_bbox
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2) # Blue box for opponent

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