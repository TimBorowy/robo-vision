# arena_tracker.py
import cv2
import numpy as np

class ArenaTracker:
    """
    Manages the arena polygon and provides masking logic.
    In Phase 2, this is used to restrict YOLO inference and AprilTag
    searches to the playable area only.
    """
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Default arena polygon: full frame
        self.arena_polygon = np.array([
            [[0, 0]], [[frame_width, 0]], [[frame_width, frame_height]], [[0, frame_height]]
        ], dtype=np.int32)

        self.cached_arena_mask = None

    def set_arena_polygon(self, polygon_points):
        """Sets the bounds of the arena. Movement outside this polygon is ignored."""
        if len(polygon_points) < 3:
            return
        self.arena_polygon = np.array([[[p[0], p[1]]] for p in polygon_points], dtype=np.int32)
        self.cached_arena_mask = None

    def get_arena_mask(self, frame_shape):
        """Generates or retrieves the cached binary mask of the arena area."""
        if self.cached_arena_mask is None or self.cached_arena_mask.shape != frame_shape[:2]:
            self.cached_arena_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
            cv2.fillPoly(self.cached_arena_mask, [self.arena_polygon], 255)
        return self.cached_arena_mask

    def draw_arena(self, display_frame):
        """Draws the arena boundary."""
        cv2.polylines(display_frame, [self.arena_polygon], True, (0, 255, 0), 2)