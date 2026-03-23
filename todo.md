Lets restructure and refactor some code. I want all values to to with control to be moved to RobotController.

Is it posible to seperate each system as a seperate subsystem?

How the new system will work:

- draw a polygon, everything outside of this polygon should be ignored. Everything inside is inside of the arena, we should track this.

- We always try seperate all moving elements from the background.
- There should be at least 2 robots in the arena, ours, and the opponent but sometimes there are more we call this a "rumble"
- mark each moving object with a green yellow square, you should do this always.

Subsystems:

KrakenTracker
position and orient our robot called "Kraken" do this with a few systems I want to be able to tweak and turn of if needed.
- position with hsv color
- position and orient with aruco markers

Opponent tracker
- position with optional hsv color tracking
- other wise, target the closest moving object

Code should be split logically in different files provide structure and code seperation


# Phase 2 Yolov8

Act as an expert Python Computer Vision and Robotics Software Engineer. 

I have an overhead camera tracking system for a 1v1 robot combat/competition game. Currently, the system uses OpenCV with ArUco markers, Background Subtraction (MOG2), and HSV color tracking to track my robot ("Kraken") and the opponent robot. 

We are pivoting the architecture to Phase 2 to drastically improve reliability against lighting changes, shadows, and motion blur. 

**Here is the Plan of Action for this session:**
1. **Use Current Architecture as a Guide:** Keep the modular structure (`main.py`, `arena_tracker`, `kraken_tracker`, `opponent_tracker`). We want to retain the concept of an "Arena Polygon Mask" (ignoring everything outside a defined playable area).
2. **Remove the Control Loop:** Strip out all the `RobotController`, ELRS serial communication, and control math (angle/distance). We are isolating the codebase to perfect the vision pipeline first. We will re-implement a smarter PID control loop later.
3. **Ditch MOG2 and HSV:** Completely remove the `cv2.createBackgroundSubtractorMOG2` and all HSV color thresholding logic. They are too sensitive to environmental lighting changes.
4. **Implement YOLOv8 Nano:** Replace the opponent detection (and generic object detection) with a YOLOv8 Nano model. We will assume the model outputs bounding boxes for the class "robot". 
5. **Implement AprilTags:** Replace the ArUco marker tracking with AprilTag tracking for our robot ("Kraken"). AprilTags are slightly more robust to motion blur and lighting gradients.

**The New Tracking Hierarchy & Logic:**
*   **Our Bot (Kraken):** Tracked primarily by AprilTags. 2 on each sides on the top and 2 on the bottom of the robot. This provides exact Position and Heading.
*   **The Opponent:** YOLOv8 Nano will detect *all* robots in the arena. We will take the YOLO bounding boxes, check if they overlap with our AprilTag's location, and if they do, we filter that box out (since it's Kraken). The *remaining* YOLO bounding box is the opponent.

Please generate the refactored Python code for this new Phase 2 architecture. Focus on clean, modular, object-oriented design. I do not have a trained YOLOv8 model named yet, can we use a generic model? and if not or if it's better to create our own, explain to be how i can do this. I have a large dataset of images of robots and overhead video footage of robots moving in the arena. I have a 3060TI in one PC or a 5060TI in another pc available to run this code or generate anyting. `robot_model.pt`.
