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
