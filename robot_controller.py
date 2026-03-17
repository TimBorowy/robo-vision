# robot_controller.py
import serial
import struct
import time

class RobotController:
    """
    Handles serial communication with the ELRS TX module to send RC commands.
    Encapsulates all RC-related configuration values.
    """
    # --- RC Channel Mapping ---
    CHANNEL_THROTTLE = 0   # Corresponds to Channel 1 on your radio
    CHANNEL_STEERING = 1   # Corresponds to Channel 2
    CHANNEL_MODE_SWITCH = 4 # Corresponds to Channel 5 (Aux1) - for Manual/Autonomous
    CHANNEL_KILL_SWITCH = 7 # Corresponds to Channel 8 (Aux4) - for Emergency Stop

    # --- Default RC Values (1000-2000 PWM range) ---
    RC_MIN = 1000
    RC_CENTER = 1500
    RC_MAX = 2000

    # --- Autonomous Control Parameters ---
    AUTONOMOUS_SPEED_FORWARD = 1600 # Example forward speed
    AUTONOMOUS_SPEED_STOP = 1500    # Stop speed
    AUTONOMOUS_TURN_LEFT = 1200     # Example left turn value (turn left)
    AUTONOMOUS_TURN_RIGHT = 1800    # Example right turn value (turn right)
    AUTONOMOUS_TURN_STRAIGHT = 1500 # Straight steering

    # --- ELRS Serial Port Configuration ---
    # ELRS_BAUDRATE is a fixed standard, so it can remain a class attribute.

    ELRS_BAUDRATE = 420000 # Standard ELRS serial baudrate

    def __init__(self, serial_port, enable_comms=True): # ELRS_SERIAL_PORT now passed here
        self.serial_port = serial_port # Store as instance attribute
        self.enable_comms = enable_comms
        self.ser = None # Serial port object

        if self.enable_comms:
            self._init_serial_elrs()
        else:
            print("RobotController: Serial communication for robot control is DISABLED by configuration.")

    def _init_serial_elrs(self):
        """Initializes the serial connection to the ELRS TX module."""
        if self.ser is not None and self.ser.is_open:
            self.ser.close() # Close existing connection if any
        try:
            self.ser = serial.Serial(self.serial_port, self.ELRS_BAUDRATE, timeout=0.05)
            print(f"RobotController: Connected to ELRS module on {self.serial_port} at {self.ELRS_BAUDRATE} baud.")
        except serial.SerialException as e:
            print(f"RobotController: Error connecting to ELRS module on {self.serial_port}: {e}")
            print("RobotController: Please ensure the Radiomaster MT-12 is connected, configured for USB Serial (CRSF),")
            print(f"RobotController: and that the correct COM port ({self.serial_port}) is specified in main.py.")
            self.ser = None

    def _map_pwm_to_crsf(self, pwm_val):
        """Maps a 1000-2000 PWM value to an 11-bit CRSF value (0-2048)."""
        return int((pwm_val - self.RC_MIN) * 2.048)

    def send_commands(self, throttle, steering, mode_switch, kill_switch, num_channels=16):
        """
        Sends RC commands via CRSF protocol to the ELRS module.
        Values for throttle, steering, mode_switch, kill_switch are expected in 1000-2000 PWM range.
        """
        if not self.enable_comms:
            return # Do nothing if comms are disabled

        if self.ser is None or not self.ser.is_open:
            # Try to re-initialize if disconnected
            self._init_serial_elrs()
            if self.ser is None or not self.ser.is_open:
                print("RobotController: ELRS module not connected, cannot send commands.")
                return

        channels_crsf = [self._map_pwm_to_crsf(self.RC_CENTER)] * num_channels # Initialize all channels to CRSF center

        channels_crsf[self.CHANNEL_THROTTLE] = self._map_pwm_to_crsf(throttle)
        channels_crsf[self.CHANNEL_STEERING] = self._map_pwm_to_crsf(steering)
        channels_crsf[self.CHANNEL_MODE_SWITCH] = self._map_pwm_to_crsf(mode_switch)
        channels_crsf[self.CHANNEL_KILL_SWITCH] = self._map_pwm_to_crsf(kill_switch)
        
        channels_crsf = [max(0, min(2047, ch)) for ch in channels_crsf]

        payload = b''
        for ch in channels_crsf:
            payload += struct.pack('<H', ch) # <H for unsigned short, little-endian

        packet_type = 0x16 # CRSF type for RC channel data
        payload_length = len(payload) + 1 # +1 for CRC byte

        header = struct.pack('>BB', payload_length, packet_type) # >BB for 2 unsigned chars, big-endian

        # Calculate CRC8 (simple XOR CRC for CRSF)
        crc_data = bytearray([payload_length, packet_type]) + payload
        crc = 0
        for byte in crc_data:
            crc ^= byte
        
        full_packet = b'\xC8' + header + payload + bytes([crc])

        try:
            self.ser.write(full_packet)
        except serial.SerialException as e:
            print(f"RobotController: Error writing to ELRS module: {e}")
            self.ser.close()
            self.ser = None # Mark as disconnected

    def close_serial(self):
        """Closes the serial connection if it's open."""
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
            print("RobotController: ELRS module serial port closed.")
        self.ser = None