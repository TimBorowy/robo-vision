import serial
import struct
import time

class RobotController:
    """
    Handles serial communication with the ELRS TX module to send RC commands.
    """
    def __init__(self, serial_port, baudrate,
                 channel_throttle, channel_steering, channel_mode_switch, channel_kill_switch,
                 rc_min, rc_center, rc_max,
                 autonomous_speed_forward, autonomous_speed_stop,
                 autonomous_turn_left, autonomous_turn_right, autonomous_turn_straight,
                 enable_comms=True):

        self.serial_port = serial_port
        self.baudrate = baudrate
        self.enable_comms = enable_comms
        self.ser = None # Serial port object

        self.CHANNEL_THROTTLE = channel_throttle
        self.CHANNEL_STEERING = channel_steering
        self.CHANNEL_MODE_SWITCH = channel_mode_switch
        self.CHANNEL_KILL_SWITCH = channel_kill_switch

        self.RC_MIN = rc_min
        self.RC_CENTER = rc_center
        self.RC_MAX = rc_max

        self.AUTONOMOUS_SPEED_FORWARD = autonomous_speed_forward
        self.AUTONOMOUS_SPEED_STOP = autonomous_speed_stop
        self.AUTONOMOUS_TURN_LEFT = autonomous_turn_left
        self.AUTONOMOUS_TURN_RIGHT = autonomous_turn_right
        self.AUTONOMOUS_TURN_STRAIGHT = autonomous_turn_straight

        if self.enable_comms:
            self._init_serial_elrs()
        else:
            print("Serial communication for robot control is DISABLED by configuration.")

    def _init_serial_elrs(self):
        """Initializes the serial connection to the ELRS TX module."""
        if self.ser is not None and self.ser.is_open:
            self.ser.close() # Close existing connection if any
        try:
            self.ser = serial.Serial(self.serial_port, self.baudrate, timeout=0.05)
            print(f"RobotController: Connected to ELRS module on {self.serial_port} at {self.baudrate} baud.")
        except serial.SerialException as e:
            print(f"RobotController: Error connecting to ELRS module on {self.serial_port}: {e}")
            print("RobotController: Please ensure the Radiomaster MT-12 is connected, configured for USB Serial (CRSF),")
            print("RobotController: and that the correct COM port is specified in ELRS_SERIAL_PORT.")
            self.ser = None

    def _map_pwm_to_crsf(self, pwm_val):
        """Maps a 1000-2000 PWM value to an 11-bit CRSF value (0-2048)."""
        # 1000 -> 0, 1500 -> 992, 2000 -> 2048 (approximate)
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
        
        # Ensure CRSF channel values are within valid 0-2047 range
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
        self.ser = None # Ensure it's marked as closed