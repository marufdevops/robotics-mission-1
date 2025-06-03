"""
Mobile robot to knock down the pins avoiding obstacles(wall)

Architecture:
    Finite State Machine (FSM) with four states:
    - EXPLORING: Default state for searching the environment
    - APPROACHING: Active when a pin is detected and being approached
    - AVOIDING_WALL: Triggered when walls/obstacles are detected
    - STUCK_RECOVERY: Emergency state when the robot is trapped

Design Principles:
    - Reactive control: Immediate response to sensor inputs
    - Behavior-based: Distinct behaviors for different situations
    - Subsumption-like: Higher priority states override lower ones
    - Holonomic motion: Utilizes omnidirectional movement capabilities
References:
    - Introduction to AI Robotics - Murphy, R.R
        - Discusses about sate machines architecture and the basics of sensors
    - Introduction to Autonomous Mobile Robots
        - Discusses about omnidirectional drive kinematics
    - https://automaticaddison.com/omni-directional-wheeled-robot-simulation-in-python/
    - https://www.youtube.com/watch?v=JbUNsYPJK1U (simulate a lidar sensor from scratch)
    - https://www.youtube.com/watch?v=EB0FQDWD2Gk (intro to state machine)
    - I have taken further helps from multiple LLM's
Author: Ahmed Maruf, SID: 250046920
"""

# Essential libraries for robotics control and sensor processing
import math
import random
import numpy as np
import cv2 as cv
from controller import Robot

# Create robot instance and get simulation timestep
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Actuators: Three-wheel omnidirectional wheels
w0_motor = robot.getDevice("wheel0_joint")
w1_motor = robot.getDevice("wheel1_joint")
w2_motor = robot.getDevice("wheel2_joint")
wheels = [w0_motor, w1_motor, w2_motor]

# Infrared sensors: for close-range obstacle detection (wall in this case) / collision avoidance
ir_sensors = [robot.getDevice(f"ir{i}") for i in range(1, 10)] # 9 IR sensors around robot
for irs in ir_sensors:
    irs.enable(timestep) # Enable each sensor with simulation timestep

# LiDAR sensors: for long-range 360° environment mapping
lidar = robot.getDevice("lidar")
lidar.enable(timestep)

# Vision sensor: Camera for visual feedback and debugging
camera = robot.getDevice('camera')
camera.enable(timestep)

# Motor configuration: Set motors to velocity control mode
for motor in wheels:
    motor.setPosition(math.inf)
    motor.setVelocity(0)

# Robot physical parameters
WHEEL_RADIUS = 0.063                     # Wheel radius [m]
DISTANCE_WHEEL_TO_ROBOT_CENTRE = 0.1826  # Distance from wheel to robot center [m]
MAX_SPEED = 8                            # Maximum allowed speed [m/s]

def base_apply_speeds(vx, vy, omega, wheels):
    """
    Translates desired robot velocities into individual wheel velocities.    
    Args:
        vx (float): Forward velocity in robot frame [m/s]
        vy (float): Lateral velocity in robot frame [m/s]
        omega (float): Rotational velocity [rad/s]
        wheels (list): List of motor objects controlling the wheels
    
    Returns:
        None: Directly applies velocities to wheel motors
    """
    # Convert linear velocities to wheel angular velocities
    vx /= WHEEL_RADIUS
    vy /= WHEEL_RADIUS
    # Convert robot angular velocity to wheel contribution
    omega *= DISTANCE_WHEEL_TO_ROBOT_CENTRE / WHEEL_RADIUS
    
    # Apply inverse kinematics matrix to get individual wheel velocities
    wheels[0].setVelocity(vy - omega)  # Front wheel
    wheels[1].setVelocity(-math.sqrt(0.75) * vx - 0.5 * vy - omega)  # Back-right wheel
    wheels[2].setVelocity( math.sqrt(0.75) * vx - 0.5 * vy - omega)  # Back-left wheel

def ir_volt_to_meter(V):
    """
    Converts IR sensor voltage readings to distance measurements.    
    Args:
        V (float): IR sensor voltage reading
    
    Returns:
        float: Estimated distance in meters
    """
    return 0.1594 * math.pow(V, -0.8533) - 0.02916


# Control parameters: Behavior tuning constants

BASE_SPEED = 1.0        # Exploration speed [m/s]
APPROACH_SPEED = 1.5    # Object approach speed [m/s]
WALL_AVOID_SPEED = 1.2  # Wall avoidance speed [m/s]
WALL_THRESHOLD = 0.35   # IR sensor wall detection threshold [m]
OBJECT_RANGE = 1.8      # LiDAR object detection range [m]
STUCK_TIME = int(3000 / timestep)  # Stuck detection time [timesteps] - ~3 seconds

# State variables: Finite state machine implementation

current_state = "EXPLORING"        # Current behavior state
state_timer = 0                    # Time spent in current state [timesteps]
stuck_timer = 0                    # Time spent detecting walls [timesteps]
objects_hit = 0                    # Mission progress counter
stuck_recovery_direction = 0       # Recovery behavior direction index

def detect_wall(ir_distances):
    """
    Detects walls or obstacles using IR sensor data.
    
    Implements a threshold-based detection algorithm that checks if any
    IR sensor reading falls below the safety threshold, indicating a
    nearby obstacle.
    
    Args:
        ir_distances (list): List of IR sensor distance readings [meters]
    
    Returns:
        bool: True if wall/obstacle detected, False otherwise
    """
    for distance in ir_distances:
        if 0 < distance < WALL_THRESHOLD:  # Valid reading within danger zone
            return True
    return False

def detect_object(lidar_distances):
    """
    Detects potential target objects (pins) using LiDAR data.
    
    Analyzes the LiDAR distance array to find objects within a specific range
    and direction relative to the robot's front.

    Args:
        lidar_distances (array): 256-element LiDAR distance array covering 360°
    
    Returns:
        tuple: (detected: bool, direction: int, distance: float)
               - detected: True if an object is within range
               - direction: -1 (right), 0 (center), +1 (left)
               - distance: Distance to the closest detected object [m]
    """
    closest_distance = float('inf')
    closest_direction = 0

    # SENSOR GEOMETRY: Front 120° arc mapping to LiDAR indices
    # LiDAR has 256 rays covering 360°, so each ray = 360°/256 = 1.40625°
    # Front 120° = indices 0-42 (right side) + 214-255 (left side)
    front_indices = list(range(0, 43)) + list(range(214, 256))

    for i in front_indices:
        distance = lidar_distances[i]

        # DISTANCE FILTERING: Remove noise and irrelevant detections
        if 0.3 < distance < OBJECT_RANGE and distance < closest_distance:
            closest_distance = distance

            # DIRECTION CLASSIFICATION: Map LiDAR index to steering direction
            # Based on sensor geometry and robot coordinate frame
            if i < 21 or i > 235: # Front-center region (±30°)
                closest_direction = 0 # Straight ahead
            elif i < 43: # Front-right region (30°-60°)
                closest_direction = -1  # Turn left to approach
            else: # Front-left region (300°-330°)
                closest_direction = 1 # Turn right to approach

    # RETURN DETECTION RESULT
    if closest_distance < OBJECT_RANGE:
        return True, closest_direction, closest_distance
    return False, 0, 0.0

print("START MISSION")

# Main control loop
while robot.step(timestep) != -1:
    """
    Main control loop implementing the finite state machine.
    
    Each iteration:
    1. Processes sensor data
    2. Updates state variables
    3. Determines appropriate behavior based on current state
    4. Applies motion commands to the robot
    5. Updates visualizations
    """

    # SENSOR DATA ACQUISITION AND PROCESSING    
    
    # LiDAR processing: Convert range image to distance array
    lidar_data = np.array(lidar.getRangeImage()).reshape((1,256))  # Convert to numpy array
    lidar_distances = lidar_data[0]  # Extract distance array (256 rays)

    # IR SENSOR DATA: Proximity measurements for collision avoidance
    ir_sensor_values = [ir_volt_to_meter(irs.getValue()) for irs in ir_sensors]  # Convert voltage to distance

    # CAMERA DATA: Visual feedback for debugging and monitoring
    camera_image = np.frombuffer(camera.getImage(), dtype=np.uint8).reshape(128,128,4)  # RGBA format

    # HIGH-LEVEL PERCEPTION: Detect environmental features
    wall_detected = detect_wall(ir_sensor_values)  # Proximity-based wall detection
    object_detected, object_direction, object_distance = detect_object(lidar_distances)  # LiDAR-based object detection

    # STATE MANAGEMENT
    state_timer += 1 # Increment time in current state

    # STUCK DETECTION: Monitor wall contact duration for recovery triggering
    if wall_detected:
        stuck_timer += 1 # Accumulate stuck time
    else:
        stuck_timer = 0 # Reset when not in contact with walls

    # VISUALIZATION: Update sensor displays for debugging
    cv.imshow('lidar', cv.resize(lidar_data, [600, 40]))
    cv.imshow('camera', cv.resize(camera_image, [256,256]))

    # BEHAVIOR SELECTION AND MOTION CONTROL

    # Initialize motion commands to zero (stationary default)
    vx, vy, omega = 0, 0, 0 # Linear velocities [m/s] and angular velocity [rad/s]

    # STATE MACHINE IMPLEMENTATION

    # STATE 1: STUCK RECOVERY - Highest priority state
    # Activated when stuck against a wall for too long
    if stuck_timer > STUCK_TIME:
        if current_state != "STUCK_RECOVERY":
            print(f"STUCK! Trying 4-directional recovery")
            current_state = "STUCK_RECOVERY"
            state_timer = 0
            stuck_recovery_direction = 0

        # Systematic 4-direction escape sequence
        direction_time = 40  # Duration to try each direction [timesteps]
        cycle_time = state_timer % (direction_time * 4)  # Current position in cycle

        # DIRECTION SEQUENCE: Forward → Right → Backward → Left
        if cycle_time < direction_time:  # Phase 1: Forward movement
            vx, vy, omega = WALL_AVOID_SPEED, 0, 0
        elif cycle_time < direction_time * 2:  # Phase 2: Right movement
            vx, vy, omega = 0, -WALL_AVOID_SPEED, 0
        elif cycle_time < direction_time * 3:  # Phase 3: Backward movement
            vx, vy, omega = -WALL_AVOID_SPEED, 0, 0
        else:  # Phase 4: Left movement
            vx, vy, omega = 0, WALL_AVOID_SPEED, 0

        # RECOVERY COMPLETION: Exit after full cycle
        if state_timer > direction_time * 4:
            current_state = "EXPLORING"
            state_timer = 0
            stuck_timer = 0

    # STATE 2: AVOIDING_WALL - Second priority state
    # Activated when IR sensors detect a wall/obstacle
    elif wall_detected:
        if current_state != "AVOIDING_WALL":
            print("Wall detected! Avoiding...")
            current_state = "AVOIDING_WALL"
            state_timer = 0

        # Combined escape vector with rotation
        vx = -WALL_AVOID_SPEED      # Backward movement (primary escape direction)
        vy = WALL_AVOID_SPEED * 0.7 # Sideways drift (secondary escape component)
        omega = 2.5 # Rotational component (orientation change)

        # AVOIDANCE COMPLETION: Exit when wall no longer detected and sufficient time elapsed
        if state_timer > 60 and not wall_detected:
            current_state = "EXPLORING"  # Return to exploration
            state_timer = 0 # Reset state timer

    # STATE 3: APPROACHING - Third priority state
    # Activated when LiDAR detects a potential target object
    elif object_detected:
        if current_state != "APPROACHING":
            print(f"Object detected at {object_distance:.2f}m, direction {object_direction}")
            current_state = "APPROACHING"
            state_timer = 0

        # Direction-based approach vectors
        if object_direction == -1:  # Target to the right
            vx, vy = APPROACH_SPEED * 0.8, -APPROACH_SPEED * 0.4 # Forward-right vector
        elif object_direction == 1:  # Target to the left
            vx, vy = APPROACH_SPEED * 0.8, APPROACH_SPEED * 0.4 # Forward-left vector
        else:  # Target straight ahead
            vx, vy = APPROACH_SPEED, 0 # Pure forward movement
        omega = 0

        # APPROACH COMPLETION: Mission progress tracking and state transition
        if state_timer > 80:
            objects_hit += 1
            print(f"Object hit! Total: {objects_hit}")
            current_state = "EXPLORING"
            state_timer = 0

    # STATE 4: EXPLORING - Lowest priority state
    else:
        if current_state != "EXPLORING":
            current_state = "EXPLORING"
            state_timer = 0

        vx = BASE_SPEED * (0.7 + 0.3 * random.random())  # 70-100% forward speed, random

        vy = BASE_SPEED * (random.random() - 0.5) * 0.5  # ±25% lateral speed, random sideways movement

        omega = (random.random() - 0.5) * 1.0            # random ±0.5 rad/s rotation for exploration

    base_apply_speeds(vx, vy, omega, wheels)  # Apply motion commands to motors

    cv.waitKey(1)
print("Mission completed!")
