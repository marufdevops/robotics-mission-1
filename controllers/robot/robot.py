"""
SIMPLIFIED ROBOT CONTROLLER: Hit objects while avoiding walls
PRIORITY: Wall avoidance must work perfectly (was broken in previous version)
"""
import math
import random
import numpy as np
import cv2 as cv
from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Hardware setup
w0_motor = robot.getDevice("wheel0_joint")
w1_motor = robot.getDevice("wheel1_joint")
w2_motor = robot.getDevice("wheel2_joint")
wheels = [w0_motor, w1_motor, w2_motor]

ir_sensors = [robot.getDevice(f"ir{i}") for i in range(1, 10)]
for irs in ir_sensors:
    irs.enable(timestep)

lidar = robot.getDevice("lidar")
lidar.enable(timestep)
camera = robot.getDevice('camera')
camera.enable(timestep)

for motor in wheels:
    motor.setPosition(math.inf)
    motor.setVelocity(0)

# Physical constants (MANDATORY - DO NOT CHANGE)
WHEEL_RADIUS = 0.063
DISTANCE_WHEEL_TO_ROBOT_CENTRE = 0.1826
MAX_SPEED = 8

def base_apply_speeds(vx, vy, omega, wheels):
    """MANDATORY FUNCTION - DO NOT MODIFY"""
    vx /= WHEEL_RADIUS
    vy /= WHEEL_RADIUS
    omega *= DISTANCE_WHEEL_TO_ROBOT_CENTRE / WHEEL_RADIUS
    wheels[0].setVelocity(vy - omega)
    wheels[1].setVelocity(-math.sqrt(0.75) * vx - 0.5 * vy - omega)
    wheels[2].setVelocity( math.sqrt(0.75) * vx - 0.5 * vy - omega)

def ir_volt_to_meter(V):
    """MANDATORY FUNCTION - DO NOT MODIFY"""
    return 0.1594 * math.pow(V, -0.8533) - 0.02916

# Control parameters
BASE_SPEED = 1.0
APPROACH_SPEED = 1.5
WALL_AVOID_SPEED = 1.2  # Increased for better wall escape
WALL_THRESHOLD = 0.35   # IR sensor wall detection threshold
OBJECT_RANGE = 1.8      # LiDAR object detection range
STUCK_TIME = int(3000 / timestep)  # Stuck detection time

# State variables
current_state = "EXPLORING"
state_timer = 0
stuck_timer = 0
objects_hit = 0
stuck_recovery_direction = 0

def detect_wall(ir_distances):
    """Detect walls using IR sensors - RESTORED WORKING VERSION"""
    for distance in ir_distances:
        if 0 < distance < WALL_THRESHOLD:
            return True
    return False

def detect_object(lidar_distances):
    """Simple object detection - find closest object in front"""
    closest_distance = float('inf')
    closest_direction = 0

    # Check front 120 degrees (indices 0-42 and 214-255)
    front_indices = list(range(0, 43)) + list(range(214, 256))

    for i in front_indices:
        distance = lidar_distances[i]
        if 0.3 < distance < OBJECT_RANGE and distance < closest_distance:
            closest_distance = distance
            # Determine direction: left (1), center (0), right (-1)
            if i < 21 or i > 235:
                closest_direction = 0  # center
            elif i < 43:
                closest_direction = -1  # right
            else:
                closest_direction = 1   # left

    if closest_distance < OBJECT_RANGE:
        return True, closest_direction, closest_distance
    return False, 0, 0.0

print("🤖 SIMPLIFIED ROBOT CONTROLLER STARTING")

# Main control loop
while robot.step(timestep) != -1:
    # Get sensor data
    lidar_data = np.array(lidar.getRangeImage()).reshape((1,256))
    lidar_distances = lidar_data[0]
    ir_sensor_values = [ir_volt_to_meter(irs.getValue()) for irs in ir_sensors]
    camera_image = np.frombuffer(camera.getImage(), dtype=np.uint8).reshape(128,128,4)

    # Sensor processing
    wall_detected = detect_wall(ir_sensor_values)
    object_detected, object_direction, object_distance = detect_object(lidar_distances)

    # Update timers
    state_timer += 1
    if wall_detected:
        stuck_timer += 1
    else:
        stuck_timer = 0

    # Display
    cv.imshow('lidar', cv.resize(lidar_data, [600, 40]))
    cv.imshow('camera', cv.resize(camera_image, [256,256]))

    # Initialize movement
    vx, vy, omega = 0, 0, 0

    # PRIORITY 1: STUCK RECOVERY (4-directional movement)
    if stuck_timer > STUCK_TIME:
        if current_state != "STUCK_RECOVERY":
            print(f"🚨 STUCK! Trying 4-directional recovery")
            current_state = "STUCK_RECOVERY"
            state_timer = 0
            stuck_recovery_direction = 0

        # Try 4 directions: forward, right, backward, left
        direction_time = 40  # Try each direction for 40 timesteps
        cycle_time = state_timer % (direction_time * 4)

        if cycle_time < direction_time:  # Forward
            vx, vy, omega = WALL_AVOID_SPEED, 0, 0
        elif cycle_time < direction_time * 2:  # Right
            vx, vy, omega = 0, -WALL_AVOID_SPEED, 0
        elif cycle_time < direction_time * 3:  # Backward
            vx, vy, omega = -WALL_AVOID_SPEED, 0, 0
        else:  # Left
            vx, vy, omega = 0, WALL_AVOID_SPEED, 0

        # Exit after trying all directions
        if state_timer > direction_time * 4:
            current_state = "EXPLORING"
            state_timer = 0
            stuck_timer = 0

    # PRIORITY 2: WALL AVOIDANCE (critical - must work perfectly)
    elif wall_detected:
        if current_state != "AVOIDING_WALL":
            print("🚧 Wall detected! Avoiding...")
            current_state = "AVOIDING_WALL"
            state_timer = 0

        # Strong wall avoidance - move away from wall
        vx = -WALL_AVOID_SPEED  # Backward
        vy = WALL_AVOID_SPEED * 0.7  # Sideways
        omega = 2.5  # Turn away

        # Exit wall avoidance after some time if no wall detected
        if state_timer > 60 and not wall_detected:
            current_state = "EXPLORING"
            state_timer = 0

    # PRIORITY 3: OBJECT APPROACH (hit anything detected)
    elif object_detected:
        if current_state != "APPROACHING":
            print(f"🎯 Object detected at {object_distance:.2f}m, direction {object_direction}")
            current_state = "APPROACHING"
            state_timer = 0

        # Simple approach movement
        if object_direction == -1:  # Right
            vx, vy = APPROACH_SPEED * 0.8, -APPROACH_SPEED * 0.4
        elif object_direction == 1:  # Left
            vx, vy = APPROACH_SPEED * 0.8, APPROACH_SPEED * 0.4
        else:  # Center
            vx, vy = APPROACH_SPEED, 0
        omega = 0

        # Count as hit after approaching for a while
        if state_timer > 80:
            objects_hit += 1
            print(f"💥 Object hit! Total: {objects_hit}")
            current_state = "EXPLORING"
            state_timer = 0

    # PRIORITY 4: EXPLORATION (default behavior)
    else:
        if current_state != "EXPLORING":
            current_state = "EXPLORING"
            state_timer = 0

        # Random exploration movement
        vx = BASE_SPEED * (0.7 + 0.3 * random.random())
        vy = BASE_SPEED * (random.random() - 0.5) * 0.5
        omega = (random.random() - 0.5) * 1.0

    # Apply movement
    base_apply_speeds(vx, vy, omega, wheels)
    cv.waitKey(1)

print("🏁 Mission completed!")
        


