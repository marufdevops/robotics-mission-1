"""
SIMPLE PIN HUNTER - Enhanced Original Template
==============================================

Based on original working template with minimal improvements:
- Faster movement speeds for quicker pin elimination
- Simple obstacle avoidance to prevent getting stuck
- Basic pin detection using proximity
- Reliable movement without complex logic

Keep it simple, keep it working!
"""

import math
import random
import numpy as np
import cv2 as cv
from controller import Robot

# Initialize robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Motors
w0_motor = robot.getDevice("wheel0_joint")
w1_motor = robot.getDevice("wheel1_joint")
w2_motor = robot.getDevice("wheel2_joint")
wheels = [w0_motor, w1_motor, w2_motor]

# Sensors
ir_sensors_names = ["ir1", "ir2", "ir3", "ir4", "ir5", "ir6", "ir7", "ir8", "ir9"]
ir_sensors = [robot.getDevice(f"{s}") for s in ir_sensors_names]
for irs in ir_sensors:
    irs.enable(timestep)

lidar = robot.getDevice("lidar")
lidar.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)
gps = robot.getDevice('gps')
gps.enable(timestep)
camera = robot.getDevice('camera')
camera.enable(timestep)

# Initialize motors
for motor in wheels:
    motor.setPosition(math.inf)
    motor.setVelocity(0)

# Robot parameters
WHEEL_RADIUS = 0.063
DISTANCE_WHEEL_TO_ROBOT_CENTRE = 0.1826
MAX_SPEED = 8

# Enhanced parameters for faster pin hunting
BASE_SPEED = 2.0  # Increased from original slow speed
WALL_THRESHOLD = 0.2  # Simple wall detection threshold

print("🎯 ENHANCED SIMPLE PIN HUNTER STARTED!")
print("Fast and reliable pin elimination")

# Simple utility functions
def apply_speeds(vx, vy, omega, wheels):
    """Apply velocities to robot wheels - original template function"""
    vx /= WHEEL_RADIUS
    vy /= WHEEL_RADIUS
    omega *= DISTANCE_WHEEL_TO_ROBOT_CENTRE / WHEEL_RADIUS
    wheels[0].setVelocity(vy - omega)
    wheels[1].setVelocity(-math.sqrt(0.75) * vx - 0.5 * vy - omega)
    wheels[2].setVelocity( math.sqrt(0.75) * vx - 0.5 * vy - omega)

def ir_volt_to_meter(V):
    """Convert IR voltage to distance - original template function"""
    return 0.1594 * math.pow(V, -0.8533) - 0.02916

# Simple pin detection using LiDAR
def detect_pin(lidar_data):
    """Simple pin detection - look for close objects in front"""
    # Look at front area of LiDAR
    front_indices = list(range(0, 32)) + list(range(224, 256))

    closest_distance = float('inf')
    closest_index = 0

    for i in front_indices:
        if 0.2 < lidar_data[i] < 1.5:  # Pin detection range
            if lidar_data[i] < closest_distance:
                closest_distance = lidar_data[i]
                closest_index = i

    if closest_distance < 1.5:
        # Determine direction
        if closest_index > 128:
            direction = 1  # Left
        else:
            direction = -1  # Right
        return True, direction

    return False, 0

# Simple wall detection
def check_wall(ir_values):
    """Simple wall detection using IR sensors"""
    return any(d < WALL_THRESHOLD for d in ir_values if d > 0)

# Main control loop - simple and reliable
print("Starting simple pin hunting mission...")
pins_hit = 0

# Simple state tracking
wall_avoid_timer = 0
pin_approach_timer = 0

# Stuck detection and recovery
stuck_timer = 0
STUCK_THRESHOLD = int(3000 / timestep)  # 3 seconds worth of timesteps
recovery_timer = 0
in_recovery = False
recovery_rotation = 0

while robot.step(timestep) != -1:
    # Get sensor data
    lidar_data = np.array(lidar.getRangeImage()).reshape((1, 256))
    ir_values = [ir_volt_to_meter(irs.getValue()) for irs in ir_sensors]
    camera_image = np.frombuffer(camera.getImage(), dtype=np.uint8).reshape(128, 128, 4)
    gps_position = gps.getValues()

    # Simple sensor visualization
    cv.imshow('LiDAR', cv.resize(lidar_data, [600, 40]))
    cv.imshow('Camera', cv.resize(camera_image, [256, 256]))

    # Status display
    status_img = np.zeros((80, 400, 3), dtype=np.uint8)
    cv.putText(status_img, f'Pins Hit: {pins_hit}', (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv.putText(status_img, f'Position: ({gps_position[0]:.1f}, {gps_position[1]:.1f})',
               (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.imshow('Status', status_img)
    cv.waitKey(1)

    # Simple behavior logic
    vx, vy, omega = 0, 0, 0

    # Check for walls
    wall_detected = check_wall(ir_values)

    # Check for pins
    pin_detected, pin_direction = detect_pin(lidar_data[0])

    # STUCK DETECTION AND RECOVERY MECHANISM
    if in_recovery:
        # Execute recovery sequence
        recovery_timer += 1

        if recovery_timer <= 30:  # First 30 timesteps: rotate
            print(f"🔄 Recovery rotation... ({recovery_timer}/30)")
            vx, vy = 0, 0  # Stop linear movement
            omega = recovery_rotation  # Rotate to escape

        elif recovery_timer <= 50:  # Next 20 timesteps: move forward
            print(f"➡️ Recovery forward movement... ({recovery_timer-30}/20)")
            vx = BASE_SPEED * 0.8  # Move forward to escape
            vy, omega = 0, 0

        else:  # Recovery complete
            print("✅ Recovery sequence completed - resuming normal behavior")
            in_recovery = False
            recovery_timer = 0
            stuck_timer = 0  # Reset stuck timer

    elif wall_detected:
        # Increment stuck timer when avoiding walls
        stuck_timer += 1

        # Check if we've been stuck too long
        if stuck_timer >= STUCK_THRESHOLD:
            print("🚨 STUCK DETECTED! Starting recovery sequence...")
            in_recovery = True
            recovery_timer = 0
            # Generate random rotation direction and magnitude (45-90 degrees)
            rotation_angle = random.uniform(45, 90) * (math.pi / 180)  # Convert to radians
            recovery_rotation = rotation_angle * random.choice([-1, 1])  # Random direction
            print(f"Recovery rotation: {recovery_rotation:.2f} rad")
        else:
            # Normal wall avoidance
            print(f"Wall detected - avoiding (stuck timer: {stuck_timer}/{STUCK_THRESHOLD})")
            vx = -BASE_SPEED * 0.5  # Back up
            vy = BASE_SPEED * 0.3   # Move sideways
            omega = 1.0             # Turn
            wall_avoid_timer += 1

            # Count pins hit when backing away from walls (simple proximity detection)
            if wall_avoid_timer > 20:  # After backing up for a bit
                # Check if we're close to where a pin might have been
                close_objects = sum(1 for d in lidar_data[0] if 0.1 < d < 0.5)
                if close_objects < 2:  # Fewer objects nearby, might have hit something
                    pins_hit += 1
                    print(f"Pin hit detected! Total: {pins_hit}")
                wall_avoid_timer = 0

    elif pin_detected:
        # Reset stuck timer when successfully detecting and approaching pins
        stuck_timer = 0

        # Move toward detected pin
        print(f"Pin detected - approaching (direction: {pin_direction})")
        vx = BASE_SPEED * 0.8  # Move forward
        if pin_direction > 0:
            vy = BASE_SPEED * 0.3  # Move left
        else:
            vy = -BASE_SPEED * 0.3  # Move right
        omega = 0
        pin_approach_timer += 1

        # Simple pin hit detection after approaching
        if pin_approach_timer > 30:
            pins_hit += 1
            print(f"Pin approach completed! Total: {pins_hit}")
            pin_approach_timer = 0

    else:
        # Reset stuck timer when freely exploring (no walls detected)
        stuck_timer = 0

        # Explore - enhanced random movement
        print("Exploring...")
        vx = BASE_SPEED * (0.7 + random.random() * 0.3)  # Forward movement
        vy = BASE_SPEED * (random.random() - 0.5) * 0.5  # Random sideways
        omega = (random.random() - 0.5) * 0.8            # Random turning

    # Apply movement
    apply_speeds(vx, vy, omega, wheels)

    # Status update
    if robot.getTime() % 2.0 < 0.1:
        print(f"Status: {pins_hit} pins hit, Position: ({gps_position[0]:.1f}, {gps_position[1]:.1f})")

print(f"Mission completed! Total pins hit: {pins_hit}")