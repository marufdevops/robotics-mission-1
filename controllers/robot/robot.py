import time
import math
import random

import numpy as np
import cv2 as cv

from controller import Robot

robot = Robot()

timestep = int(robot.getBasicTimeStep())


w0_motor = robot.getDevice("wheel0_joint")
w1_motor = robot.getDevice("wheel1_joint")
w2_motor = robot.getDevice("wheel2_joint")
wheels = [w0_motor, w1_motor, w2_motor]

ir_sensors_names = [ "ir1", "ir2", "ir3", "ir4", "ir5", "ir6", "ir7", "ir8", "ir9"]
ir_sensors = [ robot.getDevice(f"{s}") for s in ir_sensors_names ]
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

for motor in wheels:
    motor.setPosition(math.inf)
    motor.setVelocity(0)

WHEEL_RADIUS = 0.063                     # [m]
DISTANCE_WHEEL_TO_ROBOT_CENTRE = 0.1826  # [m]
MAX_SPEED = 8                            # [m/s]


def base_apply_speeds(vx, vy, omega, wheels):
  vx /= WHEEL_RADIUS
  vy /= WHEEL_RADIUS
  omega *= DISTANCE_WHEEL_TO_ROBOT_CENTRE / WHEEL_RADIUS
  wheels[0].setVelocity(vy - omega)
  wheels[1].setVelocity(-math.sqrt(0.75) * vx - 0.5 * vy - omega)
  wheels[2].setVelocity( math.sqrt(0.75) * vx - 0.5 * vy - omega)

def ir_volt_to_meter(V):
    return 0.1594 * math.pow(V, -0.8533) - 0.02916;


while (robot.step(timestep) != -1):

    vx = random.random()-0.5
    vy = random.random()-0.5

    timesteps = 0
    print(vx, vy)
    while timesteps < 5_000/timestep:
        timesteps += 1

        # LiDAR data
        lidar_data = np.array(lidar.getRangeImage()).reshape((1,256))
        cv.imshow('lidar', cv.resize(lidar_data, [600, 40]))
        # IR sensors
        ir_sensors_values = [ ir_volt_to_meter(irs.getValue()) for irs in ir_sensors]
        # Camera
        camera_image = np.frombuffer(camera.getImage(), dtype=np.uint8).reshape(128,128,4)
        print(type(camera_image))
        print(camera_image.shape)
        cv.imshow('camera', cv.resize(camera_image, [256,256]))
        
        base_apply_speeds(vx, vy, 0., wheels)

        cv.waitKey(1)
        robot.step(timestep)
    print("next!")
        


