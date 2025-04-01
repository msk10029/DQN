import numpy as np
import torch
import cv2
from config import *
import carla

import carla

from collections import deque

class FrameStack:
    """Maintains a stack of the last `STACK_SIZE` frames."""
    
    def __init__(self):
        self.frames = deque(maxlen=STACK_SIZE)  # Stores last `STACK_SIZE` frames

    def push(self, frame):
        """Adds a new frame and ensures the stack has `STACK_SIZE` frames."""
        self.frames.append(frame)

    def get_stacked_state(self): 
        """Returns stacked frames as a NumPy array (shape: STACK_SIZE x H x W)."""
        frames = list(self.frames)  # Copy frames to avoid modifying original deque
        
        # Fill missing frames with the first available frame
        while len(frames) < STACK_SIZE:
            frames.insert(0, frames[0])  # Prepend first frame to maintain correct order

        return np.stack(frames, axis=0)

    def reset(self):
        """Clears the frame stack (for new episodes)."""
        self.frames.clear()


def compute_reward(vehicle, lane_invasion, collision, action_idx):
    """
    Computes reward based on:
    - Lane invasion
    - Collision
    - Speed control
    - Distance to centerline
    - Junction handling (Updating centerline after junction)
    """

    reward = 0
    done = False  # Track if episode should end

    # Get vehicle position & nearest waypoint
    vehicle_location = vehicle.get_transform().location
    waypoint = vehicle.get_world().get_map().get_waypoint(
        vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving
    )

    velocity = vehicle.get_velocity()
    speed = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5  # Speed in m/s

    # Encourage the vehicle to maintain an optimal speed (5-15 m/s)
    if 5 <= speed <= 10:
        reward += 5  # Reward for driving at a good speed
    elif speed < 5:
        reward -= 3  # Penalize for moving too slow
    elif speed > 15:
        reward -= 4  # Penalize for driving too fast

    # Check if the vehicle is at a junction
    if waypoint.is_junction:
        # print("Vehicle is at a junction! Encouraging correct movement.")

        # Encourage movement through the junction
        if speed > 2.0:
            reward += 3  # Reward for moving through the junction

        if action_idx not in [2]:
            reward = -5

        # # If the vehicle is aligned with the lane, "straight" action is best
        # next_waypoints = waypoint.next(5.0)  # Get the next road direction
        # if len(next_waypoints) > 0:
        #     next_wp = next_waypoints[0]
        #     road_direction = next_wp.transform.get_forward_vector()
        #     vehicle_direction = vehicle.get_transform().get_forward_vector()
            
        #     # Compute alignment with the road
        #     alignment = road_direction.dot(vehicle_direction)

        #     # If the agent is aligned, reinforce "straight" action
        #     if alignment > 0.9 and action_idx == 2:  # Nearly perfect alignment
        #         reward += 5
        #         print("Perfect alignment!")  
        #     elif alignment > 0.5 and action_idx in [1, 2, 3]:  # Slightly misaligned
        #         reward += 2 
        #         print("Slight misaligned!") 
        #     elif alignment < 0.1:
        #         reward = -15
        #         done = True
        #         print("Misaligned completely. Ending Episode!!!")
        #         return reward, done, lane_invasion  
                
        #     else:  
        #         reward -= 5  # Penalize for incorrect movement at junction
                

    else:
        # Detect when the vehicle has exited a junction and update centerline
        if hasattr(vehicle, "was_in_junction") and vehicle.was_in_junction:
            # print("Vehicle has exited the junction! Updating centerline.")
            waypoint = vehicle.get_world().get_map().get_waypoint(
                vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving
            )

        vehicle.was_in_junction = False  # Reset flag

        # Compute perpendicular distance to lane centerline
        distance_to_center = vehicle_location.distance(waypoint.transform.location)
        # print (distance_to_center)

        # Reward staying close to the lane center
        if distance_to_center < 0.5:
            reward += 5  # Perfect lane following 
        elif distance_to_center < 1.0:
            reward += 2  # Slightly off-center
        else:
            reward -= 7  # Too far from center

        # End episode if vehicle is completely off the lane (e.g., >2m away)
        if distance_to_center > 1.4:
            # print("Vehicle is too far from the lane! Ending episode.")
            reward -= 20  # Harsh penalty for leaving the lane
            done = True  # End the episode
            return reward, done, lane_invasion
        
    #  Penalize lane invasion
    if lane_invasion:
        # print("Lane invasion detected! Applying penalty.")
        reward -= 8
        lane_invasion = False

    # Penalize collisions strongly and end episode
    if collision:
        reward -= 20
        done = True  
        return reward, done, lane_invasion  # Stop here if collision occurs

    return reward, done, lane_invasion



def apply_action(vehicle, action_idx):
    """
    Applies the selected action to the vehicle.
    """
    action = ACTION_SPACE[action_idx]
    control = carla.VehicleControl()
    control.steer = action['steer']
    control.throttle = action['throttle']
    control.brake = action['brake']
    vehicle.apply_control(control)

def preprocess_image(image):
    """
    Converts RGB segmentation image to a grayscale binary image (for DQN input).
    """
    img = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, 2]

    # Create blank mask
    mask = np.zeros((image.height, image.width), dtype=np.uint8)

    # Highlight lanes (tag 24) and roads (tag 1)
    mask[img == 24] = 255  # Lanes (White)
    mask[img == 1] = 128   # Roads (Gray)

    # Resize for DQN input
    processed_frame = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_NEAREST)

    return processed_frame

def normalize_state(state):
    """
    Normalizes the lane segmentation image (0 to 1 range).
    """
    return state / 255.0

def format_state_for_dqn(state):
    """
    Converts a NumPy array to a PyTorch tensor and adds necessary dimensions.
    """
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,64,64)
    return state
