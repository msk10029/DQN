import carla
import numpy as np
import random
import cv2
import time
from config import *
from utils import compute_reward, apply_action, preprocess_image
# import pygame

class CarlaEnv:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(400.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle = None
        self.lane_sensor = None
        self.rgb_frame = None
        self.collision_sensor = None
        self.latest_lane_frame = np.zeros((64, 64), dtype=np.uint8)
        self.lane_invasion_flag = False
        self.collision_flag = False

        # Initialize Pygame for displaying two camera feeds side-by-side
        # pygame.init()
        # self.WINDOW_WIDTH = 640
        # self.WINDOW_HEIGHT = 640
        # self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        # pygame.display.set_caption("Chase Camera (Left) + Segmentation (Right)")

        # Global frames for chase camera & segmentation camera
        self.rgb_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        self.seg_frame = np.zeros((320, 320, 3), dtype=np.uint8)

    def no_render(self):
        settings = self.world.get_settings()
        settings.no_rendering_mode = True  # Disable rendering
        self.world.apply_settings(settings)

    def render(self):
        settings = self.world.get_settings()
        settings.no_rendering_mode = False # rendering
        self.world.apply_settings(settings)

    

    def reset(self):
        """Respawns vehicle and sensors at a new location."""
        if self.vehicle:
            self.cleanup()
        
        # Spawn vehicle
        vehicle_bp = self.blueprint_library.filter("model3")[0]
        spawn_points = self.world.get_map().get_spawn_points()[1]
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points)

        if self.vehicle is None:
            print("Failed to spawn vehicle! Retrying...")
            return self.reset()  # Try again
        # print("Out")
        # Attach lane invasion sensor
        self.lane_sensor = self.add_lane_invasion_sensor(self.vehicle)
        
        # Attach RGB Camera
        self.attach_chase_camera(self.vehicle)
        
        # Attach Segmentation Camera
        self.attach_segmentation_camera(self.vehicle)

        self.collision_sensor = self.attach_collision_sensor()
        
        self.lane_invasion_flag = False
        self.collision_flag = False
        time.sleep(1)  # Let sensors initialize
        return self.latest_lane_frame  # Initial state for DQN
    
    # def cleanup(self):
    #     """Destroys all CARLA actors."""
    #     if self.vehicle:
    #         self.vehicle.destroy()
    #     if self.lane_sensor:
    #         self.lane_sensor.destroy()
    def attach_collision_sensor(self):
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle
        )
        collision_sensor.listen(lambda event: self.process_collision(event))
        return collision_sensor

    def process_collision(self, event):
        self.collision_flag = True
        print(f"Collision detected! Impulse: {event.normal_impulse}")
        # self.collision_hist.append(event)

    def add_lane_invasion_sensor(self, vehicle):
        """Attaches a lane invasion sensor."""
        sensor_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        sensor = self.world.spawn_actor(sensor_bp, carla.Transform(), attach_to=vehicle)
        sensor.listen(lambda event: self.lane_invasion_callback(event))
        return sensor

    def lane_invasion_callback(self, event):
        """Callback for lane invasion sensor."""
        # print("Lane invasion detected!")
        self.lane_invasion_flag = True

    def rgb_camera_callback(self, image):
            """Convert CARLA image to numpy array and update the global frame."""
            array = np.frombuffer(image.raw_data, dtype=np.uint8)  # Extract raw pixel data
            array = array.reshape((image.height, image.width, 4))  # Convert to (H, W, 4)
            self.rgb_frame = array[:, :, :3]  # Remove the alpha channel (RGBA -> RGB)

    def attach_chase_camera(self, vehicle):
        """Attaches a third-person RGB camera."""
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '640')
        camera_bp.set_attribute('fov', '110')
        
        transform = carla.Transform(carla.Location(x=-6.0, z=3.0), carla.Rotation(pitch=-15))
        self.rgb_camera = self.world.spawn_actor(camera_bp, transform, attach_to=vehicle)
        # def rgb_camera_callback(image):
        #     """Convert CARLA image to numpy array and update the global frame."""
        #     array = np.frombuffer(image.raw_data, dtype=np.uint8)  # Extract raw pixel data
        #     array = array.reshape((image.height, image.width, 4))  # Convert to (H, W, 4)
        #     self.rgb_frame = array[:, :, :3]  # Remove the alpha channel (RGBA -> RGB)
        self.rgb_camera.listen(lambda image: self.rgb_camera_callback(image))

    def attach_segmentation_camera(self, vehicle):
        """Attaches a segmentation camera."""
        camera_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', '320')
        camera_bp.set_attribute('image_size_y', '320')
        
        transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.seg_camera = self.world.spawn_actor(camera_bp, transform, attach_to=vehicle)

        def process_segmentation(image):
            # img = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, 2]
            # mask = np.zeros((image.height, image.width,30),dtype=np.uint8)
            # mask[img == 24] = (157, 234, 50)
            # mask[img == 1] = (128, 64, 128)
            #self.latest_lane_frame = cv2.resize(mask, (64, 64))
            self.latest_lane_frame = preprocess_image(image)
            # self.seg_frame = mask

        self.seg_camera.listen(process_segmentation)
        

    def cleanup(self):
        """Destroy vehicle and all attached sensors."""
        if self.vehicle:
            self.vehicle.destroy()
        
        # Destroy all sensors (RGB, Chase Camera, Segmentation, etc.)
        for sensor in [self.collision_sensor, self.seg_camera, self.rgb_camera, self.lane_sensor]:  
            if sensor is not None:
                sensor.destroy()
        
        # Set all sensor references to None to prevent memory leaks
        self.collision_sensor = None
        self.lane_sensor = None
        self.seg_camera = None
        self.rgb_camera = None  

        print("Environment cleaned up successfully.")

    # def draw_debug_info_on_pygame(self, surface):
    #     """Draws lane centerline, vehicle position, and perpendicular distance on Pygame window."""
    #     vehicle_location = self.vehicle.get_transform().location
    #     world = self.vehicle.get_world()

    #     # Get nearest waypoint on the lane (NOT the road center)
    #     waypoint = world.get_map().get_waypoint(
    #         vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving
    #     )
        
    #     next_waypoints = waypoint.next(2.0)
    #     if len(next_waypoints) == 0:
    #         return

    #     next_waypoint = next_waypoints[0]

    #     # Convert to NumPy arrays
    #     p1 = np.array([waypoint.transform.location.x, waypoint.transform.location.y])
    #     p2 = np.array([next_waypoint.transform.location.x, next_waypoint.transform.location.y])
    #     p_vehicle = np.array([vehicle_location.x, vehicle_location.y])

    #     # Compute perpendicular projection point on the lane centerline
    #     t = np.dot(p_vehicle - p1, p2 - p1) / np.dot(p2 - p1, p2 - p1)
    #     projection = p1 + t * (p2 - p1)

    #     # Convert world coordinates to screen (Pygame) coordinates
    #     def world_to_screen(world_pos, screen_size=(800, 600)):
    #         x, y = int(world_pos[0] * 10) % screen_size[0], int(world_pos[1] * 10) % screen_size[1]
    #         return x, y

    #     vehicle_screen = world_to_screen(p_vehicle)
    #     waypoint_screen = world_to_screen(p1)
    #     projected_screen = world_to_screen(projection)

    #     # Draw debug info on the chase camera image
    #     pygame.draw.circle(surface, (255, 0, 0), vehicle_screen, 5)  # Red dot (Vehicle)
    #     pygame.draw.line(surface, (0, 255, 0), waypoint_screen, world_to_screen(p2), 2)  # Green line (Lane center)
    #     pygame.draw.line(surface, (0, 0, 255), vehicle_screen, projected_screen, 2)  # Blue line (Perpendicular distance)

    #     # Add text labels for debugging
    #     font = pygame.font.Font(None, 24)
    #     projected_location = carla.Location(x=projection[0], y=projection[1], z=vehicle_location.z)

    #     # Now this works correctly:
    #     distance_to_center = vehicle_location.distance(projected_location)
    #     text_surface = font.render(f"Distance to Centerline: {distance_to_center:.2f}m", True, (255, 255, 255))
    #     surface.blit(text_surface, (10, 10))  # Display text on top-left corner

    def step(self, action_idx):
        """Executes action, updates environment, returns next state, reward, and done flag."""
        apply_action(self.vehicle, action_idx)
        time.sleep(0.1)  # Allow environment to update

        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         raise KeyboardInterrupt


        # surface = pygame.surfarray.make_surface(np.transpose(self.rgb_frame, (1,0,2)))
        # #self.draw_debug_info_on_pygame(surface)
        # self.screen.blit(surface, (0, 0))
        # pygame.display.flip()
        
        done = False
        # if self.lane_invasion_flag or self.collision_flag:
        #     done = True  # End episode if lane is invaded

        # print(self.lane_invasion_flag)
        
        reward, done, self.lane_invasion_flag = compute_reward(self.vehicle, self.lane_invasion_flag, self.collision_flag, action_idx)
        return self.latest_lane_frame, reward, done