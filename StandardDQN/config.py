CARLA_PATH = r"D:\Mohsin\ISAE Supaero\DQL Self Driving Cars\CARLA\CARLA_0.9.15\WindowsNoEditor"
MAIN_PATH = r"D:\Mohsin\ISAE Supaero\DQL Self Driving Cars\CARLA codes\Lane_Following"
# ==============
# General Config
# ==============
NUM_EPISODES = 1000 # Total episodes for training
MAX_EPISODE_STEPS = 600  # Max steps before an episode ends

# ==============
# DQN Hyperparameters
# ==============
LEARNING_RATE = 0.00003  # Learning rate for optimizer
GAMMA = 0.98  # Discount factor for future rewards
EPSILON = 1.0 # Initial exploration rate
EPSILON_DECAY = 0.997  # Decay factor for epsilon
MIN_EPSILON = 0.1  # Minimum epsilon value for exploration

# ==============
# Experience Replay
# ==============
MEMORY_SIZE = 100000  # Max number of experiences stored
BATCH_SIZE = 64 # Number of experiences per training step


TARGET_UPDATE_FREQ = 500

STACK_SIZE = 4

# ==============
# Action Space
# ==============
# ACTION_SPACE = [
#     {'steer': -0.3, 'throttle': 0.5, 'brake': 0.0},  # Turn Left
#     {'steer': -0.1, 'throttle': 0.5, 'brake': 0.0},  # Slight Left
#     {'steer':  0.0, 'throttle': 0.5, 'brake': 0.0},  # Straight
#     {'steer':  0.1, 'throttle': 0.5, 'brake': 0.0},  # Slight Right
#     {'steer':  0.3, 'throttle': 0.5, 'brake': 0.0},  # Turn Right
#     {'steer':  0.0, 'throttle': 0.0, 'brake': 0.5},  # Full Stop/Brake
# ]

ACTION_SPACE = [
    {'steer': -0.3, 'throttle': 0.5, 'brake': 0.0},  # Sharp Left
    {'steer': -0.1, 'throttle': 0.5, 'brake': 0.0},  # Slight Left
    {'steer':  0.0, 'throttle': 0.5, 'brake': 0.0},  # Straight Fast
    {'steer':  0.1, 'throttle': 0.5, 'brake': 0.0},  # Slight Right
    {'steer':  0.3, 'throttle': 0.5, 'brake': 0.0},  # Sharp Right
    {'steer':  0.0, 'throttle': 0.3, 'brake': 0.0},  # Slow Forward 
    {'steer':  0.0, 'throttle': 0.0, 'brake': 0.5},  # Full Stop
]