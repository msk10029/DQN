import torch
import numpy as np
from carla_env import CarlaEnv
from dqn_agent import DQN
from utils import FrameStack
import time

# ===========================
# Load Pre-Trained Model
# ===========================

CHECKPOINT_PATH = "./data/dqn_checkpoint_2.pth"  # Change if using a different file

# Load saved model checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print("Model in checkpoint")

# Initialize policy network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN().to(device)
policy_net.load_state_dict(checkpoint["model_state_dict"])
policy_net.eval()  # Set to evaluation mode (no training)

print("Model loaded successfully!")


env = CarlaEnv()
env.render()  # Enable rendering to see driving behavior
frame_stack = FrameStack()

state = env.reset()  # Get the initial state
frame_stack.reset()
frame_stack.push(state)
stacked_state = frame_stack.get_stacked_state()

done = False
total_reward = 0
steps = 0

print("Running trained agent...")

while not done:
    # Convert state to tensor
    state_tensor = torch.tensor(stacked_state, dtype=torch.float32, device=device).unsqueeze(0)  # Add batch dim

    # Select the best action using the trained model (no epsilon-greedy)
    with torch.no_grad():
        action_idx = policy_net(state_tensor).argmax(dim=1).item()  # Choose the action with max Q-value

    # Step the environment with the selected action
    next_state, reward, done = env.step(action_idx)

    # Update state
    frame_stack.push(next_state)
    stacked_state = frame_stack.get_stacked_state()

    total_reward += reward
    steps += 1

    print(f"Step: {steps}, Action: {action_idx}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

    time.sleep(0.05)  # Small delay for smooth rendering

env.cleanup()
print(f"Run complete! Total Reward: {total_reward:.2f}")
