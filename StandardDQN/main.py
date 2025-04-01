from carla_env import CarlaEnv
from dqn_agent import DQNAgent
from utils import *
import torch
import numpy as np
from metrics_logger import *
from save_load import save_checkpoint, load_checkpoint
import gc
# import os
# import time
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque

# Initialize live plotting variables
window_size = 50  # Moving average window size
rewards_history = []
losses_history = []
moving_avg_rewards = deque(maxlen=window_size)
moving_avg_losses = deque(maxlen=window_size)

# Function to update live plot
def update_plot():
    plt.clf()
    plt.subplot(2, 1, 1)  # Reward plot
    plt.plot(rewards_history, alpha=0.3, label="Episode Reward")
    plt.plot(pd.Series(rewards_history).rolling(window_size).mean(), 'r', label=f"Moving Average ({window_size} episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    
    plt.subplot(2, 1, 2)  # Loss plot
    plt.plot(losses_history, alpha=0.3, label="Episode Loss")
    plt.plot(pd.Series(losses_history).rolling(window_size).mean(), 'r', label=f"Moving Average ({window_size} episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.pause(0.01)  # Pause to update the plot

#-----------------------------------------------------------------------
# Main Loop
#-----------------------------------------------------------------------

# Initialize environment and agent
env = CarlaEnv()
env.no_render()
agent = DQNAgent()
logger = MetricsLogger("./data/carla_training_metrics_fix_9.csv")
frame_stack = FrameStack()


# start_episode = load_checkpoint(agent)

for episode in range(1, NUM_EPISODES + 1):
    print(f"Starting Episode {episode}/{NUM_EPISODES}")
    total_reward = 0
    total_loss = 0
    steps = 0

    state = env.reset()  # Respawn vehicle & sensor
    done = False
    frame_stack.reset()
    frame_stack.push(state)
    stacked_state = frame_stack.get_stacked_state()

    agent.check()

    while not done:
        action_idx = agent.select_action(stacked_state)
        try:
            next_state, reward, done = env.step(action_idx)
            reward = np.sign(reward) * np.log1p(abs(reward))
        except RuntimeError as e:
            print(f"RuntimeError in step:{e}")
            env.cleanup()
            raise e
        frame_stack.push(next_state)  # Push new frame into stack
        stacked_next_state = frame_stack.get_stacked_state()

        agent.store_experience(stacked_state, action_idx, reward, stacked_next_state, done)
        loss = agent.train()

        stacked_state = stacked_next_state  # Move to the next stacked state
        total_reward += reward
        total_loss += loss


    avg_loss = total_loss / max(steps, 1)
    logger.log_metrics(episode, total_reward, avg_loss, agent.epsilon)

    agent.epsilon = max(MIN_EPSILON, agent.epsilon * EPSILON_DECAY)

    print(f"Episode {episode}: Reward = {total_reward}, Loss = {avg_loss}, Epsilon = {agent.epsilon}")

    rewards_history.append(total_reward)
    losses_history.append(avg_loss)

    update_plot()

    # if episode % 500 == 0:
    #     save_checkpoint(agent, episode)

    if episode % 10 == 0:  # Every 10 episodes
        gc.collect()
        torch.cuda.empty_cache()

    # if episode % 600 == 0:
    #     env.cleanup()
    #     restart_carla()
    #     print("Initializing new Environment!!")
    #     env = CarlaEnv()

print("Training complete!")
plt.show()