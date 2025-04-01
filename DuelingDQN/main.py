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
import time

# Initialize live plotting variables
window_size = 50  # Moving average window size
rewards_history = []
losses_history = []
q_value_history = []
moving_avg_rewards = deque(maxlen=window_size)
moving_avg_losses = deque(maxlen=window_size)

# Function to update live plot
def update_plot():
    plt.clf()
    plt.subplot(3, 1, 1)  # Reward plot
    plt.plot(rewards_history, alpha=0.3, label="Episode Reward")
    plt.plot(pd.Series(rewards_history).rolling(window_size).mean(), 'r', label=f"Moving Average ({window_size} episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    
    plt.subplot(3, 1, 2)  # Loss plot
    plt.plot(losses_history, alpha=0.3, label="Episode Loss")
    plt.plot(pd.Series(losses_history).rolling(window_size).mean(), 'r', label=f"Moving Average ({window_size} episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(3, 1, 3)  # Loss plot
    plt.plot(q_value_history, alpha=0.3, label="Q Values")
    plt.plot(pd.Series(q_value_history).rolling(window_size).mean(), 'r', label=f"Moving Average ({window_size} episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Q-values")
    plt.legend()
    
    plt.pause(0.01)  # Pause to update the plot

#-----------------------------------------------------------------------
# Main Loop
#-----------------------------------------------------------------------

# Initialize environment and agent
env = CarlaEnv()
env.no_render()
agent = DQNAgent()
logger = MetricsLogger()
frame_stack = FrameStack()
reward_tracker = RunningMeanStd()

total_training_time = 0
# start_episode = load_checkpoint(agent)

# print('Loading Checkpoint!!')
# episode = load_checkpoint(agent)
# print('Checkpoint Loaded!!')

for episode in range(1, NUM_EPISODES + 1):
    start_time = time.time()
    print(f"Starting Episode {episode}/{NUM_EPISODES}")
    total_reward = 0
    total_reward_2 = 0
    total_loss = 0
    steps = 0

    state = env.reset(episode)  # Respawn vehicle & sensor
    done = False
    frame_stack.reset()
    frame_stack.push(state)
    stacked_state = frame_stack.get_stacked_state()
    steps = 0

    episode_q_values = []

    while not done:
        if (steps >= MAX_EPISODE_STEPS):
            break
        else:
            action_idx = agent.select_action(stacked_state)
            try:
                next_state, reward, done = env.step(action_idx)
                # reward_tracker.update(reward)
                # reward = reward_tracker.normalize(reward)
                reward_2 = reward
                reward = np.sign(reward) * np.log1p(abs(reward))
            except RuntimeError as e:
                print(f"RuntimeError in step:{e}")
                env.cleanup()
                raise e
            frame_stack.push(next_state)  # Push new frame into stack
            stacked_next_state = frame_stack.get_stacked_state()

            agent.store_experience(stacked_state, action_idx, reward, stacked_next_state, done)
            loss, avg_q_value = agent.train()
            episode_q_values.append(avg_q_value)

            stacked_state = stacked_next_state  # Move to the next stacked state
            total_reward += reward
            total_reward_2 += reward_2
            total_loss += loss

            steps+=1

    end_time = time.time()
    episode_time = end_time - start_time
    total_training_time += episode_time
    avg_loss = total_loss / max(steps, 1)
    if len(episode_q_values) > 0:
        logger.log_metrics(episode, total_reward, avg_loss, agent.epsilon, np.mean(episode_q_values), total_training_time)
    else:
        logger.log_metrics(episode, total_reward, avg_loss, agent.epsilon, 0, total_training_time)
    agent.epsilon = max(MIN_EPSILON, agent.epsilon * EPSILON_DECAY)

    print(f"Episode {episode}: Reward = {total_reward}, Reward_Raw = {total_reward_2}, Loss = {avg_loss}, Epsilon = {agent.epsilon}, Steps Taken = {steps}")

    rewards_history.append(total_reward)
    losses_history.append(avg_loss)
    if len(episode_q_values) > 0:
        q_value_history.append(np.mean(episode_q_values))
    

    update_plot()

    if episode % 250 == 0:
        save_checkpoint(agent, episode)
        print("Successfully Saved!!")

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