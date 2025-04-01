import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from config import *
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Saves a transition with stacked frames."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Returns a batch of stacked states."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (np.array(states, dtype=np.float32), 
                np.array(actions), 
                np.array(rewards, dtype=np.float32), 
                np.array(next_states, dtype=np.float32), 
                np.array(dones))

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(STACK_SIZE, 32, 5, stride=2)  # Change from 1 → STACK_SIZE
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2)
        self.fc1 = nn.Linear(10816, 512)
        self.fc2 = nn.Linear(512, len(ACTION_SPACE))

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(capacity=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.steps_done = 0
    
    def check(self):
        print(self.device)

    def select_action(self, state):
        """Selects action using ε-greedy policy (supports frame stacking)."""
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # Add batch dimension
        if random.random() < self.epsilon:
            return random.randint(0, len(ACTION_SPACE) - 1)  # Explore
        with torch.no_grad():
            return self.policy_net(state).argmax(dim=1).item() 

    def store_experience(self, state, action, reward, next_state, done):
        """Stores experiences in replay buffer using stacked frames."""
        self.memory.push(state, action, reward, next_state, done)

    def train(self):
       
        if len(self.memory) < BATCH_SIZE:
            return 0  # Not enough experiences to train

        
        # batch = random.sample(self.memory, BATCH_SIZE)
        # states, actions, rewards, next_states, dones = zip(*batch)

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        
        # states, next_states should each be a list of (1, C, H, W) Tensors
        # We'll cat them along dim=0 => (BATCH_SIZE, C, H, W)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)  # Shape: (BATCH_SIZE, STACK_SIZE, C, H, W)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        
        # shape of q_values => (BATCH_SIZE, num_actions), we gather along action dimension
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))  # (BATCH_SIZE, 1)
        

       
        
        # with torch.no_grad():
        #     # max Q-value for next states from target_net
        #     max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]  # (BATCH_SIZE, 1)
        #     target_q_values = rewards + GAMMA * (1 - dones) * max_next_q

        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

        # Calculate loss
        loss = nn.functional.mse_loss(q_values, target_q_values.unsqueeze(1))

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1

        # Update target network periodically
        if self.steps_done % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

