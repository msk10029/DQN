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
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (state.astype(np.float32),
            action,
            reward.astype(np.float32),
            next_state.astype(np.float32),
            done)

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, stride=2)  # Conv layer 1
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2) # Conv layer 2

        #self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2)

        # test_input = torch.zeros(1, 1, input_shape[1], input_shape[2])
        # conv_out = self.conv3(self.conv2(self.conv1(test_input)))
        # self.conv_out_size = int(np.prod(conv_out.shape)) 

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
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.steps_done = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, len(ACTION_SPACE) - 1)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)
            if state.dim() == 2:
                state = state.unsqueeze(0).unsqueeze(0)
            elif state.dim() == 3:
                state = state.unsqueeze(0)
            # state = state.expand(BATCH_SIZE, -1, -1, -1)
            return self.policy_net(state).argmax(dim=1).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        #Trains the DQN agent using a batch of experiences from replay memory.
        if len(self.memory) < BATCH_SIZE:
            return 0  # Not enough experiences to train

        # 1. Sample a batch of experiences from memory
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 2. Convert them into torch Tensors
        # states, next_states should each be a list of (1, C, H, W) Tensors
        # We'll cat them along dim=0 => (BATCH_SIZE, C, H, W)
        states = [torch.tensor(state, dtype=torch.float32).to(self.device) for state in states]
        states = torch.cat(states).to(self.device)

        states = states.view(-1, 1, 64, 64)

        next_states = [torch.tensor(ns, dtype=torch.float32).to(self.device) for ns in next_states]
        next_states = torch.cat(next_states).to(self.device)
        next_states = next_states.view(-1, 1, 64, 64)

        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)  # (BATCH_SIZE, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)  # (BATCH_SIZE, 1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)  # (BATCH_SIZE, 1)

        # 3. Compute current Q-values from policy network
        # shape of q_values => (BATCH_SIZE, num_actions), we gather along action dimension
        q_values = self.policy_net(states).gather(1, actions)  # (BATCH_SIZE, 1)

        # 4. Compute target Q-values
        # If done = 1, then no future reward => (1 - dones) zeroes out the next Q
        with torch.no_grad():
            # max Q-value for next states from target_net
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]  # (BATCH_SIZE, 1)
            target_q_values = rewards + GAMMA * (1 - dones) * max_next_q

        # 5. Calculate loss (MSE)
        loss = nn.functional.mse_loss(q_values, target_q_values)

        # 6. Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1

        # (Optional) Update target network periodically
        if self.steps_done % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

