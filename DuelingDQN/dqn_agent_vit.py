import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from config import *
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms

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


class DuelingDQN_ViT(nn.Module):
    def __init__(self):
        super(DuelingDQN_ViT, self).__init__()

        # Load a pre-trained Vision Transformer (ViT) as the feature extractor
        self.vit = models.vit_b_16(pretrained=True)  # ViT-Base with 16x16 patches
        self.vit.heads = nn.Identity()  # Remove classification head

        # Fully connected layers for Dueling DQN
        self.fc1 = nn.Linear(768, 512)  # ViT outputs 768-d feature vector
        self.fc2 = nn.Linear(512, len(ACTION_SPACE))

        # Separate Value and Advantage Streams
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, len(ACTION_SPACE))

    def forward(self, x):
        x = self.vit(x)  # Pass image through ViT
        x = torch.relu(self.fc1(x))

        value = self.value_stream(x)  # State Value V(s)
        advantage = self.advantage_stream(x)  # Advantage A(s, a)

        # Compute final Q-values using dueling trick
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values



class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DuelingDQN_ViT().to(self.device)
        self.target_net = DuelingDQN_ViT().to(self.device)
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
        #Trains the DQN agent using a batch of experiences from replay memory.
        if len(self.memory) < BATCH_SIZE:
            return 0, 0 # Not enough experiences to train

        # 1. Sample a batch of experiences from memory
        # batch = random.sample(self.memory, BATCH_SIZE)
        # states, actions, rewards, next_states, dones = zip(*batch)

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # 2. Convert them into torch Tensors
        # states, next_states should each be a list of (1, C, H, W) Tensors
        # We'll cat them along dim=0 => (BATCH_SIZE, C, H, W)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)  # Shape: (BATCH_SIZE, STACK_SIZE, C, H, W)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # 3. Compute current Q-values from policy network
        # shape of q_values => (BATCH_SIZE, num_actions), we gather along action dimension
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))  # (BATCH_SIZE, 1)
        avg_q_value = q_values.mean().item()

        # 4. Compute target Q-values
        # If done = 1, then no future reward => (1 - dones) zeroes out the next Q
        # with torch.no_grad():
        #     # max Q-value for next states from target_net
        #     max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]  # (BATCH_SIZE, 1)
        #     target_q_values = rewards + GAMMA * (1 - dones) * max_next_q

        # Use policy_net to select the best action
        next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)  

        # Use target_net to evaluate that action
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1).detach()  

        # Compute target Q-value
        target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

        # 5. Calculate loss (MSE)
        loss = nn.functional.mse_loss(q_values, target_q_values.unsqueeze(1))

        # 6. Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.steps_done += 1

        # (Optional) Update target network periodically
        if self.steps_done % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item(), avg_q_value

