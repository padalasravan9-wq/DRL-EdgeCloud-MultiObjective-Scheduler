"""
DQN Agent for Edge–Cloud Multi-Objective Scheduler
Compatible with hyperparameters.yaml
"""

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


# =========================================================
# Neural Network
# =========================================================
class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super(DQNNetwork, self).__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# =========================================================
# Replay Buffer
# =========================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# =========================================================
# DQN Agent
# =========================================================
class DQNAgent:
    def __init__(self, state_dim, action_dim, config, device="cpu"):
        self.device = torch.device(device)

        # Hyperparameters
        self.gamma = config["training"]["gamma"]
        self.batch_size = config["training"]["batch_size"]
        self.lr = config["training"]["learning_rate"]

        self.epsilon = config["exploration"]["epsilon_start"]
        self.epsilon_min = config["exploration"]["epsilon_min"]
        self.epsilon_decay = config["exploration"]["epsilon_decay"]

        hidden_layers = config["network"]["hidden_layers"]
        buffer_size = config["training"]["replay_buffer_size"]

        # Networks
        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)

        self.action_dim = action_dim
        self.learn_step_counter = 0
        self.target_update_freq = config["training"]["target_update_frequency"]

    # =====================================================
    # Action Selection (ε-greedy)
    # =====================================================
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    # =====================================================
    # Store transition
    # =====================================================
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    # =====================================================
    # Learning step
    # =====================================================
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q values
        q_values = self.policy_net(states).gather(1, actions)

        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss
        loss = self.criterion(q_values, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Target network update
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    # =====================================================
    # Save model
    # =====================================================
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    # =====================================================
    # Load model
    # =====================================================
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
