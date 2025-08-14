import torch
import torch.nn as nn
from torch.optim import Adam
from collections import deque
import random
import numpy as np

class SpeciesAI(nn.Module):
    def __init__(self, input_size, output_size, intelligence=1.0):
        super(SpeciesAI, self).__init__()
        
        # Scale model size based on intelligence (1.0 is average)
        # Base hidden size is 64, scaled by intelligence (capped at 5B params)
        self.intelligence = max(0.1, min(5.0, intelligence))  # Cap between 0.1-5.0
        
        # Calculate layer sizes - this gives us models from ~100M to ~5B parameters
        # Using a logarithmic scale so intelligence differences are meaningful
        scaled_size = int(64 * (2 ** (self.intelligence * 2)))
        scaled_size = min(scaled_size, 8192)  # Cap at 8192 to prevent OOM
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Dynamic architecture based on intelligence
        if self.intelligence < 0.5:
            # Very simple model (~100M params)
            self.layers = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, output_size)
            )
        elif self.intelligence < 1.5:
            # Medium complexity (~500M params)
            self.layers = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, output_size)
            )
        else:
            # High complexity (up to 5B params)
            self.layers = nn.Sequential(
                nn.Linear(input_size, scaled_size),
                nn.ReLU(),
                nn.Linear(scaled_size, scaled_size),
                nn.ReLU(),
                nn.Linear(scaled_size, scaled_size//2),
                nn.ReLU(),
                nn.Linear(scaled_size//2, output_size)
            )
        
        # Memory for experience replay
        self.memory = deque(maxlen=int(10000 * self.intelligence))
        self.optimizer = Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x):
        return self.layers(x)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
        actions = torch.LongTensor(np.array([t[1] for t in minibatch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch]))
        
        current_q = self(states).gather(1, actions.unsqueeze(1))
        next_q = self(next_states).max(1)[0].detach()
        target = rewards + (0.99 * next_q * (1 - dones))
        
        loss = self.loss_fn(current_q.squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()