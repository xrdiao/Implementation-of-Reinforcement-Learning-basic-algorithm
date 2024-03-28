import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch import nn

from DQN import DQN
import torch


class QNetwork(nn.Module):
    def __init__(self, action_size, state_size, hidden_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.advantage = nn.Linear(hidden_size, action_size)
        self.value = nn.Linear(hidden_size, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        advantages = self.advantage(x)
        values = self.value(x)

        mean_advantage = torch.mean(advantages) if advantages.dim() < 2 else torch.mean(advantages, dim=1, keepdim=True)
        return values + (advantages - mean_advantage)


class DuelingDQN(DQN):
    def __init__(self, env_, gamma_, alpha_, explosion_step_, epsilon_):
        super(DuelingDQN, self).__init__(env_, gamma_, alpha_, explosion_step_, epsilon_)
        self.name = 'DuelingDQN'

        self.target = QNetwork(self.action_size, self.state_size, 16)
        self.eval = QNetwork(self.action_size, self.state_size, 16)
