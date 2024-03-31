from DQN import DQN
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from PG import Actor


class Critic(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        # 输出为1，因为一个状态对应一个价值
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class PPO(DQN):
    def __init__(self, env_, gamma_, alpha_, explosion_step_, epsilon_):
        super(PPO, self).__init__(env_, gamma_, alpha_, explosion_step_, epsilon_)
