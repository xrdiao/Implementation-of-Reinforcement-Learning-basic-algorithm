from PPO_clip import PPOClip
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class Actor(nn.Module):
    def __init__(self, action_size, state_size, hidden_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = F.sigmoid(self.fc2(x))
        std = F.tanh(self.fc3(x))
        return mu, std


class PPOContinuous(PPOClip):
    def __init__(self, env_, gamma_, alpha_, explosion_step_, epsilon_):
        super(PPOContinuous, self).__init__(env_, gamma_, alpha_, explosion_step_, epsilon_)
