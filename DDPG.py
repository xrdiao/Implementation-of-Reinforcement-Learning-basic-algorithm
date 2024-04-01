from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from relay_buffer import ReplayBuffer
from DQN import DQN


class Actor(nn.Module):
    '''
    Deterministic actor network
    :return action(s) [b,1]
    '''

    def __init__(self, state_size, hidden_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # input [state]
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Critic(nn.Module):
    '''
    Deterministic critic network
    :return Q(s,a), [b,1]
    '''

    def __init__(self, action_size, state_size, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # input [states, actions]
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DDPG(DQN):
    def __init__(self, env_, gamma_, alpha_, explosion_step_, epsilon_):
        super(DDPG, self).__init__(env_, gamma_, alpha_, explosion_step_, epsilon_)
