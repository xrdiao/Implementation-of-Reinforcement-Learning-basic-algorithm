import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from DQN import DQN
import torch


class QNetwork(nn.Module):
    def __init__(self, action_size, state_size, hidden_size, hidden_size2):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.advantage = nn.Linear(hidden_size2, action_size)
        self.value = nn.Linear(hidden_size2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        advantages = self.advantage(x)
        values = self.value(x)

        return values + advantages - torch.mean(advantages, dim=1, keepdim=True)


class DuelingDQN(DQN):
    def __init__(self, env_, gamma_, alpha_, explosion_step_, epsilon_):
        super(DuelingDQN, self).__init__(env_, gamma_, alpha_, explosion_step_, epsilon_)
        self.name = 'DuelingDQN'

        # 训练的方法没有问题，如果不收敛，问题应该出在模型或者学习率上，大概率是后者（调参），其render结果和DQN的不太一样
        self.target = QNetwork(self.action_size, self.state_size, self.hidden_size*2, self.hidden_size).to(self.device)
        self.eval = QNetwork(self.action_size, self.state_size, self.hidden_size*2, self.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.eval.parameters(), lr=0.001)

