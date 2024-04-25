import torch.optim
from torch.distributions import Categorical

from DQN import DQN

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, action_size, state_size, hidden_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class Critic(nn.Module):
    def __init__(self, action_size, state_size, hidden_size=64):
        super(Critic, self).__init__()

        self.C1 = nn.Linear(state_size + action_size, hidden_size)
        self.C2 = nn.Linear(hidden_size, hidden_size)
        self.C3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        v = torch.tanh(self.C1(x))
        v = torch.tanh(self.C2(v))
        v = self.C3(v)
        return v


class SAC(DQN):
    def __init__(self, env_, gamma_, alpha_, explosion_step_, epsilon_):
        super(SAC, self).__init__(env_, gamma_, alpha_, explosion_step_, epsilon_)
        self.hidden_size = 64

        self.actor = Actor(self.action_size, self.state_size, self.hidden_size).to(self.device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic1 = Critic(self.action_size, self.state_size, self.hidden_size).to(self.device)
        self.critic2 = Critic(self.action_size, self.state_size, self.hidden_size).to(self.device)

        self.critic1_target = Critic(self.action_size, self.state_size, self.hidden_size).to(self.device)
        self.critic2_target = Critic(self.action_size, self.state_size, self.hidden_size).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.optim_critic1 = torch.optim.Adam(self.critic1.parameters(), lr=self.learning_rate)
        self.optim_critic2 = torch.optim.Adam(self.critic2.parameters(), lr=self.learning_rate)

        self.log_alpha = torch.tensor(np.log(self.alpha), dtype=torch.float, requires_grad=True, device=self.device)
        self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)
        self.target_entropy = torch.tensor(-self.action_size, dtype=torch.float, requires_grad=True, device=self.device)
        self.tau = 0.005

        self.name = 'SAC'

    def choose_action(self, state_, epsilon):
        # epsilon is useless
        state_ = torch.tensor(state_, dtype=torch.float).view(1, -1).to(self.device)

        probs = self.actor(state_)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item()

    def q_target(self, next_state, reward, dones):
        with torch.no_grad():
            next_probs = self.actor(next_state)
            dist = Categorical(next_probs)
            next_action = dist.sample()
            log_prob = dist.log_prob(next_action)

            q1_value = self.critic1_target(next_state, next_action)
            q2_value = self.critic2_target(next_state, next_action)

            min_q = torch.min(q1_value, q2_value)  # 其实就是价值函数

            # 原式中求的是Q、熵的均值。
            td_target = reward + self.gamma * (min_q + self.log_alpha.exp() * log_prob) * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1 - self.tau) + param.data * self.tau)

    def learn(self, state_, action_, reward_, next_state_, dones_):
        state_, action_, reward_, next_state_, dones_ = self.numpy2tensor(state_, action_, reward_, next_state_, dones_)

        td_target = self.q_target(next_state_, reward_, dones_)
        q1 = self.critic1(state_, action_)
        loss_critic1 = torch.mean(F.mse_loss(q1, td_target.detach()))
        q2 = self.critic2(state_, action_)
        loss_critic2 = torch.mean(F.mse_loss(q2, td_target.detach()))

        self.optim_critic1.zero_grad()
        self.optim_critic2.zero_grad()
        loss_critic1.backward()
        loss_critic2.backward()
        self.optim_critic1.step()
        self.optim_critic2.step()

        # 这里更新时的目的是最小化分布间的KL散度，所以用的是分布的entropy而不是经验中的action，这个Q其实是求V用的，所以是期望的形式
        probs = self.actor(state_)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        q1 = self.critic1(state_, action)
        q2 = self.critic2(state_, action)
        min_q = torch.min(q1, q2)

        self.optim_actor.zero_grad()
        loss_actor = torch.mean(self.log_alpha.exp() * log_prob - min_q)
        loss_actor.backward()
        self.optim_actor.step()

        # 自动调节alpha但结果容易发散
        # self.optim_alpha.zero_grad()
        # loss_alpha = torch.mean(self.log_alpha * (entropy - self.target_entropy).detach())
        # loss_alpha.backward()
        # self.optim_alpha.step()

        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        return loss_critic1.item()
