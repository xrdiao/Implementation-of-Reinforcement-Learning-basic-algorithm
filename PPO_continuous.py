import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from PPO_clip import PPOClip
import numpy as np


class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()

        self.C1 = nn.Linear(state_size, 64)
        self.C2 = nn.Linear(64, 64)
        self.C3 = nn.Linear(64, 1)

    def forward(self, x):
        v = torch.tanh(self.C1(x))
        v = torch.tanh(self.C2(v))
        v = self.C3(v)
        return v


class Actor(nn.Module):
    def __init__(self, action_size, state_size, action_bound):
        super(Actor, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, action_size)
        self.fc_mu.weight.data.mul_(0.1)
        self.fc_mu.bias.data.mul_(0.0)

        self.fc_std = nn.Linear(64, action_size)
        self.action_log_std = nn.Parameter(torch.ones(1, action_size) * 0)
        self.action_bound = action_bound

    def forward(self, x):
        x = self.fc1(x)
        mu = self.action_bound * torch.tanh(self.fc_mu(x))
        return mu

    def get_dist(self, state):
        mu = self.forward(state)
        action_log_std = self.action_log_std.expand_as(mu)
        action_std = torch.exp(action_log_std)

        dist = Normal(mu, action_std)
        return dist

    def deterministic_act(self, state):
        return self.forward(state)


class PPOContinuous(PPOClip):
    def __init__(self, env_, gamma_, alpha_, explosion_step_, epsilon_):
        super(PPOContinuous, self).__init__(env_, gamma_, alpha_, explosion_step_, epsilon_)

        self.hidden_size = 64
        self.action_bound = self.env.action_space.high[0]

        self.actor = Actor(self.action_size, self.state_size, self.action_bound).to(self.device)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=2e-5)
        self.critic = Critic(self.state_size).to(self.device)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=2e-5)

        self.update_batch_size = 64
        self.entropy_coef = 0.5
        self.entropy_decay = 0.5

        self.name = 'PPOContinuous'

    def choose_action(self, state_, epsilon_):
        state_ = torch.tensor(state_, dtype=torch.float).view(1, -1).to(self.device)

        if epsilon_ == 0:
            mu = self.actor(state_)
            return [mu.item()]

        dist = self.actor.get_dist(state_)
        action = dist.sample().item()
        return [action]

    def update(self):
        self.entropy_coef = self.entropy_coef * self.entropy_decay
        states, actions, rewards, next_states, dones = None, None, None, None, None
        for trajectory in self.memory:
            state, action, reward, next_state, done = trajectory['states'], trajectory['actions'], trajectory[
                'rewards'], trajectory['next_states'], trajectory['dones']
            state, _, reward, next_state, done = self.numpy2tensor(state, action, reward, next_state, done)
            action = torch.tensor(np.array(action), dtype=torch.float).view(-1, 1).to(self.device)

            if states is None:
                states, actions, rewards, next_states, dones = state, action, reward, next_state, done
            else:
                states = torch.cat((states, state))
                actions = torch.cat((actions, action))
                rewards = torch.cat((rewards, reward))
                next_states = torch.cat((next_states, next_state))
                dones = torch.cat((dones, done))

        with torch.no_grad():
            action_dicts = self.actor.get_dist(states)
            log_old_prob = action_dicts.log_prob(actions).detach()

            values = self.critic(states)
            targets = rewards + self.gamma * self.critic(next_states) * (1 - dones)
            deltas = targets - values
            advantages = self.cal_advantages(deltas).detach()
            td_targets = (advantages + values).to(self.device)

        optim_iter_num = int(math.ceil(states.shape[0] / self.update_batch_size))
        for _ in range(10):
            # 打乱顺序
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.device)
            states, actions, td_targets, advantages, log_old_prob = states[perm].clone(), actions[perm].clone(), \
                td_targets[perm].clone(), advantages[perm].clone(), log_old_prob[perm].clone()

            for i in range(optim_iter_num):
                index = slice(i * self.update_batch_size, min((i + 1) * self.update_batch_size, states.shape[0]))

                action_dicts = self.actor.get_dist(states[index])
                dist_entropy = torch.sum(action_dicts.entropy(), dim=1).view(-1, 1)

                log_new_prob = action_dicts.log_prob(actions[index])

                ratio = torch.exp(log_new_prob - log_old_prob[index])
                left = ratio * advantages[index]
                right = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages[index]

                self.optimizer_actor.zero_grad()
                loss_actor = (-torch.mean(torch.min(left, right) - dist_entropy * self.entropy_coef)).to(
                    self.device)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                loss_actor.backward()
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                loss_critic = torch.mean(F.mse_loss(td_targets[index].detach(), self.critic(states[index]))).to(
                    self.device)
                loss_critic.backward()
                self.optimizer_critic.step()
