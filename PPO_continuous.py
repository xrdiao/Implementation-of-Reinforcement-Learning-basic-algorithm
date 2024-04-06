import torch
import torch.nn as nn
import torch.nn.functional as F

from PPO_clip import PPOClip
import numpy as np


class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.fc1(x)
        return x


class Actor(nn.Module):
    def __init__(self, action_size, state_size, action_bound):
        super(Actor, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, action_size)
        self.fc_std = nn.Linear(256, action_size)
        self.action_bound = action_bound

    def forward(self, x):
        x = self.fc1(x)
        mu = self.action_bound * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x)) + 0.001
        return mu, std


class PPOContinuous(PPOClip):
    def __init__(self, env_, gamma_, alpha_, explosion_step_, epsilon_):
        super(PPOContinuous, self).__init__(env_, gamma_, alpha_, explosion_step_, epsilon_)

        self.hidden_size = 64
        self.action_bound = self.env.action_space.high[0]

        self.actor = Actor(self.action_size, self.state_size, self.action_bound).to(self.device)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=2e-5)
        self.critic = Critic(self.state_size).to(self.device)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=2e-5)

        self.name = 'PPOContinuous'

    def choose_action(self, state_, epsilon_):
        state_ = torch.tensor(state_, dtype=torch.float).view(1, -1).to(self.device)

        if epsilon_ == 0:
            mu, _ = self.actor(state_)
            return [mu.item()]

        mu, std = self.actor(state_)
        action = torch.distributions.Normal(mu, std).sample().item()
        return [action]

    def update(self):
        for trajectory in self.memory:
            states, actions, rewards, next_states, dones = trajectory['states'], trajectory['actions'], trajectory[
                'rewards'], trajectory['next_states'], trajectory['dones']
            states, _, rewards, next_states, dones_ = self.numpy2tensor(states, actions, rewards, next_states, dones)
            actions = torch.tensor(np.array(actions), dtype=torch.float).view(-1, 1).to(self.device)

            with torch.no_grad():
                mu, std = self.actor(states)
                action_dicts = torch.distributions.normal.Normal(mu, std)
                log_old_prob = action_dicts.log_prob(actions).detach()

                values = self.critic(states)
                targets = rewards + self.gamma * self.critic(next_states) * (1 - dones_)
                deltas = targets - values
                advantages = self.cal_advantages(deltas).detach()

            for step in range(10):
                mu, std = self.actor(states)
                action_dicts = torch.distributions.Normal(mu, std)

                log_new_prob = action_dicts.log_prob(actions)

                ratio = torch.exp(log_new_prob - log_old_prob)
                left = ratio * advantages
                right = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages

                self.optimizer_actor.zero_grad()
                loss_actor = -torch.mean(torch.min(left, right)).to(self.device)
                loss_actor.backward()
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                loss_critic = torch.mean(F.mse_loss(targets.detach(), self.critic(states))).to(self.device)
                loss_critic.backward()
                self.optimizer_critic.step()
