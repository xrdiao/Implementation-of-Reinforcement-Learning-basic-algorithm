import torch
import torch.nn as nn
import torch.nn.functional as F

from PPO_clip import PPOClip
from PPO import Critic


class Actor(nn.Module):
    def __init__(self, action_size, state_size, hidden_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, action_size)
        self.fc_std = nn.Linear(hidden_size, action_size)

    # 前向传播
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        mu = self.fc_mu(x)
        mu = 2 * torch.tanh(mu)
        std = self.fc_std(x)
        std = F.softplus(std)
        return mu, std + 0.001


class PPOContinuous(PPOClip):
    def __init__(self, env_, gamma_, alpha_, explosion_step_, epsilon_):
        super(PPOContinuous, self).__init__(env_, gamma_, alpha_, explosion_step_, epsilon_)

        self.hidden_size = 64
        self.actor = Actor(self.action_size, self.state_size, self.hidden_size).to(self.device)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=2e-5)
        self.critic = Critic(self.state_size, self.hidden_size).to(self.device)
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
            states, actions, rewards, next_states, dones_ = self.numpy2tensor(states, actions, rewards, next_states,
                                                                              dones)

            values = self.critic(states)
            targets = rewards + self.gamma * self.critic(next_states) * (1 - dones_)
            deltas = targets - values
            advantages = self.cal_advantages(deltas).detach()

            mu, std = self.actor(states)
            action_dicts = torch.distributions.Normal(mu, std)
            log_old_prob = action_dicts.log_prob(actions).detach()

            for step in range(10):
                mu, std = self.actor(states)
                action_dicts = torch.distributions.Normal(mu, std)

                log_new_prob = action_dicts.log_prob(actions)

                ratio = torch.exp(log_new_prob - log_old_prob)
                left = ratio * advantages
                right = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages

                self.optimizer_actor.zero_grad()
                loss_actor = torch.mean(-torch.min(left, right)).to(self.device)
                loss_actor.backward()
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                loss_critic = torch.mean(F.mse_loss(targets.detach(), self.critic(states))).to(self.device)
                loss_critic.backward()
                self.optimizer_critic.step()
