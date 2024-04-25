import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal

from SAC import SAC


class BetaActor(nn.Module):
    def __init__(self, action_size, state_size, action_bound):
        super(BetaActor, self).__init__()
        hidden_size = 64

        self.l1 = nn.Linear(state_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.alpha_head = nn.Linear(hidden_size, action_size)
        self.beta_head = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))

        alpha = F.softplus(self.alpha_head(a)) + 1.0
        beta = F.softplus(self.beta_head(a)) + 1.0

        return alpha, beta

    def get_dist(self, state):
        alpha, beta = self.forward(state)
        dist = Beta(alpha, beta)
        return dist

    def deterministic_act(self, state):
        alpha, beta = self.forward(state)
        mode = alpha / (alpha + beta)
        return mode


class GaussianActorMuSigma(nn.Module):
    def __init__(self, action_size, state_size, action_bound):
        super(GaussianActorMuSigma, self).__init__()
        hidden_size = 64

        self.fc1 = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

        self.fc_mu = nn.Linear(hidden_size, action_size)
        self.fc_std = nn.Linear(hidden_size, action_size)

        self.action_bound = action_bound

    def forward(self, x):
        x = self.fc1(x)
        mu = self.action_bound * torch.tanh(self.fc_mu(x))
        sigma = F.softplus(self.fc_std(x))
        return mu, sigma

    def get_dist(self, state):
        mu, sigma = self.forward(state)
        dist = Normal(mu, sigma)
        return dist, dist.entropy()

    def deterministic_act(self, state):
        mu, sigma = self.forward(state)
        return mu


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


class SACContinuous(SAC):
    def __init__(self, env_, gamma_, alpha_, explosion_step_, epsilon_):
        super(SACContinuous, self).__init__(env_, gamma_, alpha_, explosion_step_, epsilon_)

        self.hidden_size = 64

        self.actor = GaussianActorMuSigma(self.action_size, self.state_size, self.hidden_size).to(self.device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic1 = Critic(self.action_size, self.state_size, self.hidden_size).to(self.device)
        self.critic2 = Critic(self.action_size, self.state_size, self.hidden_size).to(self.device)

        self.critic1_target = Critic(self.action_size, self.state_size, self.hidden_size).to(self.device)
        self.critic2_target = Critic(self.action_size, self.state_size, self.hidden_size).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.optim_critic1 = torch.optim.Adam(self.critic1.parameters(), lr=self.learning_rate)
        self.optim_critic2 = torch.optim.Adam(self.critic2.parameters(), lr=self.learning_rate)

    def choose_action(self, state_, epsilon):
        # epsilon is useless
        state_ = torch.tensor(state_, dtype=torch.float).view(1, -1).to(self.device)

        dist, _ = self.actor.get_dist(state_)
        action = dist.sample()
        return [action.item()]

    def q_target(self, next_state, reward, dones):
        dist, entropy = self.actor.get_dist(next_state)
        next_action = dist.sample()
        log_prob = dist.log_prob(next_action)

        q1_value = self.critic1_target(next_state, next_action)
        q2_value = self.critic2_target(next_state, next_action)

        min_q = torch.min(q1_value, q2_value)

        # 原式中求的是Q、熵的均值。
        td_target = reward + self.gamma * (min_q + self.log_alpha.exp() * log_prob) * (1 - dones)
        return td_target
