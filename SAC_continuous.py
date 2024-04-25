from collections import deque

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

        self.name = 'SAC_continuous'

    def choose_action(self, state_, epsilon):
        # epsilon is useless
        state_ = torch.tensor(state_, dtype=torch.float).view(1, -1).to(self.device)

        dist, _ = self.actor.get_dist(state_)
        action = dist.sample()
        return [action.item()]

    def q_target(self, next_state, reward, dones):
        with torch.no_grad():
            dist, entropy = self.actor.get_dist(next_state)
            next_action = dist.sample()
            log_prob = dist.log_prob(next_action)

            next_action = next_action.view(-1, 1)
            q1_value = self.critic1_target(next_state, next_action)
            q2_value = self.critic2_target(next_state, next_action)

            min_q = torch.min(q1_value, q2_value)

            # 原式中求的是Q、熵的均值。
            td_target = reward + self.gamma * (min_q - self.log_alpha.exp() * log_prob) * (1 - dones)
        return td_target

    def learn(self, state_, action_, reward_, next_state_, dones_):
        state_, action_, reward_, next_state_, dones_ = self.numpy2tensor(state_, action_, reward_, next_state_, dones_)

        td_target = self.q_target(next_state_, reward_, dones_)
        q1 = self.critic1(state_, action_)
        q2 = self.critic2(state_, action_)
        loss_critic1 = torch.mean(F.mse_loss(q1, td_target.detach()))
        loss_critic2 = torch.mean(F.mse_loss(q2, td_target.detach()))

        self.optim_critic1.zero_grad()
        self.optim_critic2.zero_grad()
        loss_critic1.backward()
        loss_critic2.backward()
        self.optim_critic1.step()
        self.optim_critic2.step()

        # 这里更新时的目的是最小化分布间的KL散度，所以用的是分布的entropy而不是经验中的action，这个Q其实是求V用的，所以是期望的形式
        dist, _ = self.actor.get_dist(state_)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        action = action.view(-1, 1)
        q1 = self.critic1(state_, action)
        q2 = self.critic2(state_, action)
        q = torch.min(q1, q2)

        self.optim_actor.zero_grad()
        loss_actor = torch.mean(self.log_alpha.exp() * log_prob - q.detach())
        loss_actor.backward()
        self.optim_actor.step()

        # 自动调节alpha但结果容易发散
        self.optim_alpha.zero_grad()
        loss_alpha = -torch.mean(self.log_alpha.exp() * (log_prob + self.target_entropy).detach())
        loss_alpha.backward()
        self.optim_alpha.step()

        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        return loss_critic1.item()

    def train(self, episodes_, pretrain=False):
        max_reward = -100000000
        rewards_set = deque(maxlen=1000)

        for episode in range(episodes_):
            state = self.env.reset()
            loss_sum = 0
            sum_reward = 0

            while True:
                action = self.choose_action(state, self.epsilon)
                next_state, reward, done, info = self.env.step(action)

                self.memory.add(state, action, reward, next_state, done)

                if self.memory.size() > 200:
                    states, rewards, actions, next_states, dones = self.memory.sample(self.load_size)
                    loss_sum = loss_sum + self.learn(states, rewards, actions, next_states, dones)

                state = next_state
                sum_reward = sum_reward + reward

                if done:
                    break
            rewards_set.append(sum_reward)

            if max_reward < sum_reward:
                max_reward = sum_reward
                torch.save(self.eval.state_dict(), self.get_path())

            if episode % 100 == 0 and episode != 0:
                print("Episode {}, epsilon: {}, loss: {}, reward:{}".format(episode, self.epsilon, loss_sum,
                                                                            sum(rewards_set) / len(rewards_set)))
