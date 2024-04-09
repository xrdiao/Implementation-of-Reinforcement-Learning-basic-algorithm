from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from relay_buffer import ReplayBuffer
from DQN import DQN
from PPO_continuous import *


class Actor(nn.Module):
    '''
    Deterministic actor network
    :return action(s) [b,1]
    '''

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
        self.fc2 = nn.Linear(256, action_size)
        self.bound = action_bound

    def forward(self, x):
        # input [state]
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x * self.bound


class Critic(nn.Module):
    '''
    Deterministic critic network
    :return Q(s,a), [b,1]
    '''

    def __init__(self, action_size, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(action_size + state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, states, actions):
        # input [states, actions]
        x = torch.cat([states, actions], dim=1)
        x = self.fc1(x)
        return x


class DDPG(DQN):
    def __init__(self, env_, gamma_, alpha_, explosion_step_, epsilon_):
        super(DDPG, self).__init__(env_, gamma_, alpha_, explosion_step_, epsilon_)
        self.hidden_size = 64
        self.action_bound = self.env.action_space.high[0]

        self.actor = Actor(self.action_size, self.state_size, self.action_bound).to(self.device)
        self.actor_target = Actor(self.action_size, self.state_size, self.action_bound).to(
            self.device)
        self.critic = Critic(self.action_size, self.state_size).to(self.device)
        self.critic_target = Critic(self.action_size, self.state_size).to(self.device)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=2e-5)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=2e-5)

        self.name = 'DDPG'

    def choose_action(self, state_, epsilon_):
        state_ = torch.tensor(state_, dtype=torch.float).view(1, -1).to(self.device)
        action = self.actor_target(state_)
        if epsilon_ == 0:
            return [action.item()]
        action = action.item() + self.alpha * np.random.randn(self.action_size)  # 增加探索
        return action

    def load_model(self, addition='_actor'):
        a = self.get_path(addition)
        self.actor.load_state_dict(torch.load(self.get_path(addition)))

    def learn(self, state_, action_, reward_, next_state_, dones_):
        states, _, rewards, next_states, dones = self.numpy2tensor(state_, action_, reward_, next_state_, dones_)
        actions = torch.tensor(np.array(action_), dtype=torch.float).view(-1, 1).to(self.device)

        next_actions = self.actor_target(next_states)
        Q_values = self.critic(states, actions)
        next_Q = self.critic_target(next_states, next_actions)

        targets = rewards + self.gamma * next_Q * (1 - dones)

        self.optimizer_critic.zero_grad()
        critic_loss = F.mse_loss(Q_values, targets.detach()).to(self.device)
        critic_loss.backward()
        self.optimizer_critic.step()

        self.optimizer_actor.zero_grad()
        action_ = self.actor(states)
        actor_loss = - torch.mean(self.critic(states, action_)).to(self.device)
        actor_loss.backward()
        self.optimizer_actor.step()

    def train(self, episodes_, pretrain=False):
        # 和DQN的训练是一样的
        if pretrain:
            self.load_model()

        for episode in range(episodes_):
            state = self.env.reset()
            sum_reward = 0
            max_reward = -10000000

            while True:
                action = self.choose_action(state, self.epsilon)
                next_state, reward, done, info = self.env.step(action)

                self.memory.add(state, action, reward, next_state, done)

                if self.memory.size() > 200:
                    states, rewards, actions, next_states, dones = self.memory.sample(self.load_size)
                    self.learn(states, rewards, actions, next_states, dones)

                state = next_state
                sum_reward = sum_reward + reward

                if done:
                    break
            self.reward_buffer.append(sum_reward)

            if max_reward < sum_reward:
                torch.save(self.critic.state_dict(), self.get_path('_critic'))
                torch.save(self.actor.state_dict(), self.get_path('_actor'))

            if episode % 200 == 0 and episode != 0:
                self.critic_target.load_state_dict(self.critic.state_dict())
                self.actor_target.load_state_dict(self.actor.state_dict())

            if episode % 1000 == 0:
                print("Episode {}, reward:{}".format(episode, sum(self.reward_buffer) / len(self.reward_buffer)))
                self.reward_buffer.clear()
