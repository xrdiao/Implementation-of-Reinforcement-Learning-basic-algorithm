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

    def __init__(self, action_size, state_size, hidden_size, action_bound):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
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

    def __init__(self, action_size, state_size, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, states, actions):
        # input [states, actions]
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DDPG(DQN):
    def __init__(self, env_, gamma_, alpha_, explosion_step_, epsilon_):
        super(DDPG, self).__init__(env_, gamma_, alpha_, explosion_step_, epsilon_)
        self.action_bound = self.env.action_space.high[0]

        self.actor = Actor(self.action_size, self.state_size, self.hidden_size, self.action_bound).to(self.device)
        self.actor_target = Actor(self.action_size, self.state_size, self.hidden_size, self.action_bound).to(
            self.device)
        self.critic = Critic(self.action_size, self.state_size, self.hidden_size).to(self.device)
        self.critic_target = Critic(self.action_size, self.state_size, self.hidden_size).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_optimizer = optim.Adam(self.critic.parameters())

        self.name = 'DDPG'

    def choose_action(self, state_, epsilon_):
        state_ = torch.tensor(state_, dtype=torch.float).view(1, -1)
        action = self.actor_target(state_)
        if epsilon_ == 0:
            return [action.item()]
        action = action.item() + self.alpha * np.random.randn(self.action_size)  # 增加探索
        return action

    def learn(self, state_, action_, reward_, next_state_, dones_):
        states, actions, rewards, next_states, dones = self.numpy2tensor(state_, action_, reward_, next_state_, dones_)

        next_actions = self.actor_target(next_states)
        Q_values = self.critic(states, actions)
        next_Q = self.critic_target(next_states, next_actions)

        targets = rewards + self.gamma * next_Q

        self.critic_optimizer.zero_grad()
        critic_loss = F.mse_loss(Q_values, targets.detach()).to(self.device)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        action_ = self.actor(states)
        actor_loss = - torch.mean(self.critic(states, action_)).to(self.device)
        actor_loss.backward()
        self.actor_optimizer.step()

    def train(self, episodes_, pretrain=False):
        # 其实和DQN的训练是一样的
        if pretrain:
            self.load_model()

        for episode in range(episodes_):
            state = self.env.reset()
            sum_reward = 0

            while True:
                action = self.choose_action(state, self.epsilon)
                next_state, reward, done, info = self.env.step(action)

                self.memory.add(state, int(action), reward, next_state, done)

                if self.memory.size() > 200:
                    states, rewards, actions, next_states, dones = self.memory.sample(self.load_size)
                    self.learn(states, rewards, actions, next_states, dones)

                state = next_state
                sum_reward = sum_reward + reward

                if done:
                    break
            self.reward_buffer.append(sum_reward)

            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)
            if episode % 200 == 0 and episode != 0:
                self.critic_target.load_state_dict(self.critic.state_dict())
                self.actor_target.load_state_dict(self.actor.state_dict())

            if episode % 1000 == 0:
                print("Episode {}, reward:{}".format(episode, sum(self.reward_buffer) / len(self.reward_buffer)))
                torch.save(self.critic.state_dict(), 'models/' + self.name + 'critic.pth')
                torch.save(self.actor.state_dict(), 'models/' + self.name + 'actor.pth')
                self.reward_buffer.clear()
