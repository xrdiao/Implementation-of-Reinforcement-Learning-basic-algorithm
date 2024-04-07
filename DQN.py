from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from relay_buffer import ReplayBuffer
from gym import spaces


class Model(torch.nn.Module):
    def __init__(self, action_size, state_size, hidden_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        # x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN:
    def __init__(self, env_, gamma_, alpha_, explosion_step_, epsilon_):
        self.env = env_
        self.alpha = alpha_
        self.epsilon = epsilon_
        self.hidden_size = 16

        if isinstance(self.env.observation_space, spaces.Box):
            self.state_size = self.env.observation_space.shape[0]
        else:
            self.state_size = not self.env.observation_space.n
        if isinstance(self.env.action_space, spaces.Box):
            self.action_size = self.env.action_space.shape[0]
        else:
            self.action_size = self.env.action_space.n

        # hidden = 16
        self.eval = Model(self.action_size, self.state_size, self.hidden_size)
        self.target = Model(self.action_size, self.state_size, self.hidden_size)
        self.optimizer = optim.Adam(self.eval.parameters(), lr=0.01)  # 这里加一个weight_decay直接就不收敛了，估计是因为参数本来就不多。
        self.memory = ReplayBuffer(1000)

        self.explosion_step = explosion_step_
        self.gamma = gamma_
        self.load_size = 64
        self.reward_buffer = deque(maxlen=10000)
        self.loss_buffer = deque(maxlen=10000)

        self.min_epsilon = 0.01
        self.decay_rate = 0.01
        self.max_epsilon = 1.0

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.eval.to(self.device)
        self.target.to(self.device)
        self.name = 'DQN'

    def get_path(self, addition=''):
        return 'models/' + self.name + addition + '.pth'

    def numpy2tensor(self, state_, action_, reward_, next_state_, dones_):
        action_ = torch.tensor(np.array(action_), dtype=torch.long).view(-1, 1).to(self.device)
        reward_ = torch.tensor(np.array(reward_), dtype=torch.float).view(-1, 1).to(self.device)
        dones_ = torch.tensor(np.array(dones_), dtype=torch.long).view(-1, 1).to(self.device)
        state_ = torch.tensor(np.array(state_), dtype=torch.float).to(self.device)
        next_state_ = torch.tensor(np.array(next_state_), dtype=torch.float).to(self.device)
        return state_, action_, reward_, next_state_, dones_

    def learn(self, state_, action_, reward_, next_state_, dones_):
        state_, action_, reward_, next_state_, dones_ = self.numpy2tensor(state_, action_, reward_, next_state_, dones_)

        value = self.eval(state_).gather(1, action_)
        next_value, _ = torch.max(self.target(next_state_), dim=1)

        # done意味着终止，所以done状态只有reward
        target = reward_ + self.gamma * next_value.view(-1, 1).detach() * (1 - dones_)

        self.optimizer.zero_grad()
        loss = torch.mean(F.mse_loss(target, value)).to(self.device)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def choose_action(self, state_, epsilon_):
        state_ = torch.tensor(state_, dtype=torch.float).view(1, -1)
        if np.random.uniform(0, 1) > epsilon_:
            return torch.argmax(self.eval(state_)).item()
        return self.env.action_space.sample()

    def load_model(self):
        self.eval.load_state_dict(torch.load(self.get_path()))
        self.target.load_state_dict(torch.load(self.get_path()))

    def train(self, episodes_, pretrain=False):
        if pretrain:
            self.load_model()

        for episode in range(episodes_):
            state = self.env.reset()
            loss_sum = 0
            sum_reward = 0

            while True:
                action = self.choose_action(state, self.epsilon)
                next_state, reward, done, info = self.env.step(action)

                self.memory.add(state, int(action), reward, next_state, done)

                if self.memory.size() > 200:
                    states, rewards, actions, next_states, dones = self.memory.sample(self.load_size)
                    loss_sum = loss_sum + self.learn(states, rewards, actions, next_states, dones)

                state = next_state
                sum_reward = sum_reward + reward

                if done:
                    break
            self.reward_buffer.append(sum_reward)

            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)
            if episode % 200 == 0 and episode != 0:
                self.target.load_state_dict(self.eval.state_dict())

            if episode % 1000 == 0 and episode != 0:
                print("Episode {}, epsilon: {}, loss: {}, reward:{}".format(episode, self.epsilon, loss_sum,
                                                                            sum(self.reward_buffer) / len(
                                                                                self.reward_buffer)))
                torch.save(self.eval.state_dict(), self.get_path())
                self.reward_buffer.clear()

    def plot_reward(self):
        plt.plot(self.reward_buffer)
        plt.show()

    def test(self, episodes, render=False):
        self.load_model()
        for episode in range(episodes):
            state = self.env.reset()
            total_rewards = 0
            print("****************************************************")
            print("EPISODE ", episode)

            while True:
                if render:
                    self.env.render()

                action = self.choose_action(state, 0)
                new_state, reward, done, info = self.env.step(action)
                total_rewards += reward

                if done:
                    print("Score", total_rewards)
                    break
                state = new_state
        print('end test')
        self.env.close()
