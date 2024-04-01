from PG import PolicyGradient
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from collections import deque
import random


class Critic(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # 输出为 [b,1]，因为一个状态对应一个价值
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO(PolicyGradient):
    '''
    这个只做了离散版本的PPO，连续版本的PPO只要把 '动作选择' 和 '概率计算' 中的分布换成 *高斯分布* 即可
    '''

    def __init__(self, env_, gamma_, alpha_, explosion_step_, epsilon_):
        super(PPO, self).__init__(env_, gamma_, alpha_, explosion_step_, epsilon_)

        self.critic = Critic(self.state_size, self.hidden_size).to(self.device)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=0.001)
        self.memory = deque(maxlen=50)
        self.lmbda = 1

        self.name = 'PPO'

    def KL_divergence(self, old_probs, new_probs):
        KL_divs = []
        for i in range(len(old_probs)):
            KL_div = torch.sum(old_probs[i] * torch.log(old_probs[i] / new_probs[i]))
            KL_divs.append(KL_div)
        KL_divs = torch.tensor(KL_divs, dtype=torch.float).view(-1, 1)
        return torch.mean(KL_divs)

    def update(self):
        for trajectory in self.memory:
            states, actions, rewards, next_states, dones = trajectory['states'], trajectory['actions'], trajectory[
                'rewards'], trajectory['next_states'], trajectory['dones']
            states, actions, rewards, next_states, dones_ = self.numpy2tensor(states, actions, rewards, next_states,
                                                                              dones)

            # 计算优势函数
            values = self.critic(states)
            next_values = self.critic(next_states)
            targets = rewards + self.gamma * next_values * (1 - dones_)
            deltas = targets - values

            advantage = 0
            advantages = []
            for delta in reversed(deltas):
                advantage = self.gamma * advantage + delta
                advantages.append(advantage)
            advantages.reverse()
            advantages = torch.tensor(advantages, dtype=torch.float)

            # 原始概率，和选取动作的log概率
            old_prob = self.actor(states)
            log_old_prob = torch.log(old_prob.gather(1, actions).view(-1, 1))

            # 更新Actor
            for _ in range(10):
                new_prob = self.actor(states)
                log_new_prob = torch.log(new_prob.gather(1, actions).view(-1, 1))

                # 不收敛的问题要么在这，要么是参数设置
                # KL = F.kl_div(torch.log(old_prob), new_prob, reduction='batchmean')  # 调用这个函数容易让反向传播失效
                KL_div = self.KL_divergence(old_prob, new_prob)
                ratios = torch.exp(log_new_prob - log_old_prob.detach())
                loss_actor = -torch.mean(ratios * advantages.detach()) + self.lmbda * KL_div

                # 谁能想到问题是optimizer没有归零（之前错在optimizer.zero_grad，应该是optimizer_actor.zero_grad）
                self.optimizer_actor.zero_grad()
                loss_actor.backward()
                self.optimizer_actor.step()

            for _ in range(10):
                self.optimizer_critic.zero_grad()
                loss_critic = torch.mean(F.mse_loss(targets.detach(), self.critic(states)))
                loss_critic.backward()
                self.optimizer_critic.step()

            new_prob = self.actor(states)
            KL_div = self.KL_divergence(old_prob, new_prob)
            if KL_div > 0.1:
                self.lmbda = self.lmbda * 2
            elif KL_div < 0.01:
                self.lmbda = self.lmbda / 2

    def train(self, episodes_, pretrain=False):
        if pretrain:
            self.load_model()

        for episode in range(episodes_):
            trajectory_dict = self.explore_trajectory(episodes_)

            # 我想通过buffer来利用过去的经验
            self.memory.append(trajectory_dict)
            self.reward_buffer.append(torch.sum(torch.tensor(trajectory_dict['rewards'])).item())
            self.update()
            self.memory.clear()

            if episode % 1000 == 0:
                print("Episode {}, epsilon: {}, reward:{}".format(episode, self.epsilon, sum(self.reward_buffer) / len(
                    self.reward_buffer)))
                self.reward_buffer.clear()
                torch.save(self.eval.state_dict(), self.get_path())
