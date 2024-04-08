from collections import deque

from DQN import DQN
import numpy as np
import torch
import torch.nn.functional as F
from SumTree import SumTree


class DQNPER(DQN):
    '''
    DQNPER class: DQN with Prioritized Experience Replay Memory
    '''

    def __init__(self, env_, gamma_, alpha_, explosion_step_, epsilon_):
        super(DQNPER, self).__init__(env_, gamma_, alpha_, explosion_step_, epsilon_)
        self.memory = SumTree(capacity=1024)
        self.beta = 0.2
        self.beta_increment_per_sampling = 0.01
        self.name = 'DQNPER'

    def update(self, state_, action_, reward_, next_state_, dones_, priorities, idxes):
        state_, action_, reward_, next_state_, dones_ = self.numpy2tensor(state_, action_, reward_, next_state_, dones_)
        priorities = torch.tensor(priorities, dtype=torch.float).to(self.device)

        min_priority = self.memory.min() if self.memory.min() else 1
        ISWeights = torch.pow(priorities / min_priority, -self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        value = self.eval(state_).gather(1, action_)
        next_value, _ = torch.max(self.target(next_state_), dim=1)
        target = reward_ + self.gamma * next_value.view(-1, 1).detach() * (1 - dones_)

        self.optimizer.zero_grad()

        delta = target - value
        self.memory.node_num[idxes] = np.abs(delta.squeeze().detach().cpu().numpy())
        self.memory.update()

        # tot_loss = 0
        # for i in range(len(ISWeights)):
        #     self.memory.node_num[idxes[i]] = np.abs(delta[i].detach().numpy())
        #     loss = (ISWeights[i] * delta.detach()[i] * self.eval(state_[i])[action_[i]]).to(self.device)
        #     loss.backward()
        #     tot_loss += loss.item()

        loss = torch.mean(ISWeights * F.mse_loss(target, value)).to(self.device)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, episodes_, pretrain=False):
        if pretrain:
            self.load_model()
        max_reward = -100000000
        rewards_set = deque(maxlen=1000)

        for episode in range(episodes_):
            state = self.env.reset()
            loss_sum = 0
            sum_reward = 0
            t = 0

            while True:
                action = self.choose_action(state, self.epsilon)
                next_state, reward, done, info = self.env.step(action)

                if t == 0:
                    self.memory.add(1, state, int(action), reward, next_state, done)
                else:
                    self.memory.add(self.memory.max(), state, int(action), reward, next_state, done)
                t += 1

                if self.memory.size > 200:
                    states, rewards, actions, next_states, dones, priorities, idxes = self.memory.sample(self.load_size)
                    loss_sum = loss_sum + self.update(states, rewards, actions, next_states, dones, priorities, idxes)

                state = next_state
                sum_reward = sum_reward + reward

                if done:
                    break
            self.reward_buffer.append(sum_reward)
            self.loss_buffer.append(loss_sum)
            rewards_set.append(sum_reward)

            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)
            if episode % 100 == 0 and episode != 0:
                self.target.load_state_dict(self.eval.state_dict())

            if max_reward < sum_reward:
                max_reward = sum_reward
                torch.save(self.eval.state_dict(), self.get_path())

            if episode % 1000 == 0 and episode != 0:
                print("Episode {}, epsilon: {}, loss: {}, reward:{}".format(episode, self.epsilon, loss_sum,
                                                                            sum(rewards_set) / len(rewards_set)))
