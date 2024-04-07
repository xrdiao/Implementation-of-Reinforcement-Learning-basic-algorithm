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
        self.beta = 0.5
        self.name = 'DQNPER'

    def update(self, state_, action_, reward_, next_state_, dones_, priorities, idxes):
        state_, action_, reward_, next_state_, dones_ = self.numpy2tensor(state_, action_, reward_, next_state_, dones_)
        priorities = torch.tensor(priorities, dtype=torch.float).to(self.device)
        min_priority = self.memory.min()
        if min_priority == 0:
            min_priority = 0.001
        ISWeights = np.power(priorities / min_priority, -self.beta)

        value = self.eval(state_).gather(1, action_)
        next_value, _ = torch.max(self.target(next_state_), dim=1)
        target = reward_ + self.gamma * next_value.view(-1, 1).detach() * (1 - dones_)
        delta = target - value

        tot_loss = 0
        self.optimizer.zero_grad()
        for i in range(len(ISWeights)):
            self.memory.node_num[idxes[i]] = np.abs(delta[i].detach().numpy())
            loss = -(ISWeights[i] * delta.detach()[i] * self.eval(state_[i])[action_[i]]).to(self.device)
            loss.backward()
            tot_loss += loss.item()
        self.optimizer.step()
        if self.memory.size >= 1024:
            print(1)
        return tot_loss

    def train(self, episodes_, pretrain=False):
        if pretrain:
            self.load_model()

        for episode in range(episodes_):
            state = self.env.reset()
            loss_sum = 0
            sum_reward = 0
            t = 0

            while True:
                action = self.choose_action(state, self.epsilon)
                next_state, reward, done, info = self.env.step(action)

                data_dict = {'state': state, 'action': int(action), 'reward': reward, 'next_state': next_state,
                             'done': done}
                if t == 0:
                    self.memory.add(1, data_dict)
                else:
                    self.memory.add(self.memory.max_priority(), data_dict)

                if self.memory.get_size() > 100:
                    states, rewards, actions, next_states, dones, priorities, idxes = self.memory.sample(self.load_size)
                    loss_sum = loss_sum + self.update(states, rewards, actions, next_states, dones, priorities, idxes)

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
