import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from DQN import DQN


class Actor(nn.Module):
    def __init__(self, action_size, state_size, hidden_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class PolicyGradient(DQN):
    '''
    Policy Gradient，从结果看，确实容易局部收敛和过冲，学习率难以控制
    '''

    def __init__(self, env_, gamma_, alpha_, explosion_step_, epsilon_):
        super(PolicyGradient, self).__init__(env_, gamma_, alpha_, explosion_step_, epsilon_)

        # PG与DQN的区别在于任务的目标，DQN是逼近最优价值函数，PG是找到最多的奖励，其实可以不添加这三行，但为了可读性，还是加上了
        self.actor = Actor(self.action_size, self.state_size, self.hidden_size)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=0.001)  # 这里加一个weight_decay直接就不收敛了，是因为参数本来就不多。
        self.actor.to(self.device)

        self.name = 'PG'

    def choose_action(self, state_, epsilon_):
        state_ = torch.tensor(state_, dtype=torch.float).view(1, -1)
        # 通过随机采用获取动作，不再是DQN中的选取最大值
        probs = self.actor(state_)

        if epsilon_ == 0:
            return torch.argmax(self.actor(state_)).item()

        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def learn(self, state_, action_, reward_, next_state_, dones_):
        state_, action_, reward_, next_state_, dones_ = self.numpy2tensor(state_, action_, reward_, next_state_, dones_)
        G = 0
        loss_sum = 0

        # 从PG的伪代码得到的
        self.optimizer_actor.zero_grad()
        for t in reversed(range(len(state_))):
            G = reward_[len(state_) - 1 - t] + self.gamma * G
            log_prob = torch.log(self.actor(state_).gather(1, action_[t].view(1, -1)))
            loss = - (self.alpha * (self.gamma ** t) * G * log_prob)
            loss.backward()
        self.optimizer_actor.step()

        return loss_sum

    def explore_trajectory(self, episodes_):
        state = self.env.reset()
        trajectory_dict = dict({'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []})

        for t in range(episodes_):
            action = self.choose_action(state, self.epsilon)
            next_state, reward, done, _ = self.env.step(action)

            trajectory_dict['states'].append(state)
            trajectory_dict['actions'].append(action)
            trajectory_dict['rewards'].append(reward)
            trajectory_dict['next_states'].append(next_state)
            trajectory_dict['dones'].append(done)

            state = next_state
            if done:
                break
        return trajectory_dict

    def train(self, episodes_, pretrain=False):
        if pretrain:
            self.load_model()

        # PG的训练是通过一条条完整的路径，所以先收集数据再训练，与DQN不同
        for episode in range(episodes_):
            trajectory_dict = self.explore_trajectory(episodes_)

            # 可以通过memory记录轨迹，然后用重要性方法转换概率，但这里主要是简要实现，memory留到PPO解决
            loss_sum = self.learn(trajectory_dict['states'], trajectory_dict['actions'], trajectory_dict['rewards'],
                                  trajectory_dict['next_states'], trajectory_dict['dones'])
            self.reward_buffer.append(torch.sum(torch.tensor(trajectory_dict['rewards'])).item())

            if episode % 1000 == 0:
                print("Episode {}, epsilon: {}, loss: {}, reward:{}".format(episode, self.epsilon, loss_sum,
                                                                            sum(self.reward_buffer) / len(
                                                                                self.reward_buffer)))
                torch.save(self.eval.state_dict(), self.get_path())
                self.reward_buffer.clear()
