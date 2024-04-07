from collections import deque
import matplotlib.pyplot as plt

import numpy as np


class QLearning:
    def __init__(self, env_, gamma_, alpha_, explosion_step_, epsilon_):
        self.env = env_
        self.state_size = env_.observation_space.n
        self.action_size = env_.action_space.n
        self.gamma = gamma_
        self.q_table = np.zeros([self.state_size, self.action_size])
        self.action_size = self.action_size
        self.alpha = alpha_
        self.explosion_step = explosion_step_
        self.reward_buffer = deque(maxlen=10000)
        self.epsilon = epsilon_

        self.min_epsilon = 0.01
        self.decay_rate = 0.01
        self.max_epsilon = 1.0

        self.name = 'QLearning'

    def choose_action(self, state_, epsilon_):
        if np.random.uniform(0, 1) > epsilon_:
            return np.argmax(self.q_table[state_, :])

        return self.env.action_space.sample()

    def learn(self, state_, reward_, action_, next_state_):
        '''
        :param state_: current state
        :param reward_: reward received
        :param action_: action taken
        :param next_state_: next state

        Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a)) - Q(s,a))
        '''
        #  要注意状态的使用，弄清楚哪个地方需要的是当前状态还是下一个状态，我在写代码的时候混淆了两者。
        target = reward_ + self.gamma * np.max(self.q_table[next_state_, :])
        self.q_table[state_, action_] = self.q_table[state_, action_] + self.alpha * (
                target - self.q_table[state_, action_])

    def train(self, episodes, pretrain=False):
        for episode in range(episodes):
            state = self.env.reset()
            rewards = 0

            for _ in range(self.explosion_step):
                action = self.choose_action(state, self.epsilon)

                next_state, reward, done, info = self.env.step(action)
                self.learn(state, reward, action, next_state)

                state = next_state
                rewards += reward

                if done:
                    break
            self.reward_buffer.append(rewards)
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)
            if episode % 1000 == 0:
                print("Episode {}, reward: {}, epsilon: {}".format(episode, rewards, self.epsilon))

    def test(self, episodes, explosion_step=100, render=False):
        for episode in range(episodes):
            state = self.env.reset()
            total_rewards = 0
            print("****************************************************")
            print("EPISODE ", episode)

            for step in range(explosion_step):
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

    def plot_reward_loss(self, addition=''):
        plt.plot(self.reward_buffer)
        plt.show()
