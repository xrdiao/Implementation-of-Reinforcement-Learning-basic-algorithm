import gym
from Q_learning import QLearning
from Sarsa import Sarsa
from DQN import DQN
from DDQN import DDQN


def test(learn_method, env_name):
    episodes = 5000
    explosion_step = 100
    max_epsilon = 1
    epsilon = max_epsilon
    gamma = 0.6
    alpha = 0.7
    render = True

    env = gym.make(env_name)
    method = learn_method(env, gamma, alpha, explosion_step, epsilon)

    method.train(episodes)
    method.test(3, render=render)
    # method.plot_reward()


if __name__ == '__main__':
    # test(QLearning, 'Taxi-v3')
    # test(Sarsa, 'Taxi-v3')
    # test(DQN, 'CartPole-v1')
    test(DDQN, 'CartPole-v1')
