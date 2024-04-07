import gym

from Q_learning import QLearning
from Sarsa import Sarsa
from DQN import DQN
from DDQN import DDQN
from DuelingDQN import DuelingDQN
from PG import PolicyGradient
from PPO import PPO
from PPO_clip import PPOClip
from DDPG import DDPG
from PPO_continuous import PPOContinuous


def test(learn_method, env_name):
    episodes = 5000
    explosion_step = 100
    max_epsilon = 1
    epsilon = max_epsilon
    gamma = 0.99
    alpha = 0.7
    render = True

    env = gym.make(env_name)
    agent = learn_method(env, gamma, alpha, explosion_step, epsilon)

    print('------------' + agent.name + '--------------')
    # agent.train(episodes, pretrain=False)
    agent.test(3, render=render)
    # method.plot_reward()


if __name__ == '__main__':
    # test(QLearning, 'Taxi-v3')
    # test(Sarsa, 'Taxi-v3')
    # test(DQN, 'CartPole-v1')
    # test(DDQN, 'CartPole-v1')
    # test(DuelingDQN, 'CartPole-v1')
    # test(PolicyGradient, env_name='CartPole-v1')
    # test(PPO, 'CartPole-v1')
    # test(PPOClip, 'CartPole-v1')
    # test(PPOContinuous, 'Pendulum-v0')
    # test(PPOContinuous, 'MountainCarContinuous-v0')
    # test(DDPG, 'Pendulum-v0')
    test(DDPG, 'MountainCarContinuous-v0')
