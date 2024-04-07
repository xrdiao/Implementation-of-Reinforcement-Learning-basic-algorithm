import multiprocessing
import time

import gym
import threading

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
from DQN_PER import DQNPER


def test(learn_method, env_name, episodes=10000, gamma=0.99, explosion_step=100,
         alpha=0.7, epsilon=1, render=True, pretrain=False):
    env = gym.make(env_name)
    agent = learn_method(env, gamma, alpha, explosion_step, epsilon)

    print('------------' + agent.name + '--------------')
    agent.train(episodes, pretrain=pretrain)
    agent.test(3, render=render)
    # method.plot_reward()


def multi_test(learn_methods, env_name, gammas, alphas, episodes=10000, explosion_step=100):
    processes = []
    test_num = len(learn_methods)
    for i in range(test_num):
        process = multiprocessing.Process(target=test,
                                          args=(
                                              learn_methods[i], env_name, episodes, gammas[i], explosion_step,
                                              alphas[i]))
        process.daemon = True  # 守护线程
        processes.append(process)
    for i in range(test_num):
        processes[i].start()
    time.sleep(100000)


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
    # test(DDPG, 'MountainCarContinuous-v0')
    test(DQNPER, 'CartPole-v1')

    # g = [0.99, 0.99, 0.99, 0.99, 0.99]
    # a = [0.7, 0.7, 0.7, 0.7]
    # multi_test(learn_methods=[DQN, DQN, DDQN], env_name='CartPole-v1', gammas=g, alphas=a)
