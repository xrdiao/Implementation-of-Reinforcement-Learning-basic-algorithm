import multiprocessing
import time
import numpy as np
import gym
import threading

from matplotlib import pyplot as plt

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
from SAC import SAC


def test(learn_method, env_name, addition='', episodes=10000, gamma=0.9, explosion_step=100,
         alpha=0.7, epsilon=1, render=True, pretrain=False, times=1):
    rewards_set = None
    losses_set = None
    name = None
    lr = 0

    for t in range(times):
        env = gym.make(env_name)
        agent = learn_method(env, gamma, alpha, explosion_step, epsilon)
        name = agent.name
        lr = agent.learning_rate

        print('------------' + agent.name + ' {}'.format(t) + '--------------')
        agent.train(episodes, pretrain=pretrain)
        reward, loss = agent.get_data()
        if rewards_set is None:
            rewards_set = reward
            losses_set = loss
        else:
            rewards_set = rewards_set + reward
            losses_set = losses_set + loss
        # agent.test(3, render=render)
        # agent.plot_reward_loss(addition)
    losses_set = losses_set / times
    rewards_set = rewards_set / times

    reward_path = './data/' + name + '_rewards' + addition + '{:.5f}'.format(lr)
    loss_path = './data/' + name + '_losses' + addition + '{:.5f}'.format(lr)
    np.save(reward_path, rewards_set)
    np.save(loss_path, losses_set)

    fig_path = './figs/' + name + addition + '{:.5f}'.format(lr) + '.png'
    fig, ax = plt.subplots(2, 1, figsize=(40, 20))
    ax[0].plot(rewards_set)
    ax[0].set_title('rewards')
    ax[0].set_xlabel('episodes')
    ax[1].plot(losses_set)
    ax[1].set_title('losses')
    ax[1].set_xlabel('episodes')
    plt.savefig(fig_path, dpi=200)


def multi_test(learn_methods, env_name, gammas, alphas, episodes=10000, explosion_step=100):
    processes = []
    test_num = len(learn_methods)

    for i in range(test_num):
        process = multiprocessing.Process(target=test,
                                          args=(
                                              learn_methods[i], env_name,
                                              '_gamma{}_alpha{}'.format(gammas[i], alphas[i]), episodes, gammas[i],
                                              explosion_step, alphas[i]))
        process.daemon = True  # 守护线程
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


if __name__ == '__main__':
    # test(QLearning, 'Taxi-v3')
    # test(Sarsa, 'Taxi-v3')
    # test(DQN, 'CartPole-v1')
    # test(DQNPER, 'CartPole-v1')
    # test(DDQN, 'CartPole-v1')
    # test(DuelingDQN, 'CartPole-v1')
    # test(PolicyGradient, env_name='CartPole-v1')
    # test(PPO, 'CartPole-v1')
    # test(PPOClip, 'CartPole-v1')
    # test(PPOContinuous, 'Pendulum-v0')
    # test(PPOContinuous, 'MountainCarContinuous-v0')
    # test(DDPG, 'Pendulum-v0')
    # test(DDPG, 'MountainCarContinuous-v0')
    test(SAC, 'CartPole-v1')

    # 多线程
    g = np.linspace(0.8, 1, 10)
    a = [0.7] * len(g)  # useless

    # agents = [DDQN for _ in range(len(g))]
    # multi_test(learn_methods=agents, env_name='CartPole-v1', gammas=g, alphas=a)
    # agents = [DQN for _ in range(len(g))]
    # multi_test(learn_methods=agents, env_name='CartPole-v1', gammas=g, alphas=a)
    # agents = [DQNPER for _ in range(len(g))]
    # multi_test(learn_methods=agents, env_name='CartPole-v1', gammas=g, alphas=a)
    # agents = [DuelingDQN for _ in range(len(g))]
    # multi_test(learn_methods=agents, env_name='CartPole-v1', gammas=g, alphas=a)

    # agents = [PolicyGradient for _ in range(len(g))]
    # multi_test(learn_methods=agents, env_name='CartPole-v1', gammas=g, alphas=a)

    # agents = [PPO for _ in range(len(g))]
    # multi_test(learn_methods=agents, env_name='CartPole-v1', gammas=g, alphas=a)
