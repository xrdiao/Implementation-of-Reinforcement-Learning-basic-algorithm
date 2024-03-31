import random
from collections import deque
import numpy as np


# --------------------------------------- #
# 经验回放池
# --------------------------------------- #

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def add_dict(self, trajectory_dict):
        self.buffer.append(trajectory_dict)

    def sample_dict(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return samples

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        # *transitions代表取出列表中的值，解压
        state, action, reward, next_state, done = zip(*samples)
        return np.array(state), action, reward, np.array(next_state), done

    # 目前队列长度
    def size(self):
        return len(self.buffer)
