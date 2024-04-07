import numpy as np
import sys


class SumTree:
    def __init__(self, capacity):
        assert capacity % 2 == 0, 'capacity must be 2^N'
        self.capacity = capacity
        self.node_num = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity).tolist()
        self.idx = 0
        self.size = 0

    def add(self, v, *parms):
        data = [parm for parm in parms]

        if self.idx > self.capacity - 1:
            self.idx = 0
        self.node_num[self.capacity - 1 + self.idx] = v
        self.data[self.idx] = data
        self.idx += 1
        if self.size <= self.capacity - 1:
            self.size += 1
        self.update()

    def get(self, v):
        assert v <= self.node_num[0], 'The value is out of range.'

        idx = 0
        while True:
            if idx >= len(self.node_num) - self.capacity:
                return self.node_num[idx], self.data[idx - self.capacity + 1], idx
            if v <= self.node_num[2 * idx + 1]:
                idx = 2 * idx + 1
            else:
                v = v - self.node_num[2 * idx + 1]
                idx = 2 * (idx + 1)

    def min(self):
        priority = self.node_num[self.capacity - 1:]
        return min(priority)

    def sample(self, batch_size):
        interval = self.node_num[0] / batch_size

        states, rewards, actions, next_states, dones, priorities, idxes = [], [], [], [], [], [], []

        for b in range(batch_size):
            sample_num = np.random.uniform(b * interval, (b + 1) * interval)
            priority, data, idx = self.get(sample_num)
            data = data[0]

            states.append(data['state'])
            rewards.append(data['reward'])
            actions.append(data['action'])
            next_states.append(data['next_state'])
            dones.append(data['done'])
            priorities.append(priority)
            idxes.append(idx)

        return states, rewards, actions, next_states, dones, priorities, idxes

    def max(self):
        priority = self.node_num[self.capacity - 1:]
        return max(priority)

    def update(self):
        capacity = self.capacity
        idx = len(self.node_num) - 1
        while True:
            capacity = int(0.5 * capacity)
            reversed_layer = np.zeros(capacity)
            if capacity >= 2:
                for j in range(capacity):
                    reversed_layer[j] = self.node_num[idx] + self.node_num[idx - 1]
                    idx -= 2
                reversed_layer = reversed_layer[::-1]
                self.node_num[idx - capacity + 1:idx + 1] = reversed_layer

            else:
                self.node_num[0] = self.node_num[1] + self.node_num[2]
                break

    def get_size(self):
        return self.size


if __name__ == '__main__':
    capa = 1024
    st = SumTree(capa)
    # num = [3, 10, 12, 4, 1, 2, 8, 2]
    num = np.random.rand(capa) * 3

    for i, n in enumerate(num):
        data_ = {'num': i}
        st.add(n, data_)
    st.max_priority()
    print(st.get(2))
    # print(st.sample(6))
    print(1)
