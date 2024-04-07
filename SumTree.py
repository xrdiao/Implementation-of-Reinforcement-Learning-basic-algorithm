import numpy as np
import matplotlib.pyplot as plt


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.node_num = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity).tolist()
        self.idx = 0

    def add(self, v, data):
        if self.idx > self.capacity - 1:
            self.idx = 0
        self.node_num[self.capacity - 1 + self.idx] = v
        self.data[self.idx] = data
        self.idx += 1
        self.update()

    def get(self, v):
        assert v <= self.node_num[0], 'The value is out of range.'

        idx = 0
        while True:
            if idx >= len(self.node_num) - self.capacity:
                return self.node_num[idx], self.data[idx - self.capacity + 1]
            if v <= self.node_num[2 * idx + 1]:
                idx = 2 * idx + 1
            else:
                v = v - self.node_num[2 * idx + 1]
                idx = 2 * (idx + 1)

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


st = SumTree(8)
num = [3, 10, 12, 4, 1, 2, 8, 2, 4]

for i, n in enumerate(num):
    data_ = {'num': i}
    st.add(n, data_)
print(st.get(2))
print(1)
