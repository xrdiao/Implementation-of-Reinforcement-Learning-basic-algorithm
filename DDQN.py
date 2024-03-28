import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from DQN import DQN
import torch


class DDQN(DQN):
    def __init__(self, env_, gamma_, alpha_, explosion_step_, epsilon_):
        super(DDQN, self).__init__(env_, gamma_, alpha_, explosion_step_, epsilon_)
        self.name = 'DDQN'
        self.time2switch = 0
        self.switch = False

    def choose_action(self, state_, epsilon_):
        if np.random.uniform(0, 1) > epsilon_:
            if self.switch:
                return torch.argmax(self.target(state_)).item()
            else:
                return torch.argmax(self.eval(state_)).item()

        return self.env.action_space.sample()

    def learn(self, state_, action_, reward_, next_state_, dones_):
        action_ = torch.tensor(action_, dtype=torch.long).view(-1, 1).to(self.device)
        dones_ = torch.tensor(dones_, dtype=torch.long).view(-1, 1).to(self.device)
        reward_ = torch.tensor(reward_, dtype=torch.float).view(-1, 1).to(self.device)

        if self.switch:
            # Q_target(s,a) = reward(s,a) + gamma * Q_eval(s',argmax(Q_target(s',a'))
            _, arg = torch.max(self.target(next_state_), dim=1)
            next_value = self.eval(next_state_).gather(1, arg.view(-1, 1))
            optimizer = optim.Adam(self.target.parameters(), 0.01)
        else:
            # Q_eval(s,a) = reward(s,a) + gamma * Q_target(s',argmax(Q_eval(s',a'))
            _, arg = torch.max(self.eval(next_state_), dim=1)
            next_value = self.target(next_state_).gather(1, arg.view(-1, 1))
            optimizer = optim.Adam(self.eval.parameters(), 0.01)

        target = reward_ + self.gamma * next_value.view(-1, 1).detach() * (1-dones_)
        value = self.eval(state_).gather(1, action_)

        loss = torch.mean(F.mse_loss(target, value))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if self.time2switch % 200 == 0 and self.time2switch > 0:
            self.switch = not self.switch

        self.time2switch += 1
        return loss.item()
