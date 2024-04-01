from Q_learning import QLearning


class Sarsa(QLearning):
    def __init__(self, env_, gamma_, alpha_, explosion_step_, epsilon_):
        super().__init__(env_, gamma_, alpha_, explosion_step_, epsilon_)

    def learn(self, state_, reward_, action_, next_state_):
        '''
        :param state_: current state
        :param reward_: reward received
        :param action_: action taken
        :param next_state_: next state

        Q(s,a) = Q(s,a) + alpha * (reward + gamma * Q(s',a') - Q(s,a)), a' = pi(a|s)
        '''
        #  要注意状态的使用，弄清楚哪个地方需要的是当前状态还是下一个状态，我在写代码的时候混淆了两者。
        next_action = self.choose_action(next_state_, 0)
        target = reward_ + self.gamma * self.q_table[next_state_, next_action]
        self.q_table[state_, action_] = self.q_table[state_, action_] + self.alpha * (
                target - self.q_table[state_, action_])
