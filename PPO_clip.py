from PPO import PPO
import torch
import torch.nn.functional as F


class PPOClip(PPO):
    def __init__(self, env_, gamma_, alpha_, explosion_step_, epsilon_):
        super(PPOClip, self).__init__(env_, gamma_, alpha_, explosion_step_, epsilon_)
        self.eps = 0.2

        self.name = 'PPOClip'

    def update(self):
        for trajectory in self.memory:
            states, actions, rewards, next_states, dones = trajectory['states'], trajectory['actions'], trajectory[
                'rewards'], trajectory['next_states'], trajectory['dones']
            states, actions, rewards, next_states, dones_ = self.numpy2tensor(states, actions, rewards, next_states,
                                                                              dones)

            # 计算优势函数
            values = self.critic(states)
            targets = rewards + self.gamma * self.critic(next_states) * (1 - dones_)
            deltas = targets - values

            advantage = 0
            advantages = []
            for delta in reversed(deltas):
                advantage = self.gamma * advantage + delta
                advantages.append(advantage)
            advantages.reverse()
            advantages = torch.tensor(advantages, dtype=torch.float)

            log_old_prob = torch.log(self.actor(states).gather(1, actions).view(-1, 1)).detach()

            for step in range(10):
                log_new_prob = torch.log(self.actor(states).gather(1, actions).view(-1, 1))
                ratio = torch.exp(log_new_prob - log_old_prob)
                left = ratio * advantages
                right = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages

                loss_actor = torch.mean(-torch.min(left, right))
                loss_critic = torch.mean(F.mse_loss(targets.detach(), self.critic(states)))

                self.optimizer_critic.zero_grad()
                self.optimizer_actor.zero_grad()

                loss_critic.backward()
                loss_actor.backward()

                self.optimizer_actor.step()
                self.optimizer_critic.step()
