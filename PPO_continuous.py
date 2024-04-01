from PPO_clip import PPOClip


class PPOContinuous(PPOClip):
    def __init__(self, env_, gamma_, alpha_, explosion_step_, epsilon_):
        super(PPOContinuous, self).__init__(env_, gamma_, alpha_, explosion_step_, epsilon_)
