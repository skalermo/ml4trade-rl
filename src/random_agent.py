import gymnasium


class RandomAgent:
    def __init__(self, env: gymnasium.Env, *args, **kwargs):
        self.env = env
        self.action_space = env.action_space

    def learn(self, total_timesteps: int, *args, **kwargs):
        self.env.reset()
        for _ in range(total_timesteps):
            self.env.step(self.action_space.sample())

    def predict(self, *args, **kwargs):
        return self.action_space.sample(), None
