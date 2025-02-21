import gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

class RenderCallback(BaseCallback):
    def __init__(self, env, render_freq=1000, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.env = env
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0: 
            self.env.render()
        return True


train_env = make_vec_env("CartPole-v1", n_envs=1)

model = PPO("MlpPolicy", train_env, verbose=1)


model.learn(total_timesteps=10000, callback=RenderCallback(train_env))

train_env.close()