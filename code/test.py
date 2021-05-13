# import gym, time
# import matplotlib.pyplot as plt
# import torch as th
# from torch import nn
import time
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import os 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# #For frames : render_mode="rgb_array"
# param = {"num_levels": 1, "distribution_mode": "easy", "render_mode": "human"}
# env = gym.make("procgen:procgen-fruitbot-v0", **param)
# # env = gym.make('CartPole-v1')
# obs = env.reset()
# while True:
#     action = env.action_space.sample()
#     obs, rewards, done, info = env.step(action)
#     env.render()
#     plt.show()
#     time.sleep(0.2)
#     print("_")
    

#     if done:
#         print(rewards, done)
#         break

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor

log_dir = "log/"
os.makedirs(log_dir, exist_ok=True)

param = {"num_levels": 1, "distribution_mode": "easy", "render_mode": "human"}
env = gym.make("procgen:procgen-fruitbot-v0", **param)
env = Monitor(env, log_dir)

# env = gym.make('CartPole-v1')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=5000)


obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    # time.sleep(0.5)
env.close()

results_plotter.plot_results([log_dir], 10e6, results_plotter.X_TIMESTEPS, "Breakout")





