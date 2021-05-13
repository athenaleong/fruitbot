import os, time

import gym
import numpy as np
import matplotlib.pyplot as plt

import stable_baselines
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import PPO2
# from stable_baselines import results_plotter
# from stable_baselines.bench import Monitor
# from stable_baselines import DDPG, TD3, A2C, ACER
# from stable_baselines.ddpg.policies import LnMlpPolicy
# from stable_baselines.bench import Monitor
# from stable_baselines.results_plotter import load_results, ts2xy
# from stable_baselines.common.noise import AdaptiveParamNoiseSpec, NormalActionNoise
# from stable_baselines.common.callbacks import BaseCallback

from stable_baselines import PPO2

print("version:", stable_baselines.__version__)
import tensorflow as tf

def build_impala_cnn(unscaled_images, depths=[16,32,32], **conv_kwargs):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """

    layer_num = 0

    def get_layer_num_str():
        nonlocal layer_num
        num_str = str(layer_num)
        layer_num += 1
        return num_str

    def conv_layer(out, depth):
        return tf.layers.conv2d(out, depth, 3, padding='same', name='layer_' + get_layer_num_str())

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value

        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = tf.cast(unscaled_images, tf.float32) / 255.

    for depth in depths:
        out = conv_sequence(out, depth)

    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu, name='layer_' + get_layer_num_str())

    return out

# PPO2(CnnPolicy, env, verbose=0,
#                 gamma=0.999, n_steps=256, ent_coef=0.01,
#                 learning_rate=5e-4, vf_coef=0.5, max_grad_norm=0.5,
#                 lam=0.95, nminibatches=8, noptepochs=3, cliprange=0.2, cliprange_vf=None,
#                 #verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None))
#                 policy_kwargs={"cnn_extractor": build_impala_cnn})

filename = "bestmodel"
model = PPO2.load(filename)

env_param = {"num_levels": 50, "distribution_mode": "easy"}
env = gym.make("procgen:procgen-fruitbot-v0", **env_param)

obs = env.reset()
# prev_screen = env.render(mode='rgb_array')
# plt.imshow(prev_screen)
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    # screen = env.render(mode='rgb_array')

    # plt.imshow(screen)
    # ipythondisplay.clear_output(wait=True)
    # ipythondisplay.display(plt.gcf())
    if done:
        obs = env.reset()
    # if dones:
    #     env.reset()
    # time.sleep(0.5)
env.close()
# ipythondisplay.clear_output(wait=True)