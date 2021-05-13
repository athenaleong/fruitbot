DRIVE_PATH = "/tmp"

import tensorflow as tf
from baselines.ppo2 import ppo2
from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from baselines import logger
from mpi4py import MPI

import time, os, gym


import numpy as np
import tensorflow as tf
from baselines.a2c import utils
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch

def build_model(unscaled_images, depths=[16,32,32], bn=False, **conv_kwargs):
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

    if bn:
      def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = tf.layers.batch_normalization(out)
        out = residual_block(out)
        out = residual_block(out)
        return out
    else:
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



def main(num_envs, num_levels, timesteps_per_proc, bn, log_dir, save_interval=1):
    conv_fn = lambda x: build_model(x, depths=[16,32,32], bn=bn, emb_size=256)

    log_dir = log_dir + '/MultiPPO_{}lvl'.format(num_levels)
    if bn:
        log_dir += "_BN"
    os.makedirs(log_dir, exist_ok=True)
    log_dir = os.path.join(log_dir + "/", time.strftime("%d%m%y_%H:%M:%S", time.localtime()))
    os.makedirs(log_dir, exist_ok=True)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    env_name = "fruitbot"
    distribution_mode = "easy"
    start_level = 0
    test_worker_interval = 0
    is_test_worker=False
    nsteps = 256

    learning_rate = 5e-4
    ent_coef = .01
    gamma = .999
    lam = .95
    nminibatches = 8
    ppo_epochs = 3
    clip_range = .2
    use_vf_clipping = True

    mpi_rank_weight = 0 if is_test_worker else 1
    num_levels = 0 if is_test_worker else num_levels

    if log_dir is not None:
        log_comm = comm.Split(1 if is_test_worker else 0, 0)
        format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
        logger.configure(comm=log_comm, dir=log_dir, format_strs=format_strs)

    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=num_levels, start_level=500, distribution_mode=distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )

    venv = VecNormalize(venv=venv, ob=False)

    eval_env = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=500, start_level=0, distribution_mode=distribution_mode)
    eval_env = VecExtractDictObs(eval_env, "rgb")

    eval_env = VecMonitor(
        venv=eval_env, filename=None, keep_buf=100,
    )

    eval_env = VecNormalize(venv=eval_env, ob=False)


    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    logger.info("training")
    model = ppo2.learn(
        env=venv,
        eval_env=eval_env,
        network=conv_fn,
        total_timesteps=timesteps_per_proc,
        save_interval=save_interval,
        nsteps=nsteps,
        nminibatches=nminibatches,
        lam=lam,
        gamma=gamma,
        noptepochs=ppo_epochs,
        log_interval=1,
        ent_coef=ent_coef,
        mpi_rank_weight=mpi_rank_weight,
        clip_vf=use_vf_clipping,
        comm=comm,
        lr=learning_rate,
        cliprange=clip_range,
        update_fn=None,
        init_fn=None,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    return model


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--num_levels', type=int, default=0)
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('--bn', type=bool, default=False)

    args = parser.parse_args()

    main(num_envs=args.num_envs,
        num_levels=args.num_levels,
        timesteps_per_proc=args.timesteps,
        log_dir=args.log_dir,
        bn=args.bn,
        save_interval=args.save_interval)
