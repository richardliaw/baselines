#!/usr/bin/env python

import os
from mpi4py import MPI
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys

def train(env_id, num_timesteps, seed):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    rank = MPI.COMM_WORLD.Get_rank()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(5 + rank)
    U.make_session(num_cpu=1).__enter__()
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make(env_id)
    if rank != 0: logger.set_level(logger.DISABLED)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=256, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir() and 
        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn, 
            max_timesteps=num_timesteps,
            timesteps_per_batch=80000,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=20, optim_stepsize=1e-4, optim_batchsize=4096,
            gamma=0.995, lam=0.95, schedule='linear',
        )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Humanoid-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    train(args.env, num_timesteps=100e6, seed=args.seed)


if __name__ == '__main__':
    main()
