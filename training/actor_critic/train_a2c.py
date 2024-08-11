import argparse
import logging

import torch
import gymnasium as gym

from agent.actor_critic.a2c import DiscreteActionA2C
from agent.experience.lrev_buffer import ExhaustingBatchLREVBuffer
from agent.multiprocess.multiprocess_env import MultiprocessEnv
from agent.net.discrete_action.actor_critic_net import DAFCSPV


device = torch.device("cpu")


def ac_model_fn(nS, nA):
    return DAFCSPV(nS, nA, device=device, hidden_dims=(256, 128))


def ac_optimizer_fn(net, lr):
    return torch.optim.RMSprop(net.parameters(), lr=lr)


def make_envs_fn(mef, mea, s, n):
    return MultiprocessEnv(mef, mea, s, n)


def make_env_fn(env_name):
    return gym.make(env_name, render_mode='rgb_array')


def experience_buffer_fn():
    return ExhaustingBatchLREVBuffer()


def train_a2c(args: argparse.Namespace, env_info: dict, seed: int, logger: logging.Logger):
    ac_model_max_grad_norm = 1
    ac_optimizer_lr = 0.001

    policy_loss_weight = 1.0
    value_loss_weight = 0.6

    entropy_loss_weight = 0.001

    max_n_steps = 10
    n_workers = 24
    tau = 0.95
    gamma = 0.99

    agent = DiscreteActionA2C(name=args.model_name,
                              ac_model_fn=ac_model_fn,
                              ac_model_max_grad_norm=ac_model_max_grad_norm,
                              ac_optimizer_fn=ac_optimizer_fn,
                              ac_optimizer_lr=ac_optimizer_lr,
                              policy_loss_weight=policy_loss_weight,
                              value_loss_weight=value_loss_weight,
                              entropy_loss_weight=entropy_loss_weight,
                              max_n_steps=max_n_steps,
                              n_workers=n_workers,
                              tau=tau,
                              experience_buffer_fn=experience_buffer_fn,
                              make_envs_fn=make_envs_fn,
                              make_env_fn=make_env_fn,
                              make_env_kwargs={'env_name': args.env_name},
                              seed=seed,
                              gamma=gamma,
                              params_out_path=args.params_out_path,
                              video_out_path=args.video_out_path)

    result, final_eval_score = agent.train(max_minutes=env_info['max_minutes'],
                                           max_episodes=env_info['max_episodes'],
                                           goal_mean_100_reward=env_info['goal_mean_100_reward'],
                                           log_period_n_secs=args.log_period_n_secs,
                                           logger=logger)

    return agent, result, final_eval_score
