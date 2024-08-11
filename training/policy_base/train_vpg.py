import argparse
import logging

import torch
import gymnasium as gym

from agent.experience.lrev_buffer import ExhaustingLREVBuffer
from agent.net.discrete_action.policy_net import DAFCSP
from agent.net.value_net import FCV
from agent.policy_base.vpg import DiscreteActionVPG


def train_vpg(args: argparse.Namespace, env_info: dict, seed: int, device: torch.device, logger: logging.Logger):
    policy_model_fn = lambda nS, nA: DAFCSP(nS, nA, device=device, hidden_dims=(128, 64))
    policy_model_max_grad_norm = 1
    policy_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
    policy_optimizer_lr = 0.0005

    value_model_fn = lambda nS: FCV(nS, device=device, hidden_dims=(256, 128))
    value_model_max_grad_norm = float('inf')
    value_optimizer_fn = lambda net, lr: torch.optim.RMSprop(net.parameters(), lr=lr)
    value_optimizer_lr = 0.0007

    make_env_fn = lambda env_name: gym.make(env_name, render_mode='rgb_array')
    experience_buffer_fn = lambda: ExhaustingLREVBuffer()

    entropy_loss_weight = 0.001
    gamma = 1.0

    agent = DiscreteActionVPG(name=args.model_name,
                              policy_model_fn=policy_model_fn,
                              policy_model_max_grad_norm=policy_model_max_grad_norm,
                              policy_optimizer_fn=policy_optimizer_fn,
                              policy_optimizer_lr=policy_optimizer_lr,
                              value_model_fn=value_model_fn,
                              value_model_max_grad_norm=value_model_max_grad_norm,
                              value_optimizer_fn=value_optimizer_fn,
                              value_optimizer_lr=value_optimizer_lr,
                              entropy_loss_weight=entropy_loss_weight,
                              experience_buffer_fn=experience_buffer_fn,
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
