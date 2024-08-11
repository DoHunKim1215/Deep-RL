import argparse
import logging

import torch
import gymnasium as gym

from agent.experience.lrev_buffer import ExhaustingLRBuffer
from agent.net.discrete_action.policy_net import DAFCSP
from agent.policy_base.reinforce import DiscreteActionREINFORCE


def train_reinforce(args: argparse.Namespace, env_info: dict, seed: int, device: torch.device, logger: logging.Logger):
    policy_model_fn = lambda nS, nA: DAFCSP(nS, nA, device=device, hidden_dims=(128, 64))
    policy_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
    policy_optimizer_lr = 0.0005

    make_env_fn = lambda env_name: gym.make(env_name, render_mode='rgb_array')
    experience_buffer_fn = lambda: ExhaustingLRBuffer()

    gamma = 1.0

    agent = DiscreteActionREINFORCE(name=args.model_name,
                                    policy_model_fn=policy_model_fn,
                                    policy_optimizer_fn=policy_optimizer_fn,
                                    policy_optimizer_lr=policy_optimizer_lr,
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
