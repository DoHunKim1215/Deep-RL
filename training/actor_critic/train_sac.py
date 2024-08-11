import argparse
import logging

import torch
import gymnasium as gym

from agent.actor_critic.sac import ContinuousActionSAC
from agent.experience.sars_buffer import ReplayBuffer
from agent.net.continuous_action.policy_net import CAFCGP
from agent.net.continuous_action.q_net import CAFCQ


def train_sac(args: argparse.Namespace, env_info: dict, seed: int, device: torch.device, logger: logging.Logger):
    policy_model_fn = lambda nS, bounds: CAFCGP(nS, bounds, device=device, hidden_dims=(512, 512, 256))
    policy_max_grad_norm = float('inf')
    policy_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
    policy_optimizer_lr = 0.0003

    value_model_fn = lambda nS, nA: CAFCQ(nS, nA, device=device, hidden_dims=(512, 512, 256))
    value_max_grad_norm = float('inf')
    value_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
    value_optimizer_lr = 0.0005

    make_env_fn = lambda env_name: gym.make(env_name, render_mode='rgb_array')
    experience_buffer_fn = lambda: ReplayBuffer(capacity=1000000, batch_size=256)

    n_warmup_batches = 10
    update_target_every_steps = 1
    tau = 0.001
    gamma = 0.99

    agent = ContinuousActionSAC(name=args.model_name,
                                policy_model_fn=policy_model_fn,
                                policy_max_grad_norm=policy_max_grad_norm,
                                policy_optimizer_fn=policy_optimizer_fn,
                                policy_optimizer_lr=policy_optimizer_lr,
                                value_model_fn=value_model_fn,
                                value_max_grad_norm=value_max_grad_norm,
                                value_optimizer_fn=value_optimizer_fn,
                                value_optimizer_lr=value_optimizer_lr,
                                experience_buffer_fn=experience_buffer_fn,
                                n_warmup_batches=n_warmup_batches,
                                update_target_every_steps=update_target_every_steps,
                                tau=tau,
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
