import argparse
import logging

import torch
import gymnasium as gym

from agent.actor_critic.ddpg import ContinuousActionDDPG
from agent.experience.sars_buffer import ReplayBuffer
from agent.exploration_strategy.continuous_action import GaussianNoiseStrategy, GreedyStrategy, \
    DecayingGaussianNoiseStrategy
from agent.net.continuous_action.policy_net import CAFCDP
from agent.net.continuous_action.q_net import CAFCQ


def train_ddpg(args: argparse.Namespace, env_info: dict, seed: int, device: torch.device, logger: logging.Logger):
    policy_model_fn = lambda nS, bounds: CAFCDP(nS, bounds, device=device, hidden_dims=(512, 512, 256))
    policy_max_grad_norm = float('inf')
    policy_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
    policy_optimizer_lr = 0.0003

    value_model_fn = lambda nS, nA: CAFCQ(nS, nA, device=device, hidden_dims=(512, 512, 256))
    value_max_grad_norm = float('inf')
    value_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
    value_optimizer_lr = 0.0003

    training_strategy_fn = lambda bounds: DecayingGaussianNoiseStrategy(bounds,
                                                                        init_noise_ratio=0.5,
                                                                        min_noise_ratio=0.1,
                                                                        decay_steps=200000)
    evaluation_strategy_fn = lambda bounds: GreedyStrategy(bounds)

    make_env_fn = lambda env_name: gym.make(env_name, render_mode='rgb_array')
    experience_buffer_fn = lambda: ReplayBuffer(capacity=1000000, batch_size=256)

    n_warmup_batches = 5
    target_update_period_n_steps = 1
    tau = 0.005
    gamma = 0.99

    agent = ContinuousActionDDPG(name=args.model_name,
                                 policy_model_fn=policy_model_fn,
                                 policy_max_grad_norm=policy_max_grad_norm,
                                 policy_optimizer_fn=policy_optimizer_fn,
                                 policy_optimizer_lr=policy_optimizer_lr,
                                 value_model_fn=value_model_fn,
                                 value_max_grad_norm=value_max_grad_norm,
                                 value_optimizer_fn=value_optimizer_fn,
                                 value_optimizer_lr=value_optimizer_lr,
                                 experience_buffer_fn=experience_buffer_fn,
                                 training_strategy_fn=training_strategy_fn,
                                 evaluation_strategy_fn=evaluation_strategy_fn,
                                 n_warmup_batches=n_warmup_batches,
                                 update_target_every_steps=target_update_period_n_steps,
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
