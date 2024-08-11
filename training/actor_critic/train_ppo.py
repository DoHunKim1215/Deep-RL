import argparse
import logging

import torch
import gymnasium as gym

from agent.actor_critic.ppo import DiscreteActionPPO
from agent.experience.episode_buffer import EpisodeBuffer
from agent.multiprocess.multiprocess_env import MultiprocessEnv
from agent.net.discrete_action.policy_net import DAFCSP
from agent.net.value_net import FCV


def train_ppo(args: argparse.Namespace, env_info: dict, seed: int, device: torch.device, logger: logging.Logger):
    policy_model_fn = lambda nS, nA: DAFCSP(nS, nA, device=device, hidden_dims=(128, 64))
    policy_model_max_grad_norm = float('inf')
    policy_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
    policy_optimizer_lr = 0.0003
    policy_optimization_epochs = 80
    policy_sample_ratio = 0.8
    policy_clip_range = 0.1
    policy_stopping_kl = 0.02

    value_model_fn = lambda nS: FCV(nS, device=device, hidden_dims=(256, 128))
    value_model_max_grad_norm = float('inf')
    value_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
    value_optimizer_lr = 0.0005
    value_optimization_epochs = 80
    value_sample_ratio = 0.8
    value_clip_range = float('inf')
    value_stopping_mse = 25

    make_env_fn = lambda env_name: gym.make(env_name, render_mode='rgb_array')
    make_envs_fn = lambda mef, mea, s, n: MultiprocessEnv(mef, mea, s, n)
    episode_buffer_fn = lambda sd, g, t, nw, me, mes: EpisodeBuffer(sd, g, t, nw, me, mes, device=device)
    max_buffer_episodes = 48
    max_buffer_episode_steps = 1000

    entropy_loss_weight = 0.01
    tau = 0.97
    n_workers = 24
    gamma = 0.99

    agent = DiscreteActionPPO(name=args.model_name,
                              policy_model_fn=policy_model_fn,
                              policy_model_max_grad_norm=policy_model_max_grad_norm,
                              policy_optimizer_fn=policy_optimizer_fn,
                              policy_optimizer_lr=policy_optimizer_lr,
                              policy_optimization_epochs=policy_optimization_epochs,
                              policy_sample_ratio=policy_sample_ratio,
                              policy_clip_range=policy_clip_range,
                              policy_stopping_kl=policy_stopping_kl,
                              value_model_fn=value_model_fn,
                              value_model_max_grad_norm=value_model_max_grad_norm,
                              value_optimizer_fn=value_optimizer_fn,
                              value_optimizer_lr=value_optimizer_lr,
                              value_optimization_epochs=value_optimization_epochs,
                              value_sample_ratio=value_sample_ratio,
                              value_clip_range=value_clip_range,
                              value_stopping_mse=value_stopping_mse,
                              episode_buffer_fn=episode_buffer_fn,
                              max_buffer_episodes=max_buffer_episodes,
                              max_buffer_episode_steps=max_buffer_episode_steps,
                              entropy_loss_weight=entropy_loss_weight,
                              tau=tau,
                              n_workers=n_workers,
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
