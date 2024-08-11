import argparse

import gymnasium as gym
import torch

from agent.actor_critic.a3c import DiscreteActionA3C
from agent.experience.lrev_buffer import ExhaustingLREVBuffer
from agent.multiprocess.shared_optim import SharedAdam, SharedRMSprop
from agent.net.discrete_action.policy_net import DAFCSP
from agent.net.value_net import FCV


device = torch.device("cpu")


def policy_model_fn(nS, nA):
    return DAFCSP(nS, nA, device=device, hidden_dims=(128, 64))


def policy_optimizer_fn(net, lr):
    return SharedAdam(net.parameters(), lr=lr)


def value_model_fn(nS):
    return FCV(nS, device=device, hidden_dims=(256, 128))


def value_optimizer_fn(net, lr):
    return SharedRMSprop(net.parameters(), lr=lr)


def make_env_fn(env_name):
    return gym.make(env_name, render_mode='rgb_array')


def experience_buffer_fn():
    return ExhaustingLREVBuffer()


def train_a3c(args: argparse.Namespace, env_info: dict, seed: int):
    policy_model_max_grad_norm = 1
    policy_optimizer_lr = 0.0005
    value_model_max_grad_norm = float('inf')
    value_optimizer_lr = 0.0007

    entropy_loss_weight = 0.001

    max_n_steps = 50
    n_workers = 8
    tau = 1.0
    gamma = 1.0

    agent = DiscreteActionA3C(name=args.model_name,
                              policy_model_fn=policy_model_fn,
                              policy_model_max_grad_norm=policy_model_max_grad_norm,
                              policy_optimizer_fn=policy_optimizer_fn,
                              policy_optimizer_lr=policy_optimizer_lr,
                              value_model_fn=value_model_fn,
                              value_model_max_grad_norm=value_model_max_grad_norm,
                              value_optimizer_fn=value_optimizer_fn,
                              value_optimizer_lr=value_optimizer_lr,
                              entropy_loss_weight=entropy_loss_weight,
                              max_n_steps=max_n_steps,
                              n_workers=n_workers,
                              tau=tau,
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
                                           log_period_n_secs=args.log_period_n_secs)

    return agent, result, final_eval_score
