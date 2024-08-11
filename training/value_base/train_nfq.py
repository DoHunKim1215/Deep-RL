import argparse
import logging

import torch
import gymnasium as gym

from agent.experience.sars_buffer import ExhaustingSARSBuffer
from agent.exploration_strategy.discrete_action import EGreedyStrategy, GreedyStrategy
from agent.net.discrete_action.q_net import DAFCQ
from agent.value_base.nfq import DiscreteActionNFQ


def train_nfq(args: argparse.Namespace, env_info: dict, seed: int, device: torch.device, logger: logging.Logger):
    value_model_fn = lambda nS, nA: DAFCQ(nS, nA, device=device, hidden_dims=(512, 128))
    value_optimizer_fn = lambda net, lr: torch.optim.RMSprop(net.parameters(), lr=lr)
    value_optimizer_lr = 0.0005

    training_strategy_fn = lambda: EGreedyStrategy(epsilon=0.5)
    evaluation_strategy_fn = lambda: GreedyStrategy()

    make_env_fn = lambda env_name: gym.make(env_name, render_mode='rgb_array')
    experience_buffer_fn = lambda: ExhaustingSARSBuffer()

    batch_size = 1024
    epochs = 40
    gamma = 1.0

    agent = DiscreteActionNFQ(name=args.model_name,
                              value_model_fn=value_model_fn,
                              value_optimizer_fn=value_optimizer_fn,
                              value_optimizer_lr=value_optimizer_lr,
                              training_strategy_fn=training_strategy_fn,
                              evaluation_strategy_fn=evaluation_strategy_fn,
                              experience_buffer_fn=experience_buffer_fn,
                              batch_size=batch_size,
                              epochs=epochs,
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
