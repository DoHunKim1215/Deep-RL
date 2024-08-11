import gc
import logging
import random
from itertools import count
from typing import Callable, Any

import numpy as np
import torch
from torch import nn

from agent.agent import Agent
import gymnasium as gym

from agent.experience.lrev_buffer import ExhaustingLRBuffer
from agent.net.discrete_action.policy_net import DiscreteActionPolicyNetwork
from utils.log import Statistics


class DiscreteActionREINFORCE(Agent):
    """
    REINFORCE for discrete action space
    paper: Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. (1992)
    authors: Ronald J. Williams
    """

    def __init__(self,
                 name: str,
                 policy_model_fn: Callable[[int, int], DiscreteActionPolicyNetwork],
                 policy_optimizer_fn: Callable[[nn.Module, float], torch.optim.Optimizer],
                 policy_optimizer_lr: float,
                 experience_buffer_fn: Callable[[], ExhaustingLRBuffer],
                 make_env_fn: Callable[[Any], gym.Env],
                 make_env_kwargs: dict,
                 seed: int,
                 gamma: float,
                 params_out_path: str,
                 video_out_path: str):
        super(DiscreteActionREINFORCE, self).__init__(name=name,
                                                      make_env_fn=make_env_fn,
                                                      make_env_kwargs=make_env_kwargs,
                                                      gamma=gamma,
                                                      seed=seed,
                                                      params_out_path=params_out_path,
                                                      video_out_path=video_out_path)

        self.policy_model_fn = policy_model_fn
        self.policy_model = None

        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr
        self.policy_optimizer = None

        self.experience_buffer_fn = experience_buffer_fn
        self.experience_buffer = None

    def optimize_model(self):
        T = len(self.experience_buffer)
        log_probs, rewards = self.experience_buffer.sample()

        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:T - t] * rewards[t:]) for t in range(T)])
        discounts = torch.FloatTensor(discounts).unsqueeze(1).to(self.policy_model.device)
        returns = torch.FloatTensor(returns).unsqueeze(1).to(self.policy_model.device)

        policy_loss = -(discounts * returns * log_probs).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    def interaction_step(self, state, env):
        action, is_exploratory, log_prob, _ = self.policy_model.select_action_info(state)
        new_state, reward, is_terminal, is_truncated, _ = env.step(action)
        self.experience_buffer.store(log_prob, reward)
        self.stats.add_one_step_data(float(reward), is_exploratory)
        return new_state, is_terminal or is_truncated

    def train(self, max_minutes: int, max_episodes: int, goal_mean_100_reward: int,
              log_period_n_secs: int, logger: logging.Logger):
        # initialize seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # create training environment
        env: gym.Env = self.make_env_fn(**self.make_env_kwargs)
        env.reset(seed=self.seed)
        assert isinstance(env.action_space, gym.spaces.Discrete)

        nS, nA = env.observation_space.shape[0], env.action_space.n
        self.policy_model = self.policy_model_fn(nS, nA)
        self.replay_model = self.policy_model
        assert isinstance(self.policy_model, DiscreteActionPolicyNetwork)
        self.policy_optimizer = self.policy_optimizer_fn(self.policy_model, self.policy_optimizer_lr)

        self.experience_buffer = self.experience_buffer_fn()
        assert isinstance(self.experience_buffer, ExhaustingLRBuffer)

        self.stats = Statistics(max_episodes=max_episodes,
                                max_minutes=max_minutes,
                                goal_mean_100_reward=goal_mean_100_reward,
                                log_period_n_secs=log_period_n_secs,
                                logger=logger)
        self.stats.start_training()

        logger.info(f'{self.name} Training start. (seed: {self.seed})')

        for episode in range(max_episodes):
            state, _ = env.reset()
            self.stats.prepare_before_episode()

            for _ in count():
                state, is_terminal = self.interaction_step(state, env)
                if is_terminal:
                    gc.collect()
                    break

            self.optimize_model()
            self.experience_buffer.clear()

            # stats
            self.stats.calculate_elapsed_time()

            evaluation_score, _ = self.evaluate(self.policy_model, env)
            self.stats.append_evaluation_score(evaluation_score)
            self.save_checkpoint(episode, self.policy_model)

            if self.stats.process_after_episode(episode):
                break

        final_eval_score, score_std = self.evaluate(self.policy_model, env, n_episodes=100)
        result, training_time, wallclock_time = self.stats.get_total()
        logger.info('Training complete.')
        logger.info('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time, {:.2f}s wall-clock time.\n'
                    .format(final_eval_score, score_std, training_time, wallclock_time))

        env.close()
        del env

        self.get_cleaned_checkpoints()
        return result, final_eval_score

    def evaluate(self, model: nn.Module, env: gym.Env, n_episodes: int = 1):
        rs = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            rs.append(0)
            for _ in count():
                action = model.select_greedy_action(state)
                state, reward, is_terminated, is_truncated, _ = env.step(action)
                rs[-1] += reward
                if is_terminated or is_truncated:
                    break
        return np.mean(rs), np.std(rs)

    def render(self, model: nn.Module, env: gym.Env):
        frames = []
        state, _ = env.reset()
        for _ in count():
            action = model.select_greedy_action(state)
            state, reward, is_terminated, is_truncated, _ = env.step(action)
            frames.append(env.render())
            if is_terminated or is_truncated:
                break
        return frames
