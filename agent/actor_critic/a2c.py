import logging
import random
import time
from itertools import count
from typing import Callable, Any

import numpy as np
import torch
from torch import nn

from agent.agent import Agent
import gymnasium as gym

from agent.experience.lrev_buffer import ExhaustingBatchLREVBuffer
from agent.multiprocess.multiprocess_env import MultiprocessEnv
from agent.net.discrete_action.actor_critic_net import DiscreteActionActorCriticNetwork
from utils.log import MultiEnvStatistics


class DiscreteActionA2C(Agent):
    """
    (Synchronous) Advantage Actor Critic (A2C) for discrete action space
    paper: Asynchronous Methods for Deep Reinforcement Learning (PMLR 2016)
    link: https://arxiv.org/pdf/1602.01783
    authors: Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu
    """

    def __init__(self,
                 name: str,
                 ac_model_fn: Callable[[int, int], DiscreteActionActorCriticNetwork],
                 ac_model_max_grad_norm: float,
                 ac_optimizer_fn: Callable[[nn.Module, float], torch.optim.Optimizer],
                 ac_optimizer_lr: float,
                 policy_loss_weight: float,
                 value_loss_weight: float,
                 entropy_loss_weight: float,
                 max_n_steps: int,
                 n_workers: int,
                 tau: float,
                 experience_buffer_fn: Callable[[], ExhaustingBatchLREVBuffer],
                 make_envs_fn: Callable[[Callable, dict, int, int], MultiprocessEnv],
                 make_env_fn: Callable[[Any], gym.Env],
                 make_env_kwargs: dict,
                 seed: int,
                 gamma: float,
                 params_out_path: str,
                 video_out_path: str):
        super(DiscreteActionA2C, self).__init__(name=name,
                                                make_env_fn=make_env_fn,
                                                make_env_kwargs=make_env_kwargs,
                                                gamma=gamma,
                                                seed=seed,
                                                params_out_path=params_out_path,
                                                video_out_path=video_out_path)
        assert n_workers > 1
        self.ac_model_fn = ac_model_fn
        self.ac_model = None

        self.ac_model_max_grad_norm = ac_model_max_grad_norm
        self.ac_optimizer_fn = ac_optimizer_fn
        self.ac_optimizer_lr = ac_optimizer_lr
        self.ac_optimizer = None

        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
        self.entropy_loss_weight = entropy_loss_weight

        self.max_n_steps = max_n_steps
        self.n_workers = n_workers
        self.tau = tau

        self.experience_buffer_fn = experience_buffer_fn
        self.experience_buffer = None

        self.make_envs_fn = make_envs_fn

    def optimize_model(self, next_values):
        log_probs, rewards, entropies, values = self.experience_buffer.sample()

        if len(rewards.shape) == 1:
            rewards = rewards.reshape(1, -1)
        if len(values.shape) == 1:
            values = values.reshape(1, -1)

        rewards = np.concatenate([rewards, next_values.squeeze().reshape(1, -1)], axis=0)
        values = torch.cat([values, torch.Tensor(next_values).squeeze().unsqueeze(0).to(self.ac_model.device)], dim=0)

        # n-step TD
        T = len(rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([
            [np.sum(discounts[:T - t] * rewards[t:, w]) for t in range(T)] for w in range(self.n_workers)
        ])
        returns = torch.FloatTensor(returns.T[:-1]).reshape(-1).unsqueeze(1).to(self.ac_model.device)
        assert returns.size() == ((T - 1) * self.n_workers, 1)

        # discounted GAE
        np_values = values.detach().cpu().numpy()
        tau_discounts = np.logspace(0, T - 1, num=T - 1, base=self.gamma * self.tau, endpoint=False)
        advantages = rewards[:-1] + self.gamma * np_values[1:] - np_values[:-1]
        gaes = np.array([
            [np.sum(tau_discounts[:T - 1 - t] * advantages[t:, w]) for t in range(T - 1)] for w in range(self.n_workers)
        ])
        discounted_gaes = discounts[:-1] * gaes
        discounted_gaes = torch.FloatTensor(discounted_gaes.T).reshape(-1).unsqueeze(1).to(self.ac_model.device)

        values = values[:-1, ...].view(-1).unsqueeze(1)
        log_probs = log_probs.view(-1).unsqueeze(1)
        entropies = entropies.view(-1).unsqueeze(1)

        row_size = (T - 1) * self.n_workers
        assert values.size() == (row_size, 1)
        assert log_probs.size() == (row_size, 1)
        assert entropies.size() == (row_size, 1)

        value_error = returns.detach() - values
        value_loss = value_error.pow(2).mul(0.5).mean()
        policy_loss = -(discounted_gaes.detach() * log_probs).mean()
        entropy_loss = -entropies.mean()
        loss = (self.policy_loss_weight * policy_loss +
                self.value_loss_weight * value_loss +
                self.entropy_loss_weight * entropy_loss)

        self.ac_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_model.parameters(), self.ac_model_max_grad_norm)
        self.ac_optimizer.step()

    def interaction_step(self, states, envs):
        actions, is_exploratory, log_probs, entropies, values = self.ac_model.select_action_info(states)
        new_states, rewards, terminated, truncated = envs.step(actions)
        self.experience_buffer.store(log_probs, rewards, entropies, values)
        self.stats.add_one_step_data(rewards, is_exploratory)
        return new_states, np.logical_or(terminated, truncated), truncated

    def train(self, max_minutes: int, max_episodes: int, goal_mean_100_reward: int,
              log_period_n_secs: int, logger: logging.Logger):
        env = self.make_env_fn(**self.make_env_kwargs)
        env.reset(seed=self.seed)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        envs = self.make_envs_fn(self.make_env_fn, self.make_env_kwargs, self.seed, self.n_workers)
        assert isinstance(envs, MultiprocessEnv)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        nS, nA = env.observation_space.shape[0], env.action_space.n

        self.ac_model = self.ac_model_fn(nS, nA)
        self.replay_model = self.ac_model
        assert isinstance(self.ac_model, DiscreteActionActorCriticNetwork)
        self.ac_optimizer = self.ac_optimizer_fn(self.ac_model, self.ac_optimizer_lr)

        self.experience_buffer = self.experience_buffer_fn()
        assert isinstance(self.experience_buffer, ExhaustingBatchLREVBuffer)

        self.stats = MultiEnvStatistics(n_workers=self.n_workers,
                                        max_episodes=max_episodes,
                                        max_minutes=max_minutes,
                                        goal_mean_100_reward=goal_mean_100_reward,
                                        log_period_n_secs=log_period_n_secs,
                                        logger=logger)
        self.stats.start_training()

        states = envs.reset_all()

        logger.info(f'{self.name} Training start. (seed: {self.seed})')

        # collect n_steps rollout
        episode, n_steps_start = 0, 0
        for step in count(start=1):
            states, terminated, truncated = self.interaction_step(states, envs)

            if terminated.sum() or step - n_steps_start == self.max_n_steps:
                is_failure = np.logical_and(terminated, np.logical_not(truncated))
                next_values = self.ac_model.evaluate_state(states).detach().cpu().numpy() * (1 - is_failure)
                self.optimize_model(next_values)
                self.experience_buffer.clear()
                n_steps_start = step

            # stats
            if terminated.sum():
                episode_done = time.time()
                self.save_checkpoint(episode, self.ac_model)
                evaluation_score, _ = self.evaluate(self.ac_model, env)

                for i in range(self.n_workers):
                    if terminated[i]:
                        states[i] = envs.reset(rank=i)
                        self.stats.reset_worker_finishing_episode(episode, i, evaluation_score, episode_done)
                        episode += 1

                if self.stats.process_after_episode_done(episode, terminated):
                    break

        final_eval_score, score_std = self.evaluate(self.ac_model, env, n_episodes=100)
        result, training_time, wallclock_time = self.stats.get_result()
        logger.info('Training complete.')
        logger.info('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time, {:.2f}s wall-clock time.\n'
                    .format(final_eval_score, score_std, training_time, wallclock_time))

        env.close()
        del env
        envs.close()
        del envs

        self.get_cleaned_checkpoints()
        return result, final_eval_score

    def evaluate(self, model: DiscreteActionActorCriticNetwork, env: gym.Env, n_episodes=1):
        results = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            results.append(0)
            for _ in count():
                action = model.select_greedy_action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                results[-1] += reward
                if terminated or truncated:
                    break
        return np.mean(results), np.std(results)

    def render(self, model, env) -> list:
        frames = []
        state, _ = env.reset()
        for _ in count():
            action = model.select_greedy_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            frames.append(env.render())
            if terminated or truncated:
                break
        return frames
