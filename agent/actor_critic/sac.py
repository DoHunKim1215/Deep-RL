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

from agent.experience.sars_buffer import ReplayBuffer
from agent.net.continuous_action.policy_net import ContinuousActionStochasticPolicyNetwork
from agent.net.continuous_action.q_net import ContinuousActionQNetwork
from utils.log import Statistics


class ContinuousActionSAC(Agent):
    """
    Soft Actor-Critic (SAC) for continuous action spaces.
    paper: Soft actor-critic: Off policy maximum entropy deep reinforcement learning with a stochastic actor (2018)
    authors: Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine
    link: https://arxiv.org/pdf/1801.01290
    """

    def __init__(self,
                 name: str,
                 policy_model_fn: Callable[[int, Any], ContinuousActionStochasticPolicyNetwork],
                 policy_max_grad_norm: float,
                 policy_optimizer_fn: Callable[[nn.Module, float], torch.optim.Optimizer],
                 policy_optimizer_lr: float,
                 value_model_fn: Callable[[int, int], ContinuousActionQNetwork],
                 value_max_grad_norm: float,
                 value_optimizer_fn: Callable[[nn.Module, float], torch.optim.Optimizer],
                 value_optimizer_lr: float,
                 experience_buffer_fn: Callable[[], ReplayBuffer],
                 n_warmup_batches: int,
                 update_target_every_steps: int,
                 tau: float,
                 make_env_fn: Callable[[Any], gym.Env],
                 make_env_kwargs: dict,
                 seed: int,
                 gamma: float,
                 params_out_path: str,
                 video_out_path: str):
        super(ContinuousActionSAC, self).__init__(name=name,
                                                  make_env_fn=make_env_fn,
                                                  make_env_kwargs=make_env_kwargs,
                                                  gamma=gamma,
                                                  seed=seed,
                                                  params_out_path=params_out_path,
                                                  video_out_path=video_out_path)

        self.policy_model_fn = policy_model_fn
        self.policy_model = None

        self.policy_max_grad_norm = policy_max_grad_norm
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr
        self.policy_optimizer = None

        self.value_model_fn = value_model_fn
        self.target_value_a_model = None
        self.online_value_a_model = None
        self.target_value_b_model = None
        self.online_value_b_model = None

        self.value_max_grad_norm = value_max_grad_norm
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.value_a_optimizer = None
        self.value_b_optimizer = None

        self.experience_buffer_fn = experience_buffer_fn
        self.experience_buffer = None

        self.min_samples = 0

        self.update_target_every_steps = update_target_every_steps
        self.n_warmup_batches = n_warmup_batches
        self.tau = tau

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences.decompose()

        # step alpha
        current_actions, log_probs, _ = self.policy_model.select_action_info(states)
        target_alpha = (log_probs + self.policy_model.target_entropy).detach()
        alpha_loss = -(self.policy_model.logalpha * target_alpha).mean()

        self.policy_model.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.policy_model.alpha_optimizer.step()

        alpha = self.policy_model.logalpha.exp()

        # step Q
        next_actions, next_log_probs, _ = self.policy_model.select_action_info(next_states)
        q_spap_a = self.target_value_a_model(next_states, next_actions)
        q_spap_b = self.target_value_b_model(next_states, next_actions)
        q_spap = torch.min(q_spap_a, q_spap_b) - alpha * next_log_probs
        target_q_sa = (rewards + self.gamma * q_spap * (1 - is_terminals)).detach()
        q_sa_a = self.online_value_a_model(states, actions)
        q_sa_b = self.online_value_b_model(states, actions)
        qa_loss = (q_sa_a - target_q_sa).pow(2).mul(0.5).mean()
        qb_loss = (q_sa_b - target_q_sa).pow(2).mul(0.5).mean()

        self.value_a_optimizer.zero_grad()
        qa_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_value_a_model.parameters(), self.value_max_grad_norm)
        self.value_a_optimizer.step()
        self.value_b_optimizer.zero_grad()
        qb_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_value_b_model.parameters(), self.value_max_grad_norm)
        self.value_b_optimizer.step()

        # step policy
        current_q_sa_a = self.online_value_a_model(states, current_actions)
        current_q_sa_b = self.online_value_b_model(states, current_actions)
        current_q_sa = torch.min(current_q_sa_a, current_q_sa_b)
        policy_loss = (alpha * log_probs - current_q_sa).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.policy_max_grad_norm)
        self.policy_optimizer.step()


    def interaction_step(self, state: np.ndarray, env: gym.Env):
        if len(self.experience_buffer) < self.min_samples:
            action, exploration_ratio = self.policy_model.select_random_action(state)
        else:
            action, exploration_ratio = self.policy_model.select_action(state)

        new_state, reward, terminated, truncated, _ = env.step(action)
        self.experience_buffer.store(state, action, reward, new_state, float(terminated and not truncated))
        self.stats.add_one_step_data(float(reward), exploration_ratio)
        return new_state, terminated or truncated

    def update_value_networks(self, tau: float):
        for target, online in zip(self.target_value_a_model.parameters(), self.online_value_a_model.parameters()):
            target.data.copy_((1.0 - tau) * target.data + tau * online.data)
        for target, online in zip(self.target_value_b_model.parameters(), self.online_value_b_model.parameters()):
            target.data.copy_((1.0 - tau) * target.data + tau * online.data)

    def train(self, max_minutes: int, max_episodes: int, goal_mean_100_reward: int,
              log_period_n_secs: int, logger: logging.Logger):
        # initialize environment
        env = self.make_env_fn(**self.make_env_kwargs)
        env.reset(seed=self.seed)
        assert isinstance(env.action_space, gym.spaces.Box)

        # initialize random seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        nS, nA = env.observation_space.shape[0], env.action_space.shape[0]
        action_bounds = env.action_space.low, env.action_space.high

        self.target_value_a_model = self.value_model_fn(nS, nA)
        self.online_value_a_model = self.value_model_fn(nS, nA)
        self.target_value_b_model = self.value_model_fn(nS, nA)
        self.online_value_b_model = self.value_model_fn(nS, nA)
        assert isinstance(self.target_value_a_model, ContinuousActionQNetwork)
        self.update_value_networks(tau=1.0)

        self.policy_model = self.policy_model_fn(nS, action_bounds)
        self.replay_model = self.policy_model
        assert isinstance(self.policy_model, ContinuousActionStochasticPolicyNetwork)

        self.value_a_optimizer = self.value_optimizer_fn(self.online_value_a_model, self.value_optimizer_lr)
        self.value_b_optimizer = self.value_optimizer_fn(self.online_value_b_model, self.value_optimizer_lr)
        self.policy_optimizer = self.policy_optimizer_fn(self.policy_model, self.policy_optimizer_lr)

        self.experience_buffer = self.experience_buffer_fn()
        assert isinstance(self.experience_buffer, ReplayBuffer)

        self.min_samples = self.experience_buffer.batch_size * self.n_warmup_batches

        logger.info(f'{self.name} Training start. (seed: {self.seed})')

        self.stats = Statistics(max_episodes=max_episodes,
                                max_minutes=max_minutes,
                                goal_mean_100_reward=goal_mean_100_reward,
                                log_period_n_secs=log_period_n_secs,
                                logger=logger)
        self.stats.start_training()

        for episode in range(max_episodes):
            self.stats.prepare_before_episode()
            state, _ = env.reset()

            for _ in count():
                state, is_terminal = self.interaction_step(state, env)
                if len(self.experience_buffer) > self.min_samples:
                    experiences = self.experience_buffer.sample()
                    experiences.load(self.policy_model.device)
                    self.optimize_model(experiences)

                if self.stats.get_total_steps() % self.update_target_every_steps == 0:
                    self.update_value_networks(tau=self.tau)

                if is_terminal:
                    gc.collect()
                    break

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

    def evaluate(self, model: ContinuousActionStochasticPolicyNetwork, env: gym.Env, n_episodes=1):
        result = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            result.append(0)
            for _ in count():
                action, _ = model.select_greedy_action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                result[-1] += reward
                if terminated or truncated:
                    break
        return np.mean(result), np.std(result)

    def render(self, model: ContinuousActionStochasticPolicyNetwork, env: gym.Env):
        frames = []
        state, _ = env.reset()
        for _ in count():
            action, _ = model.select_greedy_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            frames.append(env.render())
            if terminated or truncated:
                break
        return frames
