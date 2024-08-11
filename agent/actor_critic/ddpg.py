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
from agent.exploration_strategy.continuous_action import ContinuousActionStrategy
from agent.net.continuous_action.policy_net import ContinuousActionDeterministicPolicyNetwork
from agent.net.continuous_action.q_net import ContinuousActionQNetwork
from utils.log import Statistics


class ContinuousActionDDPG(Agent):
    """
    Deep Deterministic Policy Gradient (DDPG) for continuous action spaces.
    paper: Continuous control with deep reinforcement learning (2015)
    authors: Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra
    link: https://arxiv.org/pdf/1509.02971
    """

    def __init__(self,
                 name: str,
                 policy_model_fn: Callable[[int, Any], ContinuousActionDeterministicPolicyNetwork],
                 policy_max_grad_norm: float,
                 policy_optimizer_fn: Callable[[nn.Module, float], torch.optim.Optimizer],
                 policy_optimizer_lr: float,
                 value_model_fn: Callable[[int, int], ContinuousActionQNetwork],
                 value_max_grad_norm: float,
                 value_optimizer_fn: Callable[[nn.Module, float], torch.optim.Optimizer],
                 value_optimizer_lr: float,
                 training_strategy_fn: Callable[[Any], ContinuousActionStrategy],
                 evaluation_strategy_fn: Callable[[Any], ContinuousActionStrategy],
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
        super(ContinuousActionDDPG, self).__init__(name=name,
                                                   make_env_fn=make_env_fn,
                                                   make_env_kwargs=make_env_kwargs,
                                                   gamma=gamma,
                                                   seed=seed,
                                                   params_out_path=params_out_path,
                                                   video_out_path=video_out_path)

        self.policy_model_fn = policy_model_fn
        self.target_policy_model = None
        self.online_policy_model = None

        self.policy_max_grad_norm = policy_max_grad_norm
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr
        self.policy_optimizer = None

        self.value_model_fn = value_model_fn
        self.target_value_model = None
        self.online_value_model = None

        self.value_max_grad_norm = value_max_grad_norm
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.value_optimizer = None

        self.training_strategy_fn = training_strategy_fn
        self.training_strategy = None
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.evaluation_strategy = None

        self.experience_buffer_fn = experience_buffer_fn
        self.experience_buffer = None

        self.min_samples = 0
        self.n_warmup_batches = n_warmup_batches
        self.update_target_every_steps = update_target_every_steps
        self.tau = tau

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences.decompose()

        # training q-network
        argmax_a_q_sp = self.target_policy_model(next_states)
        max_a_q_sp = self.target_value_model(next_states, argmax_a_q_sp)
        target_q_sa = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)
        q_sa = self.online_value_model(states, actions)
        td_error = q_sa - target_q_sa.detach()
        value_loss = td_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_value_model.parameters(), self.value_max_grad_norm)
        self.value_optimizer.step()

        # training policy network
        argmax_a_q_s = self.online_policy_model(states)
        max_a_q_s = self.online_value_model(states, argmax_a_q_s)
        policy_loss = -max_a_q_s.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_policy_model.parameters(), self.policy_max_grad_norm)
        self.policy_optimizer.step()

    def interaction_step(self, state: np.ndarray, env: gym.Env):
        action, ratio_noise_injected = self.training_strategy.select_action(
            self.online_policy_model, state, len(self.experience_buffer) < self.min_samples
        )
        new_state, reward, terminated, truncated, _ = env.step(action)
        self.experience_buffer.store(state, action, reward, new_state, float(terminated and not truncated))
        self.stats.add_one_step_data(float(reward), ratio_noise_injected)
        return new_state, terminated or truncated

    def update_networks(self, tau: float):
        """
        Polyak averaging
        """
        for target, online in zip(self.target_value_model.parameters(), self.online_value_model.parameters()):
            target.data.copy_((1.0 - tau) * target.data + tau * online.data)
        for target, online in zip(self.target_policy_model.parameters(), self.online_policy_model.parameters()):
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

        self.target_value_model = self.value_model_fn(nS, nA)
        self.online_value_model = self.value_model_fn(nS, nA)
        assert isinstance(self.target_value_model, ContinuousActionQNetwork)
        self.target_policy_model = self.policy_model_fn(nS, action_bounds)
        self.online_policy_model = self.policy_model_fn(nS, action_bounds)
        self.replay_model = self.online_policy_model
        assert isinstance(self.target_policy_model, ContinuousActionDeterministicPolicyNetwork)
        self.update_networks(tau=1.0)
        self.value_optimizer = self.value_optimizer_fn(self.online_value_model, self.value_optimizer_lr)
        self.policy_optimizer = self.policy_optimizer_fn(self.online_policy_model, self.policy_optimizer_lr)

        self.experience_buffer = self.experience_buffer_fn()
        assert isinstance(self.experience_buffer, ReplayBuffer)
        self.training_strategy = self.training_strategy_fn(action_bounds)
        assert isinstance(self.training_strategy, ContinuousActionStrategy)
        self.evaluation_strategy = self.evaluation_strategy_fn(action_bounds)
        assert isinstance(self.evaluation_strategy, ContinuousActionStrategy)

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
                    experiences.load(self.target_policy_model.device)
                    self.optimize_model(experiences)

                if self.stats.get_total_steps() % self.update_target_every_steps == 0:
                    self.update_networks(tau=self.tau)

                if is_terminal:
                    gc.collect()
                    break

            self.stats.calculate_elapsed_time()
            evaluation_score, _ = self.evaluate(self.online_policy_model, env)
            self.stats.append_evaluation_score(evaluation_score)
            self.save_checkpoint(episode, self.online_policy_model)

            if self.stats.process_after_episode(episode):
                break

        final_eval_score, score_std = self.evaluate(self.online_policy_model, env, n_episodes=100)
        result, training_time, wallclock_time = self.stats.get_total()
        logger.info('Training complete.')
        logger.info('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time, {:.2f}s wall-clock time.\n'
                    .format(final_eval_score, score_std, training_time, wallclock_time))

        env.close()
        del env

        self.get_cleaned_checkpoints()
        return result, final_eval_score

    def evaluate(self, model: ContinuousActionDeterministicPolicyNetwork, env: gym.Env, n_episodes=1):
        result = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            result.append(0)
            for _ in count():
                action, _ = self.evaluation_strategy.select_action(model, state)
                state, reward, terminated, truncated, _ = env.step(action)
                result[-1] += reward
                if terminated or truncated:
                    break
        return np.mean(result), np.std(result)

    def render(self, model: ContinuousActionDeterministicPolicyNetwork, env: gym.Env):
        frames = []
        state, _ = env.reset()
        for _ in count():
            action, _ = self.evaluation_strategy.select_action(model, state)
            state, reward, terminated, truncated, _ = env.step(action)
            frames.append(env.render())
            if terminated or truncated:
                break
        return frames
