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

from agent.experience.lrev_buffer import ExhaustingLREVBuffer
from agent.net.discrete_action.policy_net import DiscreteActionPolicyNetwork
from agent.net.value_net import ValueNetwork
from utils.log import Statistics


class DiscreteActionVPG(Agent):
    """
    REINFORCE with baseline (Vanilla Policy Gradient; VPG) for discrete action space
    """

    def __init__(self,
                 name: str,
                 policy_model_fn: Callable[[int, int], DiscreteActionPolicyNetwork],
                 policy_model_max_grad_norm: float,
                 policy_optimizer_fn: Callable[[nn.Module, float], torch.optim.Optimizer],
                 policy_optimizer_lr: float,
                 value_model_fn: [[int], ValueNetwork],
                 value_model_max_grad_norm: float,
                 value_optimizer_fn: Callable[[nn.Module, float], torch.optim.Optimizer],
                 value_optimizer_lr: float,
                 entropy_loss_weight: float,
                 experience_buffer_fn: Callable[[], ExhaustingLREVBuffer],
                 make_env_fn: Callable[[Any], gym.Env],
                 make_env_kwargs: dict,
                 seed: int,
                 gamma: float,
                 params_out_path: str,
                 video_out_path: str):
        super(DiscreteActionVPG, self).__init__(name=name,
                                                make_env_fn=make_env_fn,
                                                make_env_kwargs=make_env_kwargs,
                                                gamma=gamma,
                                                seed=seed,
                                                params_out_path=params_out_path,
                                                video_out_path=video_out_path)

        self.policy_model_fn = policy_model_fn
        self.policy_model = None

        self.policy_model_max_grad_norm = policy_model_max_grad_norm
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr
        self.policy_optimizer = None

        self.value_model_fn = value_model_fn
        self.value_model = None

        self.value_model_max_grad_norm = value_model_max_grad_norm
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.value_optimizer = None

        self.entropy_loss_weight = entropy_loss_weight

        self.experience_buffer_fn = experience_buffer_fn
        self.experience_buffer = None

    def optimize_model(self, last_value: float):
        T = len(self.experience_buffer) + 1
        log_probs, rewards, entropies, values = self.experience_buffer.sample()

        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        rewards = np.append(rewards, last_value)
        returns = np.array([np.sum(discounts[:T - t] * rewards[t:]) for t in range(T)])
        discounts = torch.FloatTensor(discounts[:-1]).unsqueeze(1).to(self.policy_model.device)
        returns = torch.FloatTensor(returns[:-1]).unsqueeze(1).to(self.policy_model.device)

        value_error = returns - values
        policy_loss = -(discounts * value_error.detach() * log_probs).mean()
        entropy_loss = -self.entropy_loss_weight * entropies.mean()
        self.policy_optimizer.zero_grad()
        (policy_loss + entropy_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.policy_model_max_grad_norm)
        self.policy_optimizer.step()

        value_loss = value_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.value_model_max_grad_norm)
        self.value_optimizer.step()

    def interaction_step(self, state, env):
        action, is_exploratory, log_prob, entropy = self.policy_model.select_action_info(state)
        new_state, reward, is_terminal, is_truncated, _ = env.step(action)
        self.experience_buffer.store(log_prob, reward, entropy, self.value_model(state))
        self.stats.add_one_step_data(float(reward), is_exploratory)
        return new_state, is_terminal or is_truncated, is_terminal and not is_truncated

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

        self.value_model = self.value_model_fn(nS)
        assert isinstance(self.value_model, ValueNetwork)
        self.value_optimizer = self.value_optimizer_fn(self.value_model, self.value_optimizer_lr)

        self.experience_buffer = self.experience_buffer_fn()
        assert isinstance(self.experience_buffer, ExhaustingLREVBuffer)

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

            is_failure = False
            for _ in count():
                state, is_terminal, is_failure = self.interaction_step(state, env)
                if is_terminal:
                    gc.collect()
                    break

            last_value = 0 if is_failure else self.value_model(state).detach().item()
            self.optimize_model(last_value)
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
