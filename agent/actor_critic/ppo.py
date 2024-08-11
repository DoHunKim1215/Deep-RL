import logging
import random
import time
from itertools import count
from typing import Callable, Any

import numpy as np
import torch
import gymnasium as gym
from torch import nn

from agent.agent import Agent
from agent.experience.episode_buffer import EpisodeBuffer
from agent.multiprocess.multiprocess_env import MultiprocessEnv
from agent.net.discrete_action.policy_net import DiscreteActionPolicyNetwork
from agent.net.value_net import ValueNetwork


class DiscreteActionPPO(Agent):
    EPS = 1e-6

    def __init__(self,
                 name: str,
                 policy_model_fn: Callable[[int, Any], DiscreteActionPolicyNetwork],
                 policy_model_max_grad_norm: float,
                 policy_optimizer_fn: Callable[[nn.Module, float], torch.optim.Optimizer],
                 policy_optimizer_lr: float,
                 policy_optimization_epochs: int,
                 policy_sample_ratio: float,
                 policy_clip_range: float,
                 policy_stopping_kl: float,
                 value_model_fn: Callable[[int], ValueNetwork],
                 value_model_max_grad_norm: float,
                 value_optimizer_fn: Callable[[nn.Module, float], torch.optim.Optimizer],
                 value_optimizer_lr: float,
                 value_optimization_epochs: int,
                 value_sample_ratio: float,
                 value_clip_range: float,
                 value_stopping_mse: float,
                 episode_buffer_fn: Callable[..., EpisodeBuffer],
                 max_buffer_episodes: int,
                 max_buffer_episode_steps: int,
                 entropy_loss_weight: float,
                 tau: float,
                 n_workers: int,
                 make_envs_fn: Callable[[Callable, dict, int, int], MultiprocessEnv],
                 make_env_fn: Callable[[Any], gym.Env],
                 make_env_kwargs: dict,
                 seed: int,
                 gamma: float,
                 params_out_path: str,
                 video_out_path: str):
        super(DiscreteActionPPO, self).__init__(name=name,
                                                make_env_fn=make_env_fn,
                                                make_env_kwargs=make_env_kwargs,
                                                gamma=gamma,
                                                seed=seed,
                                                params_out_path=params_out_path,
                                                video_out_path=video_out_path)
        assert n_workers > 1
        assert max_buffer_episodes >= n_workers

        self.make_envs_fn = make_envs_fn

        self.policy_model_fn = policy_model_fn
        self.policy_model_max_grad_norm = policy_model_max_grad_norm
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr
        self.policy_optimization_epochs = policy_optimization_epochs
        self.policy_sample_ratio = policy_sample_ratio
        self.policy_clip_range = policy_clip_range
        self.policy_stopping_kl = policy_stopping_kl
        self.policy_model = None
        self.policy_optimizer = None

        self.value_model_fn = value_model_fn
        self.value_model_max_grad_norm = value_model_max_grad_norm
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.value_optimization_epochs = value_optimization_epochs
        self.value_sample_ratio = value_sample_ratio
        self.value_clip_range = value_clip_range
        self.value_stopping_mse = value_stopping_mse
        self.value_model = None
        self.value_optimizer = None

        self.episode_buffer_fn = episode_buffer_fn
        self.max_buffer_episodes = max_buffer_episodes
        self.max_buffer_episode_steps = max_buffer_episode_steps
        self.episode_buffer = None

        self.entropy_loss_weight = entropy_loss_weight
        self.tau = tau
        self.n_workers = n_workers

    def optimize_model(self):
        states, actions, returns, gaes, log_probs = self.episode_buffer.get_stacks()
        values = self.value_model(states).squeeze().detach()
        gaes = (gaes - gaes.mean()) / (gaes.std() + self.EPS)
        n_samples = len(actions)

        for _ in range(self.policy_optimization_epochs):
            batch_size = int(self.policy_sample_ratio * n_samples)
            batch_idxs = np.random.choice(n_samples, batch_size, replace=False)
            states_batch = states[batch_idxs]
            actions_batch = actions[batch_idxs]
            gaes_batch = gaes[batch_idxs]
            log_probs_batch = log_probs[batch_idxs]

            log_probs_pred, entropies_pred = self.policy_model.get_predictions(states_batch, actions_batch)

            ratios = (log_probs_pred - log_probs_batch).exp()
            pi_obj = gaes_batch * ratios
            pi_obj_clipped = gaes_batch * ratios.clamp(1.0 - self.policy_clip_range, 1.0 + self.policy_clip_range)
            policy_loss = -torch.min(pi_obj, pi_obj_clipped).mean()
            entropy_loss = -entropies_pred.mean() * self.entropy_loss_weight

            self.policy_optimizer.zero_grad()
            (policy_loss + entropy_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.policy_model_max_grad_norm)
            self.policy_optimizer.step()

            with torch.no_grad():
                log_probs_pred_all, _ = self.policy_model.get_predictions(states, actions)
                kl = (log_probs - log_probs_pred_all).mean()
                if kl.item() > self.policy_stopping_kl:
                    break

        for _ in range(self.value_optimization_epochs):
            batch_size = int(self.value_sample_ratio * n_samples)
            batch_idxs = np.random.choice(n_samples, batch_size, replace=False)
            states_batch = states[batch_idxs]
            returns_batch = returns[batch_idxs]
            values_batch = values[batch_idxs]

            values_pred = self.value_model(states_batch).squeeze()
            values_pred_clipped = values_batch + (values_pred - values_batch).clamp(-self.value_clip_range,
                                                                                    self.value_clip_range)
            v_loss = (returns_batch - values_pred).pow(2)
            v_loss_clipped = (returns_batch - values_pred_clipped).pow(2)
            value_loss = torch.max(v_loss, v_loss_clipped).mul(0.5).mean()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.value_model_max_grad_norm)
            self.value_optimizer.step()

            with torch.no_grad():
                values_pred_all = self.value_model(states).squeeze()
                mse = (values - values_pred_all).pow(2).mul(0.5).mean()
                if mse.item() > self.value_stopping_mse:
                    break

    def train(self, max_minutes: int, max_episodes: int, goal_mean_100_reward: int,
              log_period_n_secs: int, logger: logging.Logger):
        training_start, last_debug_time = time.time(), float('-inf')

        env = self.make_env_fn(**self.make_env_kwargs)
        env.reset(seed=self.seed)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        envs = self.make_envs_fn(self.make_env_fn, self.make_env_kwargs, self.seed, self.n_workers)
        assert isinstance(envs, MultiprocessEnv)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        nS, nA = env.observation_space.shape[0], env.action_space.n
        episode_timestep_list = []
        episode_reward_list = []
        episode_exploration_list = []
        episode_seconds_list = []
        evaluation_scores_list = []

        self.policy_model = self.policy_model_fn(nS, nA)
        self.replay_model = self.policy_model
        assert isinstance(self.policy_model, DiscreteActionPolicyNetwork)
        self.policy_optimizer = self.policy_optimizer_fn(self.policy_model, self.policy_optimizer_lr)

        self.value_model = self.value_model_fn(nS)
        assert isinstance(self.value_model, ValueNetwork)
        self.value_optimizer = self.value_optimizer_fn(self.value_model, self.value_optimizer_lr)

        self.episode_buffer = self.episode_buffer_fn(nS, self.gamma, self.tau,
                                                     self.n_workers,
                                                     self.max_buffer_episodes,
                                                     self.max_buffer_episode_steps)
        assert isinstance(self.episode_buffer, EpisodeBuffer)

        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0
        episode = 0

        logger.info(f'{self.name} Training start. (seed: {self.seed})')

        while True:
            episode_timestep, episode_reward, episode_exploration, episode_seconds = \
                self.episode_buffer.fill(envs, self.policy_model, self.value_model)

            n_ep_batch = len(episode_timestep)
            episode_timestep_list.extend(episode_timestep)
            episode_reward_list.extend(episode_reward)
            episode_exploration_list.extend(episode_exploration)
            episode_seconds_list.extend(episode_seconds)
            self.optimize_model()
            self.episode_buffer.clear()

            # stats
            evaluation_score, _ = self.evaluate(self.policy_model, env)
            evaluation_scores_list.extend([evaluation_score, ] * n_ep_batch)
            for e in range(episode, episode + n_ep_batch):
                self.save_checkpoint(e, self.policy_model)
            training_time += episode_seconds.sum()

            mean_10_reward = np.mean(episode_reward_list[-10:])
            std_10_reward = np.std(episode_reward_list[-10:])
            mean_100_reward = np.mean(episode_reward_list[-100:])
            std_100_reward = np.std(episode_reward_list[-100:])
            mean_100_eval_score = np.mean(evaluation_scores_list[-100:])
            std_100_eval_score = np.std(evaluation_scores_list[-100:])
            mean_100_exp_rat = np.mean(episode_exploration_list[-100:])
            std_100_exp_rat = np.std(episode_exploration_list[-100:])

            total_step = int(np.sum(episode_timestep_list))
            wallclock_elapsed = time.time() - training_start
            result[episode:episode + n_ep_batch] = total_step, mean_100_reward, \
                mean_100_eval_score, training_time, wallclock_elapsed

            episode += n_ep_batch

            # debug stuff
            reached_debug_time = time.time() - last_debug_time >= log_period_n_secs
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            reached_max_episodes = episode + self.max_buffer_episodes >= max_episodes
            reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
            training_is_over = reached_max_minutes or reached_max_episodes or reached_goal_mean_reward

            if reached_debug_time or training_is_over:
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
                debug_message = '[{}] EP {:04}, STEP {:07}, '
                debug_message += 'REWARD[-10:] {:05.1f}\u00B1{:05.1f}, '
                debug_message += 'REWARD[-100:] {:05.1f}\u00B1{:05.1f}, '
                debug_message += 'EXP_RATE[-100:] {:02.1f}\u00B1{:02.1f}, '
                debug_message += 'EVAL[-100:] {:05.1f}\u00B1{:05.1f}'
                debug_message = debug_message.format(
                    elapsed_str, episode - 1, total_step, mean_10_reward, std_10_reward,
                    mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                    mean_100_eval_score, std_100_eval_score)
                logger.info(debug_message)
                last_debug_time = time.time()

            if training_is_over:
                logger.info('--> reached max_minutes {} max_episodes {} goal_mean_reward {}'
                            .format('(o)' if reached_max_minutes else '(x)',
                                    '(o)' if reached_max_episodes else '(x)',
                                    '(o)' if reached_goal_mean_reward else '(x)'))
                break

        final_eval_score, score_std = self.evaluate(self.policy_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start
        logger.info('Training complete.')
        logger.info('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time, {:.2f}s wall-clock time.\n'
                    .format(final_eval_score, score_std, training_time, wallclock_time))

        env.close()
        del env
        envs.close()
        del envs

        self.get_cleaned_checkpoints()
        return result, final_eval_score

    def evaluate(self, model, env, n_episodes=1):
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

    def render(self, model, env):
        frames = []
        state, _ = env.reset()
        for _ in count():
            action = model.select_greedy_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            frames.append(env.render())
            if terminated or truncated:
                break
        return frames
