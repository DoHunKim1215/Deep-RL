import gc
import logging
import random
from itertools import count
from typing import Callable, Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn


from agent.agent import Agent
from agent.experience.sars_buffer import PrioritizedReplayBuffer
from agent.experience.tuple import SARSTuple
from agent.exploration_strategy.discrete_action import DiscreteActionStrategy
from agent.net.discrete_action.q_net import DiscreteActionQNetwork
from utils.log import Statistics


class DiscreteActionDDQNWithPER(Agent):
    """
    Double Deep Q-Network (DDQN) for discrete action (DA) space + Prioritized Experience Replay (PER)

    paper: Deep Reinforcement Learning with Double Q-learning (2015)
    authors: Hado van Hasselt, Arthur Guez, David Silver

    paper: Prioritized Experience Replay (2015)
    authors: Tom Schaul, John Quan, Ioannis Antonoglou, David Silver
    """

    def __init__(self,
                 name: str,
                 value_model_fn: Callable[[int, int], DiscreteActionQNetwork],
                 value_optimizer_fn: Callable[[nn.Module, float], torch.optim.Optimizer],
                 value_optimizer_lr: float,
                 training_strategy_fn: Callable[[], DiscreteActionStrategy],
                 evaluation_strategy_fn: Callable[[], DiscreteActionStrategy],
                 experience_buffer_fn: Callable[[], PrioritizedReplayBuffer],
                 n_warmup_batches: int,
                 target_update_period_n_steps: int,
                 tau: float,
                 max_gradient_norm: float,
                 make_env_fn: Callable[[Any], gym.Env],
                 make_env_kwargs: dict,
                 seed: int,
                 gamma: float,
                 params_out_path: str,
                 video_out_path: str):
        super(DiscreteActionDDQNWithPER, self).__init__(name=name,
                                                        make_env_fn=make_env_fn,
                                                        make_env_kwargs=make_env_kwargs,
                                                        gamma=gamma,
                                                        seed=seed,
                                                        params_out_path=params_out_path,
                                                        video_out_path=video_out_path)

        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.online_model = None
        self.target_model = None
        self.value_optimizer = None

        self.max_gradient_norm = max_gradient_norm
        self.tau = tau

        self.training_strategy_fn = training_strategy_fn
        self.training_strategy = None
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.evaluation_strategy = None

        self.experience_buffer_fn = experience_buffer_fn
        self.experience_buffer = None

        self.n_warmup_batches = n_warmup_batches
        self.target_update_period_n_steps = target_update_period_n_steps

    def optimize_model(self, idxs: np.ndarray, weights: np.ndarray, experiences: SARSTuple):
        states, actions, rewards, next_states, is_failures = experiences.decompose()

        argmax_a_q_sp = self.online_model(next_states).detach().max(dim=1)[1]
        q_sp = self.target_model(next_states).detach()
        max_a_q_sp = q_sp[np.arange(q_sp.shape[0]), argmax_a_q_sp].unsqueeze(1)
        target_q_s = rewards + self.gamma * max_a_q_sp * (1. - is_failures)
        q_sa = self.online_model(states).gather(dim=1, index=actions.type(torch.LongTensor).to(self.online_model.device))

        weights = torch.from_numpy(weights).to(torch.float32).to(self.online_model.device)
        td_errors = q_sa - target_q_s
        value_loss = (weights * td_errors).pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), self.max_gradient_norm)
        self.value_optimizer.step()

        priorities = np.abs(td_errors.detach().cpu().numpy())
        self.experience_buffer.update(idxs, priorities)

    def interaction_step(self, state: np.ndarray, env: gym.Env):
        action = self.training_strategy.select_action(self.online_model, state)
        new_state, reward, is_terminal, is_truncated, _ = env.step(action)
        self.experience_buffer.store(state, action, float(reward), new_state, float(is_terminal and not is_truncated))
        self.stats.add_one_step_data(float(reward), self.training_strategy.exploratory_action_taken)
        return new_state, is_terminal or is_truncated

    def update_target_model(self, tau: float):
        for target, online in zip(self.target_model.parameters(), self.online_model.parameters()):
            target.data.copy_((1.0 - tau) * target.data + tau * online.data)

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
        self.online_model = self.value_model_fn(nS, nA)
        self.target_model = self.value_model_fn(nS, nA)
        assert isinstance(self.online_model, DiscreteActionQNetwork)
        assert isinstance(self.target_model, DiscreteActionQNetwork)
        self.update_target_model(tau=1.0)
        self.replay_model = self.online_model
        self.value_optimizer = self.value_optimizer_fn(self.online_model, self.value_optimizer_lr)

        self.training_strategy = self.training_strategy_fn()
        assert isinstance(self.training_strategy, DiscreteActionStrategy)
        self.evaluation_strategy = self.evaluation_strategy_fn()
        assert isinstance(self.evaluation_strategy, DiscreteActionStrategy)

        self.experience_buffer = self.experience_buffer_fn()
        assert isinstance(self.experience_buffer, PrioritizedReplayBuffer)

        self.stats = Statistics(max_episodes=max_episodes,
                                max_minutes=max_minutes,
                                goal_mean_100_reward=goal_mean_100_reward,
                                log_period_n_secs=log_period_n_secs,
                                logger=logger)
        self.stats.start_training()

        logger.info(f'{self.name} Training start. (seed: {self.seed})')

        min_samples = self.experience_buffer.batch_size * self.n_warmup_batches
        for episode in range(0, max_episodes):
            state, _ = env.reset()
            self.stats.prepare_before_episode()

            for _ in count():
                state, is_terminal = self.interaction_step(state, env)

                if len(self.experience_buffer) >= min_samples:
                    idxs, weights, experiences = self.experience_buffer.sample()
                    experiences.load(self.online_model.device)
                    self.optimize_model(idxs, weights, experiences)

                if self.stats.get_total_steps() % self.target_update_period_n_steps == 0:
                    self.update_target_model(tau=self.tau)

                if is_terminal:
                    gc.collect()
                    break

            # stats
            self.stats.calculate_elapsed_time()

            evaluation_score, _ = self.evaluate(self.online_model, env)
            self.stats.append_evaluation_score(evaluation_score)
            self.save_checkpoint(episode, self.online_model)

            if self.stats.process_after_episode(episode):
                break

        final_eval_score, score_std = self.evaluate(self.online_model, env, n_episodes=100)
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
                action = self.evaluation_strategy.select_action(model, state)
                state, reward, is_terminated, is_truncated, _ = env.step(action)
                rs[-1] += reward
                if is_terminated or is_truncated:
                    break
        return np.mean(rs), np.std(rs)

    def render(self, model: nn.Module, env: gym.Env):
        frames = []
        state, _ = env.reset()
        for _ in count():
            action = self.evaluation_strategy.select_action(model, state)
            state, reward, is_terminated, is_truncated, _ = env.step(action)
            frames.append(env.render())
            if is_terminated or is_truncated:
                break
        return frames
