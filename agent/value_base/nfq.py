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
from agent.experience.sars_buffer import ExhaustingSARSBuffer
from agent.experience.tuple import SARSTuple
from agent.exploration_strategy.discrete_action import DiscreteActionStrategy
from agent.net.discrete_action.q_net import DiscreteActionQNetwork
from utils.log import Statistics


class DiscreteActionNFQ(Agent):
    """
    Neural Fitted Q iteration (NFQ) for discrete action (DA) space
    paper: Neural Fitted Q Iteration – First Experiences with a Data Efficient Neural Reinforcement Learning Method (ECML 2005)
    authors: Martin Riedmiller
    """

    def __init__(self,
                 name: str,
                 value_model_fn: Callable[[int, int], DiscreteActionQNetwork],
                 value_optimizer_fn: Callable[[nn.Module, float], torch.optim.Optimizer],
                 value_optimizer_lr: float,
                 training_strategy_fn: Callable[[], DiscreteActionStrategy],
                 evaluation_strategy_fn: Callable[[], DiscreteActionStrategy],
                 experience_buffer_fn: Callable[[], ExhaustingSARSBuffer],
                 batch_size: int,
                 epochs: int,
                 make_env_fn: Callable[[Any], gym.Env],
                 make_env_kwargs: dict,
                 seed: int,
                 gamma: float,
                 params_out_path: str,
                 video_out_path: str):
        super(DiscreteActionNFQ, self).__init__(name=name,
                                                make_env_fn=make_env_fn,
                                                make_env_kwargs=make_env_kwargs,
                                                gamma=gamma,
                                                seed=seed,
                                                params_out_path=params_out_path,
                                                video_out_path=video_out_path)

        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.value_model = None
        self.value_optimizer = None

        self.training_strategy_fn = training_strategy_fn
        self.training_strategy = None
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.evaluation_strategy = None

        self.experience_buffer_fn = experience_buffer_fn
        self.experience_buffer = None

        self.batch_size = batch_size
        self.epochs = epochs

    def optimize_model(self, experiences: SARSTuple):
        states, actions, rewards, next_states, is_failures = experiences.decompose()

        max_a_q_sp = self.value_model(next_states).detach().max(dim=1)[0].unsqueeze(dim=1)
        target_q_s = rewards + self.gamma * max_a_q_sp * (1. - is_failures)
        q_sa = self.value_model(states).gather(dim=1, index=actions.type(torch.LongTensor).to(self.value_model.device))

        td_errors = q_sa - target_q_s
        value_loss = td_errors.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def interaction_step(self, state: np.ndarray, env: gym.Env):
        action = self.training_strategy.select_action(self.value_model, state)
        new_state, reward, is_terminal, is_truncated, _ = env.step(action)
        self.experience_buffer.store(state, action, float(reward), new_state, float(is_terminal and not is_truncated))
        self.stats.add_one_step_data(float(reward), self.training_strategy.exploratory_action_taken)
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
        self.value_model = self.value_model_fn(nS, nA)
        assert isinstance(self.value_model, DiscreteActionQNetwork)
        self.replay_model = self.value_model
        self.value_optimizer = self.value_optimizer_fn(self.value_model, self.value_optimizer_lr)

        self.training_strategy = self.training_strategy_fn()
        assert isinstance(self.training_strategy, DiscreteActionStrategy)
        self.evaluation_strategy = self.evaluation_strategy_fn()
        assert isinstance(self.evaluation_strategy, DiscreteActionStrategy)

        self.experience_buffer = self.experience_buffer_fn()
        assert isinstance(self.experience_buffer, ExhaustingSARSBuffer)

        self.stats = Statistics(max_episodes=max_episodes,
                                max_minutes=max_minutes,
                                goal_mean_100_reward=goal_mean_100_reward,
                                log_period_n_secs=log_period_n_secs,
                                logger=logger)
        self.stats.start_training()

        logger.info(f'{self.name} Training start. (seed: {self.seed})')

        for episode in range(0, max_episodes):
            state, _ = env.reset()
            self.stats.prepare_before_episode()

            for _ in count():
                state, is_terminal = self.interaction_step(state, env)

                if len(self.experience_buffer) >= self.batch_size:
                    experiences = self.experience_buffer.sample()
                    experiences.load(self.value_model.device)
                    for _ in range(self.epochs):
                        self.optimize_model(experiences)
                    self.experience_buffer.clear()

                if is_terminal:
                    gc.collect()
                    break

            # stats
            self.stats.calculate_elapsed_time()

            evaluation_score, _ = self.evaluate(self.value_model, env)
            self.stats.append_evaluation_score(evaluation_score)
            self.save_checkpoint(episode, self.value_model)

            if self.stats.process_after_episode(episode):
                break

        final_eval_score, score_std = self.evaluate(self.value_model, env, n_episodes=100)
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
