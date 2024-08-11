import gc
import random
from itertools import count
from typing import Callable, Any

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
from torch import nn

from agent.agent import Agent
from agent.experience.lrev_buffer import ExhaustingLREVBuffer
from agent.net.discrete_action.policy_net import DiscreteActionPolicyNetwork
from agent.net.value_net import ValueNetwork
from utils.log import MultiLearnerStatistics

class DiscreteActionA3C(Agent):
    """
    Asynchronous Advantage Actor Critic (A3C) for discrete action space
    paper: Asynchronous Methods for Deep Reinforcement Learning (PMLR 2016)
    link: https://arxiv.org/pdf/1602.01783
    authors: Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu
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
                 max_n_steps: int,
                 n_workers: int,
                 tau: float,
                 experience_buffer_fn: Callable[[], ExhaustingLREVBuffer],
                 make_env_fn: Callable[[Any], gym.Env],
                 make_env_kwargs: dict,
                 seed: int,
                 gamma: float,
                 params_out_path: str,
                 video_out_path: str):
        super(DiscreteActionA3C, self).__init__(name=name,
                                                make_env_fn=make_env_fn,
                                                make_env_kwargs=make_env_kwargs,
                                                gamma=gamma,
                                                seed=seed,
                                                params_out_path=params_out_path,
                                                video_out_path=video_out_path)
        assert n_workers > 1
        self.policy_model_fn = policy_model_fn
        self.shared_policy_model = None

        self.policy_model_max_grad_norm = policy_model_max_grad_norm
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr
        self.shared_policy_optimizer = None

        self.value_model_fn = value_model_fn
        self.shared_value_model = None

        self.value_model_max_grad_norm = value_model_max_grad_norm
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.shared_value_optimizer = None

        self.entropy_loss_weight = entropy_loss_weight
        self.max_n_steps = max_n_steps
        self.n_workers = n_workers
        self.tau = tau

        self.experience_buffer_fn = experience_buffer_fn

        self.n_active_workers = None
        self.get_out_signal = None

    def optimize_model_using_n_step(self,
                                    experience_buffer: ExhaustingLREVBuffer,
                                    local_policy_model: DiscreteActionPolicyNetwork,
                                    local_value_model: ValueNetwork,
                                    last_value: float):
        T = len(experience_buffer) + 1
        log_probs, rewards, entropies, values = experience_buffer.sample()

        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        rewards = np.append(rewards, last_value)
        returns = np.array([np.sum(discounts[:T - t] * rewards[t:]) for t in range(T)])
        discounts = torch.FloatTensor(discounts[:-1]).unsqueeze(1).to(local_policy_model.device)
        returns = torch.FloatTensor(returns[:-1]).unsqueeze(1).to(local_policy_model.device)

        value_error = returns - values
        policy_loss = -(discounts * value_error.detach() * log_probs).mean()
        entropy_loss = -self.entropy_loss_weight * entropies.mean()
        self.shared_policy_optimizer.zero_grad()
        (policy_loss + entropy_loss).backward()
        torch.nn.utils.clip_grad_norm_(local_policy_model.parameters(), self.policy_model_max_grad_norm)
        for param, shared_param in zip(local_policy_model.parameters(), self.shared_policy_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        self.shared_policy_optimizer.step()
        local_policy_model.load_state_dict(self.shared_policy_model.state_dict())

        value_loss = value_error.pow(2).mul(0.5).mean()
        self.shared_value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_value_model.parameters(), self.value_model_max_grad_norm)
        for param, shared_param in zip(local_value_model.parameters(), self.shared_value_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        self.shared_value_optimizer.step()
        local_value_model.load_state_dict(self.shared_value_model.state_dict())

    def optimize_model_using_gae(self,
                                 experience_buffer: ExhaustingLREVBuffer,
                                 local_policy_model: DiscreteActionPolicyNetwork,
                                 local_value_model: ValueNetwork,
                                 last_value: float):
        """
        Optimize models using generalized advantage estimation (GAE)
        paper: High-Dimensional Continuous Control Using Generalized Advantage Estimation (ICRL 2016)
        authors: John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel
        link: https://arxiv.org/pdf/1506.02438
        """
        T = len(experience_buffer) + 1
        log_probs, rewards, entropies, values = experience_buffer.sample()

        rewards = np.append(rewards, last_value)
        values = torch.cat([values, torch.FloatTensor([[last_value, ], ]).to(local_policy_model.device)], dim=0)

        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:T - t] * rewards[t:]) for t in range(T)])

        np_values = values.view(-1).detach().cpu().numpy()
        tau_discounts = np.logspace(0, T - 1, num=T - 1, base=self.gamma * self.tau, endpoint=False)
        advantages = rewards[:-1] + self.gamma * np_values[1:] - np_values[:-1]
        gaes = np.array([np.sum(tau_discounts[:T - 1 - t] * advantages[t:]) for t in range(T - 1)])

        discounts = torch.FloatTensor(discounts[:-1]).unsqueeze(1).to(local_policy_model.device)
        returns = torch.FloatTensor(returns[:-1]).unsqueeze(1).to(local_policy_model.device)
        gaes = torch.FloatTensor(gaes).unsqueeze(1).to(local_policy_model.device)

        policy_loss = -(discounts * gaes.detach() * log_probs).mean()
        entropy_loss = -entropies.mean()
        loss = policy_loss + self.entropy_loss_weight * entropy_loss
        self.shared_policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_policy_model.parameters(), self.policy_model_max_grad_norm)
        for param, shared_param in zip(local_policy_model.parameters(), self.shared_policy_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        self.shared_policy_optimizer.step()
        local_policy_model.load_state_dict(self.shared_policy_model.state_dict())

        value_error = returns - values[:-1, ...]
        value_loss = value_error.pow(2).mul(0.5).mean()
        self.shared_value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_value_model.parameters(), self.value_model_max_grad_norm)
        for param, shared_param in zip(local_value_model.parameters(), self.shared_value_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        self.shared_value_optimizer.step()
        local_value_model.load_state_dict(self.shared_value_model.state_dict())

    def optimize_model_fn(self) \
            -> Callable[[ExhaustingLREVBuffer, DiscreteActionPolicyNetwork, ValueNetwork, float], None]:
        if self.tau == 1.:
            return self.optimize_model_using_n_step
        else:
            return self.optimize_model_using_gae

    @staticmethod
    def interaction_step(state: np.ndarray,
                         env: gym.Env,
                         policy_model: DiscreteActionPolicyNetwork,
                         value_model: ValueNetwork,
                         experience_buffer: ExhaustingLREVBuffer):
        action, is_exploratory, log_prob, entropy = policy_model.select_action_info(state)
        new_state, reward, is_terminal, is_truncated, _ = env.step(action)
        experience_buffer.store(log_prob, reward, entropy, value_model(state))
        return new_state, reward, is_terminal, is_truncated, is_exploratory

    def work(self, rank):
        # new worker starts to work
        self.n_active_workers.add_(1)

        # initialize environment
        local_seed = self.seed + rank
        env = self.make_env_fn(**self.make_env_kwargs)
        env.reset(seed=local_seed)

        # initialize rank-wise seed
        torch.manual_seed(local_seed)
        np.random.seed(local_seed)
        random.seed(local_seed)

        # set up local model
        nS, nA = env.observation_space.shape[0], env.action_space.n
        local_policy_model = self.policy_model_fn(nS, nA)
        local_policy_model.load_state_dict(self.shared_policy_model.state_dict())
        local_value_model = self.value_model_fn(nS)
        local_value_model.load_state_dict(self.shared_value_model.state_dict())

        experience_buffer = self.experience_buffer_fn()
        assert isinstance(experience_buffer, ExhaustingLREVBuffer)

        optimize_model = self.optimize_model_fn()

        self.stats.prepare_before_work(rank=rank)
        while not self.get_out_signal:
            self.stats.prepare_before_episode(rank=rank)
            n_steps_start = 0
            state, _ = env.reset()

            for step in count(start=1):
                state, reward, is_terminal, is_truncated, is_exploratory = self.interaction_step(
                    state, env, local_policy_model, local_value_model, experience_buffer
                )
                self.stats.add_one_step_data(rank=rank, reward=float(reward), is_exploration=is_exploratory)
                if is_terminal or step - n_steps_start == self.max_n_steps:
                    is_failure = is_terminal and not is_truncated
                    last_value = 0 if is_failure else local_value_model(state).detach().item()
                    optimize_model(experience_buffer, local_policy_model, local_value_model, last_value)
                    experience_buffer.clear()
                    n_steps_start = step

                if is_terminal or is_truncated:
                    gc.collect()
                    break

            # save global stats
            self.stats.calculate_elapsed_time(rank=rank)
            evaluation_score, _ = self.evaluate(local_policy_model, env)
            self.stats.append_evaluation_score(rank=rank, evaluation_score=evaluation_score)
            self.save_checkpoint(self.stats.get_current_episode(rank=rank), local_policy_model)

            if self.stats.process_after_episode(rank=rank):
                self.get_out_signal.add_(1)
                break

            self.stats.go_to_next_episode(rank=rank)

        while rank == 0 and self.n_active_workers.item() > 1:
            pass

        self.stats.log_finishing_work(rank=rank)
        env.close()
        del env
        self.n_active_workers.sub_(1)

    def train(self, max_minutes: int, max_episodes: int, goal_mean_100_reward: int, log_period_n_secs: int):
        # initialize seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # create training environment
        env: gym.Env = self.make_env_fn(**self.make_env_kwargs)
        env.reset(seed=self.seed)
        assert isinstance(env.action_space, gym.spaces.Discrete)

        nS, nA = env.observation_space.shape[0], env.action_space.n
        self.shared_policy_model = self.policy_model_fn(nS, nA).share_memory()
        self.replay_model = self.shared_policy_model
        assert isinstance(self.shared_policy_model, DiscreteActionPolicyNetwork)
        self.shared_policy_optimizer = self.policy_optimizer_fn(self.shared_policy_model, self.policy_optimizer_lr)
        self.shared_value_model = self.value_model_fn(nS).share_memory()
        assert isinstance(self.shared_value_model, ValueNetwork)
        assert next(self.shared_value_model.parameters()).device == next(self.shared_policy_model.parameters()).device
        self.shared_value_optimizer = self.value_optimizer_fn(self.shared_value_model, self.value_optimizer_lr)

        self.stats = MultiLearnerStatistics(n_workers=self.n_workers,
                                            get_out_lock=mp.Lock(),
                                            max_episodes=max_episodes,
                                            goal_mean_100_reward=goal_mean_100_reward,
                                            max_minutes=max_minutes,
                                            log_period_n_secs=log_period_n_secs)

        self.n_active_workers = torch.zeros(1, dtype=torch.int).share_memory_()
        self.get_out_signal = torch.zeros(1, dtype=torch.int).share_memory_()

        self.stats.start_training()

        print(f'{self.name} Training start. (seed: {self.seed})')

        workers = [mp.Process(target=self.work, args=(rank,)) for rank in range(self.n_workers)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()

        stats, training_time, wallclock_time = self.stats.get_total()
        final_eval_score, score_std = self.evaluate(self.shared_policy_model, env, n_episodes=100)

        env.close()
        del env

        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(final_eval_score, score_std, training_time, wallclock_time))

        self.get_cleaned_checkpoints()
        return stats, final_eval_score

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
