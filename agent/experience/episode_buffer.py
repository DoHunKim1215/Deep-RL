import gc
import time

import numpy as np
import torch
import gymnasium as gym

from agent.multiprocess.multiprocess_env import MultiprocessEnv


class EpisodeBuffer:

    def __init__(self,
                 state_dim: int,
                 gamma: float,
                 tau: float,
                 n_workers: int,
                 max_episodes: int,
                 max_episode_steps: int,
                 device: torch.device):
        assert max_episodes >= n_workers

        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.n_workers = n_workers
        self.max_episodes = max_episodes
        self.max_episode_steps = max_episode_steps

        self.discounts = np.logspace(0, max_episode_steps + 1, num=max_episode_steps + 1,
                                     base=gamma, endpoint=False, dtype=np.longdouble)
        self.tau_discounts = np.logspace(0, max_episode_steps + 1, num=max_episode_steps + 1,
                                         base=gamma * tau, endpoint=False, dtype=np.longdouble)

        self.device = device

        self.states_mem = None
        self.actions_mem = None
        self.returns_mem = None
        self.gaes_mem = None
        self.log_probs_mem = None

        self.episode_steps = None
        self.episode_reward = None
        self.episode_exploration = None
        self.episode_seconds = None

        self.current_ep_idxs = None

        self.clear()

    def clear(self):
        self.states_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps, self.state_dim), dtype=np.float64)
        self.actions_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps), dtype=np.uint8)
        self.returns_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps), dtype=np.float32)
        self.gaes_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps), dtype=np.float32)
        self.log_probs_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps), dtype=np.float32)

        self.episode_steps = np.zeros(shape=self.max_episodes, dtype=np.uint16)
        self.episode_reward = np.zeros(shape=self.max_episodes, dtype=np.float32)
        self.episode_exploration = np.zeros(shape=self.max_episodes, dtype=np.float32)
        self.episode_seconds = np.zeros(shape=self.max_episodes, dtype=np.float64)

        self.states_mem[:] = np.nan
        self.actions_mem[:] = 0
        self.returns_mem[:] = np.nan
        self.gaes_mem[:] = np.nan
        self.log_probs_mem[:] = np.nan

        self.episode_steps[:] = 0
        self.episode_reward[:] = 0.
        self.episode_exploration[:] = 0.
        self.episode_seconds[:] = 0.

        self.current_ep_idxs = np.arange(self.n_workers, dtype=np.uint16)
        gc.collect()

    def fill(self, envs: MultiprocessEnv, policy_model, value_model):
        states = envs.reset_all()

        worker_rewards = np.zeros(shape=(self.n_workers, self.max_episode_steps), dtype=np.float32)
        worker_exploratory = np.zeros(shape=(self.n_workers, self.max_episode_steps), dtype=bool)
        worker_steps = np.zeros(shape=self.n_workers, dtype=np.uint16)
        worker_seconds = np.array([time.time(), ] * self.n_workers, dtype=np.float64)

        buffer_full = False
        while not buffer_full and len(self.episode_steps[self.episode_steps > 0]) < self.max_episodes / 2:
            with torch.no_grad():
                actions, log_probs, are_exploratory = policy_model.select_action_info_np(states)

            next_states, rewards, terminals, truncated = envs.step(actions)
            self.states_mem[self.current_ep_idxs, worker_steps, :] = states
            self.actions_mem[self.current_ep_idxs, worker_steps] = actions
            self.log_probs_mem[self.current_ep_idxs, worker_steps] = log_probs

            worker_exploratory[np.arange(self.n_workers), worker_steps] = are_exploratory
            worker_rewards[np.arange(self.n_workers), worker_steps] = rewards.reshape(-1,)

            for w_idx in range(self.n_workers):
                if worker_steps[w_idx] + 1 == self.max_episode_steps:
                    terminals[w_idx] = 1
                    truncated[w_idx] = True

            states = next_states
            worker_steps += 1

            if terminals.sum():
                idx_terminals = np.flatnonzero(terminals)
                next_values = np.zeros(shape=self.n_workers)

                if truncated.sum():
                    idx_truncated = np.flatnonzero(truncated)
                    with torch.no_grad():
                        next_values[idx_truncated] = value_model(next_states[idx_truncated]).squeeze().cpu().numpy()

                new_states = np.stack([envs.reset(rank=idx_terminal) for idx_terminal in idx_terminals])
                states[idx_terminals] = new_states

                for w_idx in range(self.n_workers):
                    if w_idx not in idx_terminals:
                        continue

                    e_idx = self.current_ep_idxs[w_idx]
                    T = worker_steps[w_idx]
                    self.episode_steps[e_idx] = T
                    self.episode_reward[e_idx] = worker_rewards[w_idx, :T].sum()
                    self.episode_exploration[e_idx] = worker_exploratory[w_idx, :T].mean()
                    self.episode_seconds[e_idx] = time.time() - worker_seconds[w_idx]

                    ep_rewards = np.concatenate((worker_rewards[w_idx, :T], [next_values[w_idx]]))
                    ep_discounts = self.discounts[:T + 1]
                    ep_returns = np.array([np.sum(ep_discounts[:T + 1 - t] * ep_rewards[t:]) for t in range(T)])
                    self.returns_mem[e_idx, :T] = ep_returns

                    ep_states = self.states_mem[e_idx, :T]
                    with torch.no_grad():
                        ep_values = torch.cat((
                            value_model(ep_states).squeeze(),
                            torch.tensor([next_values[w_idx]], device=self.device, dtype=torch.float32)
                        ))
                    np_ep_values = ep_values.view(-1).cpu().numpy()
                    deltas = ep_rewards[:-1] + self.gamma * np_ep_values[1:] - np_ep_values[:-1]
                    gaes = np.array([np.sum(self.tau_discounts[:T - t] * deltas[t:]) for t in range(T)])
                    self.gaes_mem[e_idx, :T] = gaes

                    worker_exploratory[w_idx, :] = 0
                    worker_rewards[w_idx, :] = 0
                    worker_steps[w_idx] = 0
                    worker_seconds[w_idx] = time.time()

                    new_ep_id = max(self.current_ep_idxs) + 1
                    if new_ep_id >= self.max_episodes:
                        buffer_full = True
                        break

                    self.current_ep_idxs[w_idx] = new_ep_id

        ep_idxs = self.episode_steps > 0
        ep_t = self.episode_steps[ep_idxs]

        self.states_mem = [row[:ep_t[i]] for i, row in enumerate(self.states_mem[ep_idxs])]
        self.states_mem = np.concatenate(self.states_mem)
        self.actions_mem = [row[:ep_t[i]] for i, row in enumerate(self.actions_mem[ep_idxs])]
        self.actions_mem = np.concatenate(self.actions_mem)
        self.returns_mem = [row[:ep_t[i]] for i, row in enumerate(self.returns_mem[ep_idxs])]
        self.returns_mem = torch.tensor(np.concatenate(self.returns_mem), device=self.device)
        self.gaes_mem = [row[:ep_t[i]] for i, row in enumerate(self.gaes_mem[ep_idxs])]
        self.gaes_mem = torch.tensor(np.concatenate(self.gaes_mem), device=self.device)
        self.log_probs_mem = [row[:ep_t[i]] for i, row in enumerate(self.log_probs_mem[ep_idxs])]
        self.log_probs_mem = torch.tensor(np.concatenate(self.log_probs_mem), device=self.device)

        ep_r = self.episode_reward[ep_idxs]
        ep_x = self.episode_exploration[ep_idxs]
        ep_s = self.episode_seconds[ep_idxs]
        return ep_t, ep_r, ep_x, ep_s

    def get_stacks(self):
        return self.states_mem, self.actions_mem, self.returns_mem, self.gaes_mem, self.log_probs_mem

    def __len__(self):
        return self.episode_steps[self.episode_steps > 0].sum()
