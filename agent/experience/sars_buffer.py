from typing import Tuple

import numpy as np

from agent.experience.tuple import SARSTuple


class SARSBuffer:
    pass


class ExhaustingSARSBuffer(SARSBuffer):
    def __init__(self):
        self.experiences = []

    def clear(self):
        self.experiences.clear()

    def store(self, state: np.ndarray, action: np.ndarray, reward: float, new_state: np.ndarray, is_failure: float):
        self.experiences.append((state, action, reward, new_state, is_failure))

    def sample(self) -> SARSTuple:
        experiences = np.array(self.experiences, dtype=object)
        batches = [np.vstack(sars) for sars in experiences.T]
        return SARSTuple(batches[0], batches[1], batches[2], batches[3], batches[4])

    def __len__(self):
        return len(self.experiences)


class ReplayBuffer(SARSBuffer):
    def __init__(self, capacity: int, batch_size: int):
        self.states = np.empty(shape=capacity, dtype=np.ndarray)
        self.actions = np.empty(shape=capacity, dtype=np.ndarray)
        self.rewards = np.empty(shape=capacity, dtype=np.float32)
        self.next_states = np.empty(shape=capacity, dtype=np.ndarray)
        self.is_failures = np.empty(shape=capacity, dtype=np.float32)

        self.capacity = capacity
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0

    def clear(self):
        self._idx = 0
        self.size = 0

    def store(self, state: np.ndarray, action: np.ndarray, reward: float, new_state: np.ndarray, is_failure: float):
        self.states[self._idx] = state
        self.actions[self._idx] = action
        self.rewards[self._idx] = reward
        self.next_states[self._idx] = new_state
        self.is_failures[self._idx] = is_failure

        self._idx = (self._idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self) -> SARSTuple:
        assert self.size >= self.batch_size
        idxs = np.random.choice(self.size, self.batch_size, replace=False)
        return SARSTuple(np.vstack(self.states[idxs]),
                         np.vstack(self.actions[idxs]),
                         np.vstack(self.rewards[idxs]),
                         np.vstack(self.next_states[idxs]),
                         np.vstack(self.is_failures[idxs]))

    def __len__(self):
        return self.size


class PrioritizedReplayBuffer(SARSBuffer):
    """
    paper: Prioritized Experience Replay (2015)
    link: https://arxiv.org/abs/1511.05952
    authors: Tom Schaul, John Quan, Ioannis Antonoglou, David Silver
    """

    EPS = 1e-6

    def __init__(self, capacity: int, batch_size: int, rank_based: bool, alpha: float, init_beta: float, beta_rate: float):
        self.capacity = capacity
        self.experiences: np.ndarray = np.empty(shape=capacity, dtype=np.ndarray)
        self.td_errors: np.ndarray = np.empty(shape=capacity, dtype=np.float64)
        self.batch_size = batch_size

        self.n_entries = 0
        self.next_idx = 0
        self.rank_based = rank_based
        self.alpha = alpha
        self.beta = init_beta
        self.init_beta = init_beta
        self.beta_rate = beta_rate

    def clear(self):
        self.n_entries = 0
        self.next_idx = 0
        self.beta = self.init_beta

    def update(self, idxs: np.ndarray, td_errors: np.ndarray):
        assert idxs.size == td_errors.size
        self.td_errors[idxs] = np.abs(td_errors)
        if self.rank_based:
            sorted_idxs = np.argsort(self.td_errors[:self.n_entries])[::-1]
            self.experiences[:self.n_entries] = self.experiences[sorted_idxs]
            self.td_errors[:self.n_entries] = self.td_errors[sorted_idxs]

    def store(self, state: np.ndarray, action: np.ndarray, reward: float, new_state: np.ndarray, is_failure: float):
        priority = 1.0
        if self.n_entries > 0:
            priority = np.max(self.td_errors[:self.n_entries])

        self.td_errors[self.next_idx] = priority
        self.experiences[self.next_idx] = np.array([state, action, reward, new_state, is_failure], dtype=object)

        self.n_entries = min(self.n_entries + 1, self.capacity)
        self.next_idx = (self.next_idx + 1) % self.capacity

    def _update_beta(self):
        self.beta = min(1.0, self.beta * self.beta_rate ** -1)
        return self.beta

    def sample(self) -> Tuple[np.ndarray, np.ndarray, SARSTuple]:
        assert self.n_entries >= self.batch_size

        self._update_beta()
        if self.rank_based:
            priorities = np.reciprocal(np.arange(self.n_entries) + 1., dtype=np.float64)
        else:  # proportional
            priorities = self.td_errors[:self.n_entries] + self.EPS
        scaled_priorities = priorities ** self.alpha
        probs = scaled_priorities / np.sum(scaled_priorities)

        weights = (self.n_entries * probs) ** (-self.beta)
        normalized_weights = weights / np.max(weights)
        idxs = np.random.choice(a=self.n_entries, size=self.batch_size, replace=False, p=probs)

        samples = self.experiences[idxs]
        samples_stacks = [np.vstack(sars) for sars in np.vstack(samples).T]
        idxs_stack = np.vstack(idxs)
        weights_stack = np.vstack(normalized_weights[idxs])
        return idxs_stack, weights_stack, SARSTuple(samples_stacks[0],
                                                    samples_stacks[1],
                                                    samples_stacks[2],
                                                    samples_stacks[3],
                                                    samples_stacks[4])

    def __len__(self):
        return self.n_entries
