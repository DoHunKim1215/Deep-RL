import numpy as np
import torch


class ExperienceTuple:
    pass


class SARSTuple(ExperienceTuple):

    def __init__(self, states, actions, rewards, next_states, is_failures):
        self.states: np.ndarray = states
        self.actions: np.ndarray = actions
        self.rewards: np.ndarray = rewards
        self.next_states: np.ndarray = next_states
        self.is_failures: np.ndarray = is_failures

    def load(self, device):
        self.states = torch.from_numpy(self.states).float().to(device)
        self.actions = torch.from_numpy(self.actions).float().to(device)
        self.next_states = torch.from_numpy(self.next_states).float().to(device)
        self.rewards = torch.from_numpy(self.rewards).float().to(device)
        self.is_failures = torch.from_numpy(self.is_failures).float().to(device)

    def decompose(self):
        return self.states, self.actions, self.rewards, self.next_states, self.is_failures
