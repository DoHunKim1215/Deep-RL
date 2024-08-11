import numpy as np
import torch


class LREVBuffer:
    pass


class ExhaustingLRBuffer(LREVBuffer):
    def __init__(self):
        self.log_probs = []
        self.rewards = []

    def clear(self):
        self.log_probs.clear()
        self.rewards.clear()

    def store(self, log_prob: torch.Tensor, reward: float):
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def sample(self):
        return torch.cat(self.log_probs), np.array(self.rewards)

    def __len__(self):
        return len(self.log_probs)


class ExhaustingLREVBuffer(LREVBuffer):
    def __init__(self):
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.values = []

    def clear(self):
        self.log_probs.clear()
        self.rewards.clear()
        self.entropies.clear()
        self.values.clear()

    def store(self, log_prob, reward, entropy, value):
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.entropies.append(entropy)
        self.values.append(value)

    def sample(self):
        return torch.cat(self.log_probs), np.array(self.rewards), torch.cat(self.entropies), torch.cat(self.values)

    def __len__(self):
        return len(self.log_probs)


class ExhaustingBatchLREVBuffer(LREVBuffer):

    def __init__(self):
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.values = []

    def clear(self):
        self.log_probs.clear()
        self.rewards.clear()
        self.entropies.clear()
        self.values.clear()

    def store(self, log_probs, rewards, entropies, values):
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.entropies.append(entropies)
        self.values.append(values)

    def sample(self):
        return (torch.stack(self.log_probs).squeeze(),
                np.array(self.rewards).squeeze(),
                torch.stack(self.entropies).squeeze(),
                torch.stack(self.values).squeeze())

    def __len__(self):
        return len(self.log_probs)