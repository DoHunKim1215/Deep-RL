from abc import abstractmethod, ABC

import numpy as np
import torch

from agent.net.continuous_action.policy_net import ContinuousActionDeterministicPolicyNetwork


class ContinuousActionStrategy(ABC):
    def __init__(self, action_bounds):
        self.low, self.high = action_bounds

    @abstractmethod
    def select_action(self, model: ContinuousActionDeterministicPolicyNetwork, state):
        pass

    @staticmethod
    def _forward(model: ContinuousActionDeterministicPolicyNetwork, state):
        with torch.no_grad():
            return model(state).detach().cpu().numpy().squeeze()


class GreedyStrategy(ContinuousActionStrategy):
    """
    Greedy Strategy
    Always chooses action with the highest value. (Always exploit)
    """

    def __init__(self, action_bounds):
        super(GreedyStrategy, self).__init__(action_bounds)

    def select_action(self, model: ContinuousActionDeterministicPolicyNetwork, state):
        action = np.clip(self._forward(model, state), self.low, self.high)
        return np.reshape(action, self.high.shape), 0.


class GaussianNoiseStrategy(ContinuousActionStrategy):
    """
    Gaussian Noise Strategy
    Add some gaussian noise to the action from deterministic policy for improving exploration.
    """
    def __init__(self, bounds, noise_ratio=0.1):
        super(GaussianNoiseStrategy, self).__init__(bounds)
        self.noise_ratio = noise_ratio
        self.action_range = self.high - self.low

    def select_action(self, model: ContinuousActionDeterministicPolicyNetwork, state, max_exploration=False):
        if max_exploration:
            noise_scale = self.action_range
        else:
            noise_scale = self.noise_ratio * self.action_range

        greedy_action = self._forward(model, state)
        noise = np.random.normal(loc=0, scale=noise_scale, size=len(self.high)).astype(np.float32)
        noisy_action = greedy_action + noise
        action = np.clip(noisy_action, self.low, self.high)
        ratio_noise_injected = np.mean(abs((greedy_action - action) / self.action_range))
        return action, ratio_noise_injected


class DecayingGaussianNoiseStrategy(ContinuousActionStrategy):
    """
    Linearly Decaying Gaussian Noise Strategy
    Add linearly decaying gaussian noise to the action from deterministic policy for improving exploration.
    """

    def __init__(self, bounds, init_noise_ratio=0.5, min_noise_ratio=0.1, decay_steps=10000):
        super(DecayingGaussianNoiseStrategy, self).__init__(bounds)
        self.step = 0
        self.noise_ratio = init_noise_ratio
        self.init_noise_ratio = init_noise_ratio
        self.min_noise_ratio = min_noise_ratio
        self.decay_steps = decay_steps
        self.action_range = self.high - self.low

    def _update_noise_ratio(self):
        noise_ratio = ((self.init_noise_ratio - self.min_noise_ratio) * (1. - self.step / self.decay_steps)
                       + self.min_noise_ratio)
        self.noise_ratio = np.clip(noise_ratio, self.min_noise_ratio, self.init_noise_ratio)
        self.step += 1

    def select_action(self, model: ContinuousActionDeterministicPolicyNetwork, state, max_exploration=False):
        if max_exploration:
            noise_scale = self.action_range
        else:
            noise_scale = self.noise_ratio * self.action_range

        greedy_action = self._forward(model, state)
        noise = np.random.normal(loc=0, scale=noise_scale, size=len(self.high)).astype(np.float32)
        noisy_action = greedy_action + noise
        action = np.clip(noisy_action, self.low, self.high)
        ratio_noise_injected = np.mean(abs((greedy_action - action) / self.action_range))

        self._update_noise_ratio()
        return action, ratio_noise_injected
