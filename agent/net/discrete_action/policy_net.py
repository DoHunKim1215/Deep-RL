from typing import Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class DiscreteActionPolicyNetwork(nn.Module):
    pass


class DAFCSP(DiscreteActionPolicyNetwork):
    """
    Fully Connected Stochastic Policy Network for discrete action (DA) space
    """

    def __init__(self,
                 input_state_dim,
                 n_output_action,
                 device,
                 hidden_dims=(32, 32),
                 activation_fc=F.relu):
        super(DAFCSP, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.output_layer = nn.Linear(hidden_dims[-1], n_output_action)

        self.device = device
        self.to(self.device)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        return self.output_layer(x)

    def select_action_info(self, state) -> Tuple[int, bool, torch.Tensor, torch.Tensor,]:
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_probs = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        is_exploratory = action != np.argmax(logits.detach().cpu().numpy())
        return action.item(), is_exploratory.item(), log_probs, entropy

    def select_action_info_np(self, state) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        logits = self.forward(state)
        np_logits: np.ndarray = logits.detach().cpu().numpy()
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        np_actions: np.ndarray = actions.detach().cpu().numpy()
        log_probs = dist.log_prob(actions)
        np_log_probs = log_probs.detach().cpu().numpy()
        are_exploratory: np.ndarray = np_actions != np.argmax(np_logits, axis=1)
        return np_actions, np_log_probs, are_exploratory

    def get_predictions(self, states, actions):
        states, actions = self._format(states), self._format(actions)
        logits = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()
        return log_probs, entropies

    def select_action(self, state):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item()

    def select_greedy_action(self, state):
        logits = self.forward(state)
        return np.argmax(logits.detach().cpu().numpy())
