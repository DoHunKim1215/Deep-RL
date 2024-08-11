from typing import Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class DiscreteActionActorCriticNetwork(nn.Module):
    pass


class DAFCSPV(DiscreteActionActorCriticNetwork):
    """
    Fully Connected Stochastic Policy And Value Network for discrete action (DA) space
    """

    def __init__(self,
                 input_state_dim,
                 n_output_action_policies,
                 device,
                 hidden_dims=(32, 32),
                 activation_fc=F.relu):
        super(DAFCSPV, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.output_policy_layer = nn.Linear(hidden_dims[-1], n_output_action_policies)
        self.output_value_layer = nn.Linear(hidden_dims[-1], 1)

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
        return self.output_policy_layer(x), self.output_value_layer(x)

    def select_action_info(self, state) -> Tuple[int, bool, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_probs = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        action = action.item() if len(action) == 1 else action.detach().cpu().numpy()
        is_exploratory = action != np.argmax(logits.detach().cpu().numpy(), axis=int(len(state) != 1))
        return action, is_exploratory, log_probs, entropy, value

    def select_action(self, state) -> int | np.ndarray:
        logits, _ = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        action = action.item() if len(action) == 1 else action.detach().cpu().numpy()
        return action

    def select_greedy_action(self, state):
        logits, _ = self.forward(state)
        return np.argmax(logits.detach().cpu().numpy())

    def evaluate_state(self, state) -> torch.Tensor:
        _, value = self.forward(state)
        return value