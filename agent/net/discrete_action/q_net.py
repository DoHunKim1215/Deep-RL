import torch
from torch import nn
import torch.nn.functional as F


class DiscreteActionQNetwork(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
        return x


class DAFCQ(DiscreteActionQNetwork):
    """
    Fully Connected Q-Network for discrete action (DA) space
    """

    def __init__(self,
                 input_state_dim,
                 n_output_actions_q,
                 device: torch.device,
                 hidden_dims=(32, 32),
                 activation_fc=F.relu):
        super(DAFCQ, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.output_layer = nn.Linear(hidden_dims[-1], n_output_actions_q)

        self.device = device
        self.to(self.device)

    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        return x


class DAFCDuelingQ(DiscreteActionQNetwork):
    """
    Fully Connected Dueling Q-Network for discrete action (DA) space
    paper: Dueling Network Architectures for Deep Reinforcement Learning (2015)
    authors: Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, Nando de Freitas
    """

    def __init__(self,
                 input_state_dim,
                 n_output_actions_q,
                 device: torch.device,
                 hidden_dims=(32, 32),
                 activation_fc=F.relu):
        super(DAFCDuelingQ, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.output_adv_layer = nn.Linear(hidden_dims[-1], n_output_actions_q)
        self.output_value_layer = nn.Linear(hidden_dims[-1], 1)

        self.device = device
        self.to(self.device)

    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        adv = self.output_adv_layer(x)
        val = self.output_value_layer(x).expand_as(adv)
        return val + adv - adv.mean(dim=1, keepdim=True).expand_as(adv)
