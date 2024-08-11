import torch
from torch import nn
import torch.nn.functional as F


class ContinuousActionQNetwork(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CAFCQ(ContinuousActionQNetwork):
    """
    Fully Connected Q-Network for continuous action (CA) space
    """

    def __init__(self,
                 input_state_dim,
                 input_action_dim,
                 device: torch.device,
                 hidden_dims=(32, 32),
                 activation_fc=F.relu):
        super(CAFCQ, self).__init__()
        self.activation_fc = activation_fc

        self.input_state_layer = nn.Linear(input_state_dim, hidden_dims[0])
        self.input_action_layer = nn.Linear(input_action_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            input_dim = hidden_dims[i] * 2 if i == 0 else hidden_dims[i]
            self.hidden_layers.append(nn.Linear(input_dim, hidden_dims[i + 1]))
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        self.device = device
        self.to(self.device)

    def _format(self, state, action):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
        u = action
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, device=self.device, dtype=torch.float32)
            if len(u.size()) == 1:
                u = u.unsqueeze(0)
        return x, u

    def forward(self, state, action):
        x, u = self._format(state, action)
        x = self.activation_fc(self.input_state_layer(x))
        u = self.activation_fc(self.input_action_layer(u))
        x = torch.cat((x, u), dim=1)
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        return x
