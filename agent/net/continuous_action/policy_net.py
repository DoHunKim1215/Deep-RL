import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ContinuousActionDeterministicPolicyNetwork(nn.Module):
    pass


class CAFCDP(ContinuousActionDeterministicPolicyNetwork):
    """
    Fully Connected Deterministic Policy Network for Continuous Action Space
    """

    def __init__(self,
                 input_dim,
                 action_bounds,
                 device: torch.device,
                 hidden_dims=(32, 32),
                 activation_fc=F.relu,
                 out_activation_fc=F.tanh):
        super(CAFCDP, self).__init__()
        self.activation_fc = activation_fc
        self.out_activation_fc = out_activation_fc
        self.env_min, self.env_max = action_bounds

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], len(self.env_max))

        self.device = device
        self.to(self.device)

        self.env_min = torch.tensor(self.env_min, device=self.device, dtype=torch.float32)
        self.env_max = torch.tensor(self.env_max, device=self.device, dtype=torch.float32)

        self.nn_min = self.out_activation_fc(torch.Tensor([float('-inf')])).to(self.device)
        self.nn_max = self.out_activation_fc(torch.Tensor([float('inf')])).to(self.device)

    def _rescale(self, x):
        return (x - self.nn_min) * (self.env_max - self.env_min) / (self.nn_max - self.nn_min) + self.env_min

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = self.out_activation_fc(self.output_layer(x))
        return self._rescale(x)


class ContinuousActionStochasticPolicyNetwork(nn.Module):
    pass


class CAFCGP(ContinuousActionStochasticPolicyNetwork):
    """
    Fully Connected Gaussian Policy Network for Continuous Action Space
    """

    def __init__(self,
                 input_dim: int,
                 action_bounds,
                 device: torch.device,
                 log_std_min=-20,
                 log_std_max=2,
                 hidden_dims=(32, 32),
                 activation_fc=F.relu,
                 entropy_lr=0.001):
        super(CAFCGP, self).__init__()
        self.activation_fc = activation_fc
        self.env_min, self.env_max = action_bounds

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.output_layer_mean = nn.Linear(hidden_dims[-1], len(self.env_max))
        self.output_layer_log_std = nn.Linear(hidden_dims[-1], len(self.env_max))

        self.device = device
        self.to(self.device)

        self.env_min = torch.tensor(self.env_min, device=self.device, dtype=torch.float32)
        self.env_max = torch.tensor(self.env_max, device=self.device, dtype=torch.float32)
        self.action_shape = self.env_max.cpu().numpy().shape

        self.nn_min = F.tanh(torch.Tensor([float('-inf')])).to(self.device)
        self.nn_max = F.tanh(torch.Tensor([float('inf')])).to(self.device)

        self.target_entropy = -np.prod(self.env_max.shape)
        self.logalpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.logalpha], lr=entropy_lr)

    def _rescale(self, x):
        return (x - self.nn_min) * (self.env_max - self.env_min) / (self.nn_max - self.nn_min) + self.env_min

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x_mean = self.output_layer_mean(x)
        x_log_std = torch.clamp(self.output_layer_log_std(x), self.log_std_min, self.log_std_max)
        return x_mean, x_log_std

    def select_action_info(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)

        pi_s = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh_action = pi_s.rsample()
        tanh_action = torch.tanh(pre_tanh_action)
        action = self._rescale(tanh_action)

        log_prob = pi_s.log_prob(pre_tanh_action) - torch.log((1 - tanh_action.pow(2)).clamp(0, 1) + epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, self._rescale(torch.tanh(mean))

    def _update_exploration_ratio(self, greedy_action, action_taken):
        env_min, env_max = self.env_min.cpu().numpy(), self.env_max.cpu().numpy()
        return np.mean(abs((greedy_action - action_taken) / (env_max - env_min)))

    def _get_greedy_action(self, mean):
        return self._rescale(torch.tanh(mean)).detach().cpu().numpy().reshape(self.action_shape)

    def select_random_action(self, state):
        mean, _ = self.forward(state)
        greedy_action = self._get_greedy_action(mean)
        random_action = np.random.uniform(low=self.env_min.cpu().numpy(), high=self.env_max.cpu().numpy())
        random_action = random_action.reshape(self.action_shape)

        exploration_ratio = self._update_exploration_ratio(greedy_action, random_action)
        return random_action, exploration_ratio

    def select_greedy_action(self, state):
        mean, _ = self.forward(state)
        greedy_action = self._get_greedy_action(mean)
        return greedy_action, 0.

    def select_action(self, state):
        mean, log_std = self.forward(state)
        greedy_action = self._get_greedy_action(mean)
        action = self._rescale(torch.tanh(torch.distributions.Normal(mean, log_std.exp()).sample()))
        action = action.detach().cpu().numpy().reshape(self.action_shape)

        exploration_ratio = self._update_exploration_ratio(greedy_action, action)
        return action, exploration_ratio
