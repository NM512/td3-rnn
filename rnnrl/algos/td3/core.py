from locale import currency
from multiprocessing.dummy import current_process
import numpy as np
import torch
import torch.nn as nn


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, activation, act_limit, recurrent, device):
        super().__init__()
        self.act_limit = act_limit
        self.recurrent = recurrent
        self.device = device
        self.activation = activation()
        self.tanh = nn.Tanh()
        if recurrent:
            self.l1 = nn.LSTM(obs_dim, hidden_dim, batch_first=True)
        else:
            self.l1 = nn.Linear(obs_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, act_dim)
        
    def forward(self, obs, hidden):
        if self.recurrent:
            self.l1.flatten_parameters()
            a, hidden = self.l1(obs, hidden)
        else:
            a, hidden = self.activation(self.l1(obs)), None

        a = self.activation(self.l2(a))
        a = self.tanh(self.l3(a))

        return self.act_limit * a, hidden
    
    def get_initialized_hidden(self):
        h_0, c_0 = None, None
        if self.recurrent:
            h_0 = torch.zeros((
                self.l1.num_layers,
                1,
                self.l1.hidden_size),
                dtype=torch.float).to(self.device)

            c_0 = torch.zeros((
                self.l1.num_layers,
                1,
                self.l1.hidden_size),
                dtype=torch.float).to(self.device)

        return (h_0, c_0)


class QFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, activation, recurrent, device):
        super().__init__()
        self.recurrent = recurrent
        self.device = device
        self.activation = activation()
        if self.recurrent:
            self.l1 = nn.LSTM(obs_dim+act_dim, hidden_dim, batch_first=True)
        else:
            self.l1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, obs, act, hidden):
        obs_act = torch.cat([obs, act], dim=-1)
        if self.recurrent:
            self.l1.flatten_parameters()
            q, hidden = self.l1(obs_act, hidden)
        else:
            q, hidden = self.activation(self.l1(obs_act)), None

        q = self.activation(self.l2(q))
        q = self.l3(q)
        return q.squeeze(-1), hidden # Critical to ensure q has right shape.

    def get_initialized_hidden(self):
        h_0, c_0 = None, None
        if self.recurrent:
            h_0 = torch.zeros((
                self.l1.num_layers,
                1,
                self.l1.hidden_size),
                dtype=torch.float).to(self.device)

            c_0 = torch.zeros((
                self.l1.num_layers,
                1,
                self.l1.hidden_size),
                dtype=torch.float).to(self.device)

        return (h_0, c_0)


class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, recurrent, hidden_dim=256,
                 activation=nn.ReLU, device='cuda'):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = Actor(obs_dim, act_dim, hidden_dim, activation, act_limit, recurrent, device)
        self.q1 = QFunction(obs_dim, act_dim, hidden_dim, activation, recurrent, device)
        self.q2 = QFunction(obs_dim, act_dim, hidden_dim, activation, recurrent, device)
        self.recurrent = recurrent
        self.device = device

    def act(self, obs, hidden):
        with torch.no_grad():
            act, hidden = self.pi(obs, hidden)
            return act.cpu().numpy(), hidden
