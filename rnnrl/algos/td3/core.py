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


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, activation, act_limit, recurrent, device):
        super().__init__()
        self.act_limit = act_limit
        self.recurrent = recurrent
        self.device = device
        self.activation = activation()
        self.tanh = nn.Tanh()
        self.norm1 = nn.LayerNorm(32)
        self.norm2 = nn.LayerNorm(32)
        self.norm3 = nn.LayerNorm(hidden_dim)
        if recurrent:
            self.obs_enc1 = nn.Linear(obs_dim, 32)
            self.obs_enc2 = nn.Linear(obs_dim, 32)
            self.rnn = nn.LSTM(32, hidden_dim, batch_first=True)
            self.l1 = nn.Linear(hidden_dim + 32, hidden_dim)
            self.l2 = nn.Linear(hidden_dim, act_dim)
        else:
            self.l1 = nn.Linear(obs_dim, hidden_dim)
            self.l2 = nn.Linear(hidden_dim, hidden_dim)
            self.l3 = nn.Linear(hidden_dim, act_dim)

        self.apply(weight_init)

    def forward(self, obs, hidden):
        if self.recurrent:
            obs_enc1 = self.obs_enc1(obs)
            obs_enc1 = self.norm1(obs_enc1)
            obs_enc1 = self.activation(obs_enc1)

            obs_enc2 = self.obs_enc2(obs)
            obs_enc2 = self.norm2(obs_enc2)
            obs_enc2 = self.activation(obs_enc2)

            self.rnn.flatten_parameters()
            h, hidden = self.rnn(obs_enc1, hidden)
            h = self.norm3(h)
            h = self.activation(h)
            h = torch.cat((h, obs_enc2), -1)
            h = self.activation(self.l1(h))
            a = self.tanh(self.l2(h))
        else:
            a, hidden = self.activation(self.l1(obs)), None
            a = self.activation(self.l2(a))
            a = self.tanh(self.l3(a))

        return self.act_limit * a, hidden

    def get_initialized_hidden(self):
        h_0, c_0 = None, None
        if self.recurrent:
            h_0 = torch.zeros((
                self.rnn.num_layers,
                1,
                self.rnn.hidden_size),
                dtype=torch.float).to(self.device)

            c_0 = torch.zeros((
                self.rnn.num_layers,
                1,
                self.rnn.hidden_size),
                dtype=torch.float).to(self.device)

        return (h_0, c_0)


class QFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, activation, recurrent, device):
        super().__init__()
        self.recurrent = recurrent
        self.device = device
        self.activation = activation()
        self.norm1 = nn.LayerNorm(32)
        self.norm2 = nn.LayerNorm(32)
        self.norm3 = nn.LayerNorm(hidden_dim)
        if recurrent:
            self.obs_enc1 = nn.Linear(obs_dim, 32)
            self.obs_enc2 = nn.Linear(obs_dim, 32)
            self.rnn = nn.LSTM(32 + act_dim, hidden_dim, batch_first=True)
            self.l1 = nn.Linear(hidden_dim + 32, hidden_dim)
            self.l2 = nn.Linear(hidden_dim, 1)
        else:
            self.l1 = nn.Linear(obs_dim + act_dim, hidden_dim)
            self.l2 = nn.Linear(hidden_dim, hidden_dim)
            self.l3 = nn.Linear(hidden_dim, 1)


    def forward(self, obs, act, hidden):
        if self.recurrent:
            obs_enc1 = self.obs_enc1(obs)
            obs_enc1 = self.norm1(obs_enc1)
            obs_enc1 = self.activation(obs_enc1)

            obs_enc2 = self.obs_enc2(obs)
            obs_enc2 = self.norm2(obs_enc2)
            obs_enc2 = self.activation(obs_enc2)

            obs_act = torch.cat([obs_enc1, act], dim=-1)
            self.rnn.flatten_parameters()
            h, hidden = self.rnn(obs_act, hidden)
            h = self.norm3(h)
            h = torch.cat((h, obs_enc2), -1)
            h = self.activation(self.l1(h))
            q = self.l2(h)
        else:
            obs_act = torch.cat([obs, act], dim=-1)
            q, hidden = self.activation(self.l1(obs_act)), None
            q = self.activation(self.l2(q))
            q = self.l3(q)
        return q.squeeze(-1), hidden # Critical to ensure q has right shape.

    def get_initialized_hidden(self):
        h_0, c_0 = None, None
        if self.recurrent:
            h_0 = torch.zeros((
                self.rnn.num_layers,
                1,
                self.rnn.hidden_size),
                dtype=torch.float).to(self.device)

            c_0 = torch.zeros((
                self.rnn.num_layers,
                1,
                self.rnn.hidden_size),
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
