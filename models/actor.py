# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
# from torch.distributions.normal import Normal
from externals.rl_games.rl_games.algos_torch._grad_distribution import GradNormal
import numpy as np

from models import model_utils


class ActorDeterministicMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg_network, device='cuda:0'):
        super(ActorDeterministicMLP, self).__init__()

        self.device = device

        self.layer_dims = [obs_dim] + cfg_network['actor_mlp']['units'] + [action_dim]

        init_ = lambda m: model_utils.init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))
                        
        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(model_utils.get_activation_func(cfg_network['actor_mlp']['activation']))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i+1]))

        self.actor = nn.Sequential(*modules).to(device)
        
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        print(self.actor)

    def get_logstd(self):
        # return self.logstd
        return None

    def forward(self, observations, deterministic = False):
        return self.actor(observations)


class ActorStochasticMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg_network, device='cuda:0'):
        super(ActorStochasticMLP, self).__init__()

        self.device = device

        self.layer_dims = [obs_dim] + cfg_network['actor_mlp']['units'] + [action_dim]

        init_ = lambda m: model_utils.init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))
        
        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            if i < len(self.layer_dims) - 2:
                modules.append(model_utils.get_activation_func(cfg_network['actor_mlp']['activation']))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i+1]))
            else:
                modules.append(model_utils.get_activation_func('identity'))
            
        self.mu_net = nn.Sequential(*modules).to(device)

        logstd = cfg_network.get('actor_logstd_init', -1.0)

        self.logstd = torch.nn.Parameter(torch.ones(action_dim, dtype=torch.float32, device=device) * logstd)

        self.action_dim = action_dim
        self.obs_dim = obs_dim

        print(self.mu_net)
        print(self.logstd)
    
    def get_logstd(self):
        return self.logstd

    def forward(self, obs, deterministic = False):
        mu = self.mu_net(obs)

        if deterministic:
            return mu
        else:
            std = self.logstd.exp() # (num_actions)
            # eps = torch.randn((*obs.shape[:-1], std.shape[-1])).to(self.device)
            # sample = mu + eps * std
            dist = GradNormal(mu, std)
            sample = dist.rsample()
            return sample
    
    def forward_with_dist(self, obs, deterministic = False):
        mu = self.mu_net(obs)
        std = self.logstd.exp() # (num_actions)
        
        dist = GradNormal(mu, std)
        eps = dist.sample_eps()
        
        if deterministic:
            eps = eps.zero_()
            
        sample = dist.eps_to_action(eps)
        
        return sample, mu, std, eps
        
    def evaluate_actions_log_probs(self, obs, actions):
        mu = self.mu_net(obs)

        std = self.logstd.exp()
        dist = GradNormal(mu, std)

        return dist.log_prob(actions)

    def forward_dist(self, obs):
        mu = self.mu_net(obs)
        std = self.logstd.exp() # (num_actions)

        return mu, std

class GradActorStochasticMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg_network, device='cuda:0'):
        super(GradActorStochasticMLP, self).__init__()

        self.device = device

        self.layer_dims = [obs_dim] + cfg_network['actor_mlp']['units']

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            modules.append(model_utils.get_activation_func(cfg_network['actor_mlp']['activation']))
            modules.append(torch.nn.LayerNorm(self.layer_dims[i + 1]))
            
        self.actor_mlp = nn.Sequential(*modules).to(device)
        
        # mu;
        out_size = self.layer_dims[-1]
        self.mu = [nn.Linear(out_size, action_dim), model_utils.get_activation_func('identity')]
        self.mu = nn.Sequential(*self.mu).to(device)
        
        # logstd;
        self.fixed_sigma = cfg_network['fixed_sigma']
        if cfg_network['fixed_sigma']:
            logstd = cfg_network.get('actor_logstd_init', -1.0)
            self.logstd = torch.nn.Parameter(torch.ones(action_dim, dtype=torch.float32, device=device) * logstd)
        else:
            self.logstd = nn.Linear(out_size, action_dim).to(device)
            
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        print(self.actor_mlp)
        print(self.mu)
        print(self.logstd)

    def forward(self, obs, deterministic = False):
        out = self.actor_mlp(obs)
        mu = self.mu(out)

        if deterministic:
            return mu
        else:
            if self.fixed_sigma:
                std = self.logstd.exp() # (num_actions)
            else:
                std = torch.exp(self.logstd(out))
            dist = GradNormal(mu, std)
            sample = dist.rsample()
            return sample
    
    def forward_with_dist(self, obs, deterministic = False):
        mu, std = self.forward_dist(obs)
            
        dist = GradNormal(mu, std)
        eps = dist.sample_eps()
        
        if deterministic:
            eps = eps.zero_()
            
        sample = dist.eps_to_action(eps)
        
        return sample, mu, std, eps
        
    def evaluate_actions_log_probs(self, obs, actions):
        mu, std = self.forward_dist(obs)
            
        dist = GradNormal(mu, std)

        return dist.log_prob(actions)

    def forward_dist(self, obs):
        out = self.actor_mlp(obs)
        mu = self.mu(out)
        if self.fixed_sigma:
            std = self.logstd.exp() # (num_actions)
        else:
            std = torch.exp(self.logstd(out))
            
        return mu, std