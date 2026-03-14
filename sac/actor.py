import copy
from typing import Tuple
from abc import ABC, abstractmethod

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(nn.Module, ABC):
    """Actor interface for deterministic policies.""" 
    def __init__(
            self,
            obs_shape: Tuple[int, ...],
            action_dim: int,
            action_scale: float=1.0
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got: {action_dim}") 
        if len(obs_shape) == 0:
            raise ValueError(f"obs_dim must be non-empty, got: {obs_shape}")

        self.obs_shape = tuple(int(element) for element in obs_shape)
        self.action_dim = action_dim
        self.action_scale = action_scale

    @abstractmethod
    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns actions with shape [B, action_dim].""" 
        raise NotImplementedError

    @torch.inference_mode()
    def act(self, s: np.ndarray | torch.Tensor, deterministic: bool=True) -> np.ndarray:
        device = next(self.parameters()).device 
        if isinstance(s, np.ndarray):
            s_t = torch.as_tensor(s, dtype=torch.float32, device=device)
        else:
            s_t = s.to(device)

        if s_t.dim() == len(self.obs_shape):
            s_t = s_t.unsqueeze(0)
        mu, log_std = self(s_t)
        
        if deterministic:
            action = mu
        else: 
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX) 
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mu, std)
            action = dist.rsample()

        action = torch.tanh(action) * self.action_scale
        action = action.detach().cpu().view(-1).numpy()

        return action

    def sample(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self(s)                                       # [B, action_dim], [B, action_dim]
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX) 
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)

        # Reparametrization trick       
        a_pre_tanh = dist.rsample()                                 # [B, action_dim]
        a_tanh = torch.tanh(a_pre_tanh)                             # [B, action_dim] 
        a = self.action_scale * a_tanh                              # [B, action_dim]

        # Compute correct log-probs
        log_probs = dist.log_prob(a_pre_tanh).sum(dim=-1)           # [B]
        
        # NOTE: Instable variant 
        #log_probs -= torch.log(
        #    self.action_scale * (1 - a_tanh.pow(2)) + EPS
        #).sum(dim=-1)                                              # [B]

        # NOTE: More stable variant
        log_probs -= (
            2*(np.log(2) - a_pre_tanh - F.softplus(-2*a_pre_tanh))
        ).sum(dim=1)                                                # [B]


        return a, log_probs

    def copy(self) -> 'Actor':
        actor = copy.deepcopy(self)
        return actor
    

class ActorMLP(Actor):
        """Squashed gaussian multilayer-perceptron.""" 
        def __init__(
                self, 
                state_dim: int, 
                h1_dim: int,
                h2_dim: int,
                action_dim: int,
                action_scale: float=1.0
        ) -> None:
            super().__init__((state_dim,), action_dim, action_scale)
            self.state_dim = state_dim
            self.h1_dim = h1_dim
            self.h2_dim = h2_dim

            self.mlp = nn.Sequential(
                nn.Linear(state_dim, h1_dim),
                nn.ReLU(True),

                nn.Linear(h1_dim, h2_dim),
                nn.ReLU(True),
            )

            self.mu = nn.Linear(h2_dim, action_dim)
            self.log_std = nn.Linear(h2_dim, action_dim)

        def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            if s.dim() == len(self.obs_shape):
                s = s.unsqueeze(0)
            
            hs = self.mlp(s) 
            mu = self.mu(hs)
            log_std = self.log_std(hs)
            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std)

            return mu, std