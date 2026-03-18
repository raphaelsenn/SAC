import copy
from typing import Tuple
from abc import ABC, abstractmethod

import numpy as np

import torch
import torch.nn as nn

from sac.utils import ensure_tensorf32


class Critic(nn.Module, ABC):
    """Critic interface for an action-value function.""" 
    def __init__(
            self, 
            obs_shape: Tuple[int, ...], 
            action_dim: int
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got: {action_dim}") 
        if len(obs_shape) == 0:
            raise ValueError(f"obs_shape must be non-empty, got: {obs_shape}")

        self.obs_shape = tuple(int(element) for element in obs_shape)
        self.action_dim = action_dim

    @abstractmethod
    def forward(
        self, 
        s: torch.Tensor, 
        a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @torch.inference_mode()
    def predict(
        self, 
        s: np.ndarray | torch.Tensor, 
        a: np.ndarray | torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        s_t = ensure_tensorf32(s, next(self.parameters()).device)
        a_t = ensure_tensorf32(a, next(self.parameters()).device)
        q1, q2 = self(s_t, a_t)
        return q1.detach().cpu().numpy(), q2.detach().cpu().numpy()

    def copy(self) -> 'Critic':
        critic = copy.deepcopy(self)
        return critic


class CriticMLP(Critic):
        def __init__(
                self, 
                state_dim: int, 
                h1_dim: int,
                h2_dim: int,
                action_dim: int,

        ) -> None:
            super().__init__((state_dim,), action_dim)
            self.state_dim = state_dim
            self.h1_dim = h1_dim
            self.h2_dim = h2_dim

            self.q1 = nn.Sequential(
                # [B, state_dim + action_dim] -> [B, h1_dim] 
                nn.Linear(state_dim + action_dim, h1_dim),
                nn.ReLU(True),
                
                # [B, h1_dim] -> [B, h2_dim] 
                nn.Linear(h1_dim, h2_dim),
                nn.ReLU(True),
                
                # [B, h2_dim] -> [B, 1] 
                nn.Linear(h2_dim, 1),
            
            )

            self.q2 = nn.Sequential(
                # [B, state_dim + action_dim] -> [B, h1_dim] 
                nn.Linear(state_dim + action_dim, h1_dim),
                nn.ReLU(True),
                
                # [B, h1_dim] -> [B, h2_dim] 
                nn.Linear(h1_dim, h2_dim),
                nn.ReLU(True),
                
                # [B, h2_dim] -> [B, 1]
                nn.Linear(h2_dim, 1),
            )
            
        def forward(
                self, 
                s: torch.Tensor, 
                a: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            x = torch.cat([s, a], dim=-1)
            q1 = self.q1(x)
            q2 = self.q2(x)
            return q1, q2