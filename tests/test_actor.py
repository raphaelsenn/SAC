import pytest

import torch
import numpy as np

from sac import ActorMLP


@pytest.fixture
def actor() -> ActorMLP:
    return ActorMLP(784, 400, 300, 10)


@pytest.fixture
def state_batch() -> torch.Tensor:
    return torch.randn(64, 784, dtype=torch.float32)


@pytest.fixture
def state() -> torch.Tensor:
    return torch.randn(784, dtype=torch.float32)


@pytest.fixture
def state_np() -> torch.Tensor:
    return np.random.randn(784).astype(np.float32)


class TestActorMLP:
    def test_dims(self, actor: ActorMLP) -> None:
        assert actor.state_dim == 784
        assert actor.h1_dim == 400
        assert actor.h2_dim == 300
        assert actor.action_dim == 10

    def test_forward_shape(self, actor: ActorMLP, state_batch: torch.Tensor) -> None:
        mu, std = actor(state_batch)
        assert mu.shape == (64, 10)
        assert std.shape == (64, 10)
        assert isinstance(mu, torch.Tensor)
        assert isinstance(std, torch.Tensor)

    def test_act_numpy_1(self, actor: ActorMLP, state: torch.Tensor) -> None:
        out = actor.act(state)
        assert isinstance(out, np.ndarray)
        assert out.shape == (10,)

        out = actor.act(state, deterministic=False)
        assert isinstance(out, np.ndarray)
        assert out.shape == (10,)


    def test_act_numpy_2(self, actor: ActorMLP, state_np: np.ndarray) -> None:
        out = actor.act(state_np)
        assert isinstance(out, np.ndarray)
        assert out.shape == (10,)
        
        out = actor.act(state_np, deterministic=False)
        assert isinstance(out, np.ndarray)
        assert out.shape == (10,)