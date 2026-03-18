import pytest

import numpy as np
import gymnasium as gym
from sac import SAC, ActorMLP, CriticMLP

ENV_ID = "MountainCarContinuous-v0"
H1_DIM = 256
H2_DIM = 256


class TestSAC:
    def test_env_agent_action_shape_and_bounds(self):
        env = gym.make(ENV_ID)  
        obs, _ = env.reset(seed=0)
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]   
        action_scale = env.action_space.high[0]
        actor = ActorMLP(state_dim, H1_DIM, H2_DIM, action_dim, action_scale)
        
        action = actor.act(obs, deterministic=True)

        action = np.asarray(action, dtype=np.float32)

        assert action.shape == env.action_space.shape
        assert np.all(np.isfinite(action))
        assert np.all(action >= env.action_space.low)
        assert np.all(action <= env.action_space.high) 
    
    def test_mountain_car(self) -> None:
        env = gym.make("MountainCarContinuous-v0")
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]   
        action_scale = env.action_space.high[0]
        actor = ActorMLP(state_dim, H1_DIM, H2_DIM, action_dim, action_scale)
        critic = CriticMLP(state_dim, H1_DIM, H2_DIM, action_dim)

        sac = SAC(
            actor, 
            critic, 
            lr_actor=0.0003,
            lr_critic=0.0003,
            timesteps=10_000,
            gradient_steps=1,
            gamma=0.99,
            tau=0.995,
            reward_scale=5.0,
            buffer_capacity=10_000,
            buffer_start_size=100,
            batch_size=32,
            eval_every=1_000_000,
            save_every=1_000_000
        )

        sac.train(env)

        returns = []
        for ep in range(3):
            obs, _ = env.reset(seed=100+ep)
            total_reward = 0.0

            for _ in range(999):
                action = np.asarray(actor.act(obs, deterministic=True), dtype=np.float32)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break

            returns.append(total_reward)
        mean_return = float(np.mean(returns))
        assert mean_return > -5.0

