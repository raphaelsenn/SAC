import os

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import gymnasium as gym

from sac.actor import Actor
from sac.critic import Critic
from sac.replay_buffer import ReplayBuffer


class SAC:
    """
    Soft Actor-Critic (SAC) algorithm.

    Reference:
    ----------
    Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
    with a Stochastic Actor, Haarnoja et al., 2018.
    https://arxiv.org/abs/1801.01290
    """
    def __init__(
            self,
            actor: Actor,
            critic: Critic,
            lr_actor: float,
            lr_critic: float,
            timesteps: int,
            gradient_steps: int,
            gamma: float,
            tau: float,
            entropy_coef: float,
            batch_size: int,
            reward_scale: float,
            weight_decay_actor: float=0.0,
            weight_decay_critic: float=0.0,
            buffer_capacity: int=1_000_000,
            buffer_start_size: int=25_000,
            n_eval_runs: int=10,
            update_target_freq: int=1_000,
            eval_every: int=5_000,
            save_every: int=10_000,
            seed: int=0,
            device: str="cpu",
            verbose: bool=True
    ) -> None:
        self.device = torch.device(device)

        # Neural nets
        self.actor = actor
        self.critic = critic
        self.critic_target = critic.copy()

        self.actor.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        # Optimizers
        self.optimizer_actor = torch.optim.Adam(
            self.actor.parameters(),
            lr=lr_actor,
            weight_decay=weight_decay_actor
        )
        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(),
            lr=lr_critic,
            weight_decay=weight_decay_critic
        )

        # Loss
        self.criterion_critic = nn.MSELoss()

        # Learning hyperparameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        self.entropy_coef = entropy_coef
        self.reward_scale = reward_scale

        # Training settings
        self.timesteps = timesteps
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.buffer_start_size = buffer_start_size
        self.buffer_capacity = buffer_capacity
        self.update_target_freq = update_target_freq

        # Evaluation and logging stuff
        self.n_eval_runs = n_eval_runs
        self.eval_every = eval_every
        self.save_every = save_every
        self.verbose = verbose
        self.seed = seed
        self.env_id = None

        # Shapes
        self.obs_shape = actor.obs_shape
        self.action_dim = actor.action_dim
        self.action_scale = actor.action_scale

        assert reward_scale > 0.0, (
            f"reward_scale needs to be > 0, got: {reward_scale}"
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            self.obs_shape,
            self.action_dim,
            buffer_capacity,
            batch_size,
            self.device
        )

        # Stats
        self.stats = {"t": [], "average_return": [], "std_return": []}

    @torch.no_grad()
    def get_action(self, x: np.ndarray, deterministic: bool=False) -> np.ndarray:
        """Expects unbatched input.""" 
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)   # [1, state_dim]
        a_t = self.actor.act(x_t, deterministic=deterministic)              # [1, action_dim]
        return a_t                                                          # [action_dim]

    def update_networks(self) -> None:
        self.actor.train(); self.critic.train()
        self.critic_target.eval() 

        # Sample random minibatch of N transitions
        for _ in range(self.gradient_steps): 
            s, a, r, s_nxt, d = self.replay_buffer.sample()

            with torch.no_grad():
                # Sample action + log probability
                a_nxt, log_prob_nxt = self.actor.sample(s_nxt)             # [B, action_dim], [B, action_dim]

                # Computing TD target 
                q1_nxt_tgt, q2_nxt_tgt = self.critic_target(s_nxt, a_nxt)  # [B, 1], [B, 1]
                q_nxt_tgt = torch.min(q1_nxt_tgt, q2_nxt_tgt).view(-1)     # [B]
                td_target = r + self.gamma * (1.0 - d) * (
                    q_nxt_tgt - self.entropy_coef * log_prob_nxt)          # [B]

            # Update critic
            # Compute critic loss
            q1, q2 = self.critic(s, a)                                      # [B, 1], [B, 1]
            loss_q1 = self.criterion_critic(q1.view(-1), td_target)         # [1]
            loss_q2 = self.criterion_critic(q2.view(-1), td_target)         # [1]

            self.optimizer_critic.zero_grad()
            loss_critic = loss_q1 + loss_q2                                 # [1]
            loss_critic.backward()
            self.optimizer_critic.step()
            
            # Update actor network
            for p in self.critic.parameters():
                p.requires_grad = False 
            
            # a_tilde is sampled from actor and needs to be differentiable w.r.t. actor params!!! 
            a_tilde, log_prob = self.actor.sample(s)

            # Computing actor loss
            q1, q2 = self.critic(s, a_tilde)                                # [B, 1], [B, 1]
            q = torch.min(q1, q2).view(-1)                                  # [B]
            loss_actor = torch.mean(self.entropy_coef*log_prob - q)         # [1]

            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            self.optimizer_actor.step()

            for p in self.critic.parameters():
                p.requires_grad = True

    @torch.no_grad()
    def update_target_networks(self) -> None:
        # Update critic (target) 
        for theta, theta_old in zip(self.critic.parameters(), self.critic_target.parameters()):
            theta_old.data.copy_(self.tau * theta_old.data + (1.0 - self.tau) * theta.data) 
        
    def train(self, env: gym.Env) -> None:
        self.explore_env(env)
        
        episode_num = 0
        done = False 
        s, _ = env.reset() 
        for t in range(self.timesteps): 
            # Sample action, observe reward and next state 
            a = self.get_action(s, deterministic=False) 
            s_nxt, r, terminated, truncated, _ = env.step(a)

            # Rescale reward, check if done 
            r_scaled = r * self.reward_scale
            done = terminated or truncated

            # Update replay memory 
            self.replay_buffer.push(s, a, r_scaled, s_nxt, terminated)

            # Update neural nets
            self.update_networks()
            if t > 0 and t % self.update_target_freq == 0: 
                self.update_target_networks()

            # Update state 
            s = s_nxt

            if done:
                s, _ = env.reset()
                episode_num += 1

            # Evaluation, checkpoints, print report
            self.handle_periodic_tasks(t, episode_num)

        self.evaluate(self.timesteps)
        self.checkpoint(self.timesteps)
        env.close()

    def explore_env(self, env: gym.Env) -> None:
        if self.env_id is None: self.env_id = env.spec.id

        s, _ = env.reset(seed=self.seed)
        env.action_space.seed(self.seed)

        done = False
        for _ in range(self.buffer_start_size): 
            a = env.action_space.sample()
            s_nxt, r, terminated, truncated, _ = env.step(a)
            
            done = terminated or truncated
            r_scaled = r * self.reward_scale

            self.replay_buffer.push(s, a, r_scaled, s_nxt, terminated)
            s = s_nxt

            if done:
                s, _ = env.reset()
                done = False

    @torch.inference_mode() 
    def evaluate(self, step: int) -> None:
        self.actor.eval(); self.critic.eval()

        done = False
        env = gym.make(self.env_id, render_mode=None)
        s, _ = env.reset(seed=self.seed + 100)
        
        rewards = np.zeros(self.n_eval_runs, dtype=np.float32)
        for n in range(self.n_eval_runs):
            while not done:
                a = self.get_action(s, deterministic=True)
                s_nxt, reward, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                s = s_nxt
                rewards[n] += reward
            done = False
            s, _ = env.reset()

        env.close()
        self.update_stats(step, rewards)

    def update_stats(self, step: int, rewards: np.ndarray) -> None:
        mean_ep_reward = float(np.mean(rewards))
        std_ep_reward = float(np.std(rewards))
        self.stats["t"].append(step)
        self.stats["average_return"].append(mean_ep_reward)
        self.stats["std_return"].append(std_ep_reward)

    def handle_periodic_tasks(self, step: int, ep_num: int) -> None:
        if step > 0 and step % self.eval_every == 0:
            self.evaluate(step)
            average_return = self.stats["average_return"][-1]
            if self.verbose: 
                print(
                    f"Total T: {step:6d}\t"
                    f"Episode: {ep_num:5d}\t"
                    f"Average Return: {average_return:10.3f}"
                )

        if step > 0 and step % self.save_every == 0:
            self.checkpoint(step) 

    def checkpoint(self, step: int) -> None:
        save_dir = f"{self.env_id}-SAC-Checkpoints-Seed{self.seed}"
        os.makedirs(save_dir, exist_ok=True)

        file_name = f"{self.env_id}-SAC-Actor-Lr{self.lr_actor}-t{step}-Seed{self.seed}.pt"
        file_name = os.path.join(save_dir, file_name) 
        torch.save(self.actor.state_dict(), file_name)
        
        file_name = f"{self.env_id}-SAC-Critic-Lr{self.lr_actor}-t{step}-Seed{self.seed}.pt"
        file_name = os.path.join(save_dir, file_name)
        torch.save(self.critic.state_dict(), file_name)

        file_name = f"{self.env_id}-SAC-Seed{self.seed}.csv"
        pd.DataFrame.from_dict(self.stats).to_csv(file_name, index=False)