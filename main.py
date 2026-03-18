from argparse import Namespace, ArgumentParser

from sac import SAC
from sac import ActorMLP, CriticMLP

import torch
import gymnasium as gym
import numpy as np


def parse_args() -> Namespace:
    parser = ArgumentParser(description="SAC training")

    parser.add_argument("--env_id", type=str, default="HalfCheetah-v5")
    parser.add_argument("--num_timesteps", type=int, default=1_000_000)
    parser.add_argument("--num_gradient_steps", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--h1_dim", type=int, default=256)
    parser.add_argument("--h2_dim", type=int, default=256)

    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_critic", type=float, default=3e-4)
    parser.add_argument("--weight_decay_actor", type=float, default=0.0)
    parser.add_argument("--weight_decay_critic", type=float, default=0.0)

    parser.add_argument("--buffer_capacity", type=int, default=1_000_000)
    parser.add_argument("--buffer_start_size", type=int, default=25_000)
    parser.add_argument("--batch_size", type=int, default=100)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.995)         # OpenAI Spinning Up version w_ <- tau * w_ + (1 - tau) * w
    parser.add_argument("--action_scale", type=float, default=1.0)
    parser.add_argument("--reward_scale", type=float, default=5.0)

    parser.add_argument("--update_target_freq", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--eval_every", type=int, default=5_000)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", default=True)

    return parser.parse_args()


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    
    env = gym.make(args.env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = ActorMLP(state_dim, args.h1_dim, args.h2_dim, action_dim, args.action_scale)
    critic = CriticMLP(state_dim, args.h1_dim, args.h2_dim, action_dim)

    sac = SAC(
        actor=actor,
        critic=critic,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        timesteps=args.num_timesteps,
        gradient_steps=args.num_gradient_steps,
        gamma=args.gamma,
        tau=args.tau,
        batch_size=args.batch_size,
        reward_scale=args.reward_scale,
        weight_decay_actor=args.weight_decay_actor,
        weight_decay_critic=args.weight_decay_critic,
        buffer_capacity=args.buffer_capacity,
        buffer_start_size=args.buffer_start_size,
        eval_every=args.eval_every,
        save_every=args.save_every,
        update_target_freq=args.update_target_freq,
        seed=args.seed,
        device=args.device,
    )
    sac.train(env)


if __name__ == "__main__":
    main()