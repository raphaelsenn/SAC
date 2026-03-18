from argparse import Namespace, ArgumentParser

import torch
import gymnasium as gym

from sac import ActorMLP


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Online Evaluation")

    parser.add_argument("--env_id", type=str, default="Pusher-v5")
    parser.add_argument("--action_scale", type=float, default=2.0)
    parser.add_argument("--h1_dim", type=int, default=256)
    parser.add_argument("--h2_dim", type=int, default=256)
    parser.add_argument("--weights", type=str, default="Pusher-v5-SAC-Checkpoints-Seed0/Pusher-v5-SAC-Actor-Lr0.0003-t3000000-Seed0.pt")

    parser.add_argument("--verbose", default=True)

    return parser.parse_args()


def play(env: gym.Env, actor: ActorMLP, n_episodes: int=100) -> None:
    for ep in range(n_episodes):
        done = False
        s, _ = env.reset()
        t = 0 
        reward_sum = 0.0
        while not done:
            a = actor.act(s, deterministic=True).flatten()
            # a = env.action_space.sample() 
            s_nxt, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            s = s_nxt
            reward_sum += reward
            t += 1 
    env.close()


def main() -> None:
    args = parse_args()

    env = gym.make(args.env_id, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = ActorMLP(state_dim, args.h1_dim, args.h2_dim, action_dim, args.action_scale)
    actor.load_state_dict(torch.load(args.weights, weights_only=True, map_location="cpu")) 
    play(env, actor)


if __name__ == "__main__":
    main()