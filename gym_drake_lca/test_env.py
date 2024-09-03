import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import envs # Import environments

import argparse
import gymnasium as gym

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env", type=str, default="LiftCube-v0",
        help="The desired environment (see 'gym_drake_lca/__init__.py').")
    args = parser.parse_args()

    # Create the environment
    env = gym.make(args.env, render_mode="human")

    # Reset the environment
    observation, info = env.reset()

    for _ in range(1000):
        # Sample random action
        action = env.action_space.sample()

        # Step the environment
        observation, reward, terminted, truncated, info = env.step(action)

        # Reset the environment if it's done
        if terminted or truncated:
            observation, info = env.reset()

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()