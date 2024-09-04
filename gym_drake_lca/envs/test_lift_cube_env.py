from gym_drake_lca.envs.lift_cube_env import LiftCubeEnv

import argparse
import pydrake
import logging

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render_mode", type=str, default="human")
    parser.add_argument("--observation_mode", type=str, default="both")
    parser.add_argument("--action_mode", type=str, default="joint")
    args = parser.parse_args()

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    pydrake.common.configure_logging()

    # Create the environment
    env = LiftCubeEnv(observation_mode=args.observation_mode,
                      action_mode=args.action_mode, render_mode=args.render_mode)

    # Reset the environment
    observation, info = env.reset()

    ax = {}
    is_displaying_images = (args.render_mode in ["rgb_array"]) or (args.observation_mode in ["image", "both"])
    if is_displaying_images:
        fig, ((_, ax["image_top"]), (ax["image_front"], ax["rgb_array"])
              ) = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all')
        plt.axis("off")  # Hide axis
        fig.show()

    for iter in range(1000):
        # Sample random action
        action = env.action_space.sample()

        print(f"\niter={iter}")
        print(f"action={action}")
        # Step the environment
        observations, reward, terminted, truncated, info = env.step(action)

        print(f"reward={reward}")
        print("observations=")
        for k, v in observations.items():
            print(f"\t{k} : {v.shape}")
            if len(v.shape) == 3:  # check if it's an image
                ax[k].imshow(v)

        if args.render_mode in ["rgb_array"]:
            # display image
            image = env.render()
            print(f"image shape:{image.shape}")

            plt.axis("off")  # Hide axis
            ax[args.render_mode].imshow(image)

        elif args.render_mode in ["human"]:
            # Render every step
            env.render()

        if is_displaying_images:
            fig.canvas.draw()
            plt.pause(0.01)

        # Reset the environment if it's done
        if terminted or truncated:
            print("Env was reset!")
            observation, info = env.reset()

    # Close the environment
    env.close()
    plt.close(fig)


if __name__ == "__main__":
    main()
