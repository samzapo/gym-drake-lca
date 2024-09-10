import gym_drake_lca.envs

import argparse
import gymnasium as gym

import numpy as np
import time

from pynput import keyboard
from pynput.keyboard import Key


class Runner:
    def __init__(self, args):
        self.is_closing = False

        # Create the environment
        self.env = gym.make(args.env, render_mode="human",
                            action_mode="ee", observation_mode="state")

        # Reset the environment
        observation, info = self.env.reset()

        # Sample random action
        sample = self.env.action_space.sample()
        sample[0] = 0.0
        sample[1] = 0.14
        sample[2] = 0.17
        sample[3] = 0.0
        self.init_action = sample
        print(f"init_action={self.init_action}")
        assert self.env.action_space.contains(self.init_action)

        self.action = self.init_action

        def on_press(key):
            try:
            #     print('alphanumeric key {0} pressed'.format(
            #         key.char))
                if key.char == 'r':
                    self.action = self.init_action
                elif key.char == 'a':
                    self.action[0] -= 0.01
                elif key.char == 'd':
                    self.action[0] += 0.01
                elif key.char == 'w':
                    self.action[3] -= 0.01
                elif key.char == 's':
                    self.action[3] += 0.01
            except AttributeError:
            #     print('special key {0} pressed'.format(
            #         key))
                if key == Key.up:
                    self.action[2] += 0.01
                elif key == Key.down:
                    self.action[2] -= 0.01
                elif key == Key.left:  # +y
                    self.action[1] += 0.01
                elif key == Key.right:
                    self.action[1] -= 0.01

        def on_release(key):
            # print('{0} released'.format(
                # key))
            if key == keyboard.Key.esc:
                self.is_closing = True
                # Stop listener
                return False

        listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release)
        listener.start()

    def run(self):
        # The event listener will be running in this block
        while True:
            # Step the environment
            print(f"action={self.action}")
            observation, reward, terminted, truncated, info = self.env.step(
                self.action)
            self.env.render()
            if self.is_closing:
                break

        # Close the environment
        self.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env", type=str, default="LiftCube-v0",
        help="The desired environment (see 'gym_drake_lca/__init__.py').")
    args = parser.parse_args()
    runner = Runner(args)
    runner.run()
