import imageio
import gymnasium as gym
import numpy as np
import gym_drake_lca

env = gym.make("LiftCube-v0")
observation, info = env.reset()
frames = []

N = 100
for i in range(N):
    print(f"{100 * i/N}%")
    env.reset()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    frames.append(image)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
imageio.mimsave("example.mp4", np.stack(frames), fps=25)
