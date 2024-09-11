# gym-drake-lca

A gym environment for [Low-Cost Robot Arm](https://github.com/AlexanderKoch-Koch/low_cost_robot) in [Drake](https://github.com/RobotLocomotion/drake).

## Installation

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html):
```bash
conda create -y -n gym-drake-lca python=3.10 && conda activate gym-drake-lca
```

Install gym-drake-lca:
```bash
pip install gym-drake-lca
```


## Quickstart

```python
# example.py
import imageio
import gymnasium as gym
import numpy as np
import gym_drake_lca

env = gym.make("LiftCube-v0")
observation, info = env.reset()
frames = []

N = 100
for i in range(N):
    env.reset()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    frames.append(image)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
imageio.mimsave("example.mp4", np.stack(frames), fps=25)
```

## Contribute

Instead of using `pip` directly, we use `poetry` for development purposes to easily track our dependencies.
If you don't have it already, follow the [instructions](https://python-poetry.org/docs/#installation) to install it.

Install the project with dev dependencies:
```bash
poetry install --all-extras
```


### Follow our style

```bash
# install pre-commit hooks
pre-commit install

# apply style and linter checks on staged files
pre-commit
```

## Acknowledgment

These instrutions are adapted from [gym-aloha](https://github.com/huggingface/gym-aloha)
This project is adapted from [gym-lowcostrobot](https://github.com/perezjln/gym-lowcostrobot)
