import os

from gymnasium.envs.registration import register

__version__ = "0.0.4"

ASSETS_PATH = os.path.join(os.path.dirname(__file__), "assets")

register(
    id="LiftCube-v0",
    entry_point="gym_drake_lca.envs:LiftCubeEnv",
    max_episode_steps=500,
)
