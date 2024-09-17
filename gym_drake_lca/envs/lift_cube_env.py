import gymnasium as gym
import numpy as np
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    MultibodyPlant,
)
from pydrake.systems.framework import (
    Context,
)

from gym_drake_lca import ASSETS_PATH

from .drake_lca_env import DrakeLcaEnv


class LiftCubeEnv(DrakeLcaEnv):
    """
    ## Description

    The robot has to lift a cube with its end-effector.

    ## Action space

    Two action modes are available: "joint" and "ee". In the "joint" mode, the action space is a 6-dimensional box
    representing the target joint angles.

    | Index | Action              | Type (unit) | Min  | Max |
    | ----- | ------------------- | ----------- | ---- | --- |
    | 0     | Shoulder pan joint  | Float (rad) | -1.0 | 1.0 |
    | 1     | Shoulder lift joint | Float (rad) | -1.0 | 1.0 |
    | 2     | Elbow flex joint    | Float (rad) | -1.0 | 1.0 |
    | 3     | Wrist flex joint    | Float (rad) | -1.0 | 1.0 |
    | 4     | Wrist roll joint    | Float (rad) | -1.0 | 1.0 |
    | 5     | Gripper joint       | Float (rad) | -1.0 | 1.0 |

    In the "ee" mode, the action space is a 4-dimensional box representing the target end-effector position and the
    gripper position.

    | Index | Action        | Type (unit) | Min  | Max |
    | ----- | ------------- | ----------- | ---- | --- |
    | 0     | X             | Float (m)   | -1.0 | 1.0 |
    | 1     | Y             | Float (m)   | -1.0 | 1.0 |
    | 2     | Z             | Float (m)   | -1.0 | 1.0 |
    | 3     | Gripper joint | Float (rad) | -1.0 | 1.0 |

    ## Observation space

    The observation space is a dictionary containing the following subspaces:

    - `"arm_qpos"`: the joint angles of the robot arm in radians, shape (6,)
    - `"arm_qvel"`: the joint velocities of the robot arm in radians per second, shape (6,)
    - `"image_front"`: the front image of the camera of size (240, 320, 3)
    - `"image_top"`: the top image of the camera of size (240, 320, 3)
    - `"cube_pos"`: the position of the cube, as (x, y, z)

    Three observation modes are available: "image" (default), "state", and "both".

    | Key             | `"image"` | `"state"` | `"both"` |
    | --------------- | --------- | --------- | -------- |
    | `"arm_qpos"`    | ✓         | ✓         | ✓        |
    | `"arm_qvel"`    | ✓         | ✓         | ✓        |
    | `"image_front"` | ✓         |           | ✓        |
    | `"image_top"`   | ✓         |           | ✓        |
    | `"cube_pos"`    |           | ✓         | ✓        |

    ## Reward

    The reward is the sum of the following terms:
        - the height of the cube above the threshold.
        - the negative distance between the end effector and the cube.


    ## Arguments

    - `observation_mode (str)`: the observation mode, can be "image", "state", or "both", default is "image", see
        section "Observation space".
    - `action_mode (str)`: the action mode, can be "joint" or "ee", default is "joint", see section "Action space".
    - `render_mode (str)`: the render mode, can be "human" or "rgb_array", default is "human".
    """

    def __init__(
        self,
        *,
        observation_mode="state",
        action_mode="joint",
        render_mode="rgb_array",
        parameters: dict | None = None,
        cube_file_path: str | None = None,
    ):
        if cube_file_path is None:
            cube_file_path = f"{ASSETS_PATH}/red_cube.sdf"

        self.cube_file_path = cube_file_path

        self.threshold_height = 0.5
        self.cube_low = np.array([-0.15, 0.10, 0.015])
        self.cube_high = np.array([0.15, 0.25, 0.015])

        super().__init__(
            observation_mode=observation_mode,
            action_mode=action_mode,
            render_mode=render_mode,
            parameters=parameters,
        )

    def add_state_to_observation_space(self, observation_subspaces):
        observation_subspaces["cube_pos"] = gym.spaces.Box(low=-10.0, high=10.0, shape=(3,))

    def add_objects_to_plant(self, plant: MultibodyPlant):
        parser = Parser(plant=plant)
        parser.AddModels(self.cube_file_path)

    def calc_reward(self, plant: MultibodyPlant, plant_context: Context) -> np.float64:
        assert self.threshold_height >= 0.0

        gripper_moving_side = plant.GetBodyByName("gripper_moving_part")
        cube = plant.GetBodyByName("cube")

        # Get the position of the cube and the distance between the end effector and the cube
        cube_pos = cube.EvalPoseInWorld(plant_context).translation()
        cube_z = cube_pos[2]
        ee_pos = gripper_moving_side.EvalPoseInWorld(plant_context).translation()
        ee_to_cube = np.linalg.norm(ee_pos - cube_pos)

        # Compute the reward
        reward_height = cube_z - self.threshold_height
        reward_distance = -ee_to_cube
        reward = reward_height + reward_distance
        return reward

    def add_state_observations(self, plant, plant_context, observations):
        cube = plant.GetBodyByName("cube")
        cube_pos = cube.EvalPoseInWorld(plant_context).translation()
        observations["cube_pos"] = cube_pos

    def handle_plant_state_reset(self, plant: MultibodyPlant, plant_context: Context):
        # Randomize a new cube position.
        cube_pos = np.random.uniform(low=self.cube_low, high=self.cube_high)
        cube_rot = RotationMatrix.MakeZRotation(np.random.uniform(0, 2 * np.pi))

        # Set the new cube position in the context.
        cube = plant.GetBodyByName("cube")
        plant.SetFreeBodyPose(plant_context, cube, RigidTransform(cube_rot, cube_pos))
