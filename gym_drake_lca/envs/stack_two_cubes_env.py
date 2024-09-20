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

    The robot has to stack the blue cube on top of the red cube.

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
    | 5     | Gripper joint | Float (rad) | -1.0 | 1.0 |

    ## Observation space

    The observation space is a dictionary containing the following subspaces:

    - `"arm_qpos"`: the joint angles of the robot arm in radians, shape (6,)
    - `"arm_qvel"`: the joint velocities of the robot arm in radians per second, shape (6,)
    - `"image_front"`: the front image of the camera of size (240, 320, 3)
    - `"image_top"`: the top image of the camera of size (240, 320, 3)
    - `"cube_red_pos"`: the position of the red cube, as (x, y, z)
    - `"cube_blue_pos"`: the position of the blue cube, as (x, y, z)

    Three observation modes are available: "image" (default), "state", and "both".

    | Key               | `"image"` | `"state"` | `"both"` |
    | ----------------- | --------- | --------- | -------- |
    | `"arm_qpos"`      | ✓         | ✓         | ✓        |
    | `"arm_qvel"`      | ✓         | ✓         | ✓        |
    | `"image_front"`   | ✓         |           | ✓        |
    | `"image_top"`     | ✓         |           | ✓        |
    | `"cube_red_pos"`  |           | ✓         | ✓        |
    | `"cube_blue_pos"` |           | ✓         | ✓        |

    ## Reward

    The reward is the opposite of the distance between the top of the red cube and the blue cube.

    ## Arguments

    - `observation_mode (str)`: the observation mode, can be "image", "state", or "both", default is "image", see
        section "Observation space".
    - `action_mode (str)`: the action mode, can be "joint" or "ee", default is "joint", see section "Action space".
    - `render_mode (str)`: the render mode, can be "human" or "rgb_array", default is None.
    """

    def __init__(
        self,
        *,
        observation_mode="state",
        action_mode="joint",
        render_mode="rgb_array",
        parameters: dict | None = None,
        blue_cube_file_path: str | None = None,
        red_cube_file_path: str | None = None,
    ):
        if blue_cube_file_path is None:
            blue_cube_file_path = f"{ASSETS_PATH}/blue_cube.sdf"
        self.blue_cube_file_path = blue_cube_file_path

        if red_cube_file_path is None:
            red_cube_file_path = f"{ASSETS_PATH}/red_cube.sdf"
        self.red_cube_file_path = red_cube_file_path

        self.red_cube_low = np.array([-0.15, 0.10, 0.0075])
        self.red_cube_high = np.array([0.15, 0.25, 0.0075])

        self.blue_cube_low = np.array([-0.15, 0.10, 0.0225])
        self.blue_cube_high = np.array([0.15, 0.25, 0.0225])
        super().__init__(
            observation_mode=observation_mode,
            action_mode=action_mode,
            render_mode=render_mode,
            parameters=parameters,
        )

    def add_state_to_observation_space(self, observation_subspaces):
        observation_subspaces["blue_cube_pos"] = gym.spaces.Box(low=-10.0, high=10.0, shape=(3,))
        observation_subspaces["red_cube_pos"] = gym.spaces.Box(low=-10.0, high=10.0, shape=(3,))

    def add_objects_to_plant(self, plant: MultibodyPlant):
        parser = Parser(plant=plant)
        model_instance = parser.AddModels(self.red_cube_file_path)
        bodies = plant.GetBodyIndices(model_instance)
        assert len(bodies) == 1
        self.red_cube_body_index = bodies[0]

        model_instance = parser.AddModels(self.blue_cube_file_path)
        bodies = plant.GetBodyIndices(model_instance)
        assert len(bodies) == 1
        self.blue_cube_body_index = bodies[0]

    def calc_reward(self, plant: MultibodyPlant, plant_context: Context) -> np.float64:
        assert self.threshold_height >= 0.0

        red_cube = plant.get_body(self.red_cube_body_index)
        blue_cube = plant.get_body(self.blue_cube_body_index)

        # Get the position of the cube and the distance between the end effector and the cube
        red_cube_pos = red_cube.EvalPoseInWorld(plant_context).translation()
        blue_cube_pos = blue_cube.EvalPoseInWorld(plant_context).translation()

        blue_box_error = np.linalg.norm(blue_cube_pos - (red_cube_pos + np.array([0.0, 0.0, 0.015])))

        # Compute the reward
        reward = -blue_box_error
        return reward

    def add_state_observations(self, plant, plant_context, observations):
        observations["red_cube_pos"] = (
            plant.get_body(self.red_cube_body_index).EvalPoseInWorld(plant_context).translation()
        )
        observations["blue_cube_pos"] = (
            plant.get_body(self.blue_cube_body_index).EvalPoseInWorld(plant_context).translation()
        )

    def handle_plant_state_reset(self, plant: MultibodyPlant, plant_context: Context):
        # Set the new cube position in the context.
        red_cube = plant.get_body(self.red_cube_body_index)
        plant.SetFreeBodyPose(
            plant_context,
            red_cube,
            RigidTransform(
                RotationMatrix.MakeZRotation(np.random.uniform(0, 2 * np.pi)),
                np.random.uniform(low=self.red_cube_low, high=self.red_cube_high),
            ),
        )

        blue_cube = plant.get_body(self.blue_cube_body_index)
        plant.SetFreeBodyPose(
            plant_context,
            blue_cube,
            RigidTransform(
                RotationMatrix.MakeZRotation(np.random.uniform(0, 2 * np.pi)),
                np.random.uniform(low=self.blue_cube_low, high=self.blue_cube_high),
            ),
        )
