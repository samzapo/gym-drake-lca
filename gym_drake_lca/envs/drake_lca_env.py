from contextlib import suppress

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from pydrake.common.value import (
    Value,
)
from pydrake.geometry import (
    ClippingRange,
    ColorRenderCamera,
    DepthRange,
    DepthRenderCamera,
    EnvironmentMap,
    EquirectangularMap,
    MakeRenderEngineVtk,
    RenderCameraCore,
    RenderEngineVtkParams,
)
from pydrake.gym import DrakeGymEnv
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.inverse_kinematics import (
    DifferentialInverseKinematicsParameters,
)
from pydrake.multibody.math import SpatialForce
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlant,
    ExternallyAppliedSpatialForce_,
    MultibodyPlant,
    MultibodyPlantConfig,
)
from pydrake.multibody.tree import (
    JacobianWrtVariable,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.controllers import (
    PidController,
)
from pydrake.systems.drawing import plot_graphviz, plot_system_graphviz
from pydrake.systems.framework import (
    Context,
    DiagramBuilder,
    EventStatus,
    LeafSystem,
)
from pydrake.systems.primitives import (
    ConstantVectorSource,
    Multiplexer,
)
from pydrake.systems.sensors import (
    CameraInfo,
    ImageRgba8U,
    RgbdSensor,
)
from pydrake.visualization import (
    ApplyVisualizationConfig,
    VisualizationConfig,
)

from gym_drake_lca import ASSETS_PATH


def add_robot(plant, parser):
    (robot_model_instance,) = parser.AddModels(f"{ASSETS_PATH}/low-cost-arm.urdf")

    identity = RigidTransform.Identity()
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link", robot_model_instance), identity)
    return robot_model_instance


def construct_robot_plant(time_step):
    # Construct Robot-only Plant
    robot_plant = MultibodyPlant(time_step)
    parser = Parser(plant=robot_plant)
    add_robot(robot_plant, parser)
    robot_plant.Finalize()
    return robot_plant


class DifferentialIKIntegrator(LeafSystem):
    def __init__(self, plant, params: DifferentialInverseKinematicsParameters):
        LeafSystem.__init__(self)
        self.params = params

        self.construct_robot_ik_plant_and_context()

        self.state_input_port_index = self.DeclareVectorInputPort(
            "estimated_state", self.plant.num_multibody_states()
        ).get_index()

        self.ee_goal_input_port_index = self.DeclareVectorInputPort(
            "ee_goal", 3 + 1
        ).get_index()  # [x, y, z, gripper]

        # TODO: Move to parameters.
        self.end_effector = self.plant.GetBodyByName("gripper_static_part")
        self.p_BoBi_B = np.array([0, 0.05, 0])

        assert self.thumb_joint.num_velocities() == 1

        self.DeclareVectorOutputPort(
            "desired_state", self.plant.num_multibody_states(), self.calc_desired_state
        )

    def SetDefaultContext(self, context):  # noqa: N802
        LeafSystem.SetDefaultContext(self, context)
        self.plant.SetDefaultContext(self.plant_context)

    def calc_desired_state(self, context, output):
        max_iter: int = 10
        iter: int = 0
        dist: np.float32 = np.inf
        dist_tol = 0.001
        while (iter < max_iter) and (dist > dist_tol):
            iter = iter + 1
            dist = self.integrate_ik(context)
        next_q_v = self.plant.GetPositionsAndVelocities(self.plant_context)
        output.set_value(next_q_v)
        # input("press to continue")

    def construct_robot_ik_plant_and_context(self):
        # Construct Robot-only Plant
        self.plant = construct_robot_plant(0)
        self.thumb_joint = self.plant.GetJointByName("joint_7")

        self.plant_context = self.plant.CreateDefaultContext()
        # Lock thumb in place (exclude from jacobian).
        self.thumb_joint.Lock(self.plant_context)

    def integrate_ik(self, context):
        q = self.plant.GetPositions(self.plant_context)
        v = self.plant.GetVelocities(self.plant_context)

        ee_goal = self.get_input_port(self.ee_goal_input_port_index).Eval(context)

        thumb_angle = ee_goal[3]

        p_WoBo_W_des = ee_goal[0:3]  # noqa: N806
        p_WoBo_W = self.end_effector.EvalPoseInWorld(self.plant_context).translation()  # noqa: N806

        dt = self.params.get_time_step()

        v_WBo_W = (p_WoBo_W_des - p_WoBo_W) / dt  # noqa: N806

        v_next = v
        q_next = q

        # Calc IK Step
        Js_v_WBo = self.plant.CalcJacobianTranslationalVelocity(  # noqa: N806
            context=self.plant_context,
            with_respect_to=JacobianWrtVariable.kV,
            frame_B=self.end_effector.body_frame(),
            p_BoBi_B=self.p_BoBi_B,
            frame_A=self.plant.world_frame(),
            frame_E=self.plant.world_frame(),
        )
        pinv = Js_v_WBo.transpose()
        with suppress(np.linalg.LinAlgError):
            pinv = np.linalg.pinv(Js_v_WBo)

        v_next = pinv @ v_WBo_W
        (v_lb, v_ub) = self.params.get_joint_velocity_limits()
        for i in range(v_next.shape[0]):
            if v_next[i] < v_lb[i]:
                v_next[i] = v_lb[i]
            elif v_ub[i] < v_next[i]:
                v_next[i] = v_ub[i]

        # integrate IK step
        q_step = dt * v_next
        q_next = q + q_step

        q_next[self.thumb_joint.position_start()] = thumb_angle
        v_next[self.thumb_joint.velocity_start()] = 0

        self.plant.SetPositions(self.plant_context, q_next)
        # Zero velocities.
        self.plant.SetVelocities(self.plant_context, v_next * 0.0)

        p_WoBo_W = self.end_effector.EvalPoseInWorld(self.plant_context).translation()  # noqa: N806
        dist = np.linalg.norm(p_WoBo_W_des - p_WoBo_W)

        return dist


def construct_lift_cube_env_default_params():
    # Tunable parameters
    image_dims = [240, 320]
    image_fov = np.pi / 4
    return {
        "camera_data": {
            "image_front": {
                "intrinsics": CameraInfo(image_dims[1], image_dims[0], image_fov),
                "extrinsics": RigidTransform(
                    RotationMatrix.MakeXRotation(np.pi / 2 + np.pi / 32).multiply(
                        RotationMatrix.MakeZRotation(-np.pi)
                    ),
                    np.array([0.0, 0.6, 0.15]),
                ),
            },
            "image_top": {
                "intrinsics": CameraInfo(image_dims[1], image_dims[0], image_fov),
                "extrinsics": RigidTransform(RotationMatrix.MakeXRotation(np.pi), np.array([0, 0.1, 0.6])),
            },
            "image_viewer": {
                "intrinsics": CameraInfo(image_dims[1], image_dims[0], image_fov),
                "extrinsics": RigidTransform(
                    RotationMatrix.MakeZRotation(np.pi / 4).multiply(
                        RotationMatrix.MakeXRotation(5 * np.pi / 8).multiply(
                            RotationMatrix.MakeZRotation(-np.pi)
                        )
                    ),
                    np.array([-0.4, 0.4, 0.3]),
                ),
            },
        },
        "observation_cameras": ["image_front", "image_top"],
        "rgb_array_camera": "image_viewer",
        "sim_time_step": 0.0,
        "gym_time_step": 0.1,
        "gym_time_limit": np.inf,
        # contact_models: 'point', 'hydroelastic_with_fallback'
        "contact_model": "hydroelastic_with_fallback",
        # contact_approximations: 'sap', 'tamsi', 'similar', 'lagged'
        "contact_approximation": "sap",
        "pid_gains": {"kp": 1.0, "ki": 0.0, "kd": 0.1},
        "lift_reward_threshold_height": 0.5,
        "obs_state_noise_magnitude": 0.0,
        "external_force_disturbances": {"magnitude": 0.0, "period": 1.0, "duration": 0.1},
        "emit_debug_printout": False,
        "ik_time_step": 0.001,
        "ik_velocity_limit_factor": 1.0,
        "joint_max_velocities": 10.0,
    }


class DrakeLcaEnv(DrakeGymEnv):
    # Derived class must implement this function
    def add_state_to_observation_space(self, observation_subspaces):
        raise NotImplementedError("add_state_to_observation_space")

    # Derived class must implement this function
    def add_state_observations(self, plant, plant_context, observations):
        raise NotImplementedError("add_state_observations")

    # Derived class must implement this function
    def add_objects_to_plant(self, plant: MultibodyPlant):
        raise NotImplementedError("add_objects_to_plant")

    # Derived class must implement this function
    def calc_reward(self, plant: MultibodyPlant, plant_context: Context) -> np.float64:
        raise NotImplementedError("calc_reward")

    # Derived class must implement this function
    def handle_plant_state_reset(self, plant: MultibodyPlant, plant_context: Context):
        raise NotImplementedError("handle_plant_state_reset")

    """
    ## Description

    A gym environment for the Low Cost Robot Arm that allowes a wrapper class to determine behavior.

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
    - (optional) `state` list items provided by derived class.

    Three observation modes are available: "image" (default), "state", and "both".

    | Key             | `"image"` | `"state"` | `"both"` |
    | --------------- | --------- | --------- | -------- |
    | `"arm_qpos"`    | ✓         | ✓         | ✓        |
    | `"arm_qvel"`    | ✓         | ✓         | ✓        |
    | `"image_front"` | ✓         |           | ✓        |
    | `"image_top"`   | ✓         |           | ✓        |
    | <derived class> |           | ✓         | ✓        |
    |       ...       |           | ✓         | ✓        |

    ## Reward

    The reward is calculated by derived class.

    ## Arguments

    - `observation_mode (str)`: the observation mode, can be "image", "state", or "both", default is "image", see
        section "Observation space".
    - `action_mode (str)`: the action mode, can be "joint" or "ee", default is "joint", see section "Action space".
    - `render_mode (str)`: the render mode, can be "human" or "rgb_array", default is "human".
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "observation_modes": ["image", "state", "both"],
        "action_modes": ["joint", "ee"],
    }

    def __init__(
        self,
        *,
        observation_mode="state",
        action_mode="joint",
        render_mode="rgb_array",
        parameters: dict | None = None,
    ):
        self.parameters = construct_lift_cube_env_default_params()
        if parameters is not None:
            for key, value in parameters.items():
                current_value = self.parameters[key]
                assert type(value) == type(current_value)
                self.parameters[key] = value

        print(f"observation_mode={observation_mode}")
        print(f"action_mode={action_mode}")
        print(f"render_mode={render_mode}")
        assert render_mode in self.metadata["render_modes"]
        assert observation_mode in self.metadata["observation_modes"]
        assert action_mode in self.metadata["action_modes"]

        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.render_mode = render_mode

        # Make simulation.
        simulator = self.construct_simulator(
            debug=self.parameters["emit_debug_printout"],
        )

        super().__init__(
            simulator=simulator,
            time_step=self.parameters["gym_time_step"],
            action_space=self.construct_action_space(),
            observation_space=self.construct_observation_space(),
            reward="reward",
            action_port_id="actions",
            observation_port_id="observations",
            render_rgb_port_id={
                "rgb_array": self.parameters["rgb_array_camera"],
                "human": None,
                "ansi": None,
            }[render_mode],
            render_mode=render_mode,
            reset_handler=self.handle_reset,
            # FIXME(samzapo): Add the following line once Drake PR #21900 lands.
            # info_handler=self.info_handler,
        )

    def info_handler(self, simulator: Simulator) -> dict:
        info = {"timestamp": simulator.get_context().get_time()}
        return info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        # FIXME(samzapo): Remove the following line once Drake PR #21900 lands.
        info["timestamp"] = self.simulator.get_context().get_time()
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        # FIXME(samzapo): Remove the following line once Drake PR #21900 lands.
        info["timestamp"] = self.simulator.get_context().get_time()
        return observation, info

    def construct_action_space(self):
        if self.action_mode == "joint":
            action_shape = 5
            return gym.spaces.Box(low=-np.pi, high=np.pi, shape=(action_shape,), dtype=np.float32)
        elif self.action_mode == "ee":
            action_shape = 4
            return gym.spaces.Box(
                low=np.array([-1, -1, 0, -np.pi]), high=np.array([1, 1, 1, np.pi]), dtype=np.float32
            )

    def construct_observation_space(self):
        max_v = self.parameters["joint_max_velocities"]
        # Set the observations space
        observation_subspaces = {
            "arm_qpos": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32),
            "arm_qvel": gym.spaces.Box(low=-max_v, high=max_v, shape=(6,), dtype=np.float32),
        }
        if self.observation_mode in ["image", "both"]:
            for camera_name, camera_data in self.parameters["camera_data"].items():
                if camera_name not in self.parameters["observation_cameras"]:
                    continue
                intrinsics = camera_data["intrinsics"]
                observation_subspaces[camera_name] = gym.spaces.Box(
                    0, 255, shape=(intrinsics.height(), intrinsics.width(), 3), dtype=np.uint8
                )

        if self.observation_mode in ["state", "both"]:
            self.add_state_to_observation_space(observation_subspaces)

        return gym.spaces.Dict(observation_subspaces)

    def add_models_to_plant(self, plant):
        parser = Parser(plant=plant)
        robot_model_instance = add_robot(plant, parser)
        self.add_ground_and_other_objects(plant, parser)
        return robot_model_instance

    def add_ground_and_other_objects(self, plant, parser):
        (ground_plane_model_instance,) = parser.AddModels(f"{ASSETS_PATH}/collision_ground_plane.sdf")

        # Weld ground to world frame.
        identity = RigidTransform.Identity()
        plant.WeldFrames(
            plant.world_frame(),
            plant.GetFrameByName("ground_plane_box", ground_plane_model_instance),
            identity,
        )
        self.add_objects_to_plant(plant)

    def construct_simulator(self, *, debug: bool):
        builder = DiagramBuilder()

        multibody_plant_config = MultibodyPlantConfig(
            time_step=self.parameters["sim_time_step"],
            contact_model=self.parameters["contact_model"],
            discrete_contact_approximation=self.parameters["contact_approximation"],
        )

        # Construct plant.
        plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)
        plant.set_name("plant")

        gravity_vector = np.array([0.0, 0.0, -9.81])
        plant.mutable_gravity_field().set_gravity_vector(gravity_vector)

        robot_model_instance = self.add_models_to_plant(plant)
        plant.Finalize()

        nq = plant.num_positions(model_instance=robot_model_instance)
        nv = plant.num_velocities(model_instance=robot_model_instance)
        assert nq == nv, "The number of position and velocity degrees of freedom should be equal."

        # Add visualizer to the plant and update only when render() is called (inf period).
        viz_config: VisualizationConfig = VisualizationConfig()
        viz_config.publish_period = np.inf
        ApplyVisualizationConfig(config=viz_config, builder=builder)

        #########################################################################
        # Actions

        gains = self.parameters["pid_gains"]
        kp = np.ones(nq) * gains["kp"]
        ki = np.ones(nq) * gains["ki"]
        kd = np.ones(nv) * gains["kd"]
        pid_controller = builder.AddSystem(PidController(kp=kp, ki=ki, kd=kd))
        pid_controller.set_name("pid_controller")

        builder.Connect(
            # [q v] all current states
            plant.get_state_output_port(robot_model_instance),
            pid_controller.get_input_port_estimated_state(),
        )

        builder.Connect(
            pid_controller.get_output_port(),  # [u] actuation
            plant.get_actuation_input_port(robot_model_instance),
        )

        if self.action_mode == "ee":
            # TODO: Add IK solver for 'ee' mode.
            ik_params = DifferentialInverseKinematicsParameters(nq, nv)

            ik_time_step = self.parameters["ik_time_step"]
            joint_max_velocities = np.ones(nv) * self.parameters["joint_max_velocities"]
            factor = self.parameters["ik_velocity_limit_factor"]
            ik_params.set_time_step(ik_time_step)
            ik_params.set_joint_velocity_limits(
                (-factor * joint_max_velocities, factor * joint_max_velocities)
            )

            differential_ik_integrator = builder.AddSystem(
                DifferentialIKIntegrator(plant=plant, params=ik_params)
            )
            differential_ik_integrator.set_name("IK")

            builder.ExportInput(
                differential_ik_integrator.get_input_port(
                    differential_ik_integrator.ee_goal_input_port_index
                ),
                "actions",
            )  # [q] desired positions

            builder.Connect(
                differential_ik_integrator.get_output_port(0),  # [q v] desired state
                pid_controller.get_input_port_desired_state(),
            )

            builder.Connect(
                # [q v] all current states
                plant.get_state_output_port(robot_model_instance),
                differential_ik_integrator.get_input_port(differential_ik_integrator.state_input_port_index),
            )

        else:
            assert self.action_mode == "joint"

            desired_velocities = builder.AddSystem(ConstantVectorSource([0] * nv))
            desired_velocities.set_name("desired_velocities")

            state_mux = builder.AddSystem(Multiplexer(input_sizes=[nq, nv]))
            state_mux.set_name("state_mux")

            builder.ExportInput(state_mux.get_input_port(0), "actions")  # [q] desired positions

            builder.Connect(
                desired_velocities.get_output_port(), state_mux.get_input_port(1)
            )  # [v] desired velocities

            builder.Connect(
                state_mux.get_output_port(0),  # [q v] desired state
                pid_controller.get_input_port_desired_state(),
            )

        #########################################################################
        # Reward

        class RewardSystem(LeafSystem):
            def __init__(self, *, calc_reward_fn):
                LeafSystem.__init__(self)
                self.calc_reward_fn = calc_reward_fn

                self.DeclareVectorInputPort("state", plant.num_multibody_states())
                self.DeclareVectorOutputPort("reward", 1, self.calc_reward)

                # FIXME: not thread safe.
                self.plant_context = plant.CreateDefaultContext()

            def calc_reward(self, context, output):
                plant_state = self.get_input_port(0).Eval(context)
                plant.SetPositionsAndVelocities(self.plant_context, plant_state)

                output[0] = self.calc_reward_fn(plant, self.plant_context)

        reward = builder.AddSystem(
            RewardSystem(calc_reward_fn=lambda plant, context: self.calc_reward(plant, context))
        )
        reward.set_name("reward")

        builder.Connect(plant.get_state_output_port(), reward.get_input_port(0))
        builder.ExportOutput(reward.get_output_port(), "reward")

        #########################################################################
        # Camera
        observation_camera_systems = {}
        if (self.render_mode in ["rgb_array"]) or (self.observation_mode in ["image", "both"]):
            environment_map: EnvironmentMap = EnvironmentMap(
                skybox=True, texture=EquirectangularMap(path=f"{ASSETS_PATH}/env_256_brick_room.jpg")
            )
            scene_graph.AddRenderer(
                "renderer",
                MakeRenderEngineVtk(
                    RenderEngineVtkParams(environment_map=environment_map, cast_shadows=True)
                ),
            )

            for camera_name, camera_data in self.parameters["camera_data"].items():
                is_observation_camera = camera_name in self.parameters["observation_cameras"]
                if is_observation_camera and (self.observation_mode not in ["image", "both"]):
                    continue

                is_monitoring_camera = camera_name is self.parameters["rgb_array_camera"]
                if is_monitoring_camera and self.render_mode not in ["rgb_array"]:
                    continue

                color_camera = ColorRenderCamera(
                    RenderCameraCore(
                        "renderer", camera_data["intrinsics"], ClippingRange(0.1, 10.0), RigidTransform()
                    ),
                    False,
                )

                depth_camera = DepthRenderCamera(color_camera.core(), DepthRange(0.1, 10.0))

                camera = builder.AddSystem(
                    RgbdSensor(
                        parent_id=scene_graph.world_frame_id(),
                        X_PB=camera_data["extrinsics"],
                        color_camera=color_camera,
                        depth_camera=depth_camera,
                    )
                )
                camera.set_name(camera_name)

                builder.Connect(scene_graph.get_query_output_port(), camera.query_object_input_port())

                if is_monitoring_camera:
                    builder.ExportOutput(camera.color_image_output_port(), camera_name)
                elif is_observation_camera:
                    observation_camera_systems[camera_name] = camera

        print(
            "Camera systems active (Note: this will impact performance): {camera_system_names}".format(
                camera_system_names=list(observation_camera_systems.keys())
            )
        )

        #########################################################################
        # Observations

        class ObservationPublisher(LeafSystem):
            def __init__(
                self,
                *,
                state_noise_magnitude: np.float32,
                camera_names: list[str],
                observation_mode: str,
                output_model_value,
                add_state_observations_fn,
            ):
                LeafSystem.__init__(self)
                self.observation_mode = observation_mode
                self.camera_input_port_index = {}
                self.output_model_value = output_model_value
                self.add_state_observations_fn = add_state_observations_fn

                for camera_name in camera_names:
                    self.camera_input_port_index[camera_name] = self.DeclareAbstractInputPort(
                        camera_name, Value(ImageRgba8U())
                    ).get_index()
                    print(
                        "{} at input index {}.".format(camera_name, self.camera_input_port_index[camera_name])
                    )

                self.state_input_port_index = self.DeclareVectorInputPort(
                    "plant_states", plant.num_multibody_states()
                ).get_index()

                def alloc_fn():
                    return Value(self.output_model_value)

                self.DeclareAbstractOutputPort("observations", alloc_fn, self.calc_observations)
                # FIXME: not thread safe.
                self.plant_context = plant.CreateDefaultContext()
                self.state_noise_magnitude = state_noise_magnitude

            def calc_observations(self, context, output):
                plant_state = self.get_input_port(self.state_input_port_index).Eval(context)

                if self.state_noise_magnitude > 0:
                    plant_state += np.random.uniform(
                        low=-self.state_noise_magnitude,
                        high=self.state_noise_magnitude,
                        size=plant.num_multibody_states(),
                    )

                plant.SetPositionsAndVelocities(self.plant_context, plant_state)

                observations = self.output_model_value
                observations["arm_qpos"] = plant.GetPositions(
                    context=self.plant_context, model_instance=robot_model_instance
                )
                observations["arm_qvel"] = plant.GetVelocities(
                    context=self.plant_context, model_instance=robot_model_instance
                )

                if self.observation_mode in ["image", "both"]:
                    for camera_name, input_port_index in self.camera_input_port_index.items():
                        observations[camera_name] = (
                            self.get_input_port(input_port_index).Eval(context).data[:, :, :3]
                        )  # remove alpha
                if self.observation_mode in ["state", "both"]:
                    self.add_state_observations_fn(plant, self.plant_context, observations)

                # Assign the output value.
                output.set_value(observations)

        obs_pub = builder.AddSystem(
            ObservationPublisher(
                state_noise_magnitude=self.parameters["obs_state_noise_magnitude"],
                camera_names=observation_camera_systems.keys(),
                observation_mode=self.observation_mode,
                output_model_value=self.construct_observation_space().sample(),
                add_state_observations_fn=lambda plant, context, obs: self.add_state_observations(
                    plant, context, obs
                ),
            )
        )
        obs_pub.set_name("obs_pub")

        for camera_name, camera_system in observation_camera_systems.items():
            print(
                "Wiring {} at input index {}.".format(
                    camera_name, obs_pub.camera_input_port_index[camera_name]
                )
            )
            builder.Connect(
                camera_system.color_image_output_port(),
                obs_pub.get_input_port(obs_pub.camera_input_port_index[camera_name]),
            )

        builder.Connect(plant.get_state_output_port(), obs_pub.get_input_port(obs_pub.state_input_port_index))
        builder.ExportOutput(obs_pub.get_output_port(), "observations")

        #########################################################################
        # EEF Disturbances

        class DisturbanceGenerator(LeafSystem):
            def __init__(self, plant, force_mag, period, duration):
                assert force_mag > 0
                assert period > 0
                assert duration > 0
                # Applies a random force [-force_mag, force_mag] at
                # the COM of the Pole body in the x direction every
                # period seconds for a given duration.
                LeafSystem.__init__(self)
                forces_cls = Value[list[ExternallyAppliedSpatialForce_[float]]]
                self.DeclareAbstractOutputPort("spatial_forces", lambda: forces_cls(), self.calc_disturbances)
                self.plant = plant
                self.gripper_body = self.plant.GetBodyByName("gripper_moving_part")
                self.force_mag = force_mag
                assert period > duration, "period: {} must be larger than duration: {}".format(
                    period, duration
                )
                self.period = period
                self.duration = duration

            def calc_disturbances(self, context, spatial_forces_vector):
                # Apply a force at COM of the Pole body.
                force = ExternallyAppliedSpatialForce_[float]()
                force.body_index = self.gripper_body.index()
                force.p_BoBq_B = self.gripper_body.default_com()
                y = context.get_time() % self.period
                max_f = self.force_mag
                if not ((y >= 0) and (y <= (self.period - self.duration))):
                    spatial_force = SpatialForce(
                        tau=[0, 0, 0],
                        f=[
                            np.random.uniform(low=-max_f, high=max_f),
                            np.random.uniform(low=-max_f, high=max_f),
                            np.random.uniform(low=-max_f, high=max_f),
                        ],
                    )
                else:
                    spatial_force = SpatialForce(tau=[0, 0, 0], f=[0, 0, 0])
                force.F_Bq_W = spatial_force
                spatial_forces_vector.set_value([force])

        disturbances = self.parameters["external_force_disturbances"]
        if disturbances["magnitude"] > 0 and disturbances["period"] > 0 and disturbances["duration"] > 0:
            disturbance_generator = builder.AddSystem(
                DisturbanceGenerator(
                    plant=plant,
                    force_mag=disturbances["magnitude"],
                    period=disturbances["period"],
                    duration=disturbances["duration"],
                )
            )
            builder.Connect(
                disturbance_generator.get_output_port(), plant.get_applied_spatial_force_input_port()
            )
            disturbance_generator.set_name("disturbance_generator")

        self.diagram = builder.Build()
        self.diagram.set_name("Diagram")

        simulator = Simulator(self.diagram)
        if self.render_mode == "human":
            simulator.set_target_realtime_rate(1.0)
        simulator.Initialize()

        if debug:

            def monitor(context, gym_time_limit=self.parameters["gym_time_limit"]):
                # Truncation: the episode duration reaches the time limit.
                if context.get_time() > gym_time_limit:
                    if debug:
                        print("Episode reached time limit.")
                    return EventStatus.ReachedTermination(self.diagram, "time limit")

                # TODO: Add penetration depth with self or environment as a penalty to the reward calculation.
                # Terminate if the robot is buried in the environment.
                max_depth = 0.01  # 1 cm

                # Get the plant context.
                plant = self.diagram.GetSubsystemByName("plant")
                plant_context = self.diagram.GetMutableSubsystemContext(plant, context)

                contact_results = plant.get_contact_results_output_port().Eval(plant_context)

                # robot-ground contact is rigid.
                for i in range(contact_results.num_point_pair_contacts()):
                    depth = contact_results.point_pair_contact_info(i).point_pair().depth

                    if depth > max_depth:
                        if debug:
                            print("Excessive Contact with Environment.")
                        return EventStatus.ReachedTermination(
                            self.diagram, "Excessive Contact with Environment."
                        )
                return EventStatus.Succeeded()

            simulator.set_monitor(monitor)

        if debug:
            # Visualize the controller plant and diagram.
            plt.figure()
            plot_graphviz(plant.GetTopologyGraphvizString())
            plt.figure()
            plot_system_graphviz(self.diagram, max_depth=2)
            plt.plot(1)
            plt.show(block=False)

        return simulator

    def handle_reset(self, simulator, diagram_context, seed):
        # Reset the Diagram context to default.
        self.diagram = simulator.get_system()
        # self.diagram.SetDefaultContext(diagram_context)

        # Set the seed.
        np.random.seed(seed)

        # Get the plant context.
        plant = self.diagram.GetSubsystemByName("plant")
        plant_context = self.diagram.GetMutableSubsystemContext(plant, diagram_context)

        self.handle_plant_state_reset(plant, plant_context)
