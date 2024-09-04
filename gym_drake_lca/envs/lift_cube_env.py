import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from pydrake.common.eigen_geometry import (
    Quaternion
)
from pydrake.common.value import (
    AbstractValue,
    Value,
)

from typing import NamedTuple

from pydrake.geometry import (
    ClippingRange,
    ColorRenderCamera,
    DepthRange,
    DepthRenderCamera,
    MakeRenderEngineVtk,
    RenderCameraCore,
    RenderEngineVtkParams,
    EnvironmentMap,
    EquirectangularMap,
)
from pydrake.gym import DrakeGymEnv
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.math import SpatialForce
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlant,
    ExternallyAppliedSpatialForce_,
    MultibodyPlant,
    MultibodyPlantConfig,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.controllers import (
    PidController,
)
from pydrake.systems.drawing import plot_graphviz, plot_system_graphviz
from pydrake.systems.framework import (
    DiagramBuilder,
    EventStatus,
    LeafSystem,
    PortDataType,
    Context,
    BasicVector,
)
from pydrake.systems.primitives import (
    ConstantVectorSource,
    Multiplexer,
    Demultiplexer,
)
from pydrake.systems.sensors import (
    CameraInfo,
    RgbdSensor,
    ImageRgba8U,
)

from pydrake.visualization import (
    ApplyVisualizationConfig,
    VisualizationConfig,
    AddDefaultVisualization,
)

from gym_drake_lca import ASSETS_PATH


def ConstructLiftCubeEnvDefaultParameters():
    # Tunable parameters
    image_dims = [240, 320]
    image_fov = np.pi/4
    return {
        "camera_data": {
            "image_front": {"intrinsics": CameraInfo(image_dims[1], image_dims[0], image_fov),
                            "extrinsics": RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2 + np.pi / 32)
                                                         .multiply(RotationMatrix.MakeZRotation(-np.pi)),
                                                         np.array([0.0, 0.6, 0.15]))},
            "image_top": {"intrinsics": CameraInfo(image_dims[1], image_dims[0], image_fov),
                          "extrinsics": RigidTransform(RotationMatrix.MakeXRotation(np.pi), np.array([0, 0.1, 0.6]))},
            "image_viewer": {"intrinsics": CameraInfo(image_dims[1], image_dims[0], image_fov),
                             "extrinsics": RigidTransform(RotationMatrix.MakeZRotation(np.pi/4)
                                                          .multiply(RotationMatrix.MakeXRotation(5 * np.pi / 8)
                                                                    .multiply(RotationMatrix.MakeZRotation(-np.pi))),
                                                          np.array([-0.4, 0.4, 0.3]))}},
        "observation_cameras": ["image_front", "image_top"],
        "rgb_array_camera": "image_viewer",
        "sim_time_step": 0.0,
        "gym_time_step": 0.05,
        "gym_time_limit": 0.5,
        # contact_models: 'point', 'hydroelastic_with_fallback'
        "contact_model": 'hydroelastic_with_fallback',
        # contact_approximations: 'sap', 'tamsi', 'similar', 'lagged'
        "contact_approximation": 'sap',
        "pid_gains": {"kp": 20.0, "ki": 0.0, "kd": 0.1},
        "lift_reward_threshold_height": 0.5,
        "obs_state_noise_magnitude": 0.0,
        "cube_pos_bounds": [np.array([-0.15, 0.10, 0.015]), np.array([0.15, 0.25, 0.015])],
        "external_force_disturbances": {"magnitude": 0.0, "period": 1.0, "duration": 0.1},
        "emit_debug_printout": False
    }


class LiftCubeEnv(DrakeGymEnv):

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
    - `render_mode (str)`: the render mode, can be "human" or "rgb_array", default is None.
    """
    metadata = {"render_modes": ["human", "rgb_array", "ansi"],
                "observation_modes": ["image", "state", "both"],
                "action_modes": ["joint", "ee"]}

    def __init__(self,
                 observation_mode="image",
                 action_mode="joint",
                 render_mode="human",
                 parameters=ConstructLiftCubeEnvDefaultParameters()):
        assert render_mode in self.metadata["render_modes"]
        assert observation_mode in self.metadata["observation_modes"]
        assert action_mode in self.metadata["action_modes"]

        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.render_mode = render_mode

        self.parameters = parameters

        # Make simulation.
        simulator = self.ConstructSimulator(
            debug=self.parameters["emit_debug_printout"])

        # Set the action space
        self.action_mode = action_mode
        action_shape = {"joint": 5, "ee": 4}[action_mode]
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_shape,), dtype=np.float32)

        super().__init__(simulator=simulator,
                         time_step=self.parameters["gym_time_step"],
                         action_space=action_space,
                         observation_space=self.ConstructObservationSpace(),
                         reward="reward",
                         action_port_id="actions",
                         observation_port_id="observations",
                         render_rgb_port_id={
                             "rgb_array": self.parameters["rgb_array_camera"], "human": None, "ansi": None}[render_mode],
                         render_mode=render_mode,
                         reset_handler=self.HandleReset)

    def ConstructObservationSpace(self):
        # Set the observations space
        observation_subspaces = {
            "arm_qpos": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32),
            "arm_qvel": gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32),
        }
        if self.observation_mode in ["image", "both"]:
            for camera_name, camera_data in self.parameters["camera_data"].items():
                if camera_name not in self.parameters["observation_cameras"]:
                    continue
                intrinsics = camera_data["intrinsics"]
                observation_subspaces[camera_name] = gym.spaces.Box(
                    0, 255, shape=(intrinsics.height(), intrinsics.width(), 3), dtype=np.uint8)
        if self.observation_mode in ["state", "both"]:
            observation_subspaces["cube_pos"] = gym.spaces.Box(
                low=-10.0, high=10.0, shape=(3,))
        return gym.spaces.Dict(observation_subspaces)

    def AddModels(self, plant):
        parser = Parser(plant=plant)
        (robot_model_instance,) = parser.AddModels(
            f"{ASSETS_PATH}/low-cost-arm.urdf")
        (ground_plane_model_instance,) = parser.AddModels(
            f"{ASSETS_PATH}/collision_ground_plane.sdf")

        parser.AddModels(f"{ASSETS_PATH}/cube.sdf")

        # Weld model instances to world frame.
        X_WI = RigidTransform.Identity()
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("base_link", robot_model_instance), X_WI)
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("ground_plane_box", ground_plane_model_instance), X_WI)
        return robot_model_instance

    def ConstructSimulator(self, debug):

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

        robot_model_instance = self.AddModels(plant)
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

        # TODO: Add IK solver for 'ee' mode.
        assert self.action_mode != "ee"

        gains = self.parameters["pid_gains"]
        kp = np.ones(nq)*gains["kp"]
        ki = np.ones(nq)*gains["ki"]
        kd = np.ones(nv)*gains["kd"]
        pid_controller = builder.AddSystem(PidController(kp=kp, ki=ki, kd=kd))
        pid_controller.set_name("pid_controller")

        desired_velocities = builder.AddSystem(ConstantVectorSource([0]*nv))
        desired_velocities.set_name("desired_velocities")

        state_mux = builder.AddSystem(Multiplexer(input_sizes=[nq, nv]))
        state_mux.set_name("state_mux")

        builder.ExportInput(state_mux.get_input_port(0),
                            "actions")  # [q] desired positions

        builder.Connect(desired_velocities.get_output_port(),
                        state_mux.get_input_port(1))  # [v] desired velocities

        builder.Connect(state_mux.get_output_port(0),  # [q v] desired state
                        pid_controller.get_input_port_desired_state())

        builder.Connect(plant.get_state_output_port(robot_model_instance),  # [q v] all current states
                        pid_controller.get_input_port_estimated_state())

        builder.Connect(pid_controller.get_output_port(),  # [u] actuation
                        plant.get_actuation_input_port(robot_model_instance))

        #########################################################################
        # Reward

        class RewardSystem(LeafSystem):
            def __init__(self, lift_reward_threshold_height: np.float32):
                LeafSystem.__init__(self)
                assert lift_reward_threshold_height >= 0.0

                self.DeclareVectorInputPort(
                    "state", plant.num_multibody_states())
                self.DeclareVectorOutputPort("reward", 1, self.CalcReward)
                # FIXME: not thread safe.
                self.plant_context = plant.CreateDefaultContext()
                self.lift_reward_threshold_height = lift_reward_threshold_height

            def CalcReward(self, context, output):
                plant_state = self.get_input_port(0).Eval(context)
                plant.SetPositionsAndVelocities(
                    self.plant_context, plant_state)

                gripper_moving_side = plant.GetBodyByName(
                    "gripper_moving_part")
                cube = plant.GetBodyByName("cube")

                # Get the position of the cube and the distance between the end effector and the cube
                cube_pos = cube.EvalPoseInWorld(
                    self.plant_context).translation()
                cube_z = cube_pos[2]
                ee_pos = gripper_moving_side.EvalPoseInWorld(
                    self.plant_context).translation()
                ee_to_cube = np.linalg.norm(ee_pos - cube_pos)

                # Compute the reward
                reward_height = cube_z - self.lift_reward_threshold_height
                reward_distance = -ee_to_cube
                reward = reward_height + reward_distance
                output[0] = reward

        reward = builder.AddSystem(RewardSystem(
            lift_reward_threshold_height=self.parameters["lift_reward_threshold_height"]))
        reward.set_name("reward")

        builder.Connect(plant.get_state_output_port(),
                        reward.get_input_port(0))
        builder.ExportOutput(reward.get_output_port(), "reward")

        #########################################################################
        # Camera
        observation_camera_systems = {}
        if (self.render_mode in ["rgb_array"]) or (self.observation_mode in ["image", "both"]):
            environment_map: EnvironmentMap = EnvironmentMap(
                skybox=True, texture=EquirectangularMap(path=f"{ASSETS_PATH}/env_256_brick_room.jpg"))
            scene_graph.AddRenderer(
                "renderer", MakeRenderEngineVtk(RenderEngineVtkParams(environment_map=environment_map, cast_shadows=True)))

            for camera_name, camera_data in self.parameters["camera_data"].items():
                is_observation_camera = (
                    camera_name in self.parameters["observation_cameras"])
                if is_observation_camera and (not self.observation_mode in ["image", "both"]):
                    continue

                is_monitoring_camera = (
                    camera_name is self.parameters["rgb_array_camera"])
                if is_monitoring_camera and not (self.render_mode in ["rgb_array"]):
                    continue

                color_camera = ColorRenderCamera(RenderCameraCore("renderer",
                                                                  camera_data["intrinsics"],
                                                                  ClippingRange(
                                                                      0.1, 10.0),
                                                                  RigidTransform()
                                                                  ), False)

                depth_camera = DepthRenderCamera(color_camera.core(),
                                                 DepthRange(0.1, 10.0))

                camera = builder.AddSystem(
                    RgbdSensor(parent_id=scene_graph.world_frame_id(),
                               X_PB=camera_data["extrinsics"],
                               color_camera=color_camera,
                               depth_camera=depth_camera))
                camera.set_name(camera_name)

                builder.Connect(scene_graph.get_query_output_port(),
                                camera.query_object_input_port())

                if is_monitoring_camera:
                    builder.ExportOutput(
                        camera.color_image_output_port(), camera_name)
                elif is_observation_camera:
                    observation_camera_systems[camera_name] = camera

        print("Camera systems active (Note: this will impact performance): {camera_system_names}".format(
            camera_system_names=list(observation_camera_systems.keys())))

        #########################################################################
        # Observations

        class ObservationPublisher(LeafSystem):
            def __init__(self, state_noise_magnitude: np.float32, camera_names: list[str], observation_mode: str, output_model_value):
                LeafSystem.__init__(self)
                self.observation_mode = observation_mode
                self.camera_input_port_index = {}
                self.output_model_value = output_model_value

                for camera_name in camera_names:
                    self.camera_input_port_index[camera_name] = self.DeclareAbstractInputPort(
                        camera_name, Value(ImageRgba8U())).get_index()
                    print(
                        f"{camera_name} at input index {self.camera_input_port_index[camera_name]}.")

                self.state_input_port_index = self.DeclareVectorInputPort(
                    "plant_states", plant.num_multibody_states()).get_index()

                def alloc_fn():
                    return Value(self.output_model_value)

                self.DeclareAbstractOutputPort(
                    "observations", alloc_fn, self.CalcObs)
                # FIXME: not thread safe.
                self.plant_context = plant.CreateDefaultContext()
                self.state_noise_magnitude = state_noise_magnitude

            def CalcObs(self, context, output):
                plant_state = self.get_input_port(
                    self.state_input_port_index).Eval(context)

                if self.state_noise_magnitude > 0:
                    plant_state += np.random.uniform(low=-self.state_noise_magnitude,
                                                     high=self.state_noise_magnitude,
                                                     size=plant.num_multibody_states())

                plant.SetPositionsAndVelocities(
                    self.plant_context, plant_state)

                observations = self.output_model_value
                observations["arm_qpos"] = plant.GetPositions(
                    context=self.plant_context, model_instance=robot_model_instance)
                observations["arm_qvel"] = plant.GetVelocities(
                    context=self.plant_context, model_instance=robot_model_instance)

                if self.observation_mode in ["image", "both"]:
                    for camera_name, input_port_index in self.camera_input_port_index.items():
                        observations[camera_name] = self.get_input_port(
                            input_port_index).Eval(context).data[:, :, :3]  # remove alpha
                if self.observation_mode in ["state", "both"]:
                    cube = plant.GetBodyByName("cube")
                    cube_pos = cube.EvalPoseInWorld(
                        self.plant_context).translation()
                    observations["cube_pos"] = cube_pos

                # Assign the output value.
                output.set_value(observations)

        obs_pub = builder.AddSystem(ObservationPublisher(state_noise_magnitude=self.parameters["obs_state_noise_magnitude"],
                                                         camera_names=observation_camera_systems.keys(),
                                                         observation_mode=self.observation_mode,
                                                         output_model_value=self.ConstructObservationSpace().sample()))
        obs_pub.set_name("obs_pub")

        for camera_name, camera_system in observation_camera_systems.items():
            print(
                f"Wiring {camera_name} at input index {obs_pub.camera_input_port_index[camera_name]}.")
            builder.Connect(camera_system.color_image_output_port(),
                            obs_pub.get_input_port(obs_pub.camera_input_port_index[camera_name]))

        builder.Connect(plant.get_state_output_port(),
                        obs_pub.get_input_port(obs_pub.state_input_port_index))
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
                forces_cls = Value[List[ExternallyAppliedSpatialForce_[float]]]
                self.DeclareAbstractOutputPort("spatial_forces",
                                               lambda: forces_cls(),
                                               self.CalcDisturbances)
                self.plant = plant
                self.gripper_body = self.plant.GetBodyByName(
                    "gripper_moving_part")
                self.force_mag = force_mag
                assert period > duration, (
                    f"period: {period} must be larger than duration: {duration}")
                self.period = period
                self.duration = duration

            def CalcDisturbances(self, context, spatial_forces_vector):
                # Apply a force at COM of the Pole body.
                force = ExternallyAppliedSpatialForce_[float]()
                force.body_index = self.gripper_body.index()
                force.p_BoBq_B = self.gripper_body.default_com()
                y = context.get_time() % self.period
                max_f = self.force_mag
                if not ((y >= 0) and (y <= (self.period - self.duration))):
                    spatial_force = SpatialForce(
                        tau=[0, 0, 0],
                        f=[np.random.uniform(low=-max_f, high=max_f),
                           np.random.uniform(low=-max_f, high=max_f),
                           np.random.uniform(low=-max_f, high=max_f),
                           ])
                else:
                    spatial_force = SpatialForce(
                        tau=[0, 0, 0],
                        f=[0, 0, 0])
                force.F_Bq_W = spatial_force
                spatial_forces_vector.set_value([force])

        disturbances = self.parameters["external_force_disturbances"]
        if 0 < disturbances["magnitude"] and 0 < disturbances["period"] and 0 < disturbances["duration"]:
            disturbance_generator = builder.AddSystem(
                DisturbanceGenerator(
                    plant=plant,
                    force_mag=disturbances["magnitude"],
                    period=disturbances["period"],
                    duration=disturbances["duration"]))
            builder.Connect(disturbance_generator.get_output_port(),
                            plant.get_applied_spatial_force_input_port())
            disturbance_generator.set_name("disturbance_generator")

        self.diagram = builder.Build()
        self.diagram.set_name("Diagram")
        simulator = Simulator(self.diagram)
        simulator.Initialize()

        def monitor(context, gym_time_limit=self.parameters["gym_time_limit"]):
            # Truncation: the episode duration reaches the time limit.
            if context.get_time() > gym_time_limit:
                if debug:
                    print("Episode reached time limit.")
                return EventStatus.ReachedTermination(
                    self.diagram,
                    "time limit")

            # TODO: Add penetration depth with self or environment as a penalty to the reward calculation.
            # Terminate if the robot is buried in the environment.
            max_depth = 0.01  # 1 cm

            # Get the plant context.
            plant = self.diagram.GetSubsystemByName("plant")
            plant_context = self.diagram.GetMutableSubsystemContext(
                plant, context)

            contact_results = plant.get_contact_results_output_port().Eval(plant_context)

            # robot-ground contact is rigid.
            for i in range(contact_results.num_point_pair_contacts()):
                depth = contact_results.point_pair_contact_info(
                    i).point_pair().depth

                if depth > max_depth:
                    if debug:
                        print("Excessive Contact with Environment.")
                    return EventStatus.Failed(self.diagram, "Excessive Contact with Environment.")
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

    def HandleReset(self, simulator, diagram_context, seed):
        # Reset the Diagram context to default.
        self.diagram = simulator.get_system()
        diagram_context = self.diagram.CreateDefaultContext()

        # Set the seed.
        np.random.seed(seed)

        # Get the plant context.
        plant = self.diagram.GetSubsystemByName("plant")
        plant_context = self.diagram.GetMutableSubsystemContext(plant,
                                                                diagram_context)
        lb, ub = self.parameters["cube_pos_bounds"]

        # Randomize a new cube position.
        cube_pos = np.random.uniform(low=lb, high=ub)

        # Set the new cube position in the context.
        cube = plant.GetBodyByName("cube")
        plant.SetFreeBodyPose(plant_context, cube, RigidTransform(cube_pos))
