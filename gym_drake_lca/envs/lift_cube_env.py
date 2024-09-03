import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from pydrake.common.eigen_geometry import (
    Quaternion
)

from pydrake.systems.sensors import (
    CameraInfo,
    ImageRgba8U,
)
from pydrake.geometry import (
    ClippingRange,
    ColorRenderCamera,
    DepthRange,
    DepthRenderCamera,
    MakeRenderEngineVtk,
    RenderCameraCore,
    RenderEngineVtkParams,
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
from pydrake.systems.drawing import plot_graphviz, plot_system_graphviz
from pydrake.systems.framework import (
    DiagramBuilder,
    EventStatus,
    LeafSystem,
    PortDataType,
)
from pydrake.systems.primitives import (
    ConstantVectorSource,
    Multiplexer,
    Demultiplexer,
)
from pydrake.systems.sensors import CameraInfo, RgbdSensor
from pydrake.systems.controllers import (
    PidController,
)
from pydrake.visualization import AddDefaultVisualization


class LiftCubeEnv(DrakeGymEnv):
    metadata = {"render_modes": ["human", "rgb_array"],
                "observation_modes": ["image", "state", "both"],
                "action_modes": ["joint", "ee"], }

    def __init__(self,
                 observation_mode="image",
                 action_mode="joint",
                 render_mode="human",
                 enable_meshcat_viz=True,
                 debug=False,
                 obs_noise=False,
                 monitoring_camera=False,
                 add_disturbances=False):
        assert render_mode in self.metadata["render_modes"]
        assert observation_mode in self.metadata["observation_modes"]
        assert action_mode in self.metadata["action_modes"]

        self.camera_intrinsics = {
            "front_camera": CameraInfo(640, 640, np.pi / 4),
            "top_camera": CameraInfo(640, 640, np.pi / 4),
            "viz_camera": CameraInfo(640, 640, np.pi / 4),
        }

        def normalize(x):
            return x / np.linalg.norm(x)

        self.X_PB = {
            "front_camera": RigidTransform(RotationMatrix.MakeYRotation(-np.pi / 2), np.array([0.049, 0.888, 0.317])),
            "top_camera": RigidTransform(RotationMatrix.Identity(), np.array([0, 0, 1])),
            "viz_camera": RigidTransform(RotationMatrix(Quaternion(wxyz=normalize([-0.15, -0.1, 0.6, 1]))), np.array([-0.1, 0.6, 0.3])),
        }

        self.observation_mode = observation_mode
        self.action_mode = action_mode
        # Gym parameters.
        self.sim_time_step = 0.001
        time_step = 0.01
        self.gym_time_limit = 5

        drake_contact_models = ['point', 'hydroelastic_with_fallback']
        self.contact_model = drake_contact_models[1]

        drake_contact_approximations = ['sap', 'tamsi', 'similar', 'lagged']
        self.contact_approximation = drake_contact_approximations[0]

        # Make simulation.
        simulator = self.ConstructSimulator(enable_meshcat_viz=enable_meshcat_viz,
                                            debug=debug,
                                            obs_noise=obs_noise,
                                            monitoring_camera=monitoring_camera,
                                            add_disturbances=add_disturbances)

        # Set the action space
        self.action_mode = action_mode
        action_shape = {"joint": 6, "ee": 4}[action_mode]
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_shape,), dtype=np.float32)

        # Set the observations space
        observation_subspaces = {
            "arm_qpos": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32),
            "arm_qvel": gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32),
        }
        if observation_mode in ["image", "both"]:
            observation_subspaces["image_front"] = gym.spaces.Box(
                0, 255, shape=(240, 320, 3), dtype=np.uint8)
            observation_subspaces["image_top"] = gym.spaces.Box(
                0, 255, shape=(240, 320, 3), dtype=np.uint8)
        if observation_mode in ["state", "both"]:
            observation_subspaces["cube_pos"] = gym.spaces.Box(
                low=-10.0, high=10.0, shape=(3,))
        observation_space = gym.spaces.Dict(observation_subspaces)

        super().__init__(simulator=simulator,
                         time_step=time_step,
                         action_space=action_space,
                         observation_space=observation_space,
                         reward="reward",
                         action_port_id="actions",
                         observation_port_id="observations",
                         render_rgb_port_id={
                             "rgb_array": "viz_camera", "human": None}[render_mode],
                         render_mode=render_mode,
                         reset_handler=self.HandleReset)

    def AddModels(self, plant):
        parser = Parser(plant=plant)
        (robot_model_instance,) = parser.AddModels(
            "gym_drake_lca/low-cost-arm.urdf")
        (ground_plane_model_instance,) = parser.AddModels(
            "gym_drake_lca/assets/collision_ground_plane.sdf")

        parser.AddModels("gym_drake_lca/assets/cube.sdf")

        # Weld model instances to world frame.
        X_WI = RigidTransform.Identity()
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("base_link", robot_model_instance), X_WI)
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("ground_plane_box", ground_plane_model_instance), X_WI)
        return robot_model_instance

    def ConstructSimulator(self,
                           enable_meshcat_viz=True,
                           debug=False,
                           obs_noise=False,
                           monitoring_camera=False,
                           add_disturbances=False):

        builder = DiagramBuilder()

        multibody_plant_config = MultibodyPlantConfig(
            time_step=self.sim_time_step,
            contact_model=self.contact_model,
            discrete_contact_approximation=self.contact_approximation,
        )

        plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)
        plant.set_name("plant")

        gravity_vector = np.array([0.0, 0.0, -9.81])
        plant.mutable_gravity_field().set_gravity_vector(gravity_vector)

        # Add assets to the plant.
        robot_model_instance = self.AddModels(plant)
        plant.Finalize()

        # Add Visualizer to Diagram
        if enable_meshcat_viz:
            AddDefaultVisualization(builder=builder)

        nq = plant.num_positions(model_instance=robot_model_instance)
        nv = plant.num_velocities(model_instance=robot_model_instance)

        #########################################################################
        # Actions

        # TODO: Add IK solver for 'ee' mode.
        assert self.action_mode != "ee"

        # TODO: This is a tunable parameter, put it somewhere accessible!
        kp = np.ones(nq)*20.0
        ki = np.zeros(nq)
        kd = np.ones(nv)*0.1
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
            def __init__(self):
                LeafSystem.__init__(self)
                # TODO: This is a tunable parameter, put it somewhere accessible!
                self.threshold_height = 0.5

                self.DeclareVectorInputPort(
                    "state", plant.num_multibody_states())
                self.DeclareVectorOutputPort("reward", 1, self.CalcReward)
                # FIXME: not thread safe.
                self.plant_context = plant.CreateDefaultContext()

            def CalcReward(self, context, output):
                plant_state = self.get_input_port(0).Eval(context)
                self.plant_context.SetContinuousState(plant_state)

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
                reward_height = cube_z - self.threshold_height
                reward_distance = -ee_to_cube
                reward = reward_height + reward_distance
                output[0] = reward

        reward = builder.AddSystem(RewardSystem())
        reward.set_name("reward")

        builder.Connect(plant.get_state_output_port(),
                        reward.get_input_port(0))
        builder.ExportOutput(reward.get_output_port(), "reward")

        #########################################################################
        # Camera

        if monitoring_camera or (self.observation_mode in ["image", "both"]):
            scene_graph.AddRenderer(
                "renderer", MakeRenderEngineVtk(RenderEngineVtkParams()))

            self.cameras = {}

            for camera_name, intrinsics in self.camera_intrinsics.items():
                if camera_name in ["top_camera", "front_camera"] and self.observation_mode not in ["image", "both"]:
                    continue
                if camera_name in ["viz_camera"] and not monitoring_camera:
                    continue

                color_camera = ColorRenderCamera(RenderCameraCore("renderer",
                                                                  intrinsics,
                                                                  ClippingRange(
                                                                      0.01, 10.0),
                                                                  RigidTransform()
                                                                  ), False)

                depth_camera = DepthRenderCamera(color_camera.core(),
                                                 DepthRange(0.01, 10.0))

                camera = builder.AddSystem(
                    RgbdSensor(parent_id=scene_graph.world_frame_id(),
                               X_PB=self.X_PB[camera_name],
                               color_camera=color_camera,
                               depth_camera=depth_camera))
                camera.set_name(camera_name)
                builder.Connect(scene_graph.get_query_output_port(),
                                camera.query_object_input_port())

                if camera_name in ["viz_camera"] and monitoring_camera:
                    builder.ExportOutput(
                        camera.color_image_output_port(), "viz_camera")
                if camera_name in ["top_camera", "front_camera"]:
                    self.camera_systems[camera_name] = camera

        #########################################################################
        # Observations

        class ObservationPublisher(LeafSystem):
            def __init__(self, noise: bool, camera_names: list[str], observation_mode: str):
                LeafSystem.__init__(self)
                self.observation_mode = observation_mode
                self.camera_input_port_index = {}

                for camera_name in camera_names:
                    self.camera_input_port_index[camera_name] = self.DeclareInputPort(
                        camera_name, PortDataType.kAbstractValued, -1).get_index()

                self.state_input_port_index = self.DeclareVectorInputPort(
                    "plant_states", plant.num_multibody_states()).get_index()

                self.DeclareAbstractOutputPort("observations", self.CalcObs)
                self.noise = noise
                # FIXME: not thread safe.
                self.plant_context = plant.CreateDefaultContext()

            def CalcObs(self, context, output):
                plant_state = self.get_input_port(
                    self.state_input_port_index).Eval(context)

                if self.noise:
                    # TODO: This is a tunable parameter, put it somewhere accessible!
                    plant_state += np.random.uniform(low=-0.01,
                                                     high=0.01,
                                                     size=plant.num_multibody_states())

                self.plant_context.SetContinuousState(plant_state)

                output = {
                    "arm_qpos": plant.GetPositions(context=self.plant_context, model_instance=robot_model_instance),
                    "arm_qvel": plant.GetVelocities(context=self.plant_context, model_instance=robot_model_instance),
                }
                if self.observation_mode in ["image", "both"]:
                    image_name_mapping = {
                        "front_camera": "image_front", "top_camera": "image_top"}
                    for camera_name, input_port_index in self.camera_input_port_index:
                        output[image_name_mapping[camera_name]] = self.get_input_port(
                            input_port_index).Eval(context)
                if self.observation_mode in ["state", "both"]:
                    cube = plant.GetBodyByName("cube")
                    cube_pos = cube.EvalPoseInWorld(
                        self.plant_context).translation()
                    output["cube_pos"] = cube_pos

        obs_pub = builder.AddSystem(ObservationPublisher(
            noise=obs_noise, camera_names=self.camera_systems.keys(), observation_mode=self.observation_mode))
        obs_pub.set_name(obs_pub)

        for camera_name, system in self.camera_systems:
            builder.Connect(system.color_image_output_port(),
                            obs_pub.get_input_port(obs_pub.camera_input_port_index[camera_name]))

        builder.Connect(plant.get_state_output_port(),
                        obs_pub.get_input_port(0))
        builder.ExportOutput(obs_pub.get_output_port(), "observations")

        #########################################################################
        # EEF Disturbances

        class DisturbanceGenerator(LeafSystem):
            def __init__(self, plant, force_mag, period, duration):
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

        if add_disturbances:
            # TODO: This is a tunable parameter, put it somewhere accessible!
            # Applies a force of 1N every 1s for 0.1s at the COM of the gripper_moving_part body.
            disturbance_generator = builder.AddSystem(
                DisturbanceGenerator(
                    plant=plant, force_mag=1,
                    period=1, duration=0.1))
            builder.Connect(disturbance_generator.get_output_port(),
                            plant.get_applied_spatial_force_input_port())

        diagram = builder.Build()
        simulator = Simulator(diagram)
        simulator.Initialize()

        def monitor(context, gym_time_limit=self.gym_time_limit):
            # Truncation: the episode duration reaches the time limit.
            if context.get_time() > gym_time_limit:
                if debug:
                    print("Episode reached time limit.")
                return EventStatus.ReachedTermination(
                    diagram,
                    "time limit")

            return EventStatus.Succeeded()

        simulator.set_monitor(monitor)

        if debug:
            # Visualize the controller plant and diagram.
            plt.figure()
            plot_graphviz(plant.GetTopologyGraphvizString())
            plt.figure()
            plot_system_graphviz(diagram, max_depth=2)
            plt.plot(1)
            plt.show(block=False)

        return simulator

    def HandleReset(self, simulator, diagram_context, seed):
        # Set the seed.
        np.random.seed(seed)

        # TODO: This is a tunable parameter, put it somewhere accessible!
        cube_low = np.array([-0.15, 0.10, 0.015])
        cube_high = np.array([0.15, 0.25, 0.015])

        # Get the plant context.
        diagram = simulator.get_system()
        plant = diagram.GetSubsystemByName("plant")
        plant_context = diagram.GetMutableSubsystemContext(plant,
                                                           diagram_context)

        # Reset the context to default state.
        plant_context.SetContinuousState(
            [0] * (plant.num_positions() + plant.num_velocities()))

        # Randomize a new cube position.
        cube_pos = np.random.uniform(cube_low, cube_high, size=(3, 1))

        # Set the new cube position in the context.
        cube = plant.GetBodyByName("cube")
        plant.SetFreeBodyPose(plant_context, cube, RigidTransform(cube_pos))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render_mode", type=str, default="human")
    parser.add_argument("--observation_mode", type=str, default="state")
    parser.add_argument("--action_mode", type=str, default="joint")
    args = parser.parse_args()

    # Create the environment
    env = LiftCubeEnv(observation_mode=args.observation_mode,
                      action_mode=args.action_mode, render_mode=args.render_mode)

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
