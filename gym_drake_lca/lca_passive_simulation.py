# usage: e.g.: python3 gym_drake_lca/lca_passive_simulation.py --time_step 0.001 --simulation_time 9999
import argparse

from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import AddDefaultVisualization


def add_models_to_plant(mbp: MultibodyPlant):
    parser = Parser(plant=mbp)
    (lca_model_instance,) = parser.AddModels("gym_drake_lca/low-cost-arm.urdf")
    (ground_plane_model_instance,) = parser.AddModels("gym_drake_lca/assets/collision_ground_plane.sdf")

    # Weld model instances to world frame.
    identity = RigidTransform.Identity()
    mbp.WeldFrames(mbp.world_frame(), mbp.GetFrameByName("base_link", lca_model_instance), identity)
    mbp.WeldFrames(
        mbp.world_frame(), mbp.GetFrameByName("ground_plane_box", ground_plane_model_instance), identity
    )

    mbp.Finalize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target_realtime_rate",
        type=float,
        default=1.0,
        help="Desired rate relative to real time.  See documentation for "
        "Simulator::set_target_realtime_rate() for details.",
    )
    parser.add_argument(
        "--simulation_time", type=float, default=10.0, help="Desired duration of the simulation in seconds."
    )
    parser.add_argument(
        "--time_step",
        type=float,
        default=0.0,
        help="If greater than zero, the plant is modeled as a system with "
        "discrete updates and period equal to this time_step. "
        "If 0, the plant is modeled as a continuous system.",
    )
    args = parser.parse_args()

    builder = DiagramBuilder()

    # Add Multibody Plant to Diagram then add Models.
    mbp, scene_graph = AddMultibodyPlantSceneGraph(builder=builder, time_step=args.time_step)
    add_models_to_plant(mbp)

    # Add Visualizer to Diagram
    AddDefaultVisualization(builder=builder)

    # Build Diagram
    diagram = builder.Build()

    diagram_context = diagram.CreateDefaultContext()
    mbp_context = mbp.GetMyMutableContextFromRoot(diagram_context)

    # Set input ports
    mbp.get_actuation_input_port().FixValue(mbp_context, [0] * mbp.num_positions())

    # Simulate Diagram from initial State, set in Context.
    simulator = Simulator(diagram, diagram_context)
    simulator.set_publish_every_time_step(False)
    simulator.set_target_realtime_rate(args.target_realtime_rate)
    simulator.Initialize()
    simulator.AdvanceTo(args.simulation_time)


if __name__ == "__main__":
    main()
