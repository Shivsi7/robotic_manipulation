# iiwa problems for pset 1

import numpy as np
from dtsystems import (
    Constant,
    HEADLESS,
    Observer,
    PController,
    PDController,
    PDController2,
    SimpleTrajectoryFollower,
    get_meshcat,
    get_positions,
    series_composition,
    show_meshcat,
    simulate,
    plot_log,
)
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Diagram,
    DiagramBuilder,
    LogVectorOutput,
    Meshcat,
    MeshcatVisualizer,
    MultibodyPlant,
    Parser,
    plot_system_graphviz,
)

from utils.drake_models import explain_model_download_error
from utils.plotting import plt

######################################################################
## Code for students to be aware of
######################################################################

IIWA14_URL = (
    "package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"
)

Q_START = np.array([0, 1.0, 0.3, 0.7, 0, 0, 0])

# The joint-space step used in the PD-controller question: drive the arm from
# Q_START to Q_START + Q_STEP (a step on joints 1, 2 and 4).
Q_STEP = np.array([0.5, -0.3, 0, 0.4, 0, 0, 0])


# The plant integrates at 1e-4 s; the controllers run ten times slower.  The
# PD derivative term divides by CONTROLLER_DT, so it has to be the period
# actually used.  See the note above _iiwa_builder for why not 1e-4.
PLANT_TIME_STEP = 1e-4
CONTROLLER_DT = 1e-3


def iiwa_s0(q0: np.ndarray) -> list[np.ndarray]:
    """
    Initial state list for simulate(): group 0 is the plant, whose discrete
    state is [q, v].
    """
    return [np.hstack([q0, np.zeros(7)])]


# Joint configurations that put the end effector at the corners of a 0.4 m
# square in the vertical plane x = 0.5, with the end-effector frame held
# axis-aligned, found (offline) by inverse kinematics -- a topic we will
# study properly later in the term.  Closed by repeating the first corner,
# like SQUARE in cartesian_2d_robot.py.  Drive the arm through them in order
# and the end effector draws the square.
SQUARE_IIWA = [
    np.array([-0.3743, 1.2271, 0.0031, 0.861, 0.0169, -0.3627, 0.3545]),
    np.array([0.2311, 1.894, 0.8932, 1.6624, -1.2178, -0.9016, 1.1868]),
    np.array([0.9917, 1.894, 0.8958, 1.6696, -1.2453, -0.9026, 0.4584]),
    np.array([0.7547, 1.1085, 0.6942, 0.8787, -1.4182, -0.6161, 0.2746]),
]
SQUARE_IIWA.append(SQUARE_IIWA[0])

######################################################################
## Code for students to study
######################################################################


def create_IIWA14_diagram(
    torques: np.ndarray = np.zeros(7), meshcat: Meshcat | None = None
) -> tuple[Diagram, MultibodyPlant]:
    """
    Build a diagram holding the welded-base iiwa14 MultibodyPlant, with a
    Constant source applying the given joint torques at the actuation input,
    a state logger named "log_plant", and a MeshcatVisualizer when meshcat is
    given.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)
    parser = Parser(plant, scene_graph)
    try:
        parser.AddModelsFromUrl(IIWA14_URL)
    except RuntimeError as e:
        # The first load downloads the models; this explains the one common
        # way that fails (a space or such in the venv's path) before re-raising.
        explain_model_download_error(e)
        raise
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))
    plant.Finalize()

    source = builder.AddSystem(Constant(torques))
    builder.Connect(source.get_output_port(), plant.get_actuation_input_port())

    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    logger = LogVectorOutput(plant.get_state_output_port(), builder)
    logger.set_name("log_plant")

    diagram = builder.Build()
    diagram.set_name("plant and scene_graph")
    return diagram, plant


def test_const_torque(q_initial: np.ndarray, torques: np.ndarray) -> None:
    """
    Simulate the arm from q_initial under a constant joint-torque vector.
    """
    meshcat = get_meshcat()
    diagram, plant = create_IIWA14_diagram(torques=torques, meshcat=meshcat)
    if not HEADLESS:
        plt.figure(figsize=(12, 6))
        plot_system_graphviz(diagram)
        plt.show()
    simulator = simulate(diagram, iiwa_s0(q_initial), 5.0)
    plot_log(diagram, simulator, "log_plant")
    q_final = get_positions(plant, simulator)
    print(f"   Initial joint positions: {q_initial}")
    print(f"   Final joint positions:   {q_final}")
    show_meshcat()


######################################################################
##  Code for students to write
######################################################################


def create_IIWA14_diagram_with_pcontroller(
    controller_gain: float, q_desired: np.ndarray, meshcat: Meshcat | None = None
) -> tuple[Diagram, MultibodyPlant]:
    """
    Like create_IIWA14_diagram, but the joints are driven by a proportional
    position controller: compose an Observer that keeps the positions out of
    the 14-D plant state [q, v] with a PController with the given gain and
    target q_desired, wired from the plant's state output back into its
    actuation input.
    """
    builder, plant, scene_graph = _iiwa_builder()
    observer = builder.AddSystem(Observer(14, range(7)))
    controller = builder.AddSystem(PController(7, q_desired, controller_gain))
    builder.Connect(plant.get_state_output_port(), observer.get_input_port())
    builder.Connect(observer.get_output_port(), controller.get_input_port())
    builder.Connect(controller.get_output_port(), plant.get_actuation_input_port())
    _finish(builder, scene_graph, meshcat, plant, controller)
    diagram = builder.Build()
    diagram.set_name("iiwa with P controller")
    return diagram, plant


def create_IIWA14_diagram_with_pd_controller(
    controller_gain: float,
    damping_gain: float,
    q_desired: np.ndarray,
    dt: float = 0.01,
    meshcat: Meshcat | None = None,
    plant_time_step: float = PLANT_TIME_STEP,
) -> tuple[Diagram, MultibodyPlant]:
    """
    Like create_IIWA14_diagram_with_pcontroller, but with the PDController
    defined in dtsystems.py: controller_gain, damping_gain, and dt go through
    to it.

    The controller has state (the previous measurement), so simulate()'s s0
    needs a second group: pass iiwa_s0(q0) + [q0] to start it at the arm's
    actual position, or the first derivative estimate is (q0 - 0) / dt.
    """
    builder, plant, scene_graph = _iiwa_builder(plant_time_step)
    observer = builder.AddSystem(Observer(14, range(7)))
    controller = builder.AddSystem(
        PDController(7, q_desired, controller_gain, damping_gain, dt=dt)
    )
    builder.Connect(plant.get_state_output_port(), observer.get_input_port())
    builder.Connect(observer.get_output_port(), controller.get_input_port())
    builder.Connect(controller.get_output_port(), plant.get_actuation_input_port())
    _finish(builder, scene_graph, meshcat, plant, controller)
    diagram = builder.Build()
    diagram.set_name("iiwa with PD controller")
    return diagram, plant


def create_IIWA14_diagram_with_waypoints(
    waypoints: list[np.ndarray],
    controller_gain: float = 10000,
    damping_gain: float = 3000,
    epsilon: float = 0.01,
    meshcat: Meshcat | None = None,
    dt: float | None = None,
) -> tuple[Diagram, MultibodyPlant]:
    """
    Drive the arm through a sequence of joint-space waypoints: an Observer
    keeps the positions out of the 14-D plant state, a SimpleTrajectoryFollower
    over the waypoints outputs the current target, and a PDController2 turns
    target and actual into joint torques.

    The follower only advances once the arm is within epsilon of the current
    waypoint, so this controller must be stiff enough that its gravity sag
    stays well under epsilon.

    Discrete state groups, in order: 0 the plant [q, v], 1 the follower's
    waypoint index, 2 the PD controller's previous measurement.  So s0 is
    iiwa_s0(q0) + [[0], q0].
    """
    dt = CONTROLLER_DT if dt is None else dt
    builder, plant, scene_graph = _iiwa_builder()
    observer = builder.AddSystem(Observer(14, range(7)))
    follower = builder.AddSystem(SimpleTrajectoryFollower(waypoints, epsilon, dt=dt))
    controller = builder.AddSystem(
        PDController2(7, controller_gain, damping_gain, dt=dt)
    )
    builder.Connect(plant.get_state_output_port(), observer.get_input_port())
    builder.Connect(observer.get_output_port(), follower.get_input_port())
    builder.Connect(follower.get_output_port(), controller.GetInputPort("target"))
    builder.Connect(observer.get_output_port(), controller.GetInputPort("actual"))
    builder.Connect(controller.get_output_port(), plant.get_actuation_input_port())
    _finish(builder, scene_graph, meshcat, plant, controller, follower)
    diagram = builder.Build()
    diagram.set_name("iiwa with waypoint follower")
    return diagram, plant


# Why not run the controller at the plant's own 1e-4?  A discrete
# MultibodyPlant "samples" its output ports (has_sampled_output_ports()), so
# the state the controller sees is one plant step old.  That one-step delay
# is 10% of a 1e-3 controller period and harmless, but a whole period at
# 1e-4: the differenced velocity is then a full step stale and the PD loop
# of 9.4 goes unstable (max joint speed after 3 s: 2.4 rad/s at 1e-4, 0.07
# at 1e-3; with SetUseSampledOutputPorts(False) 1e-4 settles fine).
def _iiwa_builder(time_step: float = PLANT_TIME_STEP):
    """
    The part of create_IIWA14_diagram that every diagram here shares: a
    builder holding the finalized, welded-base iiwa14 plant.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    parser = Parser(plant, scene_graph)
    try:
        parser.AddModelsFromUrl(IIWA14_URL)
    except RuntimeError as e:
        explain_model_download_error(e)
        raise
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))
    plant.Finalize()
    return builder, plant, scene_graph


def _finish(builder, scene_graph, meshcat, plant, *systems):
    """
    Visualizer (if any) and loggers: "log_plant" on the plant's state, and
    log_<name> on each other system's output.
    """
    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    logger = LogVectorOutput(plant.get_state_output_port(), builder)
    logger.set_name("log_plant")
    for s in systems:
        logger = LogVectorOutput(s.get_output_port(), builder)
        logger.set_name(f"log_{s.get_name()}")


######################################################################
##  Test rigs
######################################################################


def test_pcontroller(gain: float = 100.0, T: float = 10.0) -> None:
    """
    9.3: P controller trying to hold Q_START.
    """
    meshcat = get_meshcat()
    diagram, plant = create_IIWA14_diagram_with_pcontroller(gain, Q_START, meshcat)
    if not HEADLESS:
        plt.figure(figsize=(12, 6))
        plot_system_graphviz(diagram)
        plt.show()
    simulator = simulate(diagram, iiwa_s0(Q_START), T)
    print(f"   Target joint positions: {Q_START}")
    print(f"   Final joint positions:  {get_positions(plant, simulator)}")
    plot_log(diagram, simulator, "log_plant")
    show_meshcat()


def test_pd_controller(
    gain: float = 100.0, d_gain: float = 30.0, T: float = 5.0
) -> None:
    """
    9.4: PD controller stepping the arm from Q_START to Q_START + Q_STEP.
    """
    meshcat = get_meshcat()
    q_desired = Q_START + Q_STEP
    diagram, plant = create_IIWA14_diagram_with_pd_controller(
        gain, d_gain, q_desired, dt=CONTROLLER_DT, meshcat=meshcat
    )
    if not HEADLESS:
        plt.figure(figsize=(12, 6))
        plot_system_graphviz(diagram)
        plt.show()
    simulator = simulate(diagram, iiwa_s0(Q_START) + [Q_START], T)
    print(f"   Target joint positions: {q_desired}")
    print(f"   Final joint positions:  {get_positions(plant, simulator)}")
    plot_log(diagram, simulator, "log_plant")
    show_meshcat()


def test_iiwa_waypoints(
    gain: float = 10000.0, d_gain: float = 3000.0, epsilon: float = 0.01, T: float = 5.0
) -> None:
    """
    9.5: PD controller and waypoint follower drawing SQUARE_IIWA.
    """
    meshcat = get_meshcat()
    diagram, plant = create_IIWA14_diagram_with_waypoints(
        SQUARE_IIWA, gain, d_gain, epsilon, meshcat
    )
    if not HEADLESS:
        plt.figure(figsize=(12, 6))
        plot_system_graphviz(diagram)
        plt.show()
    q0 = SQUARE_IIWA[0]
    simulator = simulate(diagram, iiwa_s0(q0) + [[0.0], q0], T)
    print(f"   Final joint positions:  {get_positions(plant, simulator)}")
    plot_log(diagram, simulator, "log_plant", "log_SimpleTrajectoryFollower")
    show_meshcat()


if __name__ == "__main__":
    # test_const_torque(Q_START, np.zeros(7))     # 9.1 zero torque
    # test_const_torque(Q_START, 20 * np.ones(7)) # 9.2 constant torque
    # test_pcontroller()                          # 9.3
    # test_pd_controller()                        # 9.4
    test_iiwa_waypoints()                         # 9.5
