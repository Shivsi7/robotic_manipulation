"""
Experiments behind the 6.4212 write-ups for problems 10 (step size) and 11
(damping).  Each stage simulates the iiwa a number of times and pickles what
the write-up needs; make_pdfs_10_11.py turns the pickles into the two PDFs.

    HEADLESS=1 python experiments_10_11.py 10a 10b 11ab 11c

Results go to $PS1_RESULTS (default: results_10_11/ next to this file).
"""

import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("HEADLESS", "1")

import numpy as np
from dtsystems import simulate
from iiwa_1 import (
    CONTROLLER_DT,
    PLANT_TIME_STEP,
    Q_START,
    Q_STEP,
    SQUARE_IIWA,
    create_IIWA14_diagram_with_pd_controller,
    create_IIWA14_diagram_with_waypoints,
    iiwa_s0,
)

RESULTS = Path(os.environ.get("PS1_RESULTS", Path(__file__).parent / "results_10_11"))
RESULTS.mkdir(parents=True, exist_ok=True)

# Straight up: every link stacked over joint 1, so the inertia about it is as
# small as it gets.  Contrast with Q_START, where the arm reaches forward.
Q_UP = np.zeros(7)


def log_of(diagram, simulator, name):
    log = {s.get_name(): s for s in diagram.GetSystems()}[name].FindLog(
        simulator.get_context()
    )
    return log.sample_times(), log.data()


def pd_step(gain, d_gain, q0, q_des, ctrl_dt, plant_h, T):
    """
    Step the PD-controlled arm from q0 to q_des; return the logged trajectory
    and the wall-clock time the simulation took.
    """
    diagram, plant = create_IIWA14_diagram_with_pd_controller(
        gain, d_gain, q_des, dt=ctrl_dt, plant_time_step=plant_h
    )
    start = time.perf_counter()
    simulator = simulate(diagram, iiwa_s0(q0) + [q0], T)
    wall = time.perf_counter() - start
    t, d = log_of(diagram, simulator, "log_plant")
    print(f"   gain={gain} d={d_gain} ctrl_dt={ctrl_dt} h={plant_h}: {wall:.1f} s wall", flush=True)
    return dict(t=t, q=d[:7], v=d[7:], wall=wall, gain=gain, d_gain=d_gain,
                ctrl_dt=ctrl_dt, plant_h=plant_h)


def stage_10a():
    """Plant time step sweep; controller at Drake's default 10 ms."""
    runs = []
    for h in (1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2):
        runs.append(pd_step(100, 30, Q_START, Q_START + Q_STEP, 1e-2, h, 4.0))
    pickle.dump(runs, open(RESULTS / "10a.pkl", "wb"))


def stage_10b():
    """Controller period sweep at the fine plant step, for two gain sets."""
    runs = []
    for dt in (1e-4, 5e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1):
        runs.append(pd_step(100, 30, Q_START, Q_START + Q_STEP, dt, PLANT_TIME_STEP, 4.0))
    for dt in (1e-3, 3e-3, 5e-3, 1e-2, 2e-2):
        runs.append(pd_step(10000, 3000, Q_START, Q_START + Q_STEP, dt, PLANT_TIME_STEP, 4.0))
    pickle.dump(runs, open(RESULTS / "10b.pkl", "wb"))


def stage_10x():
    """
    Two follow-ups.  High gains at long controller periods, to find where the
    sampled-data loop lets go; and coarse plant steps with a controller period
    several times longer, to tell integration error from the same-rate delay.
    """
    runs = []
    for dt in (5e-2, 1e-1):
        runs.append(pd_step(10000, 3000, Q_START, Q_START + Q_STEP, dt, PLANT_TIME_STEP, 4.0))
    for h in (5e-3, 1e-2):
        runs.append(pd_step(100, 30, Q_START, Q_START + Q_STEP, 5e-2, h, 4.0))
    pickle.dump(runs, open(RESULTS / "10x.pkl", "wb"))


def stage_11ab():
    """Damping sweep on a joint-1 step (gravity-free axis), two configurations."""
    step = np.array([0.5, 0, 0, 0, 0, 0, 0])
    runs = []
    for d in (0, 20, 50, 100, 200, 400):
        runs.append(dict(config="Q_START", **pd_step(1000, d, Q_START, Q_START + step, CONTROLLER_DT, PLANT_TIME_STEP, 3.0)))
    for d in (50, 100):
        runs.append(dict(config="Q_UP", **pd_step(1000, d, Q_UP, Q_UP + step, CONTROLLER_DT, PLANT_TIME_STEP, 3.0)))
    pickle.dump(runs, open(RESULTS / "11ab.pkl", "wb"))


def stage_11c():
    """The square at several damping gains: tracking error and completion time."""
    runs = []
    for d in (300, 1000, 3000, 10000):
        diagram, plant = create_IIWA14_diagram_with_waypoints(SQUARE_IIWA, 10000, d, 0.01)
        q0 = SQUARE_IIWA[0]
        start = time.perf_counter()
        simulator = simulate(diagram, iiwa_s0(q0) + [[0.0], q0], 10.0)
        wall = time.perf_counter() - start
        t, dat = log_of(diagram, simulator, "log_plant")
        tf, target = log_of(diagram, simulator, "log_SimpleTrajectoryFollower")
        q = dat[:7]
        # Forward kinematics of the end effector, subsampled.
        ctx = plant.CreateDefaultContext()
        ee_frame = plant.GetFrameByName("iiwa_link_7")
        def fk(qq):
            plant.SetPositions(ctx, qq)
            return plant.CalcRelativeTransform(ctx, plant.world_frame(), ee_frame).translation()
        sub = slice(0, None, 10)
        ee = np.array([fk(q[:, i]) for i in range(q.shape[1])[sub]])
        ee_wp = np.array([fk(w) for w in SQUARE_IIWA])
        # Which waypoint the follower was chasing at each subsampled time.
        wp = np.array(SQUARE_IIWA)
        idx_full = np.argmin(np.linalg.norm(target.T[:, None, :] - wp[None], axis=2), axis=1)
        idx = idx_full[np.searchsorted(tf, t[sub]).clip(0, len(tf) - 1)]
        print(f"   square d={d}: {wall:.1f} s wall, switches at {np.round(t[sub][np.flatnonzero(np.diff(idx)) + 1], 2)}", flush=True)
        runs.append(dict(d_gain=d, t=t[sub], q=q[:, sub], ee=ee, ee_wp=ee_wp, idx=idx, wall=wall))
    pickle.dump(runs, open(RESULTS / "11c.pkl", "wb"))


if __name__ == "__main__":
    stages = dict(a10=stage_10a, b10=stage_10b, x10=stage_10x, ab11=stage_11ab, c11=stage_11c)
    names = {"10a": "a10", "10b": "b10", "10x": "x10", "11ab": "ab11", "11c": "c11"}
    for arg in sys.argv[1:]:
        print(f"== stage {arg}", flush=True)
        stages[names[arg]]()
