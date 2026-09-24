"""
Turn the pickles written by experiments_10_11.py into the two one-page PDFs
for problems 10 and 11 (6.4212).

    python make_pdfs_10_11.py            # writes problem10.pdf, problem11.pdf
"""

import os
import pickle
import textwrap
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from iiwa_1 import SQUARE_IIWA

RESULTS = Path(os.environ.get("PS1_RESULTS", Path(__file__).parent / "results_10_11"))
HERE = Path(__file__).parent

plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9,
    "lines.linewidth": 1.6,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.frameon": False,
})


def load(name):
    return pickle.load(open(RESULTS / f"{name}.pkl", "rb"))


def page(title, sections, draw, out, ncols=2):
    """A letter-size page: title, bold-headed sections of prose, then a 2x2 grid of plots."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.08, 0.96, title, fontsize=13, weight="bold", va="top")
    y = 0.935
    for heading, body in sections:
        if heading:
            fig.text(0.08, y, heading, fontsize=8.5, weight="bold", va="top")
            y -= 0.0155
        lines = textwrap.wrap(body, width=110)
        fig.text(0.08, y, "\n".join(lines), fontsize=8.5, va="top", linespacing=1.35)
        y -= 0.0143 * len(lines) + 0.009
    gs = fig.add_gridspec(2, ncols, left=0.09, right=0.96, top=y - 0.024, bottom=0.04,
                          hspace=0.55, wspace=0.3 if ncols == 2 else 0.45)
    draw([fig.add_subplot(gs[i, j]) for i in range(2) for j in range(ncols)])
    fig.savefig(out)
    fig.savefig(RESULTS / out.with_suffix(".png").name, dpi=70)  # preview, outside the repo
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------- problem 10

def problem10():
    a = load("10a")                    # plant step sweep, controller at 10 ms
    b = load("10b") + load("10x")      # controller period sweep and follow-ups
    truth = a[0]
    for r in a:
        qt = np.array([np.interp(r["t"], truth["t"], truth["q"][j]) for j in range(7)])
        r["rms"] = np.sqrt(np.mean((r["q"] - qt) ** 2))
    soft = {r["ctrl_dt"]: r for r in b if r["gain"] == 100 and r["plant_h"] == 1e-4}
    h10 = next(r for r in a if r["plant_h"] == 1e-2)
    h10_slow = next(r for r in b if r["gain"] == 100 and r["plant_h"] == 1e-2)

    walls = [r["wall"] for r in b if r["plant_h"] == 1e-4]
    sections = [
        ("Setup",
         "All the runs below use the step response from problem 9.4 (PD gains 100 and 30, simulated for 4 s). The default "
         "plant step is 0.1 ms and the default controller period is 1 ms; I changed one at a time."),

        ("How the two step sizes relate",
         "The plant step is how often the physics is integrated, and the controller period defines how often a new torque "
         "is computed (with torque held constant between updates). Therefore, the controller period should be a whole "
         "number of plant steps, so that the plant moves smoothly under a constant torque, and the controller see a fresh "
         "state every time it is run."),

        ("Plant time step and error",
         "I varied plant time step and computed the RMS joint error of each run (panel A). The error is roughly "
         "proportional to the step size (doubling the step doubles the error), which makes sense given that we are using "
         "a first-order integrator."),

        ("Wall-clock time",
         f"Since the plant step directly determines how many physics steps there are, this clearly means that runtime "
         f"decreases as the plant time step increases (panel B). The controller period, on the other hand, has no "
         f"measurable effect: with the plant at 0.1 ms, every controller period from 0.1 ms to 200 ms took about "
         f"{np.mean(walls):.0f} s (the flat line in panel B). Therefore, the plant step is the only knob that makes the "
         f"simulation faster, and a 1 ms step gives a 10x speedup for a few milliradians of error."),

        ("Failure mode",
         "If the plant step is set equal to the controller period (both 10 ms), the arm never settles and just keeps "
         "oscillating (panel C). Part of this is presumably that a 10 ms integration step is coarse (panel A shows the "
         "error growing steadily with the step size), but that can't be the whole story: the same 10 ms plant step with a "
         "50 ms controller period settles fine, and setting both steps to 0.1 ms, the finest step in the sweep, gives the "
         "same instability. So the two step sizes being equal is a problem in itself. The reason is that Drake's discrete "
         "plant reports its state with a delay of one plant step. Normally this doesn't matter since the controller period "
         "is much longer than one plant step, but when the two are equal the position the controller reads is a full "
         "period old. This means the velocity estimate (which the PD controller gets by differencing consecutive "
         "positions) is also a full period out of date, and so the damping term ends up pushing in the wrong direction "
         "and adds energy instead of removing it."),

        ("Controller period",
         "For the gains in 9.4 the controller period hardly matters, since 1 ms, 10 ms and 100 ms all give nearly the same "
         "response (panel D). This makes sense because the closed loop takes about a second to settle, so even a 100 ms update is "
         "fast compared to the motion it is controlling. Therefore, as long as the controller period is several plant "
         "steps long (to avoid the failure mode above), there is little to gain from running the controller any faster "
         "than the response time of the loop requires."),
    ]

    def draw(ax):
        hs = np.array([r["plant_h"] for r in a[1:]]) * 1e3
        ax[0].plot(hs, [r["rms"] * 1e3 for r in a[1:]], "o-")
        ax[0].set_xscale("log"); ax[0].set_yscale("log")
        ax[0].set_xlabel("plant time step (ms)"); ax[0].set_ylabel("RMS joint error (mrad)")
        ax[0].set_title("A. Error grows with the plant time step")

        ax[1].plot([r["plant_h"] * 1e3 for r in a], [r["wall"] for r in a], "o-",
                   label="plant step varied")
        dts = sorted(soft)
        ax[1].plot([dt * 1e3 for dt in dts], [soft[dt]["wall"] for dt in dts], "o-",
                   label="controller period varied")
        ax[1].set_xscale("log"); ax[1].set_yscale("log")
        ax[1].set_xlabel("step size being varied (ms)"); ax[1].set_ylabel("run time for 4 s of simulation (s)")
        ax[1].set_title("B. Run time depends on the plant step only")
        ax[1].legend(fontsize=7, loc="lower left")

        ax[2].plot(a[0]["t"], a[0]["q"][0], label="plant 0.1 ms, controller 10 ms")
        ax[2].plot(h10["t"], h10["q"][0], label="plant 10 ms, controller 10 ms")
        ax[2].plot(h10_slow["t"], h10_slow["q"][0], label="plant 10 ms, controller 50 ms")
        ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("joint 1 angle (rad)")
        ax[2].set_title("C. Equal step sizes make the loop unstable"); ax[2].legend(fontsize=7)

        for dt in (1e-3, 1e-2, 1e-1):
            ax[3].plot(soft[dt]["t"], soft[dt]["q"][0], label=f"controller {dt*1e3:g} ms")
        ax[3].set_xlabel("time (s)"); ax[3].set_ylabel("joint 1 angle (rad)")
        ax[3].set_title("D. Controller period barely matters at these gains"); ax[3].legend(fontsize=7)

    page("Problem 10: Step size", sections, draw, HERE / "problem10.pdf")


# ---------------------------------------------------------------- problem 11

def problem11():
    c = load("11ab")
    d = load("11c_metrics")
    step = 0.5
    runs = {r["d_gain"]: r for r in c if r["config"] == "Q_START"}
    for r in runs.values():
        q1 = r["q"][0] - r["q"][0][0]
        r["ov"] = (q1.max() - step) / step * 100
        band = np.abs(q1 - step) > 0.02 * step
        r["ts"] = r["t"][band][-1] if band.any() else 0.0
    sq = {r["d_gain"]: r for r in d}

    sections = [
        ("Setup",
         "I applied a 0.5 rad step to joint 1 with P gain 1000 and swept the damping gain of the PD controller. I chose "
         "joint 1 because it rotates about a vertical axis, so gravity has no effect on it and the response should be a "
         "clean second-order one."),

        ("Step response",
         f"With no damping the joint rings for the entire run (panel A). Adding damping reduces the overshoot and the "
         f"ringing. At 50 the response is clearly under-damped, at 200 it has only a small overshoot and reaches the target "
         f"fastest, and at 400 it is over-damped (no overshoot but a noticeably slower approach). Panel B shows the "
         f"overshoot falling steadily with the damping gain, from {runs[0]['ov']:.0f}% with no damping to zero at 400. "
         f"The settling time (panel C, time to get within 2% of the step and stay there) behaves differently. It drops "
         f"quickly at first, is shortest at 200 ({runs[200]['ts']:.2f} s) and then rises again at 400 "
         f"({runs[400]['ts']:.2f} s) because the over-damped response is slow to arrive. Since 200 still overshoots "
         f"slightly and 400 does not, critical damping for this joint must be somewhere between the two."),

        ("Square task",
         f"I then drew the square from 9.5 (P gain 10000) at several damping gains. With damping 1000 the arm reaches each "
         f"corner quickly but overshoots, so the sides of the square bulge outwards (panel D). With 3000 the sides are "
         f"nearly straight. With 10000 they are straight as well, but the arm approaches each corner so slowly that the "
         f"whole square takes {sq[10000]['done']:.1f} s instead of {sq[3000]['done']:.1f} s (panel F). This makes sense "
         f"because the waypoint follower only moves on once the arm is within 0.01 rad of the current corner, and an "
         f"over-damped arm creeps towards its target. Panel E shows the tracking error, i.e. the distance of the end "
         f"effector from the edge it is meant to be following, over time. With 300 and 1000 the error spikes after every "
         f"corner as the arm overshoots, whereas with 3000 and 10000 it stays within a few centimetres. At 300 the arm is "
         f"so under-damped that it takes {sq[300]['done']:.1f} s (I left this one out of panel D because it wanders all "
         f"over the plot, but it is in panels E and F)."),

        ("Conclusion",
         "Therefore, more damping helps by removing overshoot and giving straight edges, but hurts by slowing down the "
         "approach to every waypoint. Past critical damping there is no accuracy left to gain, only time to lose."),
    ]

    def seg_dist(pt, a, b):
        ab = b - a
        s = np.clip(np.dot(pt - a, ab) / np.dot(ab, ab), 0, 1)
        return np.linalg.norm(pt - (a + s * ab))

    def tracking_error(r):
        """Distance from the edge being followed, NaN once the square is done."""
        idx, ee, wp = r["idx"], r["ee"], r["ee_wp"]
        err = np.full(len(r["t"]), np.nan)
        for i in range(len(err)):
            if idx[i] >= 1 and r["t"][i] <= r["done"]:
                err[i] = seg_dist(ee[i], wp[idx[i] - 1], wp[idx[i]])
        return err

    def draw(ax):
        for dg in (0, 50, 200, 400):
            ax[0].plot(runs[dg]["t"], runs[dg]["q"][0] - runs[dg]["q"][0][0], label=f"damping {dg}" if dg else "no damping")
        ax[0].axhline(step, color="k", lw=0.6)
        ax[0].set_xlim(0, 1.5); ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("joint 1 angle (rad)")
        ax[0].set_title("A. Step response of joint 1, P gain 1000")
        ax[0].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=4,
                     handlelength=1.0, handletextpad=0.4, columnspacing=0.6)

        dgs = sorted(runs)
        ax[1].plot(dgs, [runs[dg]["ov"] for dg in dgs], "o-")
        ax[1].set_xlabel("damping gain"); ax[1].set_ylabel("overshoot (%)")
        ax[1].set_title("B. Overshoot falls as damping increases")

        ax[2].plot(dgs[2:], [runs[dg]["ts"] for dg in dgs[2:]], "o-")
        ax[2].set_xlabel("damping gain"); ax[2].set_ylabel("2% settling time (s)")
        ax[2].set_title("C. Settling time is shortest near critical")

        for dg in (1000, 3000, 10000):
            ax[3].plot(sq[dg]["ee"][:, 1], sq[dg]["ee"][:, 2], label=f"damping {dg}")
        ax[3].plot(d[0]["ee_wp"][:, 1], d[0]["ee_wp"][:, 2], color="k", lw=0.6, label="target square")
        ax[3].set_aspect("equal"); ax[3].set_xlabel("y (m)"); ax[3].set_ylabel("z (m)")
        ax[3].set_title("D. End-effector path around the square"); ax[3].legend(fontsize=6.5, loc="center")

        for dg in (300, 1000, 3000, 10000):
            ax[4].plot(sq[dg]["t"], tracking_error(sq[dg]) * 100, label=f"damping {dg}")
        ax[4].set_xlabel("time (s)"); ax[4].set_ylabel("distance from edge (cm)")
        ax[4].set_title("E. Tracking error along the square"); ax[4].legend(fontsize=6.5)

        dgs = sorted(sq)
        ax[5].plot(dgs, [sq[dg]["done"] for dg in dgs], "o-")
        ax[5].set_xscale("log"); ax[5].set_xlabel("damping gain"); ax[5].set_ylabel("time to finish the square (s)")
        ax[5].set_title("F. Too much damping makes the square slow")

        for axx in ax:
            axx.title.set_size(8)

    page("Problem 11: Damping", sections, draw, HERE / "problem11.pdf", ncols=3)


if __name__ == "__main__":
    problem10()
    problem11()
