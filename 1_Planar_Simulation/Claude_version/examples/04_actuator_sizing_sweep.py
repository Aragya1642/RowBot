"""Headless sweep: where does the controller stop being the problem?

    python examples/04_actuator_sizing_sweep.py

Turns the original tool's footer claim -- "when a crest demands more than
Fmax, no gain tuning saves you" -- into an actual chart. Sweeps thrust
capacity against significant wave height and records tracking RMSE, thrust
saturation and propeller ventilation for a well-tuned PID.

Because the sim is headless and deterministic given a seed, this is
embarrassingly parallel; ``multiprocessing`` is used here for a modest
speedup without extra dependencies.
"""
from __future__ import annotations

import itertools
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from usv_seakeeper import (SpeedHoldEnv, SpeedTaskConfig, WaveConfig,
                           VesselConfig, SimConfig, PIDGains,
                           PIDSpeedController, evaluate)

HS_VALUES = np.linspace(0.5, 4.0, 8)
FMAX_VALUES = np.array([300, 450, 600, 800, 1100, 1500, 2000])
TP_VALUES = np.array([3.0, 4.0, 5.0, 6.5, 8.0, 11.0])
SEEDS = range(6)
TARGET_SPEED = 2.0
TP_FIXED = 6.5


def _env(hs: float, fmax: float, tp: float) -> SpeedHoldEnv:
    return SpeedHoldEnv(
        wave=WaveConfig(kind="jonswap", hs=float(hs), tp=float(tp)),
        vessel=VesselConfig(mass=400.0, thrust_max=float(fmax),
                            drag_coeff=45.0, slew_rate=2500.0),
        sim=SimConfig(dof="surge_heave_pitch", dt=0.05, dt_physics=0.01,
                      max_time=45.0),
        task=SpeedTaskConfig(target_speed=TARGET_SPEED, randomize_target=None),
    )


def _score(env):
    ctrl = PIDSpeedController(PIDGains(320, 60, 35), slope_feedforward=True)
    return evaluate(env, ctrl, seeds=SEEDS)


def one_cell(args):
    """Sizing grid cell: thrust capacity vs wave height."""
    hs, fmax = args
    m = _score(_env(hs, fmax, TP_FIXED))
    return (hs, fmax, m.get("rmse", np.nan), m.get("saturated_frac", np.nan),
            m.get("min_ventilation", np.nan))


def one_vent_cell(args):
    """Ventilation grid cell: wave height vs peak period.

    Worth a separate chart because ventilation turns out to track wave
    *steepness*, not wave height: in long swell the hull heaves with the
    surface (RAO -> 1), there is little relative motion, and the propeller
    stays wetted no matter how big the waves are.
    """
    hs, tp = args
    m = _score(_env(hs, 900.0, tp))
    return (hs, tp, m.get("min_ventilation", np.nan),
            m.get("mean_ventilation", np.nan), m.get("rmse", np.nan))


def main():
    grid = list(itertools.product(HS_VALUES, FMAX_VALUES))
    vgrid = list(itertools.product(HS_VALUES, TP_VALUES))
    with Pool() as pool:
        rows = pool.map(one_cell, grid)
        vrows = pool.map(one_vent_cell, vgrid)

    shape = (len(HS_VALUES), len(FMAX_VALUES))
    rmse = np.array([r[2] for r in rows]).reshape(shape)
    sat = np.array([r[3] for r in rows]).reshape(shape)
    vent = np.array([r[4] for r in rows]).reshape(shape)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [
        (rmse, "speed RMSE [m/s]", "viridis_r"),
        (sat, "fraction of time saturated", "magma"),
        (vent, "worst prop immersion", "cividis"),
    ]
    for ax, (data, label, cmap) in zip(axes, panels):
        im = ax.pcolormesh(FMAX_VALUES, HS_VALUES, data, cmap=cmap,
                           shading="nearest")
        fig.colorbar(im, ax=ax, label=label)
        ax.set_xlabel("max thrust [N]")
        ax.set_ylabel("Hs [m]")
        ax.set_title(label)
    fig.suptitle(f"Actuator sizing for {TARGET_SPEED} m/s speed hold, "
                 f"Tp={TP_FIXED} s (tuned PID+FF, "
                 f"{len(list(SEEDS))} seeds per cell)")
    fig.tight_layout()
    fig.savefig("actuator_sizing_sweep.png", dpi=130)
    print("wrote actuator_sizing_sweep.png")

    vshape = (len(HS_VALUES), len(TP_VALUES))
    vmin = np.array([r[2] for r in vrows]).reshape(vshape)
    vmean = np.array([r[3] for r in vrows]).reshape(vshape)
    fig2, axes2 = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, data, label in ((axes2[0], vmin, "worst prop immersion"),
                            (axes2[1], vmean, "mean prop immersion")):
        im = ax.pcolormesh(TP_VALUES, HS_VALUES, data, cmap="cividis",
                           shading="nearest", vmin=0.0, vmax=1.0)
        fig2.colorbar(im, ax=ax, label=label)
        ax.set_xlabel("peak period Tp [s]")
        ax.set_ylabel("Hs [m]")
        ax.set_title(label)
    fig2.suptitle("Ventilation tracks steepness, not wave height "
                  "(Fmax = 900 N)")
    fig2.tight_layout()
    fig2.savefig("ventilation_regime.png", dpi=130)
    print("wrote ventilation_regime.png")

    print("\nRMSE [m/s], rows = Hs, cols = Fmax")
    print("        " + "".join(f"{f:>8.0f}" for f in FMAX_VALUES))
    for i, hs in enumerate(HS_VALUES):
        print(f"{hs:5.2f} m " + "".join(f"{v:>8.3f}" for v in rmse[i]))


if __name__ == "__main__":
    main()
