"""Visualisation: animated scope, comparison video, live viewer.

    python examples/05_visualise.py                # writes stills + videos
    python examples/05_visualise.py --live         # interactive window
    python examples/05_visualise.py --quick        # stills only, no encoding

Rendering costs roughly 0.15 s per frame, so a 30 s clip at 20 fps takes about
90 s. Use ``max_seconds`` and ``decimate`` to trim.
"""
from __future__ import annotations

import argparse

import numpy as np


def _controllers():
    from usv_seakeeper import Controller, PIDGains, PIDSpeedController

    class SlidingMode(Controller):
        name = "sliding-mode"

        def act(self, obs, dt):
            drag_ff = obs.drag_coeff * obs.u * abs(obs.u) / obs.thrust_max
            return drag_ff + 1.2 * np.clip(obs.error / 0.15, -1.0, 1.0)

    pid = PIDSpeedController(PIDGains(420, 70, 60), slope_feedforward=True,
                             name="PID+FF")
    return pid, SlidingMode()


def stills():
    """One still per sea state, plus the worst ventilation moment."""
    from usv_seakeeper import (SpeedHoldEnv, StationKeepEnv, SpeedTaskConfig,
                               StationTaskConfig, WaveConfig, VesselConfig,
                               SimConfig, CascadePositionController, rollout,
                               save_frame)
    pid, _ = _controllers()

    for preset, vref in (("calm_harbor", 2.5), ("storm_head_sea", 1.8),
                         ("sea_state_6", 1.5), ("following_storm", 1.5)):
        env = SpeedHoldEnv(preset=preset,
                           task=SpeedTaskConfig(target_speed=vref,
                                                randomize_target=None))
        res = rollout(env, pid, seed=11)
        save_frame(res, index=res.t.size // 2, save=f"scope_{preset}.png")
        print(f"scope_{preset}.png   rmse_steady={res.metrics['rmse_steady']:.3f}"
              f"  min_immersion={res.metrics['min_ventilation']:.3f}")

    # A short, steep sea is where the propeller actually unwets -- ventilation
    # tracks wave steepness, not wave height.
    env = SpeedHoldEnv(wave=WaveConfig(kind="jonswap", hs=2.0, tp=3.5),
                       vessel=VesselConfig(mass=400.0, thrust_max=900.0,
                                           drag_coeff=50.0),
                       sim=SimConfig(max_time=45.0),
                       task=SpeedTaskConfig(target_speed=1.5,
                                            randomize_target=None))
    res = rollout(env, pid, seed=2)
    worst = int(np.argmin(res.ventilation))
    save_frame(res, index=worst, save="scope_ventilating.png")
    print(f"scope_ventilating.png at t={res.t[worst]:.1f}s, "
          f"immersion={res.ventilation[worst]:.3f}")

    env = StationKeepEnv(preset="coastal_chop",
                         task=StationTaskConfig(target_position=40.0,
                                                randomize_target=None))
    res = rollout(env, CascadePositionController(), seed=1)
    save_frame(res, index=res.t.size // 2, save="scope_station_keep.png")
    print("scope_station_keep.png")


def videos():
    from usv_seakeeper import (SpeedHoldEnv, SpeedTaskConfig, rollout,
                               animate_rollout, animate_comparison)
    pid, sliding = _controllers()

    env = SpeedHoldEnv(preset="storm_head_sea",
                       task=SpeedTaskConfig(target_speed=1.8,
                                            randomize_target=None))
    res = rollout(env, pid, seed=11)
    print("rendering scope_head_sea.mp4 ...")
    animate_rollout(res, save="scope_head_sea.mp4", fps=20, max_seconds=30.0)
    print("rendering scope_head_sea.gif ...")
    animate_rollout(res, save="scope_head_sea.gif", fps=12, decimate=2,
                    max_seconds=18.0)

    # Same seed for both controllers, or the comparison is meaningless.
    def make():
        return SpeedHoldEnv(preset="coastal_chop",
                            task=SpeedTaskConfig(target_speed=2.0,
                                                 randomize_target=None))
    traces = [rollout(make(), pid, seed=7), rollout(make(), sliding, seed=7)]
    print("rendering scope_compare.mp4 ...")
    animate_comparison(traces, save="scope_compare.mp4", fps=20)
    for tr in traces:
        print(f"  {tr.name:14} rmse={tr.metrics['rmse']:.3f}  "
              f"action_rate={tr.metrics['action_rate_rms']:.3f}")


def live():
    import matplotlib
    for backend in ("TkAgg", "QtAgg", "MacOSX"):
        try:
            matplotlib.use(backend)
            break
        except Exception:
            continue
    from usv_seakeeper.render import LiveViewer
    print(f"backend: {matplotlib.get_backend()}")
    LiveViewer(preset="coastal_chop", target_speed=2.0).show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="interactive viewer")
    ap.add_argument("--quick", action="store_true", help="stills only")
    args = ap.parse_args()

    if args.live:
        live()
        return
    import matplotlib
    matplotlib.use("Agg")
    stills()
    if not args.quick:
        videos()


if __name__ == "__main__":
    main()
