"""Quickstart: run the PID baselines and save telemetry.

    python examples/01_quickstart.py
"""
from usv_seakeeper import (SpeedHoldEnv, StationKeepEnv, SpeedTaskConfig,
                           StationTaskConfig, SimConfig, PIDGains,
                           PIDSpeedController, CascadePositionController,
                           ZeroController, rollout, evaluate)
from usv_seakeeper.plotting import plot_rollout, plot_comparison, plot_spectrum


def speed_hold():
    def make():
        return SpeedHoldEnv(preset="coastal_chop",
                            task=SpeedTaskConfig(target_speed=2.0,
                                                 randomize_target=None))

    print(make().describe(), "\n")

    controllers = [
        ZeroController(),
        PIDSpeedController(PIDGains(260, 50, 30), name="PID"),
        PIDSpeedController(PIDGains(260, 50, 30), slope_feedforward=True,
                           name="PID+FF"),
    ]

    # single episode, identical wave realisation for every controller
    results = [rollout(make(), c, seed=7) for c in controllers]
    for r in results:
        print(r.summary())
    plot_comparison(results, save="speed_hold_comparison.png")
    plot_rollout(results[-1], save="speed_hold_telemetry.png")
    results[-1].to_csv("speed_hold_telemetry.csv")

    # a single episode in an irregular sea is noisy -- average over seeds
    print("\naveraged over 10 wave realisations:")
    for c in controllers[1:]:
        evaluate(make(), c, seeds=range(10), verbose=True)


def station_keep():
    print("\n--- station keeping ---")
    env = StationKeepEnv(preset="coastal_chop",
                         task=StationTaskConfig(target_position=40.0,
                                                randomize_target=None))
    res = rollout(env, CascadePositionController(kp_pos=0.35, max_speed=2.5),
                  seed=1)
    print(res.summary())
    plot_rollout(res, save="station_keep_telemetry.png")


def spectrum():
    env = SpeedHoldEnv(preset="storm_head_sea")
    env.reset(seed=0)
    plot_spectrum(env.sim.field, save="wave_spectrum.png")
    print("\n" + env.describe())


if __name__ == "__main__":
    speed_hold()
    station_keep()
    spectrum()
    print("\nwrote: speed_hold_comparison.png, speed_hold_telemetry.png,")
    print("       speed_hold_telemetry.csv, station_keep_telemetry.png,")
    print("       wave_spectrum.png")
