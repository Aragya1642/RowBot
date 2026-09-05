"""API and integration tests.

Run with ``python tests/test_api.py`` (no pytest needed) or ``pytest tests/``.

``validate.py`` checks the *physics*. This file checks the *plumbing*: the
Gymnasium contract, and the PolicyController bridge that lets a trained policy
be scored by the same harness as a PID. That bridge is the part most likely to
break silently -- a shape or normalisation mismatch between training and
evaluation produces a policy that "works" in training and is garbage at deploy,
with no error message anywhere. So it gets a test with a stand-in policy rather
than being taken on faith.
"""
from __future__ import annotations

import sys
import numpy as np

from usv_seakeeper import (SpeedHoldEnv, StationKeepEnv, SpeedTaskConfig,
                           StationTaskConfig, SimConfig, ObsConfig,
                           RandomizeConfig, PIDGains, PIDSpeedController,
                           CascadePositionController, PolicyController,
                           Controller, CallableController, ObservationBuilder,
                           rollout, evaluate, comparison_table)

FAILURES: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    ok = bool(cond)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------
def test_gym_contract():
    """reset/step signatures, shapes, dtypes and bounds as SB3 expects them."""
    for cls in (SpeedHoldEnv, StationKeepEnv):
        env = cls(preset="coastal_chop")
        obs, info = env.reset(seed=0)
        n = env.observation_space.shape[0]
        check(f"{cls.__name__}: reset returns (obs, info)",
              isinstance(obs, np.ndarray) and isinstance(info, dict))
        check(f"{cls.__name__}: obs shape matches space",
              obs.shape == (n,), f"{obs.shape} vs ({n},)")
        check(f"{cls.__name__}: obs dtype float32", obs.dtype == np.float32,
              str(obs.dtype))
        check(f"{cls.__name__}: obs finite at reset", bool(np.all(np.isfinite(obs))))

        out = env.step(np.array([0.4], dtype=np.float32))
        check(f"{cls.__name__}: step returns 5-tuple", len(out) == 5)
        o, r, term, trunc, inf = out
        check(f"{cls.__name__}: step obs shape/dtype",
              o.shape == (n,) and o.dtype == np.float32)
        check(f"{cls.__name__}: reward is a float", isinstance(r, float))
        check(f"{cls.__name__}: flags are bool",
              isinstance(term, bool) and isinstance(trunc, bool))
        check(f"{cls.__name__}: action space is Box(-1,1,(1,))",
              env.action_space.shape == (1,)
              and float(env.action_space.low[0]) == -1.0
              and float(env.action_space.high[0]) == 1.0)


def test_episode_terminates():
    """Episodes must actually end, and at the configured horizon."""
    env = SpeedHoldEnv(preset="coastal_chop", sim=SimConfig(max_time=10.0))
    env.reset(seed=0)
    steps = 0
    while steps < 10_000:
        _, _, term, trunc, _ = env.step(0.5)
        steps += 1
        if term or trunc:
            break
    expected = int(round(10.0 / env.dt))
    check("episode truncates at max_time", steps == expected,
          f"{steps} steps, expected {expected}")


def test_obs_finite_under_extremes():
    """Observations must stay finite in the worst preset at full deflection,
    including when the episode terminates on a safety limit."""
    env = SpeedHoldEnv(preset="sea_state_6", sim=SimConfig(max_time=60.0))
    obs, _ = env.reset(seed=3)
    bad = 0
    for i in range(1200):
        a = 1.0 if (i // 7) % 2 == 0 else -1.0     # deliberately abusive
        obs, r, term, trunc, _ = env.step(a)
        if not np.all(np.isfinite(obs)) or not np.isfinite(r):
            bad += 1
        if term or trunc:
            break
    check("obs and reward finite under adversarial actions", bad == 0,
          f"{bad} non-finite samples")


def test_determinism_and_seeding():
    """Same seed -> same trajectory; different seed -> different sea."""
    def run(seed):
        env = SpeedHoldEnv(preset="coastal_chop")
        env.reset(seed=seed)
        return [float(env.step(0.5)[1]) for _ in range(80)]
    check("same seed reproduces rewards exactly", run(5) == run(5))
    check("different seeds differ", run(5) != run(6))


def test_randomization_does_not_compound():
    """Randomised params must be redrawn from the *baseline* each episode, not
    from the previous episode's values -- otherwise mass/thrust random-walk
    away over training and the task silently changes."""
    env = SpeedHoldEnv(
        preset="coastal_chop",
        randomize=RandomizeConfig(mass=(300.0, 400.0),
                                  thrust_max=(500.0, 700.0)))
    masses, thrusts = [], []
    for s in range(30):
        env.reset(seed=s)
        masses.append(env.sim.vessel.mass)
        thrusts.append(env.sim.vessel.thrust_max)
    check("randomised mass stays inside its range",
          all(300.0 <= m <= 400.0 for m in masses),
          f"[{min(masses):.1f}, {max(masses):.1f}]")
    check("randomised thrust stays inside its range",
          all(500.0 <= t <= 700.0 for t in thrusts),
          f"[{min(thrusts):.1f}, {max(thrusts):.1f}]")
    check("randomisation actually varies", len(set(masses)) > 20)


def test_policy_controller_bridge():
    """A stand-in 'trained policy' must run through the same harness as a PID.

    Covers both accepted policy shapes: an SB3-style object with .predict()
    returning (action, state), and a bare callable.
    """
    env_for_obs = SpeedHoldEnv(preset="coastal_chop")
    env_for_obs.reset(seed=0)
    n_expected = env_for_obs.obs_builder.size

    seen_shapes = []

    class FakeSB3Model:
        """Mimics SB3's predict contract, including the (action, state) tuple."""
        def predict(self, obs, deterministic=True):
            seen_shapes.append(np.asarray(obs).shape)
            # a linear policy on the normalised observation channels
            err = float(obs[0])
            integ = float(obs[-1])
            return np.array([np.clip(1.2 * err + 0.4 * integ, -1, 1)]), None

    def bare_callable(obs):
        return np.clip(1.2 * float(obs[0]), -1.0, 1.0)

    for label, policy in (("SB3-style .predict", FakeSB3Model()),
                          ("bare callable", bare_callable)):
        env = SpeedHoldEnv(preset="coastal_chop",
                           task=SpeedTaskConfig(target_speed=2.0,
                                                randomize_target=None))
        ctrl = PolicyController(policy, env.obs_builder, name=f"policy({label})")
        res = rollout(env, ctrl, seed=11)
        check(f"PolicyController runs: {label}",
              res.t.size > 100 and np.all(np.isfinite(res.u)),
              f"{res.t.size} steps, rmse={res.metrics['rmse']:.3f}")

    check("policy received the training-shaped observation",
          all(s == (n_expected,) for s in seen_shapes),
          f"expected ({n_expected},), got {set(seen_shapes)}")

    # the observation the policy sees must match what the env feeds SB3
    env = SpeedHoldEnv(preset="coastal_chop")
    obs_from_env, _ = env.reset(seed=2)
    builder = ObservationBuilder(env.sim, env.obs_builder.cfg)
    builder.reset()
    meas = env.sim.measurement()
    obs_from_builder = builder.build(env._error(meas), env.dt, meas)
    check("env obs and standalone builder obs agree",
          np.allclose(obs_from_env, obs_from_builder, atol=1e-6),
          f"max diff {np.max(np.abs(obs_from_env - obs_from_builder)):.2e}")


def test_policy_and_pid_same_harness():
    """Both controller kinds must be scorable and comparable side by side."""
    def make():
        return SpeedHoldEnv(preset="coastal_chop",
                            task=SpeedTaskConfig(target_speed=2.0,
                                                 randomize_target=None))

    class FakePolicy:
        def predict(self, obs, deterministic=True):
            return np.array([np.clip(1.5 * float(obs[0]), -1, 1)]), None

    table = {}
    for ctrl in (PIDSpeedController(PIDGains(260, 50, 30), name="PID"),
                 PolicyController(FakePolicy(), make().obs_builder,
                                  name="fake-policy")):
        table[ctrl.name] = evaluate(make(), ctrl, seeds=range(4))
    txt = comparison_table(table)
    check("PID and policy appear in one comparison table",
          "PID" in txt and "fake-policy" in txt and "rmse" in txt)
    check("both produced finite RMSE",
          all(np.isfinite(v["rmse"]) for v in table.values()))


def test_preview_wiring():
    """preview_points must change obs width and reach the controller."""
    env0 = SpeedHoldEnv(preset="coastal_chop", sim=SimConfig(preview_points=0))
    env8 = SpeedHoldEnv(preset="coastal_chop",
                        sim=SimConfig(preview_points=8, preview_distance=25.0))
    check("preview widens the observation",
          env8.obs_builder.size == env0.obs_builder.size + 8,
          f"{env0.obs_builder.size} -> {env8.obs_builder.size}")

    got = {}

    def spy(obs, dt):
        got["n"] = obs.preview.size
        got["finite"] = bool(np.all(np.isfinite(obs.preview)))
        return 0.3
    rollout(env8, CallableController(spy), seed=1)
    check("controller receives the preview array",
          got.get("n") == 8 and got.get("finite"), str(got))


def test_sensor_noise_and_latency():
    """Noise/latency must perturb the measurement but not the true state."""
    from usv_seakeeper import SensorConfig
    env = SpeedHoldEnv(preset="coastal_chop",
                       sensors=SensorConfig(speed_noise=0.05, latency_steps=3,
                                            seed=1))
    env.reset(seed=0)
    diffs = []
    for _ in range(200):
        env.step(0.5)
        diffs.append(abs(env.sim.measurement().u - env.sim.state.u))
    check("sensor model perturbs the measured speed", max(diffs) > 0.01,
          f"max |measured-true| = {max(diffs):.4f} m/s")
    check("true state stays finite with noisy sensing",
          np.isfinite(env.sim.state.u))


def test_custom_controller_subclass():
    """The documented three-line subclass path works as advertised."""
    class Mine(Controller):
        name = "mine"

        def reset(self):
            self.integral = 0.0

        def act(self, obs, dt):
            self.integral += obs.error * dt
            return 0.4 * obs.error + 0.05 * self.integral

    res = rollout(SpeedHoldEnv(preset="coastal_chop"), Mine(), seed=4)
    check("custom Controller subclass runs and is scored",
          np.isfinite(res.metrics["rmse"]) and res.name == "mine",
          f"rmse={res.metrics['rmse']:.3f}")

    class Unbounded(Controller):
        name = "unbounded"

        def act(self, obs, dt):
            return 1e6      # must be clipped, not explode the sim
    res = rollout(SpeedHoldEnv(preset="coastal_chop"), Unbounded(), seed=4)
    check("out-of-range actions are clipped safely",
          np.all(np.abs(res.action) <= 1.0) and np.all(np.isfinite(res.u)))


def test_station_keeping_actually_converges():
    """End-to-end behavioural check on the second task."""
    env = StationKeepEnv(
        preset="coastal_chop",
        task=StationTaskConfig(target_position=40.0, randomize_target=None,
                               tolerance=1.0))
    res = rollout(env, CascadePositionController(kp_pos=0.35, max_speed=2.5),
                  seed=1)
    check("station keeping reaches and holds the target",
          res.metrics["rmse_steady"] < 1.0 and "arrival_time" in res.metrics,
          f"rmse_steady={res.metrics['rmse_steady']:.3f} m, "
          f"arrival={res.metrics.get('arrival_time', float('nan')):.1f} s")


def test_render_layer():
    """The visual layer must produce real files and hold a 1:1 metre aspect.

    Rendering is where three sign/initialisation bugs surfaced, so it is
    worth a regression test rather than being treated as decoration.
    """
    import os
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    from usv_seakeeper.render import SeaScope, animate_rollout, save_frame
    from usv_seakeeper import WaveConfig, VesselConfig

    tmp = tempfile.mkdtemp()
    env = SpeedHoldEnv(preset="coastal_chop",
                       task=SpeedTaskConfig(target_speed=2.0,
                                            randomize_target=None))
    res = rollout(env, PIDSpeedController(), seed=3)

    check("rollout carries the wave field for replay",
          res.wave_field is not None and res.vessel is not None)
    # the scope redraws the sea from the field; it must match what was logged
    replay = np.array([float(res.wave_field.query(np.array([x]), t).eta[0])
                       for x, t in zip(res.x[:200], res.t[:200])])
    check("replayed surface matches logged elevation exactly",
          float(np.max(np.abs(replay - res.wave_eta[:200]))) == 0.0)

    sc = SeaScope(res.wave_field, res.vessel, thrust_max=res.thrust_max,
                  figsize=(11.0, 9.0))
    check("scope holds 1:1 vertical aspect by default",
          abs(sc._vscale - 1.0) < 0.02, f"vscale={sc._vscale:.4f}")
    check("scope zoom adapts to the sea state",
          2.0 < sc.span < 200.0, f"span={sc.span:.1f} m")

    png = save_frame(res, index=400, save=os.path.join(tmp, "f.png"))
    check("save_frame writes a non-trivial PNG",
          os.path.getsize(png) > 20_000, f"{os.path.getsize(png)//1024} KB")

    gif = animate_rollout(res, save=os.path.join(tmp, "a.gif"), fps=8,
                          decimate=10, max_seconds=4.0)
    check("animate_rollout writes a GIF",
          os.path.getsize(gif) > 10_000, f"{os.path.getsize(gif)//1024} KB")

    # calm vs heavy seas must give different framing, and neither may crash
    spans = []
    for preset in ("calm_harbor", "sea_state_6"):
        e = SpeedHoldEnv(preset=preset)
        e.reset(seed=0)
        spans.append(SeaScope(e.sim.field, e.sim.vessel,
                              thrust_max=e.sim.vessel.thrust_max).span)
    check("calm and heavy seas get different scope windows",
          spans[1] > 2.0 * spans[0], f"{spans[0]:.1f} m vs {spans[1]:.1f} m")


def test_reset_starts_on_the_surface():
    """Heave/pitch must be seeded from the local wave surface.

    Resetting to z = 0 in a developed sea drops the hull from metres in the
    air: a spurious transient and full prop ventilation at the start of every
    episode, which quietly poisons early RL training.
    """
    worst = 0.0
    first_immersion = []
    for seed in range(12):
        env = SpeedHoldEnv(preset="sea_state_6")
        env.reset(seed=seed)
        st = env.sim.state
        worst = max(worst, abs(st.z - st.wave_eta))
        first_immersion.append(st.ventilation)
    check("vessel starts on the water surface", worst < 0.25,
          f"worst |z - eta| at reset = {worst:.3f} m")
    check("no ventilation at episode start", min(first_immersion) > 0.95,
          f"min initial immersion = {min(first_immersion):.3f}")


def test_heading_convention():
    """heading=+1 must be a head sea (encounter frequency raised)."""
    from usv_seakeeper import WaveConfig, WaveField
    T, U = 6.0, 3.0
    w = 2 * np.pi / T
    k = w * w / 9.81

    def enc(h):
        f = WaveField(WaveConfig(kind="regular", amplitude=1.0, period=T,
                                 heading=h))
        t = np.linspace(0.0, 300.0, 15000)
        eta = np.array([float(f.query(np.array([U * ti]), ti).eta[0])
                        for ti in t])
        sp = np.abs(np.fft.rfft(eta - eta.mean()))
        fr = np.fft.rfftfreq(t.size, t[1] - t[0])
        return float(2 * np.pi * fr[np.argmax(sp)])

    check("heading=+1 is a head sea (w_enc > w)", enc(1) > w,
          f"w_enc={enc(1):.3f} vs w={w:.3f}")
    check("heading=-1 is a following sea (w_enc < w)", enc(-1) < w,
          f"w_enc={enc(-1):.3f} vs w={w:.3f}")


def test_config_guard():
    """A non-integer dt/dt_physics ratio must fail loudly."""
    try:
        SimConfig(dt=0.05, dt_physics=0.003)
    except ValueError:
        check("non-integer dt ratio raises", True)
    else:
        check("non-integer dt ratio raises", False, "no exception")


def test_csv_export():
    import csv
    import tempfile
    import os
    res = rollout(SpeedHoldEnv(preset="coastal_chop"),
                  PIDSpeedController(), seed=0)
    path = os.path.join(tempfile.mkdtemp(), "t.csv")
    res.to_csv(path)
    with open(path) as fh:
        rows = list(csv.reader(fh))
    check("CSV export has header + one row per step",
          len(rows) == res.t.size + 1 and rows[0][0] == "t",
          f"{len(rows)} rows for {res.t.size} steps")


def main() -> bool:
    tests = [
        test_gym_contract,
        test_episode_terminates,
        test_obs_finite_under_extremes,
        test_determinism_and_seeding,
        test_randomization_does_not_compound,
        test_policy_controller_bridge,
        test_policy_and_pid_same_harness,
        test_preview_wiring,
        test_sensor_noise_and_latency,
        test_custom_controller_subclass,
        test_station_keeping_actually_converges,
        test_render_layer,
        test_reset_starts_on_the_surface,
        test_heading_convention,
        test_config_guard,
        test_csv_export,
    ]
    for t in tests:
        print(f"\n--- {t.__name__} ---")
        t()
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return False
    print("all API/integration checks passed")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
