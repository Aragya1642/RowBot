# USV Sea-Keeper

A headless Python port of the browser-based 1-D wave/boat control bench, rebuilt
around a **pluggable controller interface**, **Gymnasium environments**, and an
**animated scope** so the same simulator serves classical control, RL training,
and visual inspection.

The design constraint throughout: a hand-written PID and a trained policy must
run against *literally the same* physics, actuator limits, and scoring. That is
what makes any comparison you produce meaningful.

```
usv_seakeeper/
  config.py         dataclasses + scenario presets
  waves.py          regular Airy + JONSWAP, single-pass vectorised query
  sim.py            USVSim -- framework-free, step(action) -> state
  observations.py   ObservationBuilder shared by envs and wrapped policies
  controllers.py    Controller ABC + PID / cascade / preview / policy wrapper
  envs.py           SpeedHoldEnv, StationKeepEnv (+ shim if no gymnasium)
  rollout.py        rollout / evaluate / compare, metrics, CSV export
  render.py         animated scope, video export, interactive live viewer
  plotting.py       static matplotlib telemetry
  validate.py       37 analytic physics checks
examples/
  01_quickstart.py               baselines, plots, CSV
  02_custom_controller.py        three ways to plug in your own
  03_train_rl.py                 SAC training + benchmark vs PID
  04_actuator_sizing_sweep.py    Hs x Fmax sizing + Hs x Tp ventilation charts
  05_visualise.py                stills, videos, live viewer
tests/
  test_api.py       34 API / integration / render checks
```

## Install

```bash
pip install -r requirements.txt          # numpy, matplotlib
pip install gymnasium                    # for the real Env API
pip install "stable-baselines3[extra]"   # only for examples/03
pip install -e .                         # add --no-build-isolation if offline
```

Everything except `examples/03` runs with numpy + matplotlib alone. `envs.py`
falls back to a minimal API-compatible shim when `gymnasium` is absent. Video
export needs `ffmpeg` for MP4; GIF works through pillow with no external binary.

## Verify before you trust it

```bash
python -m usv_seakeeper.validate     # 37 analytic physics checks
python tests/test_api.py             # 34 API / integration / render checks
```

The two suites cover different failure modes. `validate.py` checks the
**physics** against closed-form limits (dispersion for both headings, encounter
frequency, `4*sqrt(m0) = Hs`, `v = sqrt(F/Cd)`, Stokes drift, heave RAO,
free-fall of an emerged hull, RK4 order). `tests/test_api.py` checks the
**plumbing** -- the Gymnasium contract, seeding determinism, the render layer,
and in particular the `PolicyController` bridge, which is the part most likely
to break silently: a shape or normalisation mismatch between training and
evaluation gives you a policy that trains fine and is garbage at deploy, with no
error anywhere. It is tested against a stand-in policy rather than taken on
faith.

Every physics bug found while building this is covered by a check that is still
in the suite. Several were found by *looking at the animation*, which is the
main argument for having one.

## Quick start

```python
from usv_seakeeper import SpeedHoldEnv, PIDSpeedController, PIDGains, rollout

env = SpeedHoldEnv(preset="coastal_chop")
res = rollout(env, PIDSpeedController(PIDGains(260, 50, 30),
                                      slope_feedforward=True), seed=7)
print(res.summary())
res.to_csv("telemetry.csv")
```

## Watching it

```python
from usv_seakeeper import animate_rollout, animate_comparison, save_frame

animate_rollout(res, "scope.mp4", fps=20, max_seconds=30)   # or .gif
save_frame(res, index=600, save="frame.png")
animate_comparison([res_pid, res_rl], "compare.mp4")        # same seed only
```

The scope draws the true wave surface from the `WaveField` attached to the
result, the hull rotated by actual pitch, the thrust vector from the propeller,
and the prop marker coloured by immersion, with strips for speed/setpoint,
commanded vs delivered thrust, wave elevation vs hull heave, and prop immersion.

Horizontal and vertical metres map to the same number of pixels (the scope says
so in its caption) and the zoom follows the sea state, because a 5 m swell and a
0.2 m ripple need very different windows. Note the consequence: a Tp = 10.5 s sea
has a ~170 m wavelength, so a 27 m window legitimately shows only part of one
wave. That is what the wave/heave strip is for -- the local view cannot convey
the wave the vessel is riding.

Rendering costs about 0.15 s per frame, so a 30 s clip at 20 fps takes ~90 s.

For interactive tuning, closest to the original browser tool:

```python
from usv_seakeeper.render import LiveViewer
LiveViewer(preset="coastal_chop").show()    # sliders for Hs, Tp, Fmax, PID gains
```

Needs an interactive matplotlib backend; it raises a clear error under Agg.

## Plugging in a controller

One method. Return a normalised thrust in `[-1, 1]`; it is clipped for you.

```python
from usv_seakeeper import Controller

class MyController(Controller):
    name = "mine"

    def reset(self):                 # called at episode start
        self.integral = 0.0

    def act(self, obs, dt) -> float:
        self.integral += obs.error * dt
        return 0.4 * obs.error + 0.05 * self.integral
```

`obs` is a `ControlObs` with named **physical** quantities -- deliberately not
the flat RL vector, because classical controllers benefit from dimensional
access and it makes them an honest upper baseline:

| field | meaning |
|---|---|
| `error` | target - measured (m/s for speed, m for position) |
| `x, u` | surge position [m], velocity [m/s] |
| `z, w` | heave [m], heave rate [m/s] (positive up, from calm waterline) |
| `theta, q` | pitch [rad], pitch rate [rad/s] (positive bow-up) |
| `thrust` | current actuator command [N] -- a real state, it is rate limited |
| `ventilation` | prop immersion, 1 = fully wetted, 0 = fully emerged |
| `wave_slope` | local d(eta)/dx |
| `preview` | look-ahead wave slopes (empty unless `preview_points > 0`) |
| `mass, thrust_max, drag_coeff` | plant parameters, for model-based control |

Two other entry styles:

```python
from usv_seakeeper import CallableController, PolicyController

CallableController(lambda obs, dt: 0.5 * obs.error, name="P")
PolicyController(sb3_model, env.obs_builder, name="SAC")   # trained policy
```

`PolicyController` is the bridge: it rebuilds the flat training observation from
the same `ObservationBuilder`, so a policy can be scored by the same harness and
plotted on the same axes as the PID baselines.

## RL

```bash
python examples/03_train_rl.py --task speed   --steps 300000
python examples/03_train_rl.py --task station --steps 500000
python examples/03_train_rl.py --task speed --eval-only --model sac_speed
```

Two observation choices that decide whether training works at all:

1. **The actuator state is in the observation.** The thruster is rate limited,
   so commanded thrust is a genuine state variable. Omit it and the MDP is not
   Markov; the policy chatters.
2. **A running error integral is available** (`include_integral`). With
   unmodelled drag, a memoryless policy on (error, velocity) *cannot* drive
   steady-state error to zero -- same reason a P controller cannot. Either keep
   the integral channel or use a recurrent policy.

Domain randomisation is on by default in the training factory: wave phases are
reseeded every episode and Hs/Tp/mass/thrust/drag are drawn from ranges. A
policy trained on one fixed realisation scores beautifully and transfers to
nothing, because the sea is then a deterministic function of time it can
memorise. Training and evaluation seed sets are disjoint.

Throughput: ~810 env steps/s at `dt_physics=0.005`, ~1535 at `0.01` (validated
as converged). With `SubprocVecEnv` at 8 workers, 1M steps takes a few minutes.

## The model

**Waves.** Linear superposition, deep-water dispersion `w^2 = g|k|`. Regular
mode is a single Airy component; irregular is a JONSWAP spectrum (gamma = 3.3,
64 components) rescaled so `4*sqrt(m0) = Hs`.

Frequencies are jittered within each bin. Uniform spacing makes the surface
exactly periodic (~250 s at typical settings) -- fine for a browser demo, fatal
for RL. The band runs to `6*wp`, not `3*wp`: slope variance scales as
`w^4 S(w)`, so a low cutoff badly under-predicts the actual disturbance.

Orbital velocities are evaluated at the hull's height with `exp(|k|z)`, clamped
never to exceed the free surface. Not cosmetic: evaluating at the mean surface
drops the `du/dz * zeta` term, which is exactly half the Stokes drift, biasing
every station-keeping result.

**Sign conventions**, both of which had bugs worth naming:

* `heading = +1` is a **head sea** (waves toward the vessel, encounter frequency
  raised to `w + |k|U`); `-1` is a **following sea**. The phase is `kx - wt` and
  the vessel steams toward `+x`, so a head sea needs `k < 0`.
* The **horizontal** orbital velocity carries `sign(k)` and the vertical does
  not. From `phi = (A w/|k|) e^{|k|z} sin(kx - wt)`, `u = dphi/dx` picks up
  `sign(k)` while `w = dphi/dz` stays equal to `d(eta)/dt`. Sharing one `A*w`
  array between them silently inverts the heave forcing in a head sea.

Both are pinned by validation checks that run for each heading.

**Vessel.** Two modes via `SimConfig.dof`:

* `"surge"` -- 1 DOF, hull pinned to the surface (the original JS model). The
  horizontal constraint force is `m*g*sin(th)*cos(th)`, which for linear
  deep-water waves *equals* the Froude-Krylov force, since
  `-g*d(eta)/dx = du/dt` when `w^2 = g|k|`. The browser tool's "gravity along
  the slope" and its "Morison inertia" toggle were the same physics twice.
* `"surge_heave_pitch"` -- 3 DOF (default). Nonlinear surge; heave and pitch as
  linear seakeeping equations with added mass, radiation damping, and
  hydrostatic restoring taken *relative to the local wave surface*. That gives
  RAO -> 1 in long waves and, with sinc length averaging, RAO -> 0 for waves
  shorter than the hull, without tabulated diffraction coefficients.

**Wave force is averaged over the waterline length** (`sinc(kL/2)`, the exact
integral of a sinusoid over the hull). A point-force model over-predicts
short-wave excitation badly once `L/lambda >~ 0.2`.

**Hull emergence.** Every hydrodynamic term -- drag, wave excitation, radiation
damping, added mass -- is scaled by a smoothstep **wetted fraction**, and the
hydrostatic restoring saturates once the submerged volume reaches zero (one
draft above the local surface). The linear restoring `-C33*(z - eta)` is exact
while immersed and reaches several times the vessel weight on an airborne hull;
ungated, a steep sea launches the vessel 16 m into the air. With the gating, an
emerged hull free-falls at exactly `-g` and a hull at rest on calm water is in
exact equilibrium -- both asserted. The smoothstep rather than a linear clip
matters twice over: added mass genuinely varies continuously with submergence,
and a hard kink at full immersion is crossed on every wave and knocks RK4 down
from 4th order.

**Actuator.** Symmetric `+/-Fmax`, exact closed-form rate-limited zero-order
hold (so the ramp is independent of `dt_physics`), and **ventilation**: the prop
sits at `prop_x` aft and `prop_depth` down, and thrust collapses via a
smoothstep on disc immersion as the stern lifts clear.

**Integration.** RK4 on the full state vector, 200 Hz sub-steps under a 20 Hz
control interval, zero-order hold on the action. Reset seeds heave and pitch
from the local wave surface, not `z = 0`, which otherwise drops the hull from
metres in the air in a developed sea.

## Metrics

`rmse` covers the whole episode and for station keeping is dominated by the
transit from the start point -- 14.2 m in a case where the vessel actually holds
to 0.035 m. Compare controllers on **`rmse_steady`** (second half of the
episode). Also reported: `max_abs_error`, `settling_time`, `in_tolerance_frac`,
`saturated_frac`, `min_ventilation`, `control_effort`, `action_rate_rms`.

Single episodes in an irregular sea are close to meaningless -- variance between
phase realisations is large. Use `evaluate(env, ctrl, seeds=range(N))`, and the
same seed set for every controller you compare.

## Results worth knowing before you train

All numbers below are 12 wave realisations of `coastal_chop`, 2.0 m/s setpoint:

```
controller        rmse   rmse_steady  in_tol   saturated  action_rate_rms
PID              0.334      0.224      45%        0%          0.028
PID+FF           0.269      0.085      82%        7%          0.132
bang-bang        0.309      0.238      46%      100%          0.561
sliding-mode     0.216      0.066      92%       10%          0.108
```

**A pure tracking reward will train a bang-bang policy.** Bang-bang beats plain
PID on RMSE (0.309 vs 0.334) while saturated 100% of the time at 20x the
action-rate RMS. The slew limit makes that partly work in sim and it is useless
on hardware, so `w_rate` is nonzero by default and the bang-bang baseline is
kept in `examples/02` as a cautionary case. A well-tuned PID+FF does beat it, so
the trap is a badly-shaped reward rather than an unbeatable strategy.

**Sliding mode is the baseline to beat, not PID** -- RMSE 0.216 vs 0.269, and
in-tolerance 92% vs 82%. It is also *smoother* than PID+FF (action rate 0.108 vs
0.132), so it is not winning by spending more actuator. That is the bar an RL
policy has to clear for the comparison to mean anything.

**Saturation is not the binding constraint in heavy seas; ventilation is.** At
Hs = 3.5 m with 2000 N installed, the thruster saturates only 4% of the time yet
speed RMSE is 1.115 m/s. Sizing sweep (`examples/04`, Tp = 6.5 s):

```
RMSE [m/s]      Fmax:  300     600    1100    2000
Hs = 0.5 m            0.379   0.295   0.283   0.283    <- controller-limited
Hs = 1.5 m            0.580   0.382   0.316   0.315
Hs = 2.5 m            1.151   0.699   0.487   0.458
Hs = 3.5 m            3.374   2.815   1.348   1.115    <- ventilation floor
```

**Ventilation tracks steepness, not wave height.** At Hs = 2 m the worst prop
immersion is 1.000 at Tp = 11 s and 0.000 at Tp = 3 s. In long swell the hull
heaves with the surface (RAO -> 1), so there is no relative motion to unwet the
propeller, however large the waves. The design axis is period, not height --
`examples/04` charts this separately.

## Known limitations

* **No surf-riding, and none is possible.** Wave capture needs the vessel to
  approach the celerity `c = g*Tp/(2*pi)`; every preset sits at `v/c = 0.29` to
  `0.39`, so celerity exceeds attainable speed 3-4x. If you need surf-riding you
  need much shorter periods or a much faster vessel.
* **Head vs following seas barely differ here.** At identical Hs = 3.5 m,
  Tp = 9 s: RMSE 0.394 +- 0.093 (head) vs 0.403 +- 0.065 (following) --
  indistinguishable. The disturbance is wave slope, whose RMS is
  heading-independent; only the encounter frequency shifts. The real asymmetry
  in practice comes from broaching and surf-riding, and this model has neither.
  Do not design a heading study around this simulator.
* **No yaw, sway or roll.** Single-axis only, so no broaching, no course
  keeping, no beam seas. Following seas are a sign flip on `k`, not an
  overtaking-wave model.
* **Linear waves.** No breaking, slamming or green water. `USVSim` emits a
  `RuntimeWarning` above steepness 0.30 (breaking is ~0.44); motions stay
  bounded and finite past that, but are not quantitatively trustworthy. All
  shipped presets sit at 0.067-0.101.
* **Submergence past the freeboard is a swamped hull**, outside a linear
  seakeeping model. The model caps buoyancy at the freeboard-limited volume and
  stays bounded, but that regime needs a nonlinear method.
* **Preview control is currently worse than no preview** (RMSE 0.336 vs 0.269,
  in-tolerance 44% vs 82%, over 12 seeds). The
  built-in `PreviewFeedforwardController` averages previewed slope over a fixed
  fraction of the scan and feeds forward `m*g*sin(th)` from a slope the hull has
  not reached, with no time-to-arrival term. It needs to be a proper
  finite-horizon problem: advect the previewed slope by boat speed to get
  arrival times, then solve over the actuator rise time. Treat it as an open
  item, not a working baseline.
* **`slope_feedforward` is exact only in `dof="surge"`.** In 3 DOF the surge
  wave load is the Morison inertia term and gravity acts through the actual
  pitch, which lags the wave slope.
* Radiation damping is a constant damping ratio, not frequency dependent.
* RK4 crossing the actuator ramp's derivative kink gives a one-off ~0.2 mm
  position error over 10 s. Quantified in `validate.py`, not worth fixing.
