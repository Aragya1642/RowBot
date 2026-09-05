"""The pluggable controller interface.

Subclass :class:`Controller`, implement :meth:`Controller.act`, and it can be
run by :func:`usv_seakeeper.rollout.rollout` against exactly the same physics
your RL agent trains on::

    class MyController(Controller):
        def act(self, obs, dt):
            return 0.3 * obs.error          # must return a value in [-1, 1]

Included baselines:

* :class:`ZeroController`, :class:`ConstantController` -- trivial references.
* :class:`PIDSpeedController` -- speed holding, with optional wave-slope
  feedforward and ventilation compensation.
* :class:`CascadePositionController` -- station keeping as an outer position
  loop generating a speed setpoint for the inner speed loop. More robust than
  a single position PID, because the speed limit is enforced structurally
  rather than by tuning.
* :class:`PreviewFeedforwardController` -- uses the wave-radar preview.
* :class:`PolicyController` -- wraps a trained RL policy so it can be scored
  by the same harness and plotted on the same axes as the PID baselines.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Any

import numpy as np

from .config import G
from .sim import USVSim, USVState


# --------------------------------------------------------------------------
# what a controller sees
# --------------------------------------------------------------------------
@dataclass
class ControlObs:
    """Everything a classical controller might want, in physical units.

    This is deliberately *not* the flat RL observation vector -- classical
    controllers benefit from named, dimensional quantities, and giving them
    full state access makes them an honest upper baseline.
    """
    t: float
    error: float                 # target - measured, in task units
    x: float
    u: float
    z: float
    w: float
    theta: float
    q: float
    thrust: float                # current actuator command [N]
    ventilation: float
    wave_slope: float
    preview: np.ndarray          # look-ahead wave slopes (may be empty)
    # task references (whichever applies)
    target_speed: Optional[float] = None
    target_position: Optional[float] = None
    # plant parameters, for model-based controllers
    mass: float = 0.0
    thrust_max: float = 1.0
    drag_coeff: float = 0.0

    @classmethod
    def from_sim(cls, sim: USVSim, error: float,
                 target_speed: Optional[float] = None,
                 target_position: Optional[float] = None,
                 state: Optional[USVState] = None) -> "ControlObs":
        st = state if state is not None else sim.measurement()
        return cls(
            t=st.t, error=error, x=st.x, u=st.u, z=st.z, w=st.w,
            theta=st.theta, q=st.q, thrust=st.thrust_cmd,
            ventilation=st.ventilation, wave_slope=st.wave_slope,
            preview=sim.preview(), target_speed=target_speed,
            target_position=target_position, mass=sim.vessel.mass,
            thrust_max=sim.vessel.thrust_max,
            drag_coeff=sim.vessel.drag_coeff,
        )


# --------------------------------------------------------------------------
# base class
# --------------------------------------------------------------------------
class Controller(ABC):
    """Base class. ``act`` must return a normalised thrust in ``[-1, 1]``."""

    name: str = "controller"

    def reset(self) -> None:
        """Clear internal state between episodes."""

    @abstractmethod
    def act(self, obs: ControlObs, dt: float) -> float:
        ...

    def __call__(self, obs: ControlObs, dt: float) -> float:
        return float(np.clip(self.act(obs, dt), -1.0, 1.0))


class ZeroController(Controller):
    name = "zero"

    def act(self, obs: ControlObs, dt: float) -> float:
        return 0.0


class ConstantController(Controller):
    def __init__(self, value: float = 0.5):
        self.value = value
        self.name = f"constant({value:+.2f})"

    def act(self, obs: ControlObs, dt: float) -> float:
        return self.value


class CallableController(Controller):
    """Adapt a bare function ``f(obs, dt) -> action`` into a Controller."""

    def __init__(self, fn: Callable[[ControlObs, float], float], name: str = "callable"):
        self.fn = fn
        self.name = name

    def act(self, obs: ControlObs, dt: float) -> float:
        return self.fn(obs, dt)


# --------------------------------------------------------------------------
# PID
# --------------------------------------------------------------------------
@dataclass
class PIDGains:
    kp: float = 200.0
    ki: float = 40.0
    kd: float = 20.0


class PIDSpeedController(Controller):
    """Speed-hold PID in newtons, with conditional-integration anti-windup.

    Options
    -------
    slope_feedforward
        Injects ``m*g*sin(th)*cos(th)`` to pre-cancel the wave-induced surge
        load. In the 1-DOF surface-following model this is exactly the
        disturbance, so it is close to a perfect cancellation; in 3-DOF it is
        an approximation because the hull pitch lags the wave slope.
    ventilation_compensation
        Divides the command by the measured thrust-loss factor (floored) to
        recover authority when the prop is partly emerged. Realistic only if
        you actually have a submergence estimate -- a useful ablation.
    derivative_tau
        First-order low-pass on the D term. Leave at 0 for noise-free
        sensing; you will need it (0.05-0.2 s) as soon as speed noise is on.
    """

    def __init__(self, gains: Optional[PIDGains] = None,
                 slope_feedforward: bool = False,
                 ventilation_compensation: bool = False,
                 derivative_tau: float = 0.0,
                 name: Optional[str] = None):
        self.g = gains or PIDGains()
        self.slope_ff = slope_feedforward
        self.vent_comp = ventilation_compensation
        self.derivative_tau = derivative_tau
        self.name = name or ("PID+FF" if slope_feedforward else "PID")
        self.reset()

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error: Optional[float] = None
        self._deriv = 0.0

    def act(self, obs: ControlObs, dt: float) -> float:
        e = obs.error
        raw_integral = self._integral + e * dt

        if self._prev_error is None:
            d_raw = 0.0
        else:
            d_raw = (e - self._prev_error) / max(dt, 1e-9)
        self._prev_error = e
        if self.derivative_tau > 0.0:
            alpha = dt / (self.derivative_tau + dt)
            self._deriv += alpha * (d_raw - self._deriv)
        else:
            self._deriv = d_raw

        f = self.g.kp * e + self.g.ki * raw_integral + self.g.kd * self._deriv

        if self.slope_ff:
            th = np.arctan(obs.wave_slope)
            f += obs.mass * G * np.sin(th) * np.cos(th)

        if self.vent_comp:
            f /= max(obs.ventilation, 0.25)

        f_sat = float(np.clip(f, -obs.thrust_max, obs.thrust_max))
        # conditional integration: only accumulate when not driving further
        # into saturation in the direction of the error
        pushing_into_sat = (f != f_sat) and (np.sign(e) == np.sign(f))
        if not pushing_into_sat:
            self._integral = raw_integral

        return f_sat / obs.thrust_max


class CascadePositionController(Controller):
    """Station keeping: outer P(+D) position loop -> inner speed PID."""

    def __init__(self, kp_pos: float = 0.35, kd_pos: float = 0.0,
                 max_speed: float = 3.0,
                 inner: Optional[PIDSpeedController] = None,
                 name: str = "cascade-PID"):
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.max_speed = max_speed
        self.inner = inner or PIDSpeedController(
            PIDGains(kp=300.0, ki=60.0, kd=25.0), slope_feedforward=True)
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.inner.reset()
        self._v_ref = 0.0

    def act(self, obs: ControlObs, dt: float) -> float:
        v_ref = self.kp_pos * obs.error - self.kd_pos * obs.u
        v_ref = float(np.clip(v_ref, -self.max_speed, self.max_speed))
        self._v_ref = v_ref
        inner_obs = ControlObs(**{**obs.__dict__, "error": v_ref - obs.u,
                                  "target_speed": v_ref})
        return self.inner.act(inner_obs, dt)


class PreviewFeedforwardController(Controller):
    """Speed PID plus a feedforward built from the wave-radar preview.

    Averages the previewed slope over a horizon roughly matching the actuator
    rise time, so the thruster starts building force *before* the crest load
    arrives instead of chasing it. Requires ``SimConfig.preview_points > 0``.
    """

    def __init__(self, gains: Optional[PIDGains] = None,
                 preview_gain: float = 1.0,
                 horizon_fraction: float = 0.4,
                 name: str = "PID+preview"):
        self.pid = PIDSpeedController(gains or PIDGains())
        self.preview_gain = preview_gain
        self.horizon_fraction = horizon_fraction
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.pid.reset()

    def act(self, obs: ControlObs, dt: float) -> float:
        base = self.pid.act(obs, dt)
        if obs.preview.size == 0:
            return base
        n = max(1, int(round(self.horizon_fraction * obs.preview.size)))
        slope_ahead = float(np.mean(obs.preview[:n]))
        th = np.arctan(slope_ahead)
        ff = obs.mass * G * np.sin(th) * np.cos(th)
        return base + self.preview_gain * ff / obs.thrust_max


# --------------------------------------------------------------------------
# RL policy adapter
# --------------------------------------------------------------------------
class PolicyController(Controller):
    """Wrap a trained policy so the same harness can score it.

    ``policy`` may be:
      * a Stable-Baselines3 model (has ``.predict``), or
      * any callable mapping an observation array to an action.

    ``obs_builder`` must be the *same* :class:`ObservationBuilder`
    configuration used during training -- pass ``env.obs_builder`` from the
    env you trained on, or construct one with an identical ``ObsConfig``.
    """

    def __init__(self, policy: Any, obs_builder, deterministic: bool = True,
                 name: str = "RL policy"):
        self.policy = policy
        self.obs_builder = obs_builder
        self.deterministic = deterministic
        self.name = name

    def reset(self) -> None:
        self.obs_builder.reset()

    def _predict(self, obs: np.ndarray) -> float:
        if hasattr(self.policy, "predict"):
            action, _ = self.policy.predict(obs, deterministic=self.deterministic)
            return float(np.asarray(action).reshape(-1)[0])
        out = self.policy(obs)
        return float(np.asarray(out).reshape(-1)[0])

    def act(self, obs: ControlObs, dt: float) -> float:
        vec = self.obs_builder.build(obs.error, dt)
        return self._predict(vec)


__all__ = [
    "Controller", "ControlObs", "PIDGains", "ZeroController",
    "ConstantController", "CallableController", "PIDSpeedController",
    "CascadePositionController", "PreviewFeedforwardController",
    "PolicyController",
]
