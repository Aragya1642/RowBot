"""Gymnasium environments.

Two tasks, sharing one base class and one simulator:

* :class:`SpeedHoldEnv`   -- track a surge speed setpoint through waves.
* :class:`StationKeepEnv` -- reach a point and hold it.

Both expose ``action = [normalised thrust]`` in ``[-1, 1]`` and a flat
observation built by :class:`~usv_seakeeper.observations.ObservationBuilder`.

If ``gymnasium`` is not installed, a minimal API-compatible shim is used so the
module still imports and the environments still run (handy for CI and for
testing the physics); install gymnasium to use the real thing with SB3.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import (WaveConfig, VesselConfig, SensorConfig, SimConfig,
                     SpeedTaskConfig, StationTaskConfig, RewardConfig,
                     RandomizeConfig, get_preset)
from .sim import USVSim, USVState
from .observations import ObsConfig, ObservationBuilder
from .controllers import ControlObs

# --------------------------------------------------------------------------
# gymnasium (with fallback shim)
# --------------------------------------------------------------------------
try:                                                    # pragma: no cover
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:                                     # pragma: no cover
    GYMNASIUM_AVAILABLE = False

    class _Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.dtype = dtype
            if shape is None:
                shape = np.shape(low) if np.ndim(low) else (1,)
            self.shape = tuple(shape)
            self.low = np.broadcast_to(np.asarray(low, dtype), self.shape).copy()
            self.high = np.broadcast_to(np.asarray(high, dtype), self.shape).copy()

        def sample(self):
            return np.random.uniform(self.low, self.high).astype(self.dtype)

        def contains(self, x):
            x = np.asarray(x)
            return (x.shape == self.shape and np.all(x >= self.low)
                    and np.all(x <= self.high))

        def __repr__(self):
            return f"Box({self.low.min()}, {self.high.max()}, {self.shape})"

    class _Spaces:
        Box = _Box

    spaces = _Spaces()

    class _Env:
        metadata: Dict[str, Any] = {}
        observation_space = None
        action_space = None

        def reset(self, *, seed=None, options=None):
            raise NotImplementedError

        def step(self, action):
            raise NotImplementedError

        def close(self):
            pass

    class _GymModule:
        Env = _Env

    gym = _GymModule()


# --------------------------------------------------------------------------
# base
# --------------------------------------------------------------------------
class _USVEnvBase(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 wave: Optional[WaveConfig] = None,
                 vessel: Optional[VesselConfig] = None,
                 sim: Optional[SimConfig] = None,
                 sensors: Optional[SensorConfig] = None,
                 randomize: Optional[RandomizeConfig] = None,
                 reward: Optional[RewardConfig] = None,
                 obs: Optional[ObsConfig] = None,
                 preset: Optional[str] = None):
        if preset is not None:
            p_wave, p_vessel, p_sim = get_preset(preset)
            wave = wave or p_wave
            vessel = vessel or p_vessel
            sim = sim or p_sim

        self.sim = USVSim(wave=wave, vessel=vessel, sim=sim,
                          sensors=sensors, randomize=randomize)
        self.reward_cfg = reward or RewardConfig()
        self.obs_builder = ObservationBuilder(self.sim, obs)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.obs_builder.size,), dtype=np.float32)

        self._prev_action = 0.0
        self._step_count = 0
        self._episode_return = 0.0
        self._err_sq_sum = 0.0
        self._in_tol_steps = 0
        self._sat_steps = 0
        self._vent_sum = 0.0

    # -- to be provided by subclasses ---------------------------------
    def _sample_task(self) -> None:
        raise NotImplementedError

    def _error(self, state: USVState) -> float:
        raise NotImplementedError

    def _tolerance(self) -> float:
        raise NotImplementedError

    def _err_scale(self) -> float:
        raise NotImplementedError

    def _extra_penalty(self, state: USVState) -> float:
        return 0.0

    def _task_terminated(self, state: USVState) -> bool:
        return False

    # -- gym API ------------------------------------------------------
    @property
    def dt(self) -> float:
        return self.sim.cfg.dt

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None):
        self.sim.reset(seed=seed)
        self._sample_task()
        self.obs_builder.reset()
        self._prev_action = 0.0
        self._step_count = 0
        self._episode_return = 0.0
        self._err_sq_sum = 0.0
        self._in_tol_steps = 0
        self._sat_steps = 0
        self._vent_sum = 0.0

        meas = self.sim.measurement()
        obs = self.obs_builder.build(self._error(meas), self.dt, meas)
        return obs, self._info(meas, 0.0)

    def step(self, action):
        a = float(np.asarray(action).reshape(-1)[0])
        a = float(np.clip(a, -1.0, 1.0))
        state = self.sim.step(a)
        self._step_count += 1

        meas = self.sim.measurement()
        err = self._error(meas)
        true_err = self._error(state)

        r = self._reward(a, true_err, state)
        self._episode_return += r
        self._err_sq_sum += true_err ** 2
        if abs(true_err) < self._tolerance():
            self._in_tol_steps += 1
        if self.sim.saturation() > 0.99:
            self._sat_steps += 1
        self._vent_sum += state.ventilation

        crashed = self.sim.is_unsafe()
        if crashed:
            r -= self.reward_cfg.crash_penalty
            self._episode_return -= self.reward_cfg.crash_penalty

        terminated = bool(crashed or self._task_terminated(state))
        truncated = bool(state.t >= self.sim.cfg.max_time - 1e-9)

        obs = self.obs_builder.build(err, self.dt, meas)
        self._prev_action = a
        return obs, float(r), terminated, truncated, self._info(state, true_err)

    # -- reward -------------------------------------------------------
    def _reward(self, action: float, err: float, state: USVState) -> float:
        c = self.reward_cfg
        e_n = err / self._err_scale()
        if c.gaussian_error:
            r = c.w_error * float(np.exp(-e_n * e_n))
        else:
            r = -c.w_error * e_n * e_n
        r -= c.w_effort * action * action
        r -= c.w_rate * (action - self._prev_action) ** 2
        r += self._extra_penalty(state)
        if abs(err) < self._tolerance():
            r += c.bonus_in_tolerance
        return r

    # -- info ---------------------------------------------------------
    def _info(self, state: USVState, err: float) -> Dict[str, Any]:
        n = max(self._step_count, 1)
        return {
            "t": state.t,
            "x": state.x,
            "speed": state.u,
            "error": err,
            "thrust": state.thrust_cmd,
            "thrust_delivered": state.thrust_delivered,
            "ventilation": state.ventilation,
            "saturation": self.sim.saturation(),
            "rmse": float(np.sqrt(self._err_sq_sum / n)),
            "in_tolerance_frac": self._in_tol_steps / n,
            "saturated_frac": self._sat_steps / n,
            "mean_ventilation": self._vent_sum / n,
            "episode_return": self._episode_return,
        }

    # -- bridge to the classical-controller harness -------------------
    def control_obs(self, state: Optional[USVState] = None) -> ControlObs:
        st = state if state is not None else self.sim.measurement()
        return ControlObs.from_sim(self.sim, self._error(st), state=st)

    def describe(self) -> str:
        info = self.sim.info()
        lines = [f"{type(self).__name__}",
                 f"  sea            : {info['sea']}",
                 f"  slope RMS      : {info['slope_rms']:.3f}",
                 f"  recurrence     : {info['recurrence_s']:.0f} s",
                 f"  heave/pitch Tn : {info['heave_Tn']:.2f} / {info['pitch_Tn']:.2f} s",
                 f"  v_max (calm)   : {info['v_max_calm']:.2f} m/s",
                 f"  obs ({self.obs_builder.size}) : {', '.join(self.obs_builder.names)}"]
        if info["breaking"]:
            lines.append("  WARNING: requested sea state exceeds breaking steepness")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# speed holding
# --------------------------------------------------------------------------
class SpeedHoldEnv(_USVEnvBase):
    """Hold a commanded surge speed through waves."""

    def __init__(self, task: Optional[SpeedTaskConfig] = None, **kwargs):
        self.task = task or SpeedTaskConfig()
        super().__init__(**kwargs)
        self.target_speed = self.task.target_speed

    def _sample_task(self) -> None:
        if self.task.randomize_target is not None:
            lo, hi = self.task.randomize_target
            self.target_speed = float(self.sim._rng.uniform(lo, hi))
        else:
            self.target_speed = self.task.target_speed

    def _error(self, state: USVState) -> float:
        return self.target_speed - state.u

    def _tolerance(self) -> float:
        return self.task.tolerance

    def _err_scale(self) -> float:
        return self.task.err_scale

    def control_obs(self, state: Optional[USVState] = None) -> ControlObs:
        st = state if state is not None else self.sim.measurement()
        return ControlObs.from_sim(self.sim, self._error(st),
                                   target_speed=self.target_speed, state=st)


# --------------------------------------------------------------------------
# station keeping
# --------------------------------------------------------------------------
class StationKeepEnv(_USVEnvBase):
    """Transit to a point and hold station against wave drift."""

    def __init__(self, task: Optional[StationTaskConfig] = None,
                 position_bound: float = 150.0, **kwargs):
        self.task = task or StationTaskConfig()
        self.position_bound = position_bound
        super().__init__(**kwargs)
        self.target_position = self.task.target_position
        self._arrival_time: Optional[float] = None

    def _sample_task(self) -> None:
        if self.task.randomize_target is not None:
            lo, hi = self.task.randomize_target
            self.target_position = float(self.sim._rng.uniform(lo, hi))
        else:
            self.target_position = self.task.target_position
        self._arrival_time = None

    def _error(self, state: USVState) -> float:
        return self.target_position - state.x

    def _tolerance(self) -> float:
        return self.task.tolerance

    def _err_scale(self) -> float:
        return self.task.err_scale

    def _extra_penalty(self, state: USVState) -> float:
        # discourage sitting on target at speed (which means overshoot next)
        c = self.reward_cfg
        return -c.w_velocity * (state.u / 4.0) ** 2

    def _task_terminated(self, state: USVState) -> bool:
        err = abs(self._error(state))
        if self._arrival_time is None and err < self.task.tolerance and abs(state.u) < 0.2:
            self._arrival_time = state.t
        if abs(state.x - self.target_position) > self.position_bound:
            return True
        if not self.task.hold and self._arrival_time is not None:
            return True
        return False

    def _info(self, state, err):
        info = super()._info(state, err)
        info["arrival_time"] = self._arrival_time
        info["target_position"] = self.target_position
        return info

    def control_obs(self, state: Optional[USVState] = None) -> ControlObs:
        st = state if state is not None else self.sim.measurement()
        return ControlObs.from_sim(self.sim, self._error(st),
                                   target_position=self.target_position, state=st)


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------
def register_envs() -> None:
    """Register ``USVSpeedHold-v0`` and ``USVStationKeep-v0`` with Gymnasium."""
    if not GYMNASIUM_AVAILABLE:      # pragma: no cover
        return
    from gymnasium.envs.registration import register, registry
    for env_id, path in (("USVSpeedHold-v0", "usv_seakeeper.envs:SpeedHoldEnv"),
                         ("USVStationKeep-v0", "usv_seakeeper.envs:StationKeepEnv")):
        if env_id not in registry:
            register(id=env_id, entry_point=path)


__all__ = ["SpeedHoldEnv", "StationKeepEnv", "register_envs",
           "GYMNASIUM_AVAILABLE"]
