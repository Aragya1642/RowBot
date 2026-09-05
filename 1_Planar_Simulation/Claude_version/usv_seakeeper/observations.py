"""Observation construction, shared by the RL environments and by
:class:`~usv_seakeeper.controllers.PolicyController`.

Keeping this in one place is what makes it safe to drop a trained policy into
the classical-controller harness: both paths build the observation vector with
the same code, so there is no train/deploy skew.

Two design notes that matter for whether training works at all:

1. **The actuator state is part of the observation.** The thruster is rate
   limited, so the commanded thrust is a genuine state variable. Omit it and
   the MDP is not Markov, and the policy will chatter.

2. **An integral term is available.** With unmodelled drag, a memoryless
   policy on (error, velocity) *cannot* drive steady-state error to zero --
   the same reason a P controller cannot. Either include the running error
   integral (default) or use a recurrent policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .sim import USVSim, USVState


@dataclass
class ObsConfig:
    include_speed: bool = True
    include_actuator: bool = True
    include_wave_slope: bool = True
    include_attitude: bool = True       # pitch, pitch rate
    include_heave: bool = True          # heave, heave rate
    include_ventilation: bool = True
    include_integral: bool = True
    include_preview: bool = True
    include_time: bool = False

    # normalisation scales (keep observations roughly unit-variance)
    speed_scale: float = 4.0
    error_scale: float = 2.0
    heave_scale: float = 2.0
    slope_scale: float = 0.3
    pitch_scale: float = 0.5
    rate_scale: float = 2.0
    integral_scale: float = 5.0
    integral_clip: float = 10.0


class ObservationBuilder:
    """Builds a flat float32 observation vector from the simulator state."""

    def __init__(self, sim: USVSim, config: Optional[ObsConfig] = None):
        self.sim = sim
        self.cfg = config or ObsConfig()
        self._integral = 0.0
        self._names = self._build_names()

    # ------------------------------------------------------------------
    def _build_names(self) -> List[str]:
        c = self.cfg
        n = ["error"]
        if c.include_speed:
            n.append("speed")
        if c.include_actuator:
            n.append("thrust_norm")
        if c.include_wave_slope:
            n.append("wave_slope")
        if c.include_attitude:
            n += ["pitch", "pitch_rate"]
        if c.include_heave and self.sim.cfg.dof != "surge":
            n += ["heave", "heave_rate"]
        if c.include_ventilation:
            n.append("ventilation")
        if c.include_integral:
            n.append("error_integral")
        if c.include_time:
            n.append("time_frac")
        if c.include_preview:
            n += [f"preview_{i}" for i in range(self.sim.cfg.preview_points)]
        return n

    @property
    def names(self) -> List[str]:
        return list(self._names)

    @property
    def size(self) -> int:
        return len(self._names)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._integral = 0.0
        self._names = self._build_names()

    def build(self, error: float, dt: float,
              state: Optional[USVState] = None) -> np.ndarray:
        """Assemble the observation.

        ``error`` is the task error (target minus measured), in task units:
        m/s for speed holding, m for station keeping.
        """
        c = self.cfg
        st = state if state is not None else self.sim.measurement()
        sim = self.sim

        self._integral = float(np.clip(self._integral + error * dt,
                                       -c.integral_clip, c.integral_clip))

        vals = [error / c.error_scale]
        if c.include_speed:
            vals.append(st.u / c.speed_scale)
        if c.include_actuator:
            vals.append(st.thrust_cmd / max(sim.vessel.thrust_max, 1e-9))
        if c.include_wave_slope:
            vals.append(st.wave_slope / c.slope_scale)
        if c.include_attitude:
            vals += [st.theta / c.pitch_scale, st.q / c.rate_scale]
        if c.include_heave and sim.cfg.dof != "surge":
            vals += [st.z / c.heave_scale, st.w / c.rate_scale]
        if c.include_ventilation:
            vals.append(st.ventilation * 2.0 - 1.0)
        if c.include_integral:
            vals.append(self._integral / c.integral_scale)
        if c.include_time:
            vals.append(st.t / max(sim.cfg.max_time, 1e-9))
        if c.include_preview:
            pv = sim.preview()
            if pv.size:
                vals.extend(pv / c.slope_scale)

        return np.asarray(vals, dtype=np.float32)

    # convenience for controllers that want the raw integral
    @property
    def integral(self) -> float:
        return self._integral


__all__ = ["ObsConfig", "ObservationBuilder"]
