"""Framework-free USV simulator.

:class:`USVSim` is the single source of truth for the physics. It knows
nothing about Gymnasium, controllers, or plotting -- it takes a normalised
thrust command in ``[-1, 1]`` and advances one control interval.

Everything else in this package (the classical-controller harness, the RL
environments, the rollout/plotting tools) is a thin wrapper around this, so a
PID controller and a trained policy are guaranteed to be evaluated against
identical dynamics.

Coordinate conventions
----------------------
* ``x``      surge position, earth-fixed, positive forward [m]
* ``u``      surge velocity [m/s]
* ``z``      heave, positive **up**, measured from the calm waterline [m]
* ``theta``  pitch, positive **bow up** [rad]
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple, Any

import numpy as np

from .config import (G, RHO, WaveConfig, VesselConfig, SensorConfig, SimConfig,
                     RandomizeConfig)
from .waves import WaveField, WaveQuery


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------
@dataclass
class USVState:
    """Full simulator state. ``y`` holds the integrated mechanical states."""
    t: float = 0.0
    x: float = 0.0
    u: float = 0.0
    z: float = 0.0
    w: float = 0.0
    theta: float = 0.0
    q: float = 0.0
    thrust_cmd: float = 0.0      # actuator command after rate limiting [N]
    thrust_delivered: float = 0.0  # after ventilation losses [N]
    ventilation: float = 1.0     # 1 = fully immersed, 0 = fully emerged

    # instantaneous wave kinematics at the CG (for logging / observation)
    wave_eta: float = 0.0
    wave_slope: float = 0.0
    wave_u: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.u, self.z, self.w, self.theta, self.q])

    def to_dict(self) -> Dict[str, float]:
        return {
            "t": self.t, "x": self.x, "u": self.u, "z": self.z, "w": self.w,
            "theta": self.theta, "q": self.q,
            "thrust_cmd": self.thrust_cmd,
            "thrust_delivered": self.thrust_delivered,
            "ventilation": self.ventilation,
            "wave_eta": self.wave_eta, "wave_slope": self.wave_slope,
            "wave_u": self.wave_u,
        }


# --------------------------------------------------------------------------
# sensor model
# --------------------------------------------------------------------------
class SensorModel:
    """Additive Gaussian noise plus pure transport delay on measurements."""

    def __init__(self, config: SensorConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._buffer: list = []

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._buffer = []

    def measure(self, state: USVState) -> USVState:
        c = self.config
        m = replace(state)
        if c.speed_noise:
            m.u += self._rng.normal(0.0, c.speed_noise)
        if c.position_noise:
            m.x += self._rng.normal(0.0, c.position_noise)
        if c.slope_noise:
            m.wave_slope += self._rng.normal(0.0, c.slope_noise)
        if c.pitch_noise:
            m.theta += self._rng.normal(0.0, c.pitch_noise)

        if c.latency_steps > 0:
            self._buffer.append(m)
            if len(self._buffer) > c.latency_steps:
                return self._buffer.pop(0)
            return self._buffer[0]
        return m


# --------------------------------------------------------------------------
# simulator
# --------------------------------------------------------------------------
class USVSim:
    """Deterministic-given-seed USV simulator.

    Examples
    --------
    >>> sim = USVSim()
    >>> sim.reset(seed=0)                       # doctest: +ELLIPSIS
    USVState(...)
    >>> for _ in range(20):
    ...     st = sim.step(0.5)                  # 50% forward thrust
    """

    def __init__(self,
                 wave: Optional[WaveConfig] = None,
                 vessel: Optional[VesselConfig] = None,
                 sim: Optional[SimConfig] = None,
                 sensors: Optional[SensorConfig] = None,
                 randomize: Optional[RandomizeConfig] = None):
        self.wave_cfg = wave or WaveConfig()
        self.vessel = vessel or VesselConfig()
        self.cfg = sim or SimConfig()
        self.sensor_cfg = sensors or SensorConfig()
        self.rand_cfg = randomize or RandomizeConfig()

        self._base_wave = replace(self.wave_cfg)
        self._base_vessel = replace(self.vessel)

        self.field = WaveField(self.wave_cfg, hull_length=(
            self.vessel.length if self.cfg.fk_length_averaging else None))
        self.sensors = SensorModel(self.sensor_cfg)
        self.state = USVState()
        self._rng = np.random.default_rng(0)
        self._preview_offsets = self._make_preview_offsets()
        self._refresh_derived()

    # ------------------------------------------------------------------
    def _make_preview_offsets(self) -> np.ndarray:
        n = self.cfg.preview_points
        if n <= 0:
            return np.zeros(0)
        return np.linspace(self.vessel.length * 0.5,
                           self.cfg.preview_distance, n)

    def _refresh_derived(self) -> None:
        v = self.vessel
        self._m_surge = v.mass * (1.0 + v.added_mass_surge)
        self._m_heave = v.mass * (1.0 + v.added_mass_heave)
        self._i_pitch = v.pitch_inertia() * (1.0 + v.added_inertia_pitch)
        self._c33 = v.heave_stiffness()
        self._c55 = v.pitch_stiffness()
        self._b33 = 2.0 * v.damping_ratio_heave * np.sqrt(self._c33 * self._m_heave)
        self._b55 = 2.0 * v.damping_ratio_pitch * np.sqrt(self._c55 * self._i_pitch)
        # bounds on the relative vertical offset over which the linear
        # hydrostatic restoring is valid (see VesselConfig.emergence_limit)
        self._d_emerge = v.emergence_limit()
        self._d_immerse = v.immersion_limit()

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None,
              initial_speed: float = 0.0,
              initial_position: float = 0.0) -> USVState:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._apply_randomization()

        u0 = initial_speed
        if self.rand_cfg.initial_speed is not None:
            lo, hi = self.rand_cfg.initial_speed
            u0 = float(self._rng.uniform(lo, hi))

        self.state = USVState(t=0.0, x=float(initial_position), u=float(u0))

        # Start the vessel *on* the water, not at z = 0. In a developed sea
        # the surface at the spawn point can be metres away from the mean
        # level, and resetting to z = 0 drops the hull from mid-air: a large
        # spurious heave/pitch transient, full prop ventilation for the first
        # second, and an inflated max_abs_error on every episode. Seeding the
        # heave and pitch states from the local surface removes it.
        if self.cfg.dof != "surge":
            wq = self.field.query(np.array([self.state.x]), 0.0, z=0.0)
            self.state.z = float(wq.eta_avg[0])
            self.state.w = float(wq.w_avg[0])
            slope = float(wq.slope_avg[0])
            self.state.theta = math.atan(slope)
            self.state.q = float(wq.slope_rate_avg[0]) / (1.0 + slope * slope)

        self.sensors.reset(seed=None if seed is None else seed + 7919)
        self._update_wave_readout()
        return replace(self.state)

    def _apply_randomization(self) -> None:
        r = self.rand_cfg
        # restore baselines so randomisation does not compound across episodes
        self.wave_cfg = replace(self._base_wave)
        self.vessel = replace(self._base_vessel)

        def draw(rng_range):
            lo, hi = rng_range
            return float(self._rng.uniform(lo, hi))

        if r.hs is not None:
            self.wave_cfg.hs = draw(r.hs)
        if r.tp is not None:
            self.wave_cfg.tp = draw(r.tp)
        if r.heading:
            self.wave_cfg.heading = 1 if self._rng.random() < 0.5 else -1
        if r.reseed_waves:
            self.wave_cfg.seed = int(self._rng.integers(0, 2 ** 31 - 1))

        if r.mass is not None:
            self.vessel.mass = draw(r.mass)
        if r.thrust_max is not None:
            self.vessel.thrust_max = draw(r.thrust_max)
        if r.drag_coeff is not None:
            self.vessel.drag_coeff = draw(r.drag_coeff)
        if r.slew_rate is not None:
            self.vessel.slew_rate = draw(r.slew_rate)

        self.field = WaveField(self.wave_cfg, hull_length=(
            self.vessel.length if self.cfg.fk_length_averaging else None))
        self._refresh_derived()
        self._preview_offsets = self._make_preview_offsets()
        self._warn_if_outside_validity()

    def _warn_if_outside_validity(self) -> None:
        """Flag sea states where the linear model stops being trustworthy.

        Two envelopes matter. Linear superposition cannot represent crests
        much past a steepness of ~0.3 (breaking is ~0.44), and once the hull
        is submerged deeper than its freeboard the linear hydrostatic model is
        describing a swamped vessel, which really needs a nonlinear method.
        Results past these points stay bounded and finite, but should not be
        quoted as physics.
        """
        st = self.wave_cfg.steepness()
        if st > 0.30:
            warnings.warn(
                f"sea state steepness {st:.2f} is outside linear wave theory "
                f"(breaking ~0.44); motions remain bounded but are not "
                f"quantitatively trustworthy",
                RuntimeWarning, stacklevel=3)

    # ------------------------------------------------------------------
    # kinematics helpers
    # ------------------------------------------------------------------
    def _pose(self, x: float, z: float, theta: float, t: float
              ) -> Tuple[WaveQuery, float, float]:
        """Wave query at the CG and at the propeller, plus the effective
        pitch/heave used by the 1-DOF (surface-following) model."""
        xs = np.array([x, x + self.vessel.prop_x])
        wq = self.field.query(xs, t)
        return wq, float(wq.eta[0]), float(wq.slope[0])

    def _ventilation_factor(self, x: float, z: float, theta: float,
                            eta_prop: float) -> float:
        """Fraction of commanded thrust actually delivered.

        The propeller sits at ``prop_x`` behind the CG and ``prop_depth``
        below the calm waterline. On a crest in a steep sea the stern lifts
        clear, the prop ventilates, and thrust collapses -- this is *the*
        dominant high-sea-state failure mode and it is a much harder problem
        for a controller than pure actuator saturation, because the authority
        loss is state dependent rather than a fixed limit.

        The factor is referenced to the propeller *disc*, not the axis: full
        thrust once the top of the disc is submerged (axis deeper than D/2),
        zero once the bottom of the disc clears the surface, smoothstep in
        between. Anchoring it to the disc means a nominally well-submerged
        prop reads exactly 1.0 instead of sitting on the edge of the ramp.
        """
        if not self.cfg.ventilation:
            return 1.0
        v = self.vessel
        # vertical position of the prop axis in earth frame
        z_axis = z + v.prop_x * math.sin(theta) - v.prop_depth
        h = eta_prop - z_axis                 # axis submergence
        d = v.prop_diameter if v.prop_diameter > 1e-6 else 1e-6
        s = (h + 0.5 * d) / d
        if s <= 0.0:
            return 0.0
        if s >= 1.0:
            return 1.0
        return s * s * (3.0 - 2.0 * s)        # smoothstep

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------
    def _derivatives(self, y: np.ndarray, t: float, f_act: float
                     ) -> Tuple[np.ndarray, float, float]:
        """Return (dy/dt, delivered_thrust, ventilation_factor)."""
        v = self.vessel
        if self.cfg.dof == "surge":
            x, u = y[0], y[1]
            # the hull rides the surface, so evaluate orbital kinematics
            # there; the "surface" sentinel resolves z inside the query and
            # saves a second full evaluation
            wq = self.field.query(np.array([x, x + v.prop_x]), t, z="surface")
            eta = float(wq.eta_avg[0])
            slope = float(wq.slope_avg[0])
            theta = math.atan(slope)
            z = eta
            vent = self._ventilation_factor(x, z, theta, float(wq.eta[1]))
            f_del = f_act * vent

            # Surface-following (bead-on-a-wire) model: the horizontal
            # component of the constraint force is m*g*sin(th)*cos(th). For
            # linear deep-water waves this equals the Froude-Krylov force,
            # since -g * d(eta)/dx == du/dt when w^2 = g*k.
            grav_x = -v.mass * G * math.sin(theta) * math.cos(theta)
            u_orb = float(wq.u_avg[0]) if self.cfg.morison_inertia else 0.0
            a_orb = float(wq.du_dt_avg[0]) if self.cfg.morison_inertia else 0.0
            u_rel = u - u_orb
            f_drag = -v.drag_coeff * u_rel * (u_rel if u_rel >= 0.0 else -u_rel)
            f_inertia = v.mass * v.added_mass_surge * a_orb

            du = (f_del * math.cos(theta) + grav_x + f_drag + f_inertia) / self._m_surge
            return np.array([u, du, 0.0, 0.0, 0.0, 0.0]), f_del, vent

        # ---- 3-DOF: surge / heave / pitch ----
        x, u, z, w, theta, q = y
        # Never evaluate orbital kinematics above the free surface: linear
        # theory has no fluid there, and exp(|k|z) would amplify instead of
        # decay. Clamping the evaluation height to the surface is the usual
        # (Wheeler-style) fix.
        wq0 = self.field.query(np.array([x, x + v.prop_x]), t)
        z_eval = min(z, float(wq0.eta_avg[0]))
        wq = self.field.query(np.array([x, x + v.prop_x]), t, z=z_eval)
        eta_a = float(wq.eta_avg[0])
        slope_a = float(wq.slope_avg[0])
        u_orb = float(wq.u_avg[0])
        a_orb = float(wq.du_dt_avg[0])
        w_orb = float(wq.w_avg[0])
        slope_rate = float(wq.slope_rate_avg[0])

        vent = self._ventilation_factor(x, z, theta, float(wq.eta[1]))
        f_del = f_act * vent

        # Wetted fraction. Submerged volume falls as the hull rises, reaching
        # zero one draft above the local surface. Every hydrodynamic term --
        # drag, wave excitation, radiation damping, added mass -- is scaled by
        # it, or an airborne hull keeps being pushed and damped by water that
        # is not there. Leaving these ungated launches the vessel many metres
        # into the air in a steep sea.
        #
        # A smoothstep rather than a linear clip, for two reasons: added mass
        # and radiation damping genuinely vary continuously with submergence,
        # and a hard clip puts a derivative kink at full immersion that the
        # hull crosses on every wave, which knocks RK4 down from 4th order.
        d = z - eta_a
        s_wet = 1.0 - d / self._d_emerge
        if s_wet <= 0.0:
            phi = 0.0
        elif s_wet >= 1.0:
            phi = 1.0
        else:
            phi = s_wet * s_wet * (3.0 - 2.0 * s_wet)   # smoothstep

        # --- surge ---
        # Heave is free, so there is no surface constraint force. The
        # horizontal wave load is the Morison inertia term
        # m*(1+Ca)*du/dt, which drives the hull toward the local orbital
        # velocity. Thrust is rotated by the actual pitch angle.
        u_rel = u - u_orb
        f_drag = -phi * v.drag_coeff * u_rel * (u_rel if u_rel >= 0.0 else -u_rel)
        f_wave = phi * self._m_surge * a_orb if self.cfg.morison_inertia else 0.0
        m_surge_eff = v.mass * (1.0 + v.added_mass_surge * phi)
        du = (f_del * math.cos(theta) + f_drag + f_wave) / m_surge_eff

        # --- heave --- relative-motion hydrostatics:
        # restoring is proportional to submergence *relative to the local
        # wave surface*, which gives RAO -> 1 for long waves and, together
        # with sinc length averaging, RAO -> 0 for waves shorter than the
        # hull, without needing tabulated diffraction coefficients.
        #
        # The offset is clamped to the range over which that linear form is
        # valid: at one draft above the surface the submerged volume is zero,
        # so the restoring saturates at exactly the vessel weight and, with
        # added mass also gated to zero, the hull free-falls at -g. Sinking
        # past the freeboard submerges the deck and the waterplane stops
        # growing.
        d_c = d
        if d_c > self._d_emerge:
            d_c = self._d_emerge
        elif d_c < -self._d_immerse:
            d_c = -self._d_immerse
        m_heave_eff = v.mass * (1.0 + v.added_mass_heave * phi)
        dw = (-phi * self._b33 * (w - w_orb) - self._c33 * d_c) / m_heave_eff

        # --- pitch ---
        theta_wave = math.atan(slope_a)
        q_wave = slope_rate / (1.0 + slope_a * slope_a)   # d/dt arctan(slope)
        m_thrust = f_del * v.prop_depth                   # low prop -> bow-up trim
        i_pitch_eff = v.pitch_inertia() * (1.0 + v.added_inertia_pitch * phi)
        dq = (-phi * self._b55 * (q - q_wave)
              - phi * self._c55 * (theta - theta_wave)
              + m_thrust) / i_pitch_eff

        return np.array([u, du, w, dw, q, dq]), f_del, vent

    # ------------------------------------------------------------------
    # integration
    # ------------------------------------------------------------------
    def _rk4(self, y: np.ndarray, t: float, dt: float, f_of_t
             ) -> Tuple[np.ndarray, float, float]:
        """RK4 on the full mechanical state.

        ``f_of_t`` gives the actuator force at an absolute time. Evaluating it
        at each stage time (rather than freezing it per sub-step) keeps the
        rate-limited thrust ramp from dragging the scheme down to first order.
        """
        k1, f_del, vent = self._derivatives(y, t, f_of_t(t))
        k2, _, _ = self._derivatives(y + 0.5 * dt * k1, t + 0.5 * dt, f_of_t(t + 0.5 * dt))
        k3, _, _ = self._derivatives(y + 0.5 * dt * k2, t + 0.5 * dt, f_of_t(t + 0.5 * dt))
        k4, _, _ = self._derivatives(y + dt * k3, t + dt, f_of_t(t + dt))
        return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), f_del, vent

    def step(self, action: float) -> USVState:
        """Advance one control interval with a zero-order-held command.

        Parameters
        ----------
        action
            Normalised thrust command in ``[-1, 1]``. Values outside are
            clipped, so a policy with an unbounded head still behaves.
        """
        a = float(np.clip(action, -1.0, 1.0))
        f_target = a * self.vessel.thrust_max
        dt = self.cfg.dt_physics
        n = self.cfg.substeps()

        y = self.state.as_array()
        t = self.state.t
        t0 = t
        f_start = self.state.thrust_cmd
        slew = self.vessel.slew_rate
        gap = f_target - f_start

        def f_of_t(t_abs: float) -> float:
            """Exact rate-limited zero-order-hold ramp.

            Closed form rather than an Euler accumulation, so the actuator
            trajectory is independent of ``dt_physics``.
            """
            budget = slew * (t_abs - t0)
            if budget <= 0.0:
                return f_start
            if gap > budget:
                return f_start + budget
            if gap < -budget:
                return f_start - budget
            return f_target

        f_del = self.state.thrust_delivered
        vent = self.state.ventilation
        for _ in range(n):
            y, f_del, vent = self._rk4(y, t, dt, f_of_t)
            t += dt
        f_act = f_of_t(t)

        self.state.x, self.state.u = float(y[0]), float(y[1])
        self.state.z, self.state.w = float(y[2]), float(y[3])
        self.state.theta, self.state.q = float(y[4]), float(y[5])
        self.state.t = t
        self.state.thrust_cmd = f_act
        self.state.thrust_delivered = f_del
        self.state.ventilation = vent

        if self.cfg.dof == "surge":
            wq = self.field.query(np.array([self.state.x]), t)
            self.state.z = float(wq.eta_avg[0])
            self.state.theta = math.atan(float(wq.slope_avg[0]))

        self._update_wave_readout()
        return replace(self.state)

    def _update_wave_readout(self) -> None:
        wq = self.field.query(np.array([self.state.x]), self.state.t)
        self.state.wave_eta = float(wq.eta[0])
        self.state.wave_slope = float(wq.slope[0])
        self.state.wave_u = float(wq.u[0])

    # ------------------------------------------------------------------
    # sensing / preview
    # ------------------------------------------------------------------
    def measurement(self) -> USVState:
        """Noisy, optionally delayed view of the state (what a controller sees)."""
        return self.sensors.measure(self.state)

    def preview(self) -> np.ndarray:
        """Wave slope at a set of look-ahead distances ahead of the bow.

        The wave field is analytic, so this is free -- it stands in for a
        wave-radar / lidar scan and lets you build preview-feedforward or
        MPC controllers, or hand an RL policy the disturbance it is about
        to meet instead of only the one it is already in.
        """
        if self._preview_offsets.size == 0:
            return np.zeros(0)
        xs = self.state.x + self._preview_offsets
        return self.field.query(xs, self.state.t).slope_avg

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    def is_unsafe(self) -> bool:
        return (abs(self.state.theta) > self.cfg.pitch_limit
                or abs(self.state.u) > self.cfg.speed_limit
                or not np.isfinite(self.state.u))

    def saturation(self) -> float:
        return abs(self.state.thrust_cmd) / max(self.vessel.thrust_max, 1e-9)

    def max_sustainable_speed(self) -> float:
        """Calm-water top speed: sqrt(Fmax / Cd). Handy sanity check when
        choosing a target speed -- asking for more than this can never work
        regardless of gains."""
        return float(np.sqrt(self.vessel.thrust_max / self.vessel.drag_coeff))

    def info(self) -> Dict[str, Any]:
        return {
            "sea": self.field.summary(),
            "slope_rms": self.field.slope_rms(),
            "recurrence_s": self.field.recurrence_period(),
            "heave_Tn": self.vessel.heave_natural_period(),
            "pitch_Tn": self.vessel.pitch_natural_period(),
            "draft": self.vessel.draft(),
            "v_max_calm": self.max_sustainable_speed(),
            "breaking": self.wave_cfg.is_breaking(),
        }


__all__ = ["USVSim", "USVState", "SensorModel"]
