"""Configuration dataclasses for the USV Sea-Keeper simulator.

Everything the simulator needs is described by plain dataclasses so that
configs are trivially serialisable, diffable, and sweepable.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Literal, Optional, Tuple, Dict, Any
import math

G = 9.81            # gravitational acceleration [m/s^2]
RHO = 1025.0        # seawater density [kg/m^3]


# --------------------------------------------------------------------------
# Waves
# --------------------------------------------------------------------------
@dataclass
class WaveConfig:
    """Sea state description.

    Two modes:
      * ``regular``   -- a single Airy wave (amplitude / period).
      * ``jonswap``   -- a JONSWAP spectrum discretised into ``n_components``
                         linear components with random phases.
    """
    kind: Literal["regular", "jonswap", "calm"] = "jonswap"

    # regular
    amplitude: float = 0.5          # [m]
    period: float = 4.0             # [s]

    # irregular (JONSWAP)
    hs: float = 1.2                 # significant wave height [m]
    tp: float = 6.0                 # peak period [s]
    gamma: float = 3.3              # peak enhancement factor
    n_components: int = 64
    # Frequency band as multiples of the peak frequency. The upper bound
    # matters a lot: wave *slope* variance scales like omega^4 * S(omega),
    # so truncating at 3*wp (as the original JS did) badly under-predicts
    # the disturbance the controller actually has to reject.
    w_lo_factor: float = 0.3
    w_hi_factor: float = 6.0
    # Randomise frequency within each bin. Without this, uniformly spaced
    # components make the sea surface exactly periodic with period
    # 2*pi/dw (~60 s for typical settings) and an RL agent will happily
    # memorise the wave train instead of learning to reject it.
    frequency_jitter: bool = True

    seed: Optional[int] = 0         # None -> nondeterministic phases

    # Evaluate orbital velocities at the body's actual height using the
    # deep-water decay exp(|k| z) rather than at the mean surface. Required
    # to reproduce the correct Stokes drift (see WaveField.query).
    depth_stretching: bool = True

    # Direction of travel relative to the waves, with the vessel steaming
    # toward +x:
    #   +1 -> head sea      (waves run toward the vessel; encounter
    #                        frequency is raised above the wave frequency)
    #   -1 -> following sea (waves overtake the vessel; encounter frequency
    #                        is lowered, and surf-riding becomes possible)
    # Implemented as a sign on the wavenumber; see WaveField._build.
    heading: Literal[1, -1] = 1

    def steepness(self) -> float:
        """Characteristic steepness. >~0.44 is unphysical (breaking)."""
        if self.kind == "calm":
            return 0.0
        if self.kind == "regular":
            w = 2.0 * math.pi / self.period
            k = w * w / G
            return self.amplitude * k
        wp = 2.0 * math.pi / self.tp
        kp = wp * wp / G
        return 0.5 * self.hs * kp

    def is_breaking(self) -> bool:
        return self.steepness() > 0.44


# --------------------------------------------------------------------------
# Vessel
# --------------------------------------------------------------------------
@dataclass
class VesselConfig:
    """Vessel, actuator and hydrodynamic parameters."""
    mass: float = 350.0             # [kg]
    length: float = 2.6             # waterline length [m]
    beam: float = 1.0               # [m]
    block_coeff: float = 0.7        # for draft estimation

    # surge
    drag_coeff: float = 40.0        # [N s^2 / m^2], F = -Cd * v_rel * |v_rel|
    added_mass_surge: float = 0.15  # fraction of mass

    # heave / pitch (only used when dof == "surge_heave_pitch")
    added_mass_heave: float = 1.0   # fraction of mass
    added_inertia_pitch: float = 1.0
    freeboard: Optional[float] = None   # deck height above waterline [m];
                                        # defaults to 2x draft
    damping_ratio_heave: float = 0.20
    damping_ratio_pitch: float = 0.20
    gyradius_factor: float = 0.25   # radius of gyration / length

    # actuator
    thrust_max: float = 500.0       # [N], symmetric
    slew_rate: float = 2000.0       # [N/s]
    prop_depth: float = 0.18        # prop axis depth below calm waterline [m]
    prop_diameter: float = 0.15     # [m]
    prop_x: float = -1.1            # longitudinal position rel. CG [m] (stern -ve)

    def draft(self) -> float:
        return self.mass / (RHO * self.length * self.beam * self.block_coeff)

    def emergence_limit(self) -> float:
        """Relative rise at which the hull is fully out of the water.

        Submerged volume falls as V(d) = displacement - A_wp*d, so it reaches
        zero at d = displacement/A_wp. Past that the hydrostatic restoring
        must saturate at the vessel weight (free fall) instead of growing
        linearly -- otherwise an airborne hull gets yanked back with several
        times its own weight.
        """
        return (self.mass / RHO) / self.waterplane_area()

    def immersion_limit(self) -> float:
        """Relative sinkage at which the deck submerges and the waterplane
        stops growing."""
        return self.freeboard if self.freeboard is not None else 2.0 * self.draft()

    def waterplane_area(self) -> float:
        return self.length * self.beam

    def heave_stiffness(self) -> float:
        return RHO * G * self.waterplane_area()

    def pitch_stiffness(self) -> float:
        # rho*g*I_L with I_L = L^3*B/12 (longitudinal 2nd moment of waterplane)
        return RHO * G * (self.length ** 3) * self.beam / 12.0

    def pitch_inertia(self) -> float:
        return self.mass * (self.gyradius_factor * self.length) ** 2

    def heave_natural_period(self) -> float:
        m = self.mass * (1.0 + self.added_mass_heave)
        return 2.0 * math.pi * math.sqrt(m / self.heave_stiffness())

    def pitch_natural_period(self) -> float:
        i = self.pitch_inertia() * (1.0 + self.added_inertia_pitch)
        return 2.0 * math.pi * math.sqrt(i / self.pitch_stiffness())


# --------------------------------------------------------------------------
# Sensing
# --------------------------------------------------------------------------
@dataclass
class SensorConfig:
    """Measurement realism. Defaults are *noise free* -- turn these on when
    you care about sim-to-real, because a policy trained on a perfect
    250 Hz state estimate learns to differentiate a clean signal and will
    fall apart on real GNSS/IMU data."""
    speed_noise: float = 0.0        # [m/s] std
    position_noise: float = 0.0     # [m] std
    slope_noise: float = 0.0        # [-] std on measured wave slope
    pitch_noise: float = 0.0        # [rad] std
    latency_steps: int = 0          # control-step delay applied to observations
    seed: Optional[int] = 1234


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
@dataclass
class SimConfig:
    dof: Literal["surge", "surge_heave_pitch"] = "surge_heave_pitch"
    dt: float = 0.05                # control / step interval [s] (20 Hz)
    dt_physics: float = 0.005       # integrator sub-step [s] (200 Hz)
    ventilation: bool = True        # thrust loss when prop emerges
    fk_length_averaging: bool = True # average wave force over hull length
    morison_inertia: bool = True
    max_time: float = 60.0          # episode length [s]

    # preview ("wave radar") -- the wave field is analytic, so a finite
    # spatial scan ahead of the bow is available for free. Useful for
    # preview-feedforward or MPC-style controllers.
    preview_points: int = 0
    preview_distance: float = 30.0  # [m] ahead of the bow

    # safety / termination
    pitch_limit: float = math.radians(45.0)
    speed_limit: float = 15.0       # [m/s]

    def __post_init__(self) -> None:
        ratio = self.dt / self.dt_physics
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(
                f"dt ({self.dt}) must be an integer multiple of dt_physics "
                f"({self.dt_physics}); got ratio {ratio:.6f}. A non-integer "
                "ratio silently changes the effective control interval.")

    def substeps(self) -> int:
        return max(1, int(round(self.dt / self.dt_physics)))


# --------------------------------------------------------------------------
# Tasks & rewards
# --------------------------------------------------------------------------
@dataclass
class SpeedTaskConfig:
    target_speed: float = 2.0       # [m/s]
    randomize_target: Optional[Tuple[float, float]] = None
    tolerance: float = 0.15         # [m/s] "on setpoint" band
    err_scale: float = 1.0          # reward normalisation


@dataclass
class StationTaskConfig:
    target_position: float = 40.0   # [m]
    randomize_target: Optional[Tuple[float, float]] = (20.0, 80.0)
    tolerance: float = 1.0          # [m]
    err_scale: float = 10.0         # [m] reward normalisation
    hold: bool = True               # keep station after arrival (vs. terminate)


@dataclass
class RewardConfig:
    """Reward shaping weights.

    reward = -(err/err_scale)^2
             - w_effort  * (u)^2
             - w_rate    * (du/dt normalised)^2
             - w_velocity* (v/v_scale)^2      [station-keeping only]
             + bonus_in_tolerance
    """
    w_error: float = 1.0
    w_effort: float = 0.02
    w_rate: float = 0.05
    w_velocity: float = 0.10
    bonus_in_tolerance: float = 0.5
    gaussian_error: bool = False    # exp(-(e/s)^2) instead of -(e/s)^2
    crash_penalty: float = 50.0


# --------------------------------------------------------------------------
# Domain randomisation
# --------------------------------------------------------------------------
@dataclass
class RandomizeConfig:
    """Per-episode randomisation ranges. ``None`` leaves a parameter fixed.

    Wave phases are always reseeded per episode when ``reseed_waves`` is
    True, which is the single most important item here.
    """
    reseed_waves: bool = True
    hs: Optional[Tuple[float, float]] = None
    tp: Optional[Tuple[float, float]] = None
    mass: Optional[Tuple[float, float]] = None
    thrust_max: Optional[Tuple[float, float]] = None
    drag_coeff: Optional[Tuple[float, float]] = None
    slew_rate: Optional[Tuple[float, float]] = None
    initial_speed: Optional[Tuple[float, float]] = None
    heading: bool = False           # randomly flip head/following sea


# --------------------------------------------------------------------------
# Presets (ported from the original browser tool, plus a heavy case)
# --------------------------------------------------------------------------
def _preset(wave: WaveConfig, vessel: VesselConfig, sim: SimConfig) -> Dict[str, Any]:
    return {"wave": wave, "vessel": vessel, "sim": sim}


PRESETS: Dict[str, Dict[str, Any]] = {
    "calm_harbor": _preset(
        WaveConfig(kind="regular", amplitude=0.25, period=3.5),
        VesselConfig(mass=250.0, thrust_max=400.0, drag_coeff=35.0, slew_rate=2500.0),
        SimConfig(),
    ),
    "coastal_chop": _preset(
        WaveConfig(kind="jonswap", hs=1.2, tp=6.0),
        VesselConfig(mass=350.0, thrust_max=600.0, drag_coeff=45.0, slew_rate=2000.0),
        SimConfig(),
    ),
    "storm_head_sea": _preset(
        WaveConfig(kind="jonswap", hs=3.2, tp=8.0, heading=1),
        VesselConfig(mass=600.0, thrust_max=1400.0, drag_coeff=70.0, slew_rate=3000.0,
                     length=3.4, beam=1.3),
        SimConfig(max_time=90.0),
    ),
    "sea_state_6": _preset(
        WaveConfig(kind="jonswap", hs=5.0, tp=10.5, heading=1),
        VesselConfig(mass=900.0, thrust_max=2200.0, drag_coeff=95.0, slew_rate=4000.0,
                     length=4.5, beam=1.7),
        SimConfig(max_time=120.0),
    ),
    "following_sea": _preset(
        WaveConfig(kind="jonswap", hs=2.0, tp=7.0, heading=-1),
        VesselConfig(mass=400.0, thrust_max=800.0, drag_coeff=50.0),
        SimConfig(),
    ),
    # A following-sea storm. Note what this preset does NOT reproduce: true
    # surf-riding needs the vessel to approach the wave celerity
    # c = g*Tp/(2*pi), which is 14 m/s here against an attainable 4.5 m/s.
    # At v/c ~ 0.3 wave capture is unreachable, so the head/following
    # difference in this model is only the shift in encounter frequency.
    "following_storm": _preset(
        WaveConfig(kind="jonswap", hs=3.5, tp=9.0, heading=-1),
        VesselConfig(mass=600.0, thrust_max=1400.0, drag_coeff=70.0,
                     length=3.4, beam=1.3),
        SimConfig(max_time=90.0, speed_limit=25.0),
    ),
    "calm_water": _preset(
        WaveConfig(kind="calm"),
        VesselConfig(),
        SimConfig(dof="surge", ventilation=False),
    ),
}


def get_preset(name: str):
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; options: {sorted(PRESETS)}")
    p = PRESETS[name]
    return (replace(p["wave"]), replace(p["vessel"]), replace(p["sim"]))


__all__ = [
    "G", "RHO", "WaveConfig", "VesselConfig", "SensorConfig", "SimConfig",
    "SpeedTaskConfig", "StationTaskConfig", "RewardConfig", "RandomizeConfig",
    "PRESETS", "get_preset", "asdict", "replace",
]
