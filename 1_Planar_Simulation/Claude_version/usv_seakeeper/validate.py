"""Physics validation against analytic limits.

Run with ``python -m usv_seakeeper.validate``. Every check has a closed-form
expected value, so a regression in the dynamics shows up as a failed
assertion rather than as an RL policy that mysteriously stops working.
"""
from __future__ import annotations

from dataclasses import replace
from typing import List, Tuple

import warnings

import numpy as np

# This suite deliberately probes far outside the linear-theory envelope
# (near-breaking seas, 1 m waves at a 1.2 s period) to check that the model
# degrades gracefully rather than diverging, so the validity warning is
# expected here and would only obscure the results.
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module=r"usv_seakeeper\.sim")

from .config import (G, WaveConfig, VesselConfig, SimConfig, SpeedTaskConfig)
from .waves import WaveField
from .sim import USVSim


class Check:
    def __init__(self):
        self.rows: List[Tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((name, bool(ok), detail))

    def close(self, name: str, got: float, want: float, rtol: float = 0.05,
              atol: float = 0.0) -> None:
        ok = abs(got - want) <= max(atol, rtol * abs(want))
        self.add(name, ok, f"got {got:.5g}, expected {want:.5g}")

    def report(self) -> bool:
        width = max(len(r[0]) for r in self.rows) + 2
        print("\n" + "=" * (width + 46))
        print("PHYSICS VALIDATION".center(width + 46))
        print("=" * (width + 46))
        for name, ok, detail in self.rows:
            print(f"{'PASS' if ok else 'FAIL'}  {name.ljust(width)}{detail}")
        n_fail = sum(1 for _, ok, _ in self.rows if not ok)
        print("-" * (width + 46))
        print(f"{len(self.rows) - n_fail}/{len(self.rows)} passed")
        return n_fail == 0


# --------------------------------------------------------------------------
def check_spectrum(c: Check) -> None:
    """4*sqrt(m0) must equal Hs, and a long time series must have std Hs/4."""
    cfg = WaveConfig(kind="jonswap", hs=2.5, tp=7.0, n_components=200, seed=3)
    f = WaveField(cfg)
    c.close("JONSWAP Hs from spectral moment", f.hs_realised(), 2.5, rtol=0.001)

    t = np.linspace(0.0, 4000.0, 40000)
    eta = np.array([float(f.query(np.array([0.0]), ti).eta[0]) for ti in t[:8000]])
    c.close("JONSWAP time-series std == Hs/4", float(np.std(eta)), 2.5 / 4.0,
            rtol=0.08)

    # frequency jitter should push recurrence far beyond a useful episode
    # Recurrence must be far longer than any episode (>= 20x the longest
    # preset) so a policy cannot memorise the wave train.
    c.add("frequency jitter breaks recurrence",
          f.recurrence_period() > 2000.0,
          f"recurrence {f.recurrence_period():.3g} s")
    f_nojit = WaveField(replace(cfg, frequency_jitter=False))
    c.add("uniform spacing does recur (control case)",
          f_nojit.recurrence_period() < 500.0,
          f"recurrence {f_nojit.recurrence_period():.0f} s")


def check_encounter_frequency(c: Check) -> None:
    """Head vs following sea must be the right way round.

    With the vessel steaming toward +x, a head sea raises the encounter
    frequency to w + |k|U and a following sea lowers it to w - |k|U. Getting
    the wavenumber sign backwards silently swaps the two, which is easy to
    miss and changes the control problem completely (following seas allow
    surf-riding; head seas do not).
    """
    T, U = 6.0, 3.0
    w = 2 * np.pi / T
    k = w * w / G

    def measured(heading: int) -> float:
        f = WaveField(WaveConfig(kind="regular", amplitude=1.0, period=T,
                                 heading=heading))
        # Record length sets the FFT bin width (dw = 2*pi/T_record); 600 s
        # gives ~0.010 rad/s, comfortably finer than the 2% tolerance.
        t = np.linspace(0.0, 600.0, 30000)
        eta = np.array([float(f.query(np.array([U * ti]), ti).eta[0])
                        for ti in t])
        spec = np.abs(np.fft.rfft(eta - eta.mean()))
        freq = np.fft.rfftfreq(t.size, t[1] - t[0])
        return float(2 * np.pi * freq[np.argmax(spec)])

    c.close("head sea encounter frequency = w + kU", measured(1), w + k * U,
            rtol=0.02)
    c.close("following sea encounter frequency = w - kU", measured(-1),
            w - k * U, rtol=0.02)


def check_dispersion(c: Check) -> None:
    """Regular wave analytics, checked for *both* headings against the
    field's own signed wavenumber."""
    for heading, name in ((1, "head"), (-1, "following")):
        f = WaveField(WaveConfig(kind="regular", amplitude=1.0, period=6.0,
                                 heading=heading, depth_stretching=False))
        w = 2 * np.pi / 6.0
        k = float(f.k[0])                 # signed: negative in a head sea
        sgn = np.sign(k)
        x, t = 12.3, 4.7
        th = k * x - w * t
        q = f.query(np.array([x]), t)
        c.close(f"[{name}] eta = A cos(kx-wt)", float(q.eta[0]),
                np.cos(th), atol=1e-9)
        c.close(f"[{name}] slope = -Ak sin(kx-wt)", float(q.slope[0]),
                -k * np.sin(th), atol=1e-9)
        c.close(f"[{name}] u = Aw sign(k) cos(kx-wt)", float(q.u[0]),
                w * sgn * np.cos(th), atol=1e-9)
        c.close(f"[{name}] w_vert = Aw sin(kx-wt) = deta/dt",
                float(q.w[0]), w * np.sin(th), atol=1e-9)
        # Linear-theory identity: -g*d(eta)/dx == du/dt when w^2 = g|k|.
        # Holds only if the sign(k) factor on u is present, so this is the
        # check that catches a reversed wave force in a following sea.
        c.close(f"[{name}] -g*deta/dx == du/dt (FK identity)",
                float(-G * q.slope[0]), float(q.du_dt[0]), atol=1e-9)


def check_length_averaging(c: Check) -> None:
    """sinc averaging must pass long waves and reject short ones."""
    long_wave = WaveField(WaveConfig(kind="regular", amplitude=1.0, period=12.0),
                          hull_length=2.6)
    q = long_wave.query(np.array([0.0]), 0.0)
    c.close("long wave: averaged eta ~ point eta",
            float(q.eta_avg[0]), float(q.eta[0]), rtol=0.02)

    short_wave = WaveField(WaveConfig(kind="regular", amplitude=1.0, period=1.2),
                           hull_length=8.0)
    q = short_wave.query(np.array([0.0]), 0.0)
    ratio = abs(float(q.eta_avg[0]) / max(abs(float(q.eta[0])), 1e-12))
    c.add("short wave strongly attenuated by hull averaging", ratio < 0.1,
          f"eta_avg/eta = {ratio:.4f}")


def check_calm_top_speed(c: Check) -> None:
    """Full thrust in calm water -> v = sqrt(Fmax/Cd)."""
    sim = USVSim(wave=WaveConfig(kind="calm"),
                 vessel=VesselConfig(thrust_max=500.0, drag_coeff=40.0),
                 sim=SimConfig(dof="surge", ventilation=False, max_time=400.0))
    sim.reset(seed=0)
    for _ in range(int(300 / sim.cfg.dt)):
        sim.step(1.0)
    c.close("calm-water top speed = sqrt(F/Cd)", sim.state.u,
            np.sqrt(500.0 / 40.0), rtol=0.01)


def check_stokes_drift(c: Check) -> None:
    """A freely floating hull in regular waves drifts at the Stokes rate
    A^2*w*k (deep water, second order)."""
    A, T = 0.4, 5.0
    w = 2 * np.pi / T
    k = w * w / G
    # Stokes drift is along the direction of wave propagation. The default
    # heading is a head sea, so the waves run toward -x and the drift is
    # negative -- which also pins down the heading sign convention.
    expected = -A * A * w * k
    # The surface-following model reduces to a Lagrangian fluid particle only
    # in the drag-dominated limit: the constraint/FK force scales with mass,
    # so a light, high-drag body tracks the orbital velocity and the mean of
    # u along its own path is the Stokes drift.
    sim = USVSim(wave=WaveConfig(kind="regular", amplitude=A, period=T),
                 vessel=VesselConfig(mass=2.0, drag_coeff=400.0,
                                     length=0.2, beam=0.2),
                 sim=SimConfig(dof="surge", ventilation=False,
                               fk_length_averaging=False, max_time=600.0))
    sim.reset(seed=0)
    n = int(500 / sim.cfg.dt)
    for _ in range(n):
        sim.step(0.0)
    drift = sim.state.x / sim.state.t
    c.close("free-floating drift ~ Stokes drift A^2*w*k", drift, expected,
            rtol=0.05)


def check_heave_rao(c: Check) -> None:
    """Heave RAO -> 1 for waves much longer than the natural period."""
    v = VesselConfig()
    tn = v.heave_natural_period()
    sim = USVSim(wave=WaveConfig(kind="regular", amplitude=0.5, period=10 * tn),
                 vessel=v, sim=SimConfig(dof="surge_heave_pitch",
                                         ventilation=False, max_time=400.0))
    sim.reset(seed=0)
    zs, etas = [], []
    for i in range(int(120 / sim.cfg.dt)):
        st = sim.step(0.0)
        if st.t > 60.0:
            zs.append(st.z)
            etas.append(st.wave_eta)
    rao = np.ptp(zs) / max(np.ptp(etas), 1e-9)
    c.close("heave RAO -> 1 in long waves", float(rao), 1.0, rtol=0.10)

    # short waves: response should be small
    sim2 = USVSim(wave=WaveConfig(kind="regular", amplitude=0.2, period=0.25 * tn),
                  vessel=v, sim=SimConfig(dof="surge_heave_pitch",
                                          ventilation=False, max_time=200.0))
    sim2.reset(seed=0)
    zs = []
    for _ in range(int(60 / sim2.cfg.dt)):
        st = sim2.step(0.0)
        if st.t > 30.0:
            zs.append(st.z)
    c.add("heave RAO << 1 in short waves", np.ptp(zs) < 0.2 * 0.4,
          f"z p2p = {np.ptp(zs):.4f} m vs wave p2p 0.4 m")


def check_integrator_order(c: Check) -> None:
    """RK4 convergence, plus the practical question: does the default
    ``dt_physics`` matter at all?

    The order is measured on a *smooth* configuration (actuator pre-loaded so
    the rate limiter never engages). With a rate-limited ramp the command has
    a derivative kink, which caps local order at 2 in the one sub-step that
    straddles it -- a real but tiny effect, quantified in the second check.
    """
    def run(dt_phys: float, prime: bool, action: float = 0.6) -> float:
        sim = USVSim(wave=WaveConfig(kind="regular", amplitude=0.9, period=4.0),
                     vessel=VesselConfig(slew_rate=2000.0),
                     sim=SimConfig(dof="surge_heave_pitch", dt=0.4,
                                   dt_physics=dt_phys, ventilation=False,
                                   max_time=100.0))
        sim.reset(seed=0)
        if prime:
            sim.state.thrust_cmd = action * sim.vessel.thrust_max
        for _ in range(25):
            sim.step(action)
        return sim.state.x

    ref = run(2e-5, prime=True)
    e1 = abs(run(0.05, True) - ref)
    e2 = abs(run(0.025, True) - ref)
    order = np.log2(e1 / max(e2, 1e-18))
    c.add("RK4 convergence order ~4 (smooth command)", order > 3.5,
          f"measured order {order:.2f}")

    # what a user actually cares about: is the default step fine enough?
    fine = run(0.0005, prime=False)
    default = run(0.005, prime=False)
    c.add("default dt_physics=0.005 is converged (<1 mm over 10 s)",
          abs(default - fine) < 1e-3,
          f"|dx| = {abs(default - fine)*1e3:.4f} mm")


def check_stretching_flag(c: Check) -> None:
    """Turning depth stretching off should halve the wave drift -- a direct
    check that the term contributing the second half of the Stokes drift is
    the one we think it is."""
    A, T = 0.4, 5.0
    w = 2 * np.pi / T
    k = w * w / G

    def drift(stretch: bool) -> float:
        sim = USVSim(wave=WaveConfig(kind="regular", amplitude=A, period=T,
                                     depth_stretching=stretch),
                     vessel=VesselConfig(mass=2.0, drag_coeff=400.0,
                                         length=0.2, beam=0.2),
                     sim=SimConfig(dof="surge", ventilation=False,
                                   fk_length_averaging=False, max_time=600.0))
        sim.reset(seed=0)
        for _ in range(int(500 / sim.cfg.dt)):
            sim.step(0.0)
        return sim.state.x / sim.state.t

    c.close("no stretching -> half the Stokes drift",
            drift(False), -0.5 * A * A * w * k, rtol=0.25)


def check_hull_emergence(c: Check) -> None:
    """A hull thrown clear of the water must free-fall, not be catapulted.

    The linear heave restoring -C33*(z - eta) is exact only while the hull is
    immersed. Applied unclamped it reaches several times the vessel weight
    once the hull is airborne, and together with ungated radiation damping it
    launches the vessel many metres up in a steep sea. Restoring saturates at
    the weight; damping and added mass are scaled by the wetted fraction.
    """
    v = VesselConfig(mass=400.0, thrust_max=900.0, drag_coeff=50.0)

    def run(tp: float, seed: int = 2):
        sim = USVSim(wave=WaveConfig(kind="jonswap", hs=2.0, tp=tp, seed=seed),
                     vessel=v, sim=SimConfig(max_time=60.0))
        sim.reset(seed=seed)
        rise, speeds = [], []
        for _ in range(int(50 / sim.cfg.dt)):
            st = sim.step(0.5)
            rise.append(st.z - st.wave_eta)
            speeds.append(st.u)
        return np.array(rise), np.array(speeds)

    # Moderate sea, comfortably inside linear-theory validity: the hull should
    # essentially track the surface, emerging only marginally.
    rise, speeds = run(6.0)
    c.add("hull tracks the surface in a moderate sea",
          rise.max() < 3.0 * v.emergence_limit(),
          f"max rise {rise.max():.3f} m vs emergence limit "
          f"{v.emergence_limit():.3f} m")

    # Near-breaking sea. Being thrown clear of the water is real here, so the
    # bound is against divergence rather than against emergence: the motion
    # must stay bounded, the hull must come back down, and the surge must stay
    # within a few times the calm-water top speed.
    rise, speeds = run(3.0)
    v_max = np.sqrt(900.0 / 50.0)
    c.add("steep sea: hull motion bounded, no runaway",
          rise.max() < 4.0 and abs(rise[-50:]).mean() < 1.0
          and np.all(np.isfinite(rise)),
          f"max rise {rise.max():.2f} m, mean |rise| at end "
          f"{abs(rise[-50:]).mean():.3f} m")
    c.add("steep sea: surge stays within a few times v_max",
          np.abs(speeds).max() < 3.0 * v_max,
          f"max |u| = {np.abs(speeds).max():.2f} m/s vs v_max {v_max:.2f}")

    # A fully emerged hull must accelerate downward at exactly -g: restoring
    # saturates at the weight and added mass is gated off.
    calm = USVSim(wave=WaveConfig(kind="calm"), vessel=v,
                  sim=SimConfig(ventilation=False, max_time=10.0))
    calm.reset(seed=0)
    calm.state.z = 5.0 * v.emergence_limit()      # well clear of the water
    calm.state.w = 0.0
    y = calm.state.as_array()
    dy, _, _ = calm._derivatives(y, 0.0, 0.0)
    c.close("fully emerged hull free-falls at -g", float(dy[3]), -G, rtol=0.01)

    # and at rest on calm water it must be in equilibrium
    calm.reset(seed=0)
    dy, _, _ = calm._derivatives(calm.state.as_array(), 0.0, 0.0)
    c.close("hull at rest on calm water is in vertical equilibrium",
            float(dy[3]), 0.0, atol=1e-9)


def check_ventilation(c: Check) -> None:
    """The prop must lose immersion in a steep sea and not in a calm one."""
    v = VesselConfig(prop_depth=0.15, prop_diameter=0.15)
    calm = USVSim(wave=WaveConfig(kind="regular", amplitude=0.05, period=5.0),
                  vessel=v, sim=SimConfig(max_time=60.0))
    calm.reset(seed=0)
    vents = [calm.step(0.6).ventilation for _ in range(int(40 / calm.cfg.dt))]
    c.add("no ventilation in near-calm water", min(vents) > 0.999,
          f"min immersion {min(vents):.4f}")

    rough = USVSim(wave=WaveConfig(kind="jonswap", hs=3.0, tp=6.5, seed=5),
                   vessel=v, sim=SimConfig(max_time=120.0))
    rough.reset(seed=0)
    vents = [rough.step(0.8).ventilation for _ in range(int(100 / rough.cfg.dt))]
    c.add("ventilation occurs in a steep sea", min(vents) < 0.5,
          f"min immersion {min(vents):.3f}")


def check_determinism(c: Check) -> None:
    """Same seed -> identical trajectory (required for fair comparisons)."""
    def run(seed: int) -> float:
        sim = USVSim(wave=WaveConfig(kind="jonswap", hs=2.0, tp=6.0),
                     sim=SimConfig(max_time=30.0))
        sim.reset(seed=seed)
        for _ in range(300):
            sim.step(0.5)
        return sim.state.x
    c.add("reset(seed) is reproducible", run(11) == run(11),
          f"x = {run(11):.6f}")
    c.add("different seeds give different seas", run(11) != run(12))


def check_energy_sanity(c: Check) -> None:
    """Zero thrust, calm water, initial speed -> monotone decay to zero."""
    sim = USVSim(wave=WaveConfig(kind="calm"), vessel=VesselConfig(),
                 sim=SimConfig(dof="surge", ventilation=False, max_time=200.0))
    v0, T = 5.0, 150.0
    sim.reset(seed=0, initial_speed=v0)
    speeds = [sim.step(0.0).u for _ in range(int(T / sim.cfg.dt))]
    monotone = all(b <= a + 1e-9 for a, b in zip(speeds, speeds[1:]))
    c.add("quadratic drag decay is monotonic", monotone)
    # M dv/dt = -Cd v^2  ->  v(t) = v0 / (1 + Cd*v0*t/M).  Note this is a
    # 1/t decay, so the speed never actually reaches zero -- that is correct
    # physics for quadratic drag, not a bug.
    M = sim.vessel.mass * (1.0 + sim.vessel.added_mass_surge)
    analytic = v0 / (1.0 + sim.vessel.drag_coeff * v0 * T / M)
    c.close("quadratic drag matches analytic 1/t decay", speeds[-1], analytic,
            rtol=0.002)


def check_speed_limit_claim(c: Check) -> None:
    """A target above the calm-water top speed must be unreachable -- the
    actuator-sizing lesson from the original tool, now as an assertion."""
    from .envs import SpeedHoldEnv
    from .controllers import PIDSpeedController, PIDGains
    from .rollout import rollout

    env = SpeedHoldEnv(task=SpeedTaskConfig(target_speed=10.0),
                       wave=WaveConfig(kind="calm"),
                       vessel=VesselConfig(thrust_max=500.0, drag_coeff=40.0),
                       sim=SimConfig(dof="surge", ventilation=False, max_time=120.0))
    res = rollout(env, PIDSpeedController(PIDGains(1e4, 1e3, 0.0)), seed=0)
    v_max = np.sqrt(500.0 / 40.0)
    c.add("unreachable setpoint saturates and plateaus at v_max",
          res.metrics["saturated_frac"] > 0.9 and abs(res.u[-1] - v_max) < 0.1,
          f"final u {res.u[-1]:.3f} vs v_max {v_max:.3f}, "
          f"sat {res.metrics['saturated_frac']:.0%}")


def main() -> bool:
    c = Check()
    check_dispersion(c)
    check_encounter_frequency(c)
    check_spectrum(c)
    check_length_averaging(c)
    check_calm_top_speed(c)
    check_energy_sanity(c)
    check_stokes_drift(c)
    check_heave_rao(c)
    check_integrator_order(c)
    check_stretching_flag(c)
    check_hull_emergence(c)
    check_ventilation(c)
    check_determinism(c)
    check_speed_limit_claim(c)
    return c.report()


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
