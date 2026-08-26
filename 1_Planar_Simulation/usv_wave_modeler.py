#!/usr/bin/env python3
"""
USV Sea-Keeper — 1D wave & boat control modeler
================================================

A 1-DOF (surge) simulation bench for designing the controller that drives an
unmanned surface vehicle (USV) *over* waves.

Physics
-------
The boat is a surge body that rides the free surface (z = eta, pitch = atan(deta/dx)).
The disturbance the controller must beat is the along-slope component of gravity:

    (m + m_a) v_dot = F_thrust  -  C_d (v - u) |v - u|  -  m g sin(theta) cos(theta)  +  F_wave

  theta   = atan(deta/dx)                 local wave slope angle
  u       = horizontal orbital velocity   (deep-water, at the surface)
  F_wave  = m * Ca * u_dot                 Morison inertia / Froude-Krylov (optional)

Waves use deep-water dispersion (omega^2 = g k). Irregular seas are a JONSWAP
spectrum (gamma = 3.3) scaled to a chosen significant height Hs and peak period Tp.
Integration is RK4 with the commanded thrust held across each sub-step; the thruster
has a slew-rate limit and the PID has conditional-integration anti-windup.

Usage
-----
    python usv_wave_modeler.py                 # interactive GUI (needs a display)
    python usv_wave_modeler.py --headless      # run FF off/on comparison, save PNG

Or import the core and script your own controller studies:

    from usv_wave_modeler import WaveField, Vessel, USVSimulator, run_sim
    t, log = run_sim(WaveField.irregular(hs=1.5, tp=6),
                     Vessel(mass=350, fmax=600),
                     mode="speed", vref=2.0, kp=260, ki=50, kd=30, ff=True,
                     duration=40)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

G = 9.81          # gravity                      [m/s^2]
RHO = 1025.0      # seawater density             [kg/m^3]  (kept for reference)
CA = 0.15         # surge added-mass coefficient [-]


# ======================================================================
# WAVE FIELD
# ======================================================================
@dataclass
class WaveField:
    """A superposition of Airy wave components. Works for scalar or array x."""
    A: np.ndarray     # amplitudes            [m]
    w: np.ndarray     # angular frequencies   [rad/s]
    k: np.ndarray     # wavenumbers           [1/m]
    phi: np.ndarray   # phases                [rad]
    label: str = "regular"

    # ---- constructors ------------------------------------------------
    @classmethod
    def regular(cls, amp: float, period: float) -> "WaveField":
        w = 2 * np.pi / period
        k = w * w / G
        return cls(np.array([amp]), np.array([w]), np.array([k]),
                   np.array([0.0]), label="regular")

    @classmethod
    def irregular(cls, hs: float, tp: float, n: int = 26, seed: int = 1) -> "WaveField":
        """JONSWAP (gamma = 3.3) discretised into `n` components."""
        wp = 2 * np.pi / tp
        wmin, wmax = 0.35 * wp, 3.0 * wp
        edges = np.linspace(wmin, wmax, n + 1)
        w = 0.5 * (edges[:-1] + edges[1:])
        dw = edges[1] - edges[0]
        sig = np.where(w <= wp, 0.07, 0.09)
        r = np.exp(-((w - wp) ** 2) / (2 * sig ** 2 * wp ** 2))
        Sshape = (wp ** 4) / (w ** 5) * np.exp(-1.25 * (wp / w) ** 4) * np.power(3.3, r)
        m0 = np.sum(Sshape * dw)
        scale = (hs * hs) / (16.0 * m0)          # so 4*sqrt(sum S dw) == Hs
        A = np.sqrt(2.0 * Sshape * scale * dw)
        k = w * w / G
        rng = np.random.default_rng(seed)
        phi = rng.uniform(0, 2 * np.pi, size=n)
        return cls(A, w, k, phi, label="irregular")

    # ---- field queries (x scalar or 1-D array) -----------------------
    def _phase(self, x, t):
        x = np.asarray(x, dtype=float)
        # shape (Ncomp, Nx)
        return self.k[:, None] * x.reshape(-1)[None, :] - self.w[:, None] * t + self.phi[:, None], x

    def elevation(self, x, t):
        ph, x = self._phase(x, t)
        e = (self.A[:, None] * np.cos(ph)).sum(0)
        return float(e[0]) if x.ndim == 0 else e.reshape(x.shape)

    def slope(self, x, t):
        ph, x = self._phase(x, t)
        s = (-self.A[:, None] * self.k[:, None] * np.sin(ph)).sum(0)
        return float(s[0]) if x.ndim == 0 else s.reshape(x.shape)

    def orbital_u(self, x, t):
        ph, x = self._phase(x, t)
        u = (self.A[:, None] * self.w[:, None] * np.cos(ph)).sum(0)
        return float(u[0]) if x.ndim == 0 else u.reshape(x.shape)

    def orbital_udot(self, x, t):
        ph, x = self._phase(x, t)
        a = (self.A[:, None] * self.w[:, None] ** 2 * np.sin(ph)).sum(0)
        return float(a[0]) if x.ndim == 0 else a.reshape(x.shape)

    # ---- diagnostics -------------------------------------------------
    @property
    def hs(self) -> float:
        return 4.0 * np.sqrt(np.sum(0.5 * self.A ** 2))

    @property
    def max_slope(self) -> float:
        return float(np.sum(self.A * self.k))


# ======================================================================
# VESSEL
# ======================================================================
@dataclass
class Vessel:
    mass: float = 350.0     # dry mass                 [kg]
    fmax: float = 500.0     # max |thrust|             [N]
    drag: float = 40.0      # quadratic drag coeff     [N s^2 / m^2]
    slew: float = 2000.0    # thruster slew-rate limit [N/s]
    morison: bool = True    # include wave orbital coupling


# ======================================================================
# CONTROLLER
# ======================================================================
@dataclass
class PID:
    kp: float = 200.0
    ki: float = 40.0
    kd: float = 20.0
    ff: bool = False        # slope feedforward: cancel m g sin(theta)
    _integ: float = 0.0
    _prev_err: float = 0.0

    def reset(self):
        self._integ = 0.0
        self._prev_err = 0.0

    def command(self, err, dt, fmax, grav_load):
        raw_i = self._integ + err * dt
        d_err = (err - self._prev_err) / max(dt, 1e-9)
        self._prev_err = err
        u = self.kp * err + self.ki * raw_i + self.kd * d_err
        if self.ff:
            u += grav_load
        sat = float(np.clip(u, -fmax, fmax))
        # conditional integration anti-windup: only integrate if not driving
        # further into the saturated direction.
        windup = (u != sat) and (np.sign(err) == np.sign(u))
        if not windup:
            self._integ = raw_i
        return sat


# ======================================================================
# SIMULATOR
# ======================================================================
@dataclass
class USVSimulator:
    wave: WaveField
    vessel: Vessel
    pid: PID = field(default_factory=PID)
    mode: str = "speed"       # "manual" | "speed" | "position"
    manual: float = 0.0       # manual thrust command, fraction [-1, 1]
    vref: float = 2.0         # target speed     [m/s]
    xref: float = 60.0        # target position  [m]

    # --- state ---
    t: float = 0.0
    x: float = 0.0
    v: float = 0.0
    thrust: float = 0.0

    # --- accumulators / metrics ---
    effort: float = 0.0
    _err_sq: float = 0.0
    _err_n: int = 0
    crests: int = 0
    _last_slope: float = 0.0
    reached: bool = False
    reached_t: float = 0.0

    def reset(self):
        self.t = self.x = self.v = self.thrust = 0.0
        self.effort = self._err_sq = 0.0
        self._err_n = 0
        self.crests = 0
        self._last_slope = 0.0
        self.reached = False
        self.reached_t = 0.0
        self.pid.reset()

    # ---- forcing ----------------------------------------------------
    def _grav_load(self, x, t):
        """Horizontal component of gravity along the wave slope (the load)."""
        theta = np.arctan(self.wave.slope(x, t))
        return self.vessel.mass * G * np.sin(theta) * np.cos(theta)

    def _accel(self, x, v, t):
        M = self.vessel.mass * (1 + CA)
        gx = -self._grav_load(x, t)
        if self.vessel.morison:
            u = self.wave.orbital_u(x, t)
            udot = self.wave.orbital_udot(x, t)
        else:
            u = udot = 0.0
        vrel = v - u
        f_drag = -self.vessel.drag * vrel * abs(vrel)
        f_inertia = self.vessel.mass * CA * udot if self.vessel.morison else 0.0
        return (self.thrust + gx + f_drag + f_inertia) / M

    # ---- controller -------------------------------------------------
    def _command(self, dt):
        if self.mode == "manual":
            return self.manual * self.vessel.fmax
        if self.mode == "position":
            err = self.xref - self.x
        else:                          # speed
            err = self.vref - self.v
        return self.pid.command(err, dt, self.vessel.fmax, self._grav_load(self.x, self.t))

    # ---- one integration step ---------------------------------------
    def step(self, dt: float):
        # commanded thrust through slew-rate limiter
        cmd = self._command(dt)
        max_step = self.vessel.slew * dt
        self.thrust += float(np.clip(cmd - self.thrust, -max_step, max_step))

        # RK4 on (x, v); thrust held constant across the sub-step
        x, v, t = self.x, self.v, self.t
        k1 = self._accel(x, v, t)
        k2 = self._accel(x + 0.5 * dt * v, v + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self._accel(x + 0.5 * dt * (v + 0.5 * dt * k1), v + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self._accel(x + dt * (v + 0.5 * dt * k2), v + dt * k3, t + dt)
        dv = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.v = v + dv
        self.x = x + (v + 0.5 * dv) * dt
        self.t = t + dt

        # metrics
        self.effort += abs(self.thrust) * dt
        if self.mode != "manual":
            e = (self.xref - self.x) if self.mode == "position" else (self.vref - self.v)
            self._err_sq += e * e
            self._err_n += 1
        sl = self.wave.slope(self.x, self.t)
        if self._last_slope > 0 and sl <= 0 and self.v > 0.05:
            self.crests += 1
        self._last_slope = sl
        if (self.mode == "position" and not self.reached
                and abs(self.xref - self.x) < 0.5 and abs(self.v) < 0.15):
            self.reached = True
            self.reached_t = self.t

    # ---- derived readouts -------------------------------------------
    @property
    def rms_err(self) -> float:
        return float(np.sqrt(self._err_sq / self._err_n)) if self._err_n else 0.0

    @property
    def saturated(self) -> bool:
        return abs(self.thrust) >= self.vessel.fmax - 1.0

    def verdict(self) -> str:
        if self.mode == "manual":
            return "OPEN LOOP"
        if self.mode == "position":
            if self.reached:
                return "ARRIVED"
            return "STALLED" if (self.saturated and self.v < 0.05) else "TRANSIT"
        # speed
        if self.vref > 0.1 and self.v < 0.05 and self.saturated:
            return "STALLED — Fmax too low"
        if abs(self.vref - self.v) < 0.2:
            return "ON SETPOINT"
        return "THRUST SATURATED" if self.saturated else "SETTLING"


# ======================================================================
# BATCH RUNNER (headless studies)
# ======================================================================
def run_sim(wave: WaveField, vessel: Vessel, *, mode="speed", manual=0.0,
            vref=2.0, xref=60.0, kp=200, ki=40, kd=20, ff=False,
            duration=40.0, dt=0.004, sample_dt=0.02):
    """Run a simulation and return (time_array, log_dict) sampled at sample_dt."""
    sim = USVSimulator(wave, vessel, PID(kp, ki, kd, ff),
                       mode=mode, manual=manual, vref=vref, xref=xref)
    sim.reset()
    n = int(round(duration / dt))
    every = max(1, int(round(sample_dt / dt)))
    log = {k: [] for k in ("t", "x", "v", "thrust", "slope", "vref", "xref_err")}
    for i in range(n):
        sim.step(dt)
        if i % every == 0:
            log["t"].append(sim.t)
            log["x"].append(sim.x)
            log["v"].append(sim.v)
            log["thrust"].append(sim.thrust)
            log["slope"].append(sim.wave.slope(sim.x, sim.t))
            log["vref"].append(vref if mode == "speed" else np.nan)
            log["xref_err"].append((xref - sim.x) if mode == "position" else np.nan)
    return np.array(log["t"]), {k: np.array(v) for k, v in log.items()}, sim


# ======================================================================
# PRESETS
# ======================================================================
PRESETS = {
    "harbor":  dict(sea="regular", amp=0.25, per=3.5, mass=250, fmax=400, drag=35,
                    slew=2500, mode="speed", vref=2.5, kp=180, ki=35, kd=15, ff=False, morison=True),
    "chop":    dict(sea="irregular", hs=1.2, tp=6, mass=350, fmax=600, drag=45,
                    slew=2000, mode="speed", vref=2.0, kp=260, ki=50, kd=30, ff=True, morison=True),
    "storm":   dict(sea="irregular", hs=3.2, tp=8, mass=600, fmax=1400, drag=70,
                    slew=3000, mode="speed", vref=1.5, kp=420, ki=70, kd=60, ff=True, morison=True),
    "station": dict(sea="irregular", hs=1.0, tp=5, mass=350, fmax=700, drag=45,
                    slew=2500, mode="position", xref=40, kp=120, ki=8, kd=180, ff=True, morison=True),
}


def _wave_from(cfg) -> WaveField:
    if cfg["sea"] == "regular":
        return WaveField.regular(cfg["amp"], cfg["per"])
    return WaveField.irregular(cfg["hs"], cfg["tp"], seed=cfg.get("seed", 1))


# ======================================================================
# HEADLESS COMPARISON
# ======================================================================
def headless_demo(outfile="usv_ff_comparison.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wave = WaveField.irregular(hs=1.4, tp=6, seed=7)
    vessel = Vessel(mass=350, fmax=600, drag=45, slew=2000, morison=True)
    common = dict(mode="speed", vref=2.0, kp=260, ki=50, kd=30, duration=45)

    t_off, log_off, s_off = run_sim(wave, vessel, ff=False, **common)
    t_on,  log_on,  s_on  = run_sim(wave, vessel, ff=True,  **common)

    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig.suptitle("Slope feedforward off vs on  ·  Hs=1.4 m, Tp=6 s, target 2.0 m/s",
                 fontsize=13, fontweight="bold")

    ax[0].axhline(2.0, ls="--", lw=1, color="#8fb8ff", label="setpoint")
    ax[0].plot(t_off, log_off["v"], color="#c0554e", label="PID only")
    ax[0].plot(t_on,  log_on["v"],  color="#1f9e8f", label="PID + feedforward")
    ax[0].set_ylabel("speed  [m/s]"); ax[0].legend(loc="lower right", fontsize=9)

    ax[1].axhline(vessel.fmax, ls=":", lw=1, color="#999")
    ax[1].axhline(-vessel.fmax, ls=":", lw=1, color="#999")
    ax[1].plot(t_off, log_off["thrust"], color="#c0554e")
    ax[1].plot(t_on,  log_on["thrust"],  color="#1f9e8f")
    ax[1].set_ylabel("thrust  [N]")

    ax[2].plot(t_off, log_off["slope"], color="#5aa0c0")
    ax[2].set_ylabel("wave slope"); ax[2].set_xlabel("time  [s]")

    for a in ax:
        a.grid(alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outfile, dpi=130)
    print(f"saved {outfile}")
    print(f"  PID only        : RMS speed err = {s_off.rms_err:.3f} m/s, "
          f"distance = {s_off.x:6.1f} m, effort = {s_off.effort/1000:5.1f} kN·s")
    print(f"  PID + feedforward: RMS speed err = {s_on.rms_err:.3f} m/s, "
          f"distance = {s_on.x:6.1f} m, effort = {s_on.effort/1000:5.1f} kN·s")
    return outfile


# ======================================================================
# INTERACTIVE GUI
# ======================================================================
def launch_gui():
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
    from matplotlib.patches import Polygon

    BG, PANEL, SIG, TGT, WARN = "#04121a", "#0d2c3c", "#33e6cf", "#8fb8ff", "#ffb454"
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": "#052029",
        "text.color": "#dbf3f2", "axes.labelcolor": "#9fc4c9",
        "xtick.color": "#7aa6ad", "ytick.color": "#7aa6ad",
        "axes.edgecolor": "#123a4c", "font.size": 9,
    })

    # ---- live parameter/state holders ----
    # Seed from a COMPLETE default set, then overlay the starting preset, so a
    # missing key (e.g. a speed-mode preset has no xref) can never KeyError.
    defaults = dict(sea="regular", amp=0.5, per=4.0, hs=1.2, tp=6.0, seed=1,
                    mass=350, fmax=500, drag=40, slew=2000, morison=True,
                    mode="speed", manual=0, vref=2.0, xref=60,
                    kp=200, ki=40, kd=20, ff=False)
    cfg = {**defaults, **PRESETS["chop"]}; cfg["seed"] = 1
    wave = [_wave_from(cfg)]
    sim = USVSimulator(wave[0], Vessel(cfg["mass"], cfg["fmax"], cfg["drag"],
                                       cfg["slew"], cfg["morison"]),
                       PID(cfg["kp"], cfg["ki"], cfg["kd"], cfg["ff"]),
                       mode=cfg["mode"], vref=cfg["vref"], xref=cfg["xref"])
    sim.reset()
    running = [False]

    WIN = 20.0        # telemetry window [s]
    VIEW = 34.0       # scope width      [m]
    hist = {k: [] for k in ("t", "v", "vref", "thrust", "slope", "xerr")}

    fig = plt.figure(figsize=(14.5, 8.6))
    fig.canvas.manager.set_window_title("USV Sea-Keeper — Python")

    # ---- axes layout ----
    ax_sea = fig.add_axes([0.34, 0.60, 0.63, 0.36])
    ax_v   = fig.add_axes([0.34, 0.42, 0.63, 0.145])
    ax_thr = fig.add_axes([0.34, 0.25, 0.63, 0.145])
    ax_slp = fig.add_axes([0.34, 0.075, 0.63, 0.145])

    for a in (ax_sea, ax_v, ax_thr, ax_slp):
        a.tick_params(labelsize=8)

    # scope artists
    ax_sea.set_ylim(-5, 5); ax_sea.set_yticks([])
    ax_sea.set_title("Waterline scope", color="#9fc4c9", fontsize=10, loc="left")
    xs = np.linspace(0, VIEW, 240)
    (crest_line,) = ax_sea.plot([], [], color=SIG, lw=2)
    water_fill = [None]
    hull_patch = Polygon(np.zeros((4, 2)), closed=True, fc="#f4f2ea", ec="#cf9a4a", lw=1.5, zorder=5)
    ax_sea.add_patch(hull_patch)
    (mast_line,) = ax_sea.plot([], [], color="#12303c", lw=6, solid_capstyle="round", zorder=6)
    (sensor_pt,) = ax_sea.plot([], [], marker="o", ms=6, color=SIG, zorder=7)
    (thrust_line,) = ax_sea.plot([], [], color=SIG, lw=3, marker=">", ms=8, zorder=8)
    tgt_line = ax_sea.axvline(np.nan, ls="--", color=TGT, lw=1.4)
    overlay = ax_sea.text(0.012, 0.96, "", transform=ax_sea.transAxes, va="top",
                          family="monospace", fontsize=9, color=SIG)
    verdict_txt = ax_sea.text(0.985, 0.96, "", transform=ax_sea.transAxes, va="top",
                              ha="right", family="monospace", fontsize=10, fontweight="bold")
    metrics_txt = ax_sea.text(0.985, 0.05, "", transform=ax_sea.transAxes, va="bottom",
                              ha="right", family="monospace", fontsize=8.5, color="#9fc4c9")

    # telemetry artists
    (l_vref,) = ax_v.plot([], [], color=TGT, lw=1.3, ls="--")
    (l_v,)    = ax_v.plot([], [], color=SIG, lw=2)
    ax_v.set_ylabel("v [m/s]", fontsize=8)
    (l_thr,)  = ax_thr.plot([], [], color=WARN, lw=2)
    ax_thr.set_ylabel("F [N]", fontsize=8)
    (l_slp,)  = ax_slp.plot([], [], color="#5ad1ff", lw=1.6)
    ax_slp.set_ylabel("slope / err", fontsize=8); ax_slp.set_xlabel("time [s]", fontsize=8)
    for a in (ax_v, ax_thr, ax_slp):
        a.grid(alpha=0.15)

    # ================= CONTROL WIDGETS (left column) =================
    def sax(x, y, w, h):
        a = fig.add_axes([x, y, w, h]); a.set_facecolor(PANEL); return a

    # radios
    rax_sea = sax(0.035, 0.885, 0.12, 0.075)
    radio_sea = RadioButtons(rax_sea, ("regular", "irregular"),
                             active=(0 if cfg["sea"] == "regular" else 1))
    rax_sea.set_title("sea", fontsize=8, color="#9fc4c9")
    rax_mode = sax(0.175, 0.885, 0.13, 0.075)
    radio_mode = RadioButtons(rax_mode, ("manual", "speed", "position"),
                              active=("manual", "speed", "position").index(cfg["mode"]))
    rax_mode.set_title("controller", fontsize=8, color="#9fc4c9")

    # sliders
    sliders = {}
    defs = [
        ("amp", "amp [m]", 0.0, 2.0, cfg.get("amp", 0.5)),
        ("per", "period [s]", 2.0, 12.0, cfg.get("per", 4.0)),
        ("hs", "Hs [m]", 0.2, 5.0, cfg.get("hs", 1.2)),
        ("tp", "Tp [s]", 3.0, 15.0, cfg.get("tp", 6.0)),
        ("mass", "mass [kg]", 80, 1500, cfg["mass"]),
        ("fmax", "Fmax [N]", 50, 2500, cfg["fmax"]),
        ("drag", "drag", 2, 200, cfg["drag"]),
        ("slew", "slew [N/s]", 100, 8000, cfg["slew"]),
        ("vref", "v_ref [m/s]", 0, 6, cfg.get("vref", 2.0)),
        ("xref", "x_ref [m]", 10, 200, cfg.get("xref", 60)),
        ("kp", "Kp", 0, 1200, cfg["kp"]),
        ("ki", "Ki", 0, 400, cfg["ki"]),
        ("kd", "Kd", 0, 400, cfg["kd"]),
        ("manual", "manual [%]", -100, 100, 0),
    ]
    y = 0.83
    for key, lbl, lo, hi, val in defs:
        a = sax(0.11, y, 0.19, 0.022)
        sliders[key] = Slider(a, lbl, lo, hi, valinit=val, color=SIG)
        sliders[key].label.set_fontsize(8)
        sliders[key].valtext.set_fontsize(8)
        y -= 0.036

    # checks
    cax = sax(0.035, y - 0.02, 0.27, 0.05)
    checks = CheckButtons(cax, ["feedforward", "wave coupling"],
                          [cfg["ff"], cfg["morison"]])

    # buttons
    def bax(x, w, yy): return fig.add_axes([x, yy, w, 0.03])
    yb = y - 0.10
    b_run = Button(bax(0.035, 0.085, yb), "Run", color=SIG, hovercolor="#5ff0dd")
    b_run.label.set_color("#04121a")
    b_reset = Button(bax(0.128, 0.075, yb), "Reset", color=PANEL, hovercolor="#124")
    b_seed = Button(bax(0.208, 0.095, yb), "New sea", color=PANEL, hovercolor="#124")
    # presets
    yb2 = yb - 0.045
    pbtns = {}
    for i, (name, _) in enumerate(PRESETS.items()):
        pbtns[name] = Button(fig.add_axes([0.035 + i * 0.068, yb2, 0.064, 0.03]),
                             name, color=PANEL, hovercolor="#124")
        pbtns[name].label.set_fontsize(7.5)
    fig.text(0.035, yb2 + 0.036, "scenarios", fontsize=8, color="#7aa6ad")

    # ================= widget → model wiring =================
    def rebuild_wave():
        if cfg["sea"] == "regular":
            wave[0] = WaveField.regular(sliders["amp"].val, sliders["per"].val)
        else:
            wave[0] = WaveField.irregular(sliders["hs"].val, sliders["tp"].val,
                                          seed=cfg["seed"])
        sim.wave = wave[0]

    def apply_params():
        sim.vessel.mass = sliders["mass"].val
        sim.vessel.fmax = sliders["fmax"].val
        sim.vessel.drag = sliders["drag"].val
        sim.vessel.slew = sliders["slew"].val
        sim.pid.kp = sliders["kp"].val
        sim.pid.ki = sliders["ki"].val
        sim.pid.kd = sliders["kd"].val
        sim.vref = sliders["vref"].val
        sim.xref = sliders["xref"].val
        sim.manual = sliders["manual"].val / 100.0
        rebuild_wave()

    for s in sliders.values():
        s.on_changed(lambda _val: apply_params())

    def on_sea(label):
        cfg["sea"] = label; rebuild_wave()
    radio_sea.on_clicked(on_sea)

    def on_mode(label):
        cfg["mode"] = label; sim.mode = label
        sim.pid.reset(); sim.reached = False
    radio_mode.on_clicked(on_mode)

    def on_check(label):
        st = dict(zip(["feedforward", "wave coupling"], checks.get_status()))
        sim.pid.ff = st["feedforward"]
        sim.vessel.morison = st["wave coupling"]
    checks.on_clicked(on_check)

    def do_reset(_=None):
        sim.reset()
        for k in hist: hist[k].clear()
    b_reset.on_clicked(do_reset)

    def do_run(_=None):
        running[0] = not running[0]
        b_run.label.set_text("Pause" if running[0] else "Run")
    b_run.on_clicked(do_run)

    def do_seed(_=None):
        cfg["seed"] = int(np.random.randint(1, 10 ** 6)); rebuild_wave()
    b_seed.on_clicked(do_seed)

    def make_preset(name):
        def _cb(_=None):
            p = dict(PRESETS[name]); cfg.update(p); cfg.setdefault("seed", 1)
            # radios
            radio_sea.set_active(0 if p["sea"] == "regular" else 1)
            radio_mode.set_active(("manual", "speed", "position").index(p["mode"]))
            sim.mode = p["mode"]
            # checks
            cur = dict(zip(["feedforward", "wave coupling"], checks.get_status()))
            if cur["feedforward"] != p["ff"]: checks.set_active(0)
            if cur["wave coupling"] != p["morison"]: checks.set_active(1)
            # sliders (set_val triggers apply_params)
            for key in ("amp", "per", "hs", "tp", "mass", "fmax", "drag", "slew",
                        "vref", "xref", "kp", "ki", "kd"):
                if key in p:
                    sliders[key].set_val(p[key])
            apply_params(); do_reset()
            if not running[0]: do_run()
        return _cb
    for name in PRESETS:
        pbtns[name].on_clicked(make_preset(name))

    # ================= animation loop =================
    DT = 0.004
    FRAME_DT = 0.02
    hull_local = np.array([[-1.3, -0.13], [1.09, -0.13], [1.3, -0.6], [-1.3, -0.6]])

    def push_hist():
        hist["t"].append(sim.t); hist["v"].append(sim.v)
        hist["vref"].append(sim.vref if sim.mode == "speed" else np.nan)
        hist["thrust"].append(sim.thrust)
        hist["slope"].append(sim.wave.slope(sim.x, sim.t))
        hist["xerr"].append((sim.xref - sim.x) if sim.mode == "position" else np.nan)
        while hist["t"] and hist["t"][0] < sim.t - WIN:
            for k in hist: hist[k].pop(0)

    def update(_frame):
        if running[0]:
            steps = int(round(FRAME_DT / DT))
            for _ in range(steps):
                sim.step(DT)
            push_hist()

        # -- scope --
        cam = sim.x
        ax_sea.set_xlim(cam - VIEW * 0.4, cam + VIEW * 0.6)
        xw = np.linspace(cam - VIEW * 0.4, cam + VIEW * 0.6, 240)
        eta = sim.wave.elevation(xw, sim.t)
        crest_line.set_data(xw, eta)
        if water_fill[0] is not None:
            water_fill[0].remove()
        water_fill[0] = ax_sea.fill_between(xw, eta, -5, color="#0a3547", zorder=1)

        slope = sim.wave.slope(sim.x, sim.t)
        pitch = np.arctan(slope)
        z = sim.wave.elevation(sim.x, sim.t)
        c, s = np.cos(pitch), np.sin(pitch)
        R = np.array([[c, -s], [s, c]])
        verts = (hull_local @ R.T) + np.array([sim.x, z])
        hull_patch.set_xy(verts)
        mast_top = np.array([0, -0.6]) @ R.T + [sim.x, z]
        mast_hi = np.array([0, -1.9]) @ R.T + [sim.x, z]
        mast_line.set_data([mast_top[0], mast_hi[0]], [mast_top[1], mast_hi[1]])
        sensor_pt.set_data([mast_hi[0]], [mast_hi[1]])

        thr_px = (sim.thrust / sim.vessel.fmax) * 4.0
        ty = z - 0.35
        thrust_line.set_data([sim.x, sim.x + thr_px], [ty, ty])
        thrust_line.set_color(WARN if sim.saturated else SIG)
        thrust_line.set_marker(">" if thr_px >= 0 else "<")

        tgt_line.set_xdata([sim.xref, sim.xref] if sim.mode == "position" else [np.nan, np.nan])

        overlay.set_text(f"slope {slope:+.3f}\npitch {np.degrees(pitch):+5.1f}°\nspeed {sim.v:5.2f} m/s")
        vt = sim.verdict()
        col = SIG if vt in ("ON SETPOINT", "ARRIVED") else (
              "#ff6b6b" if "STALL" in vt else WARN)
        verdict_txt.set_text(vt); verdict_txt.set_color(col)
        m = (f"dist {sim.x:6.1f} m   crests {sim.crests:3d}\n"
             f"thrust {sim.thrust:6.0f} N   sat {100*min(1,abs(sim.thrust)/sim.vessel.fmax):3.0f}%\n"
             f"RMS err {sim.rms_err:.3f}   effort {sim.effort/1000:4.1f} kNs")
        metrics_txt.set_text(m)

        # -- telemetry --
        t = np.array(hist["t"])
        if len(t):
            ax_v.set_xlim(max(0, sim.t - WIN), max(WIN, sim.t))
            ax_thr.set_xlim(*ax_v.get_xlim()); ax_slp.set_xlim(*ax_v.get_xlim())
            l_v.set_data(t, hist["v"]); l_vref.set_data(t, hist["vref"])
            l_thr.set_data(t, hist["thrust"])
            if sim.mode == "position":
                l_slp.set_data(t, hist["xerr"]); ax_slp.set_ylabel("pos err [m]", fontsize=8)
            else:
                l_slp.set_data(t, hist["slope"]); ax_slp.set_ylabel("wave slope", fontsize=8)
            ax_v.set_ylim(min(-0.2, np.nanmin(hist["v"]) - 0.3), max(3, np.nanmax(hist["v"]) + 0.3))
            ax_thr.set_ylim(-sim.vessel.fmax * 1.1, sim.vessel.fmax * 1.1)
            arr = hist["xerr"] if sim.mode == "position" else hist["slope"]
            lo, hi = np.nanmin(arr), np.nanmax(arr)
            if np.isfinite(lo) and hi - lo > 1e-6:
                ax_slp.set_ylim(lo - 0.1 * abs(lo) - 0.05, hi + 0.1 * abs(hi) + 0.05)

        return ()

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, update, interval=int(FRAME_DT * 1000),
                         blit=False, cache_frame_data=False)
    fig._anim = anim  # keep a reference alive
    apply_params()
    plt.show()


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="USV 1D wave & boat control modeler")
    ap.add_argument("--headless", action="store_true",
                    help="run a feedforward off/on comparison and save a PNG")
    ap.add_argument("--out", default="usv_ff_comparison.png")
    args = ap.parse_args()
    if args.headless:
        headless_demo(args.out)
    else:
        launch_gui()


if __name__ == "__main__":
    main()