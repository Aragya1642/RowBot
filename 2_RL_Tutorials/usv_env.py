"""Long-crested irregular wavefield + a Gymnasium station-keeping environment.

Two independent pieces:

  JonswapWaveField -- pure numpy, no Gym dependency. Spectral wavefield with
                      elevation, slope, and orbital velocity queries. Usable on
                      its own for RAO benchmarking or as a disturbance module.

  WaveStationKeepEnv -- Gymnasium env wrapping the wavefield. Agent commands
                        surge thrust to hold a station against wave drift.

Conventions: SI throughout, deep water, x positive in the direction of wave
propagation, z positive up, t in seconds from episode start.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install gymnasium") from exc

G = 9.80665


# --------------------------------------------------------------------------
# Wavefield
# --------------------------------------------------------------------------

@dataclass
class SeaState:
    """Spectral sea state parameters."""

    hs: float = 2.0          # significant wave height [m]
    tp: float = 8.0          # peak period [s]
    gamma: float = 3.3       # JONSWAP peak enhancement (1.0 -> Pierson-Moskowitz)
    n_components: int = 64   # number of discrete wave components
    omega_span: tuple = (0.3, 3.0)  # integration band as multiples of omega_p


class JonswapWaveField:
    """Long-crested irregular sea built by superposing Airy components.

    Frequencies are drawn with per-bin jitter rather than placed at bin
    centres. Equal-spaced frequencies make the surface repeat with period
    2*pi/d_omega, which an RL agent will happily memorise; jitter breaks that.
    """

    def __init__(self, sea_state: SeaState, rng: np.random.Generator | None = None):
        self.sea_state = sea_state
        self.rng = rng if rng is not None else np.random.default_rng()
        self._build()

    # -- spectrum ----------------------------------------------------------

    @staticmethod
    def spectrum(omega: np.ndarray, hs: float, tp: float, gamma: float) -> np.ndarray:
        """JONSWAP one-sided variance density S(omega) [m^2 s/rad].

        Hasselmann form with the (1 - 0.287 ln gamma) normaliser, so that the
        zeroth moment recovers Hs = 4 sqrt(m0) for any gamma.
        """
        omega = np.asarray(omega, dtype=float)
        omega_p = 2.0 * np.pi / tp
        sigma = np.where(omega <= omega_p, 0.07, 0.09)
        r = np.exp(-0.5 * ((omega - omega_p) / (sigma * omega_p)) ** 2)
        a_gamma = 1.0 - 0.287 * np.log(gamma)

        with np.errstate(divide="ignore", invalid="ignore"):
            pm = (5.0 / 16.0) * hs**2 * omega_p**4 * omega**-5.0
            pm *= np.exp(-1.25 * (omega_p / omega) ** 4)
        s = a_gamma * pm * gamma**r
        return np.where(omega > 0.0, s, 0.0)

    def _build(self) -> None:
        ss = self.sea_state
        omega_p = 2.0 * np.pi / ss.tp
        lo, hi = (f * omega_p for f in ss.omega_span)

        edges = np.linspace(lo, hi, ss.n_components + 1)
        d_omega = np.diff(edges)
        # one random draw per bin, uniform inside the bin
        self.omega = edges[:-1] + d_omega * self.rng.uniform(size=ss.n_components)

        s = self.spectrum(self.omega, ss.hs, ss.tp, ss.gamma)
        self.amp = np.sqrt(2.0 * s * d_omega)          # component amplitudes [m]
        self.k = self.omega**2 / G                      # deep-water dispersion
        self.phase = self.rng.uniform(0.0, 2.0 * np.pi, size=ss.n_components)

        # realised Hs from the discretisation, useful as a sanity check
        self.hs_realised = 4.0 * np.sqrt(0.5 * np.sum(self.amp**2))

    # -- queries -----------------------------------------------------------

    def _theta(self, x: float, t: float) -> np.ndarray:
        return self.k * x - self.omega * t + self.phase

    def elevation(self, x: float, t: float) -> float:
        """Surface elevation eta [m]."""
        return float(np.sum(self.amp * np.cos(self._theta(x, t))))

    def slope(self, x: float, t: float) -> float:
        """Surface slope d(eta)/dx [rad, small-angle]."""
        return float(-np.sum(self.amp * self.k * np.sin(self._theta(x, t))))

    def elevation_rate(self, x: float, t: float) -> float:
        """d(eta)/dt at a fixed x [m/s]."""
        return float(np.sum(self.amp * self.omega * np.sin(self._theta(x, t))))

    def elevation_profile(self, xs: np.ndarray, t: float) -> np.ndarray:
        """Vectorised elevation over an array of x positions [m]."""
        xs = np.asarray(xs, dtype=float)
        theta = self.k[None, :] * xs[:, None] - self.omega[None, :] * t + self.phase[None, :]
        return np.sum(self.amp[None, :] * np.cos(theta), axis=1)

    def orbital_velocity(self, x: float, z: float, t: float) -> tuple[float, float]:
        """Horizontal and vertical particle velocity (u, w) [m/s].

        Wheeler-free: deep-water exp(kz) decay evaluated at z <= 0.
        """
        theta = self._theta(x, t)
        decay = np.exp(self.k * min(z, 0.0))
        u = np.sum(self.amp * self.omega * decay * np.cos(theta))
        w = np.sum(self.amp * self.omega * decay * np.sin(theta))
        return float(u), float(w)


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

class WaveStationKeepEnv(gym.Env):
    """Hold longitudinal station in an irregular head sea.

    State is surge (x, u) plus a linear heave oscillator driven by the local
    surface. The wave disturbance in surge is the Froude-Krylov slope force
    -m*g*d(eta)/dx, the same term that drives surf-riding, so the agent has to
    fight a genuinely wave-correlated load rather than white noise.

    Observation (8,):
        0  station error x - x_ref            [m], normalised by x_limit
        1  surge velocity u                   [m/s], normalised by u_scale
        2  heave z                            [m], normalised by hs
        3  heave rate zdot                    [m/s]
        4  local elevation eta                [m], normalised by hs
        5  local slope d(eta)/dx              [rad]
        6  local elevation rate d(eta)/dt     [m/s]
        7  previous thrust command            [-1, 1]

    Action (1,): normalised surge thrust in [-1, 1], scaled by thrust_max.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        dt: float = 0.05,
        episode_seconds: float = 120.0,
        mass: float = 350.0,
        drag_coeff: float = 45.0,
        thrust_max: float = 900.0,
        waterplane_area: float = 2.4,
        heave_damping_ratio: float = 0.35,
        x_limit: float = 12.0,
        u_scale: float = 4.0,
        action_rate_cost: float = 0.02,
        sea_state_ranges: dict | None = None,
        n_components: int = 64,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self._fig = None
        self.dt = dt
        self.max_steps = int(round(episode_seconds / dt))
        self.mass = mass
        self.drag_coeff = drag_coeff
        self.thrust_max = thrust_max
        self.waterplane_area = waterplane_area
        self.heave_damping_ratio = heave_damping_ratio
        self.x_limit = x_limit
        self.u_scale = u_scale
        self.action_rate_cost = action_rate_cost
        self.n_components = n_components

        # domain randomisation bounds, resampled every reset
        self.sea_state_ranges = sea_state_ranges or {
            "hs": (0.5, 4.0),
            "tp": (5.0, 12.0),
            "gamma": (1.0, 5.0),
        }

        # heave natural frequency from hydrostatic stiffness rho*g*Aw over mass
        self._heave_omega_n = float(np.sqrt(1025.0 * G * waterplane_area / mass))

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

        self.wave: JonswapWaveField | None = None

    # -- Gym API -----------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        opts = options or {}

        if "sea_state" in opts:
            ss = opts["sea_state"]
        else:
            lo_hi = self.sea_state_ranges
            ss = SeaState(
                hs=float(self.np_random.uniform(*lo_hi["hs"])),
                tp=float(self.np_random.uniform(*lo_hi["tp"])),
                gamma=float(self.np_random.uniform(*lo_hi["gamma"])),
                n_components=self.n_components,
            )
        self.sea_state = ss
        self.wave = JonswapWaveField(ss, rng=self.np_random)

        self.t = 0.0
        self.steps = 0
        self.x_ref = 0.0
        self.x = float(self.np_random.uniform(-1.0, 1.0))
        self.u = 0.0
        self.z = self.wave.elevation(self.x, 0.0)
        self.zdot = 0.0
        self.prev_action = 0.0

        return self._obs(), self._info()

    def step(self, action):
        a = float(np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], -1.0, 1.0))

        # --- surge: thrust, quadratic drag, wave slope force
        slope = self.wave.slope(self.x, self.t)
        f_thrust = a * self.thrust_max
        f_drag = -self.drag_coeff * self.u * abs(self.u)
        f_wave = -self.mass * G * slope
        udot = (f_thrust + f_drag + f_wave) / self.mass

        # --- heave: linear oscillator restoring toward the local surface
        omega_n = self._heave_omega_n
        eta = self.wave.elevation(self.x, self.t)
        eta_rate = self.wave.elevation_rate(self.x, self.t)
        zddot = (
            -2.0 * self.heave_damping_ratio * omega_n * (self.zdot - eta_rate)
            - omega_n**2 * (self.z - eta)
        )

        # semi-implicit Euler: velocities first, then positions
        self.u += udot * self.dt
        self.x += self.u * self.dt
        self.zdot += zddot * self.dt
        self.z += self.zdot * self.dt

        self.t += self.dt
        self.steps += 1

        err = self.x - self.x_ref
        reward = -((err / self.x_limit) ** 2)
        reward -= 0.01 * (self.u / self.u_scale) ** 2
        reward -= self.action_rate_cost * (a - self.prev_action) ** 2
        self.prev_action = a

        # terminated: the agent actually failed (physical excursion)
        # truncated: time ran out, episode is fine, bootstrap the value
        terminated = bool(abs(err) > self.x_limit)
        truncated = bool(self.steps >= self.max_steps)
        if terminated:
            reward -= 10.0

        return self._obs(), float(reward), terminated, truncated, self._info()

    # -- rendering ---------------------------------------------------------

    def _init_render(self):
        import matplotlib
        if self.render_mode == "rgb_array":
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self._plt = plt
        self._fig, self._ax = plt.subplots(figsize=(9.0, 4.0))
        self._ax.set_xlabel("x [m]")
        self._ax.set_ylabel("z [m]")
        self._ax.grid(alpha=0.2)

        (self._surface,) = self._ax.plot([], [], lw=1.6, color="#2b6ea3")
        self._fill = None
        (self._hull,) = self._ax.plot([], [], lw=2.4, color="#c2410c",
                                      solid_capstyle="round")
        self._ref = self._ax.axvline(0.0, ls="--", lw=1.0, color="#555555")
        self._readout = self._ax.text(
            0.015, 0.95, "", transform=self._ax.transAxes,
            va="top", ha="left", fontsize=9, family="monospace",
        )
        if self.render_mode == "human":
            plt.ion()
            plt.show(block=False)

    def render(self):
        if self.render_mode is None:
            return None
        if self._fig is None:
            self._init_render()

        half_window = 40.0
        xs = np.linspace(self.x - half_window, self.x + half_window, 240)
        eta = self.wave.elevation_profile(xs, self.t)

        self._surface.set_data(xs, eta)
        if self._fill is not None:
            self._fill.remove()
        self._fill = self._ax.fill_between(xs, eta, -12.0, color="#2b6ea3", alpha=0.18)

        # hull as a short line segment pitched by the local surface slope
        slope = self.wave.slope(self.x, self.t)
        pitch = np.arctan(slope)
        half_len = 2.2
        dx, dz = half_len * np.cos(pitch), half_len * np.sin(pitch)
        self._hull.set_data([self.x - dx, self.x + dx], [self.z - dz, self.z + dz])

        self._ref.set_xdata([self.x_ref, self.x_ref])
        self._ax.set_xlim(self.x - half_window, self.x + half_window)
        lim = max(3.0, 1.6 * self.sea_state.hs)
        self._ax.set_ylim(-lim, lim)
        self._readout.set_text(
            f"t    {self.t:6.1f} s\n"
            f"Hs   {self.sea_state.hs:5.2f} m   Tp {self.sea_state.tp:4.1f} s\n"
            f"err  {self.x - self.x_ref:+6.2f} m\n"
            f"u    {self.u:+6.2f} m/s\n"
            f"thr  {self.prev_action:+6.2f}"
        )

        if self.render_mode == "human":
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
            self._plt.pause(1.0 / self.metadata["render_fps"])
            return None

        self._fig.canvas.draw()
        buf = np.asarray(self._fig.canvas.buffer_rgba())
        return buf[..., :3].copy()

    def close(self):
        if self._fig is not None:
            self._plt.close(self._fig)
            self._fig = None

    # -- helpers -----------------------------------------------------------

    def _obs(self) -> np.ndarray:
        hs = max(self.sea_state.hs, 1e-3)
        return np.array(
            [
                (self.x - self.x_ref) / self.x_limit,
                self.u / self.u_scale,
                self.z / hs,
                self.zdot,
                self.wave.elevation(self.x, self.t) / hs,
                self.wave.slope(self.x, self.t),
                self.wave.elevation_rate(self.x, self.t),
                self.prev_action,
            ],
            dtype=np.float32,
        )

    def _info(self) -> dict:
        return {
            "t": self.t,
            "hs": self.sea_state.hs,
            "tp": self.sea_state.tp,
            "gamma": self.sea_state.gamma,
            "hs_realised": self.wave.hs_realised,
            "station_error": self.x - self.x_ref,
        }


if __name__ == "__main__":
    # spectral sanity check: realised Hs should track the requested Hs
    for hs, tp, gamma in [(1.0, 6.0, 1.0), (2.5, 8.0, 3.3), (4.0, 11.0, 5.0)]:
        wf = JonswapWaveField(
            SeaState(hs=hs, tp=tp, gamma=gamma, n_components=256),
            rng=np.random.default_rng(0),
        )
        print(f"requested Hs={hs:.2f}  realised={wf.hs_realised:.3f}  "
              f"error={100 * (wf.hs_realised / hs - 1):+.1f}%")

    env = WaveStationKeepEnv()
    obs, info = env.reset(seed=0)
    total = 0.0
    while True:
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        total += r
        if terminated or truncated:
            break
    print(f"\nrandom rollout: {env.steps} steps, return={total:.1f}, "
          f"terminated={terminated}, truncated={truncated}")