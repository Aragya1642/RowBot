"""Linear (Airy) wave field with deep-water dispersion.

All wave kinematics are derived from one superposition::

    eta(x, t) = sum_i A_i * cos(k_i * x - w_i * t + phi_i),   w_i^2 = g * k_i

A single call to :meth:`WaveField.query` evaluates sin/cos once and derives
elevation, slope, and the surface orbital velocity/acceleration from the same
trig terms. That matters: the RL loop calls this a few hundred times per
simulated second, and recomputing the transcendentals five times over would
dominate runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import WaveConfig, G


@dataclass
class WaveQuery:
    """Wave kinematics at one or more x-positions, at a single time."""
    eta: np.ndarray          # surface elevation [m]
    slope: np.ndarray        # d eta / d x [-]
    u: np.ndarray            # horizontal orbital velocity at z=0 [m/s]
    du_dt: np.ndarray        # horizontal orbital acceleration [m/s^2]
    w: np.ndarray            # vertical orbital velocity (= d eta / d t) [m/s]
    # Hull-length-averaged versions (equal to the above when averaging is off)
    eta_avg: np.ndarray
    slope_avg: np.ndarray
    u_avg: np.ndarray
    du_dt_avg: np.ndarray
    w_avg: np.ndarray
    slope_rate_avg: np.ndarray   # d(slope_avg)/dt [1/s]


class WaveField:
    """Discretised linear wave field.

    Parameters
    ----------
    config
        :class:`~usv_seakeeper.config.WaveConfig`.
    hull_length
        If given, an additional set of "length averaged" outputs is produced,
        where each component is scaled by ``sinc(k*L/2)``. This is the exact
        result of integrating a sinusoid over the waterline length, and it is
        what kills the response to waves much shorter than the hull. Without
        it a point-force model over-predicts high-frequency excitation badly.
    """

    def __init__(self, config: WaveConfig, hull_length: Optional[float] = None):
        self.config = config
        self.hull_length = hull_length
        self._build()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build(self) -> None:
        c = self.config
        if c.kind == "calm":
            self.A = np.zeros(0)
            self.k = np.zeros(0)
            self.w = np.zeros(0)
            self.phi = np.zeros(0)
        elif c.kind == "regular":
            w = 2.0 * np.pi / c.period
            self.A = np.array([c.amplitude])
            self.w = np.array([w])
            self.k = np.array([w * w / G])
            self.phi = np.zeros(1)
        elif c.kind == "jonswap":
            self._build_jonswap()
        else:
            raise ValueError(f"unknown wave kind {c.kind!r}")

        # Sign convention. The vessel steams toward +x and the phase is
        # (k*x - w*t), so a component with k > 0 travels toward +x, i.e. it
        # overtakes the vessel -- a *following* sea. A head sea therefore
        # needs k < 0. Hence the negation: heading=+1 (head) -> k < 0.
        # Verified by check_encounter_frequency() in validate.py, which
        # measures that a head sea raises the encounter frequency above the
        # intrinsic wave frequency and a following sea lowers it.
        self.k = -self.k * c.heading

        # hull-length averaging factor: sinc(kL/2) = sin(kL/2)/(kL/2)
        if self.hull_length:
            arg = 0.5 * np.abs(self.k) * self.hull_length
            with np.errstate(invalid="ignore", divide="ignore"):
                sinc = np.where(arg < 1e-9, 1.0, np.sin(arg) / np.where(arg == 0, 1, arg))
            self.length_factor = sinc
        else:
            self.length_factor = np.ones_like(self.A)

        # Cached derived amplitude products.
        #
        # The sign(k) factor on the *velocity* terms is essential and easy to
        # miss. From the deep-water potential
        #     phi = (A w / |k|) e^{|k| z} sin(k x - w t)
        # we get u = dphi/dx = A w sign(k) e^{|k|z} cos(theta), i.e. the
        # orbital velocity points along the direction of propagation. The
        # vertical velocity w = dphi/dz = A w e^{|k|z} sin(theta) carries no
        # such factor (it stays equal to d(eta)/dt). Omitting sign(k) leaves
        # elevation and slope correct but reverses the wave force in a
        # following sea, which is exactly the case the sign convention exists
        # to support. Pinned by the FK-identity check in validate.py, which
        # is run for both headings.
        # Only the HORIZONTAL terms take sign(k). The vertical velocity
        # w = dphi/dz = A w e^{|k|z} sin(theta) must stay equal to d(eta)/dt
        # for any propagation direction, so it uses the unsigned array.
        # Sharing one A*w array between u and w is a tempting simplification
        # and silently inverts the heave forcing in a head sea.
        sgn = np.sign(self.k)
        self._Ak = self.A * self.k
        self._Aw_h = self.A * self.w * sgn          # horizontal velocity
        self._Aw_v = self.A * self.w                # vertical velocity
        self._Aww = self.A * self.w * self.w * sgn  # horizontal acceleration
        self._A_avg = self.A * self.length_factor
        self._Ak_avg = self._Ak * self.length_factor
        self._Aw_h_avg = self._Aw_h * self.length_factor
        self._Aw_v_avg = self._Aw_v * self.length_factor
        self._Aww_avg = self._Aww * self.length_factor
        self._Akw_avg = self._Ak * self.w * self.length_factor
        self._k_abs = np.abs(self.k)

        # Batched coefficient matrices. Eleven separate mat-vec products per
        # query is dominated by numpy call overhead at these array sizes, so
        # they are stacked into four matmuls instead. Surface geometry (eta,
        # slope) uses the plain trig terms; orbital velocities use the
        # depth-stretched ones, hence the split.
        self._C_cos_surf = np.stack([self.A, self._A_avg], axis=1)
        self._C_sin_surf = np.stack([self._Ak, self._Ak_avg,
                                     self._Akw_avg], axis=1)
        self._C_cos_orb = np.stack([self._Aw_h, self._Aw_h_avg], axis=1)
        self._C_sin_orb = np.stack([self._Aww, self._Aww_avg,
                                    self._Aw_v, self._Aw_v_avg], axis=1)

    def _build_jonswap(self) -> None:
        c = self.config
        rng = np.random.default_rng(c.seed)
        wp = 2.0 * np.pi / c.tp
        w_lo, w_hi = c.w_lo_factor * wp, c.w_hi_factor * wp
        n = c.n_components
        dw = (w_hi - w_lo) / n
        edges = w_lo + dw * np.arange(n)
        if c.frequency_jitter:
            w = edges + dw * rng.uniform(0.0, 1.0, size=n)
        else:
            w = edges + 0.5 * dw

        sigma = np.where(w <= wp, 0.07, 0.09)
        r = np.exp(-((w - wp) ** 2) / (2.0 * (sigma * wp) ** 2))
        # Unnormalised JONSWAP shape; the alpha*g^2 prefactor is absorbed by
        # the Hs rescaling below.
        S = w ** -5.0 * np.exp(-1.25 * (wp / w) ** 4) * c.gamma ** r

        m0 = float(np.sum(S) * dw)
        scale = (c.hs ** 2) / (16.0 * m0)   # enforce Hs = 4*sqrt(m0)
        S = S * scale

        self.A = np.sqrt(2.0 * S * dw)
        self.w = w
        self.k = w * w / G
        self.phi = rng.uniform(0.0, 2.0 * np.pi, size=n)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def reseed(self, seed: Optional[int]) -> None:
        """Draw a new random phase set (and bin jitter) for the same spectrum."""
        self.config.seed = seed
        self._build()

    def set_spectrum(self, hs: Optional[float] = None, tp: Optional[float] = None,
                     seed: Optional[int] = None) -> None:
        if hs is not None:
            self.config.hs = float(hs)
        if tp is not None:
            self.config.tp = float(tp)
        if seed is not None:
            self.config.seed = int(seed)
        self._build()

    def query(self, x, t: float, z=None) -> WaveQuery:
        """Evaluate wave kinematics at position(s) ``x`` and time ``t``.

        Parameters
        ----------
        z
            Vertical position at which to evaluate the *orbital velocities*
            (deep-water decay ``exp(|k| z)``). Elevation and slope are surface
            geometry and are unaffected.

            This matters more than it looks. Evaluating the orbital velocity
            at the mean surface (``z = 0``) instead of at the body's actual
            height loses the ``du/dz * zeta`` term in the Lagrangian mean, and
            that term is exactly half of the Stokes drift -- so wave-induced
            drift comes out 2x too small, which directly biases any
            station-keeping task. Pass the body's heave position here.
        """
        # fast path: the simulator always passes a 1-D float array
        if type(x) is np.ndarray and x.ndim == 1:
            xs = x
        else:
            xs = np.atleast_1d(np.asarray(x, dtype=float))
        if self.A.size == 0:
            z = np.zeros_like(xs)
            return WaveQuery(*(z.copy() for _ in range(11)))

        # phase matrix: (n_x, n_components)
        psi = xs[:, None] * self.k + (self.phi - self.w * t)
        cos = np.cos(psi)
        sin = np.sin(psi)

        surf_c = cos @ self._C_cos_surf
        surf_s = sin @ self._C_sin_surf
        eta, eta_avg = surf_c[:, 0], surf_c[:, 1]
        slope, slope_avg = -surf_s[:, 0], -surf_s[:, 1]
        slope_rate_avg = surf_s[:, 2]

        if z is not None and self.config.depth_stretching:
            # "surface" means: evaluate at the local free surface, which is
            # what a hull riding the waves experiences. Resolving it here
            # avoids a second full query just to look up eta.
            z_arr = eta if isinstance(z, str) else np.atleast_1d(
                np.asarray(z, dtype=float))
            # exp(|k| z), clipped: linear theory is valid to O(kA) and short
            # components would blow up on a crest. kz <= 1 is the usual guard.
            decay = np.exp(np.minimum(z_arr[:, None] * self._k_abs, 1.0))
            cos_d, sin_d = cos * decay, sin * decay
        else:
            cos_d, sin_d = cos, sin

        orb_c = cos_d @ self._C_cos_orb
        orb_s = sin_d @ self._C_sin_orb
        u, u_avg = orb_c[:, 0], orb_c[:, 1]
        du_dt, du_dt_avg = orb_s[:, 0], orb_s[:, 1]
        w_vert, w_avg = orb_s[:, 2], orb_s[:, 3]

        return WaveQuery(eta, slope, u, du_dt, w_vert, eta_avg, slope_avg,
                         u_avg, du_dt_avg, w_avg, slope_rate_avg)

    def elevation(self, x, t: float) -> np.ndarray:
        return self.query(x, t).eta

    def slope(self, x, t: float) -> np.ndarray:
        return self.query(x, t).slope

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    def m0(self) -> float:
        """Zeroth spectral moment (variance) of the elevation."""
        return float(0.5 * np.sum(self.A ** 2))

    def hs_realised(self) -> float:
        return 4.0 * np.sqrt(self.m0())

    def slope_rms(self) -> float:
        """RMS wave slope -- this is the disturbance magnitude the controller
        actually has to reject, so it is a better difficulty metric than Hs."""
        return float(np.sqrt(0.5 * np.sum((self.A * self.k) ** 2)))

    def recurrence_period(self) -> float:
        """Time after which a uniformly sampled spectrum repeats. With
        frequency jitter enabled this is effectively infinite."""
        if self.w.size < 2:
            return float("inf")
        dw = np.diff(np.sort(self.w))
        dw = dw[dw > 1e-12]
        if dw.size == 0:
            return float("inf")
        return float(2.0 * np.pi / np.min(dw))

    def summary(self) -> str:
        c = self.config
        if c.kind == "calm":
            return "calm water"
        if c.kind == "regular":
            k = float(abs(self.k[0]))
            return (f"regular A={c.amplitude:.2f} m T={c.period:.1f} s "
                    f"lambda={2*np.pi/k:.1f} m steepness={c.amplitude*k:.3f}")
        return (f"JONSWAP Hs={self.hs_realised():.2f} m Tp={c.tp:.1f} s "
                f"gamma={c.gamma} N={c.n_components} "
                f"slope_rms={self.slope_rms():.3f} "
                f"{'head' if c.heading > 0 else 'following'} sea")


__all__ = ["WaveField", "WaveQuery"]
