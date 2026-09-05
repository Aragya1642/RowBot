"""Animated visualisation -- the replacement for the browser scope.

Three entry points, in rough order of usefulness:

* :func:`animate_rollout` -- turn any ``RolloutResult`` into an MP4/GIF. This
  is the one you want after training: watch what the policy actually does
  instead of inferring it from an RMSE number.
* :func:`save_frame` -- a single annotated still, for READMEs and papers.
* :class:`LiveViewer` -- interactive scope with sliders, closest to the
  original browser tool. Tune gains and sea state while it runs.

The scope redraws the true wave surface from the ``WaveField`` attached to the
result, so what you see is exactly the sea the controller was fighting, not a
re-simulation that might have drifted.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrow
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------
# palette, carried over from the original tool so the two look related
# --------------------------------------------------------------------------
BG = "#04121a"
PANEL = "#0a2431"
EDGE = "#123a4c"
SIGNAL = "#33e6cf"
TARGET = "#8fb8ff"
WARN = "#ffb454"
DANGER = "#ff6b6b"
HULL = "#f4f2ea"
HULL_LINE = "#cf9a4a"
CREST = "#3fe0d2"
TXT = "#dbf3f2"
TXT_DIM = "#7aa6ad"
MONO = {"family": "monospace"}


def _surface_verts(xs: np.ndarray, eta: np.ndarray,
                   floor: float = -30.0) -> np.ndarray:
    """Closed polygon for the water body under a surface profile."""
    verts = np.empty((xs.size + 2, 2))
    verts[:-2, 0] = xs
    verts[:-2, 1] = eta
    verts[-2] = (xs[-1], floor)
    verts[-1] = (xs[0], floor)
    return verts


def _dark(ax, *, spines: bool = True) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TXT_DIM, labelsize=8)
    for s in ax.spines.values():
        s.set_color(EDGE if spines else "none")
    ax.grid(alpha=0.12, color=SIGNAL, linewidth=0.5)
    ax.yaxis.label.set_color(TXT_DIM)
    ax.xaxis.label.set_color(TXT_DIM)


# --------------------------------------------------------------------------
class SeaScope:
    """Persistent matplotlib artists for one animated scope.

    Artists are created once and updated in place; recreating them per frame
    is what makes naive matplotlib animation unusably slow.
    """

    def __init__(self, field, vessel, *, thrust_max: float = 500.0,
                 target_kind: str = "speed", span: Optional[float] = None,
                 window: float = 20.0, figsize=(11.0, 8.0), title: str = "",
                 exaggeration: float = 1.0):
        self.field = field
        self.vessel = vessel
        self.thrust_max = thrust_max
        self.target_kind = target_kind
        self.window = window      # seconds of trailing telemetry
        self.exaggeration = exaggeration

        self.fig = plt.figure(figsize=figsize, facecolor=BG)
        gs = self.fig.add_gridspec(5, 1, height_ratios=[2.6, 1, 1, 0.95, 0.7],
                                   hspace=0.26, left=0.085, right=0.955,
                                   top=0.925, bottom=0.065)
        self.ax_scope = self.fig.add_subplot(gs[0])
        self.ax_speed = self.fig.add_subplot(gs[1])
        self.ax_thrust = self.fig.add_subplot(gs[2])
        self.ax_wave = self.fig.add_subplot(gs[3])
        self.ax_vent = self.fig.add_subplot(gs[4])

        self.fig.suptitle(title, color=TXT, fontsize=11, **MONO)

        # Vertical and horizontal metres must map to the same number of
        # pixels, or the waves are silently distorted and the whole point of
        # a "waterline scope" is lost. Rather than fighting matplotlib's
        # aspect machinery (which resizes the axes box and wrecks the
        # gridspec), derive the box aspect from the layout and set the limits
        # to match it exactly.
        box = self.ax_scope.get_position()
        self._box_aspect = (box.height * figsize[1]) / (box.width * figsize[0])

        # Zoom follows the sea state: a 5 m swell and a 0.2 m ripple need
        # very different windows for the vessel to be legible. Fixed once so
        # the view does not jitter frame to frame.
        eta_ref = self._reference_amplitude()
        self._v_half = max(1.45 * eta_ref / max(exaggeration, 1e-6),
                           1.15 * vessel.length * self._box_aspect, 0.7)
        self.span = float(span) if span else 2.0 * self._v_half / self._box_aspect
        # actual vertical:horizontal metre ratio being displayed
        self._vscale = (self._box_aspect * self.span) / (2.0 * self._v_half)

        self._build_scope()
        self._build_strips()

    def _reference_amplitude(self) -> float:
        """A representative crest height for framing the view."""
        cfg = getattr(self.field, "config", None)
        if cfg is not None and getattr(cfg, "kind", "") == "regular":
            return max(float(cfg.amplitude), 1e-3)
        m0 = self.field.m0()
        return max(2.1 * float(np.sqrt(m0)), 1e-3)   # ~ Hs/2, a crest

    # ------------------------------------------------------------------
    def _build_scope(self) -> None:
        ax = self.ax_scope
        _dark(ax)
        ax.set_ylabel("elevation [m]")
        ax.set_xlabel("distance from vessel [m]", fontsize=8.5)
        ax.grid(False)

        self._x_surf = np.linspace(-self.span * 0.42, self.span * 0.58, 400)
        ax.set_xlim(self._x_surf[0], self._x_surf[-1])
        ax.set_ylim(-self._v_half, self._v_half)
        self._surf_line, = ax.plot([], [], color=CREST, lw=2.0, zorder=3)
        # A Polygon whose vertices are updated in place. fill_between returns
        # a collection that cannot be mutated, so animating it means
        # remove()+recreate every frame, which is both slower and leaks
        # artists into the axes.
        self._surf_fill = Polygon(np.zeros((3, 2)), closed=True,
                                  facecolor="#0a3547", edgecolor="none",
                                  zorder=2)
        ax.add_patch(self._surf_fill)
        self._msl = ax.axhline(0.0, color=TXT_DIM, lw=0.8, ls=":", alpha=0.5,
                               zorder=1)

        self._hull = Polygon(np.zeros((4, 2)), closed=True, facecolor=HULL,
                             edgecolor=HULL_LINE, lw=1.4, zorder=6)
        ax.add_patch(self._hull)
        self._mast, = ax.plot([], [], color=SIGNAL, lw=2.2, zorder=6)
        self._sensor, = ax.plot([], [], marker="o", ms=4, color=SIGNAL,
                                zorder=7)
        self._prop, = ax.plot([], [], marker="o", ms=6, color=WARN, zorder=7,
                              mec=BG, mew=0.8)
        self._thrust_arrow, = ax.plot([], [], color=SIGNAL, lw=3.0, zorder=6,
                                      solid_capstyle="round")
        self._thrust_head, = ax.plot([], [], marker=">", ms=7, color=SIGNAL,
                                     zorder=7, ls="none")
        self._target_line = ax.axvline(np.nan, color=TARGET, lw=1.4, ls="--",
                                       alpha=0.9, zorder=4)

        self._caption = ax.text(0.988, 0.035, "", transform=ax.transAxes,
                                va="bottom", ha="right", color=TXT_DIM,
                                fontsize=8, zorder=8, **MONO)
        self._hud = ax.text(0.012, 0.96, "", transform=ax.transAxes,
                            va="top", ha="left", color=TXT_DIM, fontsize=8.5,
                            linespacing=1.6, zorder=8, **MONO)
        self._verdict = ax.text(0.988, 0.96, "", transform=ax.transAxes,
                                va="top", ha="right", color=SIGNAL,
                                fontsize=9.5, fontweight="bold", zorder=8,
                                bbox=dict(boxstyle="round,pad=0.35",
                                          fc="#0a2431", ec=SIGNAL, lw=1.0),
                                **MONO)

    def _strip(self, ax, ylabel, series):
        _dark(ax)
        ax.set_ylabel(ylabel, fontsize=8.5)
        lines = []
        for color, lw, ls in series:
            ln, = ax.plot([], [], color=color, lw=lw, ls=ls)
            lines.append(ln)
        return lines

    def _build_strips(self) -> None:
        primary = "speed [m/s]" if self.target_kind == "speed" else "position [m]"
        self._speed_lines = self._strip(
            self.ax_speed, primary,
            [(TARGET, 1.2, "--"), (SIGNAL, 1.8, "-")])
        self._thrust_lines = self._strip(
            self.ax_thrust, "thrust [N]",
            [(WARN, 1.8, "-"), ("#d98cff", 1.1, "-")])
        for sign in (1, -1):
            self.ax_thrust.axhline(sign * self.thrust_max, color=DANGER,
                                   lw=0.8, ls=":", alpha=0.6)
        self.ax_thrust.set_ylim(-self.thrust_max * 1.18,
                                self.thrust_max * 1.18)
        # Long-wave context. The scope window is a few boat lengths at 1:1,
        # but the peak wavelength in a developed sea is tens of metres, so
        # the local view alone cannot show the wave the vessel is riding.
        self._wave_lines = self._strip(
            self.ax_wave, "wave / heave\n[m]",
            [("#5ad1ff", 1.4, "--"), (SIGNAL, 1.7, "-")])
        self._vent_lines = self._strip(self.ax_vent, "prop\nimmersion",
                                       [("#a6e26a", 1.6, "-")])
        self.ax_vent.set_ylim(-0.05, 1.08)
        self.ax_vent.set_xlabel("time [s]", fontsize=8.5)
        for ax in (self.ax_speed, self.ax_thrust, self.ax_wave):
            ax.tick_params(labelbottom=False)

    # ------------------------------------------------------------------
    def update(self, res, i: int) -> Sequence:
        """Draw frame ``i`` of a :class:`RolloutResult`."""
        t = float(res.t[i])
        x = float(res.x[i])
        z = float(res.z[i])
        theta = float(res.theta[i])
        v = self.vessel

        # ---- wave surface, in world coordinates around the boat ----
        xs = x + self._x_surf
        q = self.field.query(xs, t)
        eta = np.asarray(q.eta)
        self._surf_line.set_data(self._x_surf, eta)
        self._surf_fill.set_xy(_surface_verts(self._x_surf, eta))

        # ---- hull, rotated by actual pitch about the CG ----
        L, H = v.length, max(v.length * 0.16, 0.22)
        body = np.array([[-0.5 * L, -0.18 * H], [0.42 * L, -0.18 * H],
                         [0.5 * L, 0.62 * H], [-0.5 * L, 0.62 * H]])
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        self._hull.set_xy(body @ rot.T + np.array([0.0, z]))

        mast = np.array([[0.0, 0.62 * H], [0.0, 1.7 * H]]) @ rot.T + [0.0, z]
        self._mast.set_data(mast[:, 0], mast[:, 1])
        self._sensor.set_data([mast[1, 0]], [mast[1, 1]])

        # propeller position, coloured by immersion
        vent = float(res.ventilation[i])
        pr = np.array([v.prop_x, -v.prop_depth]) @ rot.T + [0.0, z]
        self._prop.set_data([pr[0]], [pr[1]])
        self._prop.set_color(SIGNAL if vent > 0.85 else
                             (WARN if vent > 0.3 else DANGER))

        # ---- thrust vector, drawn from the stern along the thrust line ----
        f = float(res.thrust_delivered[i])
        scale = 1.9 * L / max(self.thrust_max, 1e-9)
        tip = pr + np.array([f * scale * c, f * scale * s])
        self._thrust_arrow.set_data([pr[0], tip[0]], [pr[1], tip[1]])
        saturated = abs(float(res.thrust[i])) >= self.thrust_max - 1.0
        col = WARN if saturated else SIGNAL
        self._thrust_arrow.set_color(col)
        if abs(f) > 0.02 * self.thrust_max:
            self._thrust_head.set_data([tip[0]], [tip[1]])
            self._thrust_head.set_marker(">" if f > 0 else "<")
            self._thrust_head.set_color(col)
        else:
            self._thrust_head.set_data([], [])

        # ---- target marker (station keeping only) ----
        if self.target_kind == "position":
            self._target_line.set_xdata([float(res.target[i]) - x] * 2)
        else:
            self._target_line.set_xdata([np.nan, np.nan])

        # ---- framing: boat fixed at screen centre, world scrolls past ----
        # limits are constant (set in _build_scope) so the 1:1 metre aspect
        # holds and the horizon does not jump around
        vs = ("vertical scale 1:1" if abs(self._vscale - 1.0) < 0.05
              else f"vertical exaggeration x{self._vscale:.1f}")
        self._caption.set_text(
            f"x = {x:.1f} m   |   window {self.span:.0f} m   |   {vs}")

        # ---- HUD + verdict ----
        slope = float(res.wave_slope[i])
        self._hud.set_text(
            f"t     {t:6.2f} s\n"
            f"slope {slope:+6.3f}\n"
            f"pitch {np.degrees(theta):+6.1f} deg\n"
            f"speed {float(res.u[i]):6.2f} m/s\n"
            f"immer {vent:6.2f}")
        txt, col = self._verdict_for(res, i, saturated, vent)
        self._verdict.set_text(txt)
        self._verdict.set_color(col)
        self._verdict.get_bbox_patch().set_edgecolor(col)

        # ---- trailing telemetry ----
        lo = max(0.0, t - self.window)
        m = (res.t >= lo) & (res.t <= t)
        tw = res.t[m]
        primary = res.u if self.target_kind == "speed" else res.x
        self._speed_lines[0].set_data(tw, res.target[m])
        self._speed_lines[1].set_data(tw, primary[m])
        self._thrust_lines[0].set_data(tw, res.thrust[m])
        self._thrust_lines[1].set_data(tw, res.thrust_delivered[m])
        self._wave_lines[0].set_data(tw, res.wave_eta[m])
        self._wave_lines[1].set_data(tw, res.z[m])
        self._vent_lines[0].set_data(tw, res.ventilation[m])
        wseg = np.concatenate([res.wave_eta[m], res.z[m]])
        wpad = max(0.1, 0.15 * float(np.ptp(wseg)))
        self.ax_wave.set_ylim(float(wseg.min()) - wpad, float(wseg.max()) + wpad)

        for ax in (self.ax_speed, self.ax_thrust, self.ax_wave, self.ax_vent):
            ax.set_xlim(lo, max(lo + self.window, t))
        seg = np.concatenate([primary[m], res.target[m]])
        pad = max(0.25, 0.12 * (np.ptp(seg) if seg.size else 1.0))
        self.ax_speed.set_ylim(float(np.min(seg)) - pad,
                               float(np.max(seg)) + pad)

        return (self._surf_line, self._hull, self._mast, self._thrust_arrow)

    def _verdict_for(self, res, i, saturated, vent):
        err = abs(float(res.error[i]))
        tol = 0.15 if self.target_kind == "speed" else 1.0
        if vent < 0.25:
            return "VENTILATED", DANGER
        if err < tol:
            return "ON TARGET", SIGNAL
        if saturated:
            return "SATURATED", WARN
        return "SETTLING", WARN


# --------------------------------------------------------------------------
def animate_rollout(res, save: Optional[str] = "rollout.mp4", *,
                    fps: int = 25, decimate: Optional[int] = None,
                    span: Optional[float] = None, window: float = 20.0,
                    exaggeration: float = 1.0, max_seconds: Optional[float] = None,
                    figsize=(11.0, 9.0), dpi: int = 100,
                    title: Optional[str] = None):
    """Render a rollout as a video.

    Parameters
    ----------
    save
        ``.mp4`` (needs ffmpeg) or ``.gif`` (pillow, no external binary).
        Pass ``None`` to get the ``FuncAnimation`` back for notebook display.
    decimate
        Keep every Nth simulation step. Defaults to whatever gives roughly
        real-time playback at ``fps``, so a 60 s episode makes a 60 s video.
        Raise it for a faster-than-real-time overview.
    max_seconds
        Only animate the first N simulated seconds. Rendering costs roughly
        0.15 s per frame, so a 90 s episode at 20 fps is about 4 minutes of
        wall time -- worth trimming for a quick look.
    """
    from matplotlib.animation import FuncAnimation

    if res.wave_field is None:
        raise ValueError(
            "this result has no wave field attached, so the sea surface "
            "cannot be redrawn; produce it with rollout() rather than "
            "constructing RolloutResult by hand")

    dt = float(res.t[1] - res.t[0]) if res.t.size > 1 else 0.05
    if decimate is None:
        decimate = max(1, int(round(1.0 / (fps * dt))))
    n = res.t.size
    if max_seconds is not None:
        n = min(n, int(round(max_seconds / dt)) + 1)
    frames = np.arange(0, n, decimate)

    scope = SeaScope(res.wave_field, res.vessel, thrust_max=res.thrust_max,
                     target_kind=res.target_kind, span=span, window=window,
                     figsize=figsize, exaggeration=exaggeration,
                     title=title or f"{res.name}   |   {res.sea_summary}")

    anim = FuncAnimation(scope.fig, lambda k: scope.update(res, int(k)),
                         frames=frames, interval=1000.0 / fps, blit=False)
    if save is None:
        return anim

    if save.lower().endswith(".gif"):
        anim.save(save, writer="pillow", fps=fps, dpi=max(72, dpi // 2))
    else:
        anim.save(save, writer="ffmpeg", fps=fps, dpi=dpi,
                  savefig_kwargs={"facecolor": BG})
    plt.close(scope.fig)
    return save


def save_frame(res, index: int = -1, save: str = "frame.png", *,
               span: Optional[float] = None, exaggeration: float = 1.0,
               figsize=(11.0, 9.0), dpi: int = 130,
               title: Optional[str] = None) -> str:
    """Render one annotated still from a rollout."""
    if index < 0:
        index = res.t.size + index
    scope = SeaScope(res.wave_field, res.vessel, thrust_max=res.thrust_max,
                     target_kind=res.target_kind, span=span, figsize=figsize,
                     exaggeration=exaggeration,
                     title=title or f"{res.name}   |   {res.sea_summary}")
    scope.update(res, int(index))
    scope.fig.savefig(save, dpi=dpi, facecolor=BG)
    plt.close(scope.fig)
    return save


def animate_comparison(results: Sequence, save: str = "comparison.mp4", *,
                       fps: int = 25, decimate: Optional[int] = None,
                       dpi: int = 110):
    """Stack several controllers' scopes vertically, running in lockstep.

    Only meaningful when every result came from the same seed -- otherwise
    they are riding different seas and the comparison is meaningless.
    """
    from matplotlib.animation import FuncAnimation

    n = len(results)
    scopes = []
    fig = plt.figure(figsize=(11.0, 3.1 * n), facecolor=BG)
    gs = fig.add_gridspec(n, 1, hspace=0.42, left=0.08, right=0.96,
                          top=0.94, bottom=0.07)
    for k, res in enumerate(results):
        ax = fig.add_subplot(gs[k])
        sc = _MiniScope(ax, res)
        scopes.append(sc)
    fig.suptitle("controller comparison (same sea realisation)", color=TXT,
                 fontsize=11, **MONO)

    dt = float(results[0].t[1] - results[0].t[0])
    if decimate is None:
        decimate = max(1, int(round(1.0 / (fps * dt))))
    n_frames = min(r.t.size for r in results)
    frames = np.arange(0, n_frames, decimate)

    def step(k):
        for sc in scopes:
            sc.update(int(k))
        return []

    anim = FuncAnimation(fig, step, frames=frames, interval=1000.0 / fps,
                         blit=False)
    if save.lower().endswith(".gif"):
        anim.save(save, writer="pillow", fps=fps, dpi=max(72, dpi // 2))
    else:
        anim.save(save, writer="ffmpeg", fps=fps, dpi=dpi,
                  savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    return save


class _MiniScope:
    """A single-axes scope used by :func:`animate_comparison`."""

    def __init__(self, ax, res, span: float = 40.0):
        self.ax, self.res = ax, res
        self.field, self.vessel = res.wave_field, res.vessel
        _dark(ax)
        ax.grid(False)
        self._xs = np.linspace(-span * 0.4, span * 0.6, 240)
        self._line, = ax.plot([], [], color=CREST, lw=1.7)
        self._fill = Polygon(np.zeros((3, 2)), closed=True,
                             facecolor="#0a3547", edgecolor="none", zorder=2)
        ax.add_patch(self._fill)
        ax.axhline(0.0, color=TXT_DIM, lw=0.7, ls=":", alpha=0.5)
        self._hull = Polygon(np.zeros((4, 2)), closed=True, facecolor=HULL,
                             edgecolor=HULL_LINE, lw=1.2, zorder=6)
        ax.add_patch(self._hull)
        self._arrow, = ax.plot([], [], color=SIGNAL, lw=2.6, zorder=6)
        self._label = ax.text(0.012, 0.94, "", transform=ax.transAxes,
                              va="top", color=TXT, fontsize=9, zorder=8,
                              **MONO)
        ax.set_xlim(self._xs[0], self._xs[-1])
        ax.set_ylabel("elev [m]", fontsize=8)

    def update(self, i: int) -> None:
        r = self.res
        i = min(i, r.t.size - 1)
        t, x, z, th = float(r.t[i]), float(r.x[i]), float(r.z[i]), float(r.theta[i])
        eta = np.asarray(self.field.query(x + self._xs, t).eta)
        self._line.set_data(self._xs, eta)
        self._fill.set_xy(_surface_verts(self._xs, eta))
        v = self.vessel
        L, H = v.length, max(v.length * 0.16, 0.22)
        body = np.array([[-0.5 * L, -0.18 * H], [0.42 * L, -0.18 * H],
                         [0.5 * L, 0.62 * H], [-0.5 * L, 0.62 * H]])
        c, s = np.cos(th), np.sin(th)
        self._hull.set_xy(body @ np.array([[c, -s], [s, c]]).T + [0.0, z])
        f = float(r.thrust_delivered[i])
        pr = np.array([v.prop_x, -v.prop_depth]) @ np.array([[c, -s], [s, c]]).T + [0.0, z]
        sc = 1.9 * L / max(r.thrust_max, 1e-9)
        self._arrow.set_data([pr[0], pr[0] + f * sc * c],
                             [pr[1], pr[1] + f * sc * s])
        amp = float(np.max(np.abs(eta))) if eta.size else 1.0
        half = max(1.6 * amp, 1.4 * L, 1.0)
        self.ax.set_ylim(-half, half)
        self._label.set_text(f"{r.name}   err {float(r.error[i]):+5.2f}   "
                             f"u {float(r.u[i]):4.2f} m/s")


# --------------------------------------------------------------------------
class LiveViewer:
    """Interactive scope with sliders -- the browser Control Bay, in Python.

    Needs an interactive matplotlib backend (QtAgg, TkAgg, macosx). It will
    not work under Agg, so this is for your desktop, not a headless box.
    QtAgg (``pip install pyqt6``) is noticeably smoother than TkAgg.

    >>> from usv_seakeeper.render import LiveViewer
    >>> LiveViewer(preset="coastal_chop").show()     # doctest: +SKIP

    Sliders retune the live PID and rebuild the sea state in place; the
    physics keeps running, so you can watch the effect of a gain change on
    the same wave train.

    Rendering notes
    ---------------
    This uses **blitting**: the static parts of the figure (axes frames,
    grids, ticks, labels) are drawn once into a cached background, and each
    frame only re-draws the ~13 artists that actually move. A full
    ``canvas.draw()`` of this figure costs about 70 ms -- a 14 fps ceiling,
    which also starves the GUI event loop and makes the sliders feel sticky.
    Blitting brings it to roughly 4 ms, so the frame budget stops being the
    constraint.

    Blitting requires the axes to be static, so all limits are fixed rather
    than autoscaled per frame. When a slider changes something that *must*
    move a limit (max thrust, target speed, wave height), the background is
    invalidated and recaptured on the next tick -- see ``_invalidate``.
    """

    def __init__(self, preset: str = "coastal_chop", target_speed: float = 2.0,
                 seed: int = 0, figsize=(11.5, 9.0), fps: int = 20,
                 speed: float = 1.0):
        from matplotlib.widgets import Slider, Button, CheckButtons
        from .envs import SpeedHoldEnv
        from .config import SpeedTaskConfig, SimConfig
        from .controllers import PIDSpeedController, PIDGains

        self.env = SpeedHoldEnv(
            preset=preset,
            task=SpeedTaskConfig(target_speed=target_speed,
                                 randomize_target=None),
            sim=SimConfig(max_time=1e9))
        self.env.reset(seed=seed)
        self.ctrl = PIDSpeedController(PIDGains(260, 50, 30),
                                       slope_feedforward=True)
        self.running = True
        self.fps = int(fps)
        # Sim steps per drawn frame, chosen for real-time playback rather
        # than hardcoded. `speed` is a wall-clock multiplier.
        self.steps_per_frame = max(
            1, int(round(speed / (self.fps * self.env.dt))))

        self._hist = {k: deque(maxlen=4000) for k in
                      ("t", "x", "u", "z", "theta", "error", "thrust",
                       "thrust_delivered", "ventilation", "wave_slope",
                       "target")}
        self._bg = None
        self._dirty = True
        self._frame_times = deque(maxlen=30)

        self.fig = plt.figure(figsize=figsize, facecolor=BG)
        gs = self.fig.add_gridspec(4, 2, width_ratios=[3.1, 1.0],
                                   height_ratios=[2.4, 1, 1, 0.8],
                                   hspace=0.30, wspace=0.20,
                                   left=0.07, right=0.98, top=0.93,
                                   bottom=0.06)
        self.scope = _ViewerScope(self.fig, gs, self.env)
        self._build_controls(gs, Slider, Button, CheckButtons)
        self.fig.canvas.mpl_connect("resize_event", lambda _e: self._invalidate())

    # ------------------------------------------------------------------
    def _invalidate(self) -> None:
        """Mark the cached background stale; it is recaptured next tick."""
        self._dirty = True

    def _build_controls(self, gs, Slider, Button, CheckButtons):
        panel = self.fig.add_subplot(gs[:, 1])
        panel.axis("off")
        self._sliders = {}
        specs = [
            ("Hs [m]", 0.2, 5.0, self.env.sim.wave_cfg.hs),
            ("Tp [s]", 3.0, 14.0, self.env.sim.wave_cfg.tp),
            ("Fmax [N]", 100.0, 2500.0, self.env.sim.vessel.thrust_max),
            ("v_ref [m/s]", 0.0, 5.0, self.env.target_speed),
            ("Kp", 0.0, 1200.0, self.ctrl.g.kp),
            ("Ki", 0.0, 400.0, self.ctrl.g.ki),
            ("Kd", 0.0, 400.0, self.ctrl.g.kd),
        ]
        for k, (label, lo, hi, init) in enumerate(specs):
            ax = self.fig.add_axes([0.79, 0.84 - 0.062 * k, 0.17, 0.022],
                                   facecolor=PANEL)
            sl = Slider(ax, label, lo, hi, valinit=init, color=SIGNAL)
            sl.label.set_color(TXT_DIM)
            sl.label.set_fontsize(8)
            sl.valtext.set_color(SIGNAL)
            sl.valtext.set_fontsize(8)
            sl.on_changed(self._on_slider)
            self._sliders[label] = sl

        ax_chk = self.fig.add_axes([0.79, 0.34, 0.17, 0.075], facecolor=PANEL)
        self._chk = CheckButtons(ax_chk, ["slope FF", "ventilation"],
                                 [self.ctrl.slope_ff,
                                  self.env.sim.cfg.ventilation])
        for lbl in self._chk.labels:
            lbl.set_color(TXT_DIM)
            lbl.set_fontsize(8)
        self._chk.on_clicked(self._on_check)

        ax_pause = self.fig.add_axes([0.79, 0.26, 0.080, 0.035])
        ax_reset = self.fig.add_axes([0.88, 0.26, 0.080, 0.035])
        self._b_pause = Button(ax_pause, "Pause", color=PANEL, hovercolor=EDGE)
        self._b_reset = Button(ax_reset, "Reset", color=PANEL, hovercolor=EDGE)
        for b in (self._b_pause, self._b_reset):
            b.label.set_color(TXT)
            b.label.set_fontsize(8)
        self._b_pause.on_clicked(self._toggle)
        self._b_reset.on_clicked(self._reset)

    # ------------------------------------------------------------------
    def _on_slider(self, _):
        s = self._sliders
        hs, tp = s["Hs [m]"].val, s["Tp [s]"].val
        sea_changed = (abs(hs - self.env.sim.wave_cfg.hs) > 1e-9
                       or abs(tp - self.env.sim.wave_cfg.tp) > 1e-9)
        self.env.sim.field.set_spectrum(hs=hs, tp=tp)
        fmax_changed = abs(s["Fmax [N]"].val - self.env.sim.vessel.thrust_max) > 1e-9
        self.env.sim.vessel.thrust_max = s["Fmax [N]"].val
        vref_changed = abs(s["v_ref [m/s]"].val - self.env.target_speed) > 1e-9
        self.env.target_speed = s["v_ref [m/s]"].val
        self.ctrl.g.kp = s["Kp"].val
        self.ctrl.g.ki = s["Ki"].val
        self.ctrl.g.kd = s["Kd"].val
        # Gain changes are pure data; these three move axis limits, so the
        # cached background has to be rebuilt.
        if sea_changed or fmax_changed or vref_changed:
            self.scope.set_static_limits(self.env)
            self._invalidate()

    def _on_check(self, label):
        if label == "slope FF":
            self.ctrl.slope_ff = not self.ctrl.slope_ff
        else:
            self.env.sim.cfg.ventilation = not self.env.sim.cfg.ventilation

    def _toggle(self, _):
        self.running = not self.running
        self._b_pause.label.set_text("Run" if not self.running else "Pause")
        self._invalidate()

    def _reset(self, _):
        self.env.reset(seed=int(np.random.randint(1 << 30)))
        self.ctrl.reset()
        for v in self._hist.values():
            v.clear()
        self.scope.set_static_limits(self.env)
        self._invalidate()

    # ------------------------------------------------------------------
    def _advance(self) -> None:
        for _ in range(self.steps_per_frame):
            obs = self.env.control_obs()
            a = self.ctrl(obs, self.env.dt)
            self.env.step(a)
            st = self.env.sim.state
            h = self._hist
            h["t"].append(st.t)
            h["x"].append(st.x)
            h["u"].append(st.u)
            h["z"].append(st.z)
            h["theta"].append(st.theta)
            h["error"].append(self.env.target_speed - st.u)
            h["thrust"].append(st.thrust_cmd)
            h["thrust_delivered"].append(st.thrust_delivered)
            h["ventilation"].append(st.ventilation)
            h["wave_slope"].append(st.wave_slope)
            h["target"].append(self.env.target_speed)

    def _tick(self) -> None:
        t0 = time.perf_counter()
        if self.running:
            self._advance()
        self.scope.update(self.env, self._hist, np.mean(self._frame_times)
                          if self._frame_times else 0.0)

        canvas = self.fig.canvas
        if self._dirty or self._bg is None:
            # full draw to (re)build the static background, then cache it
            canvas.draw()
            self._bg = {ax: canvas.copy_from_bbox(ax.bbox)
                        for ax in self.scope.blit_axes}
            self._dirty = False
        else:
            try:
                for ax, bg in self._bg.items():
                    canvas.restore_region(bg)
                for artist in self.scope.dynamic_artists:
                    artist.axes.draw_artist(artist)
                for ax in self._bg:
                    canvas.blit(ax.bbox)
                canvas.flush_events()
            except Exception:
                # some backends don't support region blitting; fall back
                self._bg = None
                canvas.draw_idle()
        self._frame_times.append(time.perf_counter() - t0)

    def show(self, fps: Optional[int] = None):
        if matplotlib.get_backend().lower() == "agg":
            raise RuntimeError(
                "LiveViewer needs an interactive backend; Agg cannot show a "
                "window. Try matplotlib.use('QtAgg') (pip install pyqt6) or "
                "'TkAgg' before importing pyplot, or use animate_rollout() "
                "to write a video instead.")
        if fps is not None:
            self.fps = int(fps)
        self._timer = self.fig.canvas.new_timer(interval=int(1000 / self.fps))
        self._timer.add_callback(self._tick)
        self._timer.start()
        plt.show()
        return self._timer


class _ViewerScope:
    """Scope + strips for :class:`LiveViewer`, driven by live history deques.

    All dynamic artists are created with ``animated=True`` so they are
    excluded from the cached background, and are exposed through
    ``dynamic_artists`` for the blit loop. Axis limits are set once by
    ``set_static_limits`` rather than per frame -- autoscaling would
    invalidate the background on every tick and defeat the whole scheme.
    """

    def __init__(self, fig, gs, env, window: float = 20.0):
        self.fig = fig
        self.window = window
        self.ax_scope = fig.add_subplot(gs[0, 0])
        self.ax_speed = fig.add_subplot(gs[1, 0])
        self.ax_thrust = fig.add_subplot(gs[2, 0])
        self.ax_vent = fig.add_subplot(gs[3, 0])
        self.blit_axes = (self.ax_scope, self.ax_speed, self.ax_thrust,
                          self.ax_vent)
        for ax in self.blit_axes:
            _dark(ax)
        self.ax_scope.grid(False)
        fig.suptitle("USV Sea-Keeper  |  live", color=TXT, fontsize=11, **MONO)

        # 220 rather than 300 surface samples: at 1:1 aspect over a ~25 m
        # window that is still sub-decimetre resolution, and the polygon is
        # redrawn every frame.
        self._n_surf = 220
        self._line, = self.ax_scope.plot([], [], color=CREST, lw=2.0,
                                         zorder=3, animated=True)
        self._fill = Polygon(np.zeros((3, 2)), closed=True,
                             facecolor="#0a3547", edgecolor="none",
                             zorder=2, animated=True)
        self.ax_scope.add_patch(self._fill)
        self.ax_scope.axhline(0.0, color=TXT_DIM, lw=0.8, ls=":", alpha=0.5)
        self._hull = Polygon(np.zeros((4, 2)), closed=True, facecolor=HULL,
                             edgecolor=HULL_LINE, lw=1.4, zorder=6,
                             animated=True)
        self.ax_scope.add_patch(self._hull)
        self._arrow, = self.ax_scope.plot([], [], color=SIGNAL, lw=3.0,
                                          zorder=6, animated=True)
        self._prop, = self.ax_scope.plot([], [], "o", ms=6, color=WARN,
                                         zorder=7, mec=BG, mew=0.8,
                                         animated=True)
        self._hud = self.ax_scope.text(0.012, 0.96, "", va="top",
                                       transform=self.ax_scope.transAxes,
                                       color=TXT_DIM, fontsize=8.5,
                                       linespacing=1.6, zorder=8,
                                       animated=True, **MONO)
        self._fps = self.ax_scope.text(0.988, 0.96, "", va="top", ha="right",
                                       transform=self.ax_scope.transAxes,
                                       color=TXT_DIM, fontsize=8, zorder=8,
                                       animated=True, **MONO)

        self._vref, = self.ax_speed.plot([], [], color=TARGET, lw=1.2,
                                         ls="--", animated=True)
        self._v, = self.ax_speed.plot([], [], color=SIGNAL, lw=1.8,
                                      animated=True)
        self._fc, = self.ax_thrust.plot([], [], color=WARN, lw=1.7,
                                        animated=True)
        self._fd, = self.ax_thrust.plot([], [], color="#d98cff", lw=1.1,
                                        animated=True)
        self._fmax_hi, = self.ax_thrust.plot([], [], color=DANGER, lw=0.8,
                                             ls=":", animated=True)
        self._fmax_lo, = self.ax_thrust.plot([], [], color=DANGER, lw=0.8,
                                             ls=":", animated=True)
        self._vent, = self.ax_vent.plot([], [], color="#a6e26a", lw=1.6,
                                        animated=True)

        self.dynamic_artists = (
            self._fill, self._line, self._hull, self._arrow, self._prop,
            self._hud, self._fps, self._vref, self._v, self._fc, self._fd,
            self._fmax_hi, self._fmax_lo, self._vent)

        self.ax_speed.set_ylabel("speed [m/s]", fontsize=8.5)
        self.ax_thrust.set_ylabel("thrust [N]", fontsize=8.5)
        self.ax_vent.set_ylabel("immersion", fontsize=8.5)
        self.ax_vent.set_xlabel("time in window [s]", fontsize=8.5)
        for ax in (self.ax_speed, self.ax_thrust):
            ax.tick_params(labelbottom=False)
        self.set_static_limits(env)

    # ------------------------------------------------------------------
    def set_static_limits(self, env) -> None:
        """Fix every axis limit. Called on init and whenever a slider moves
        something that needs a different range."""
        v = env.sim.vessel
        # Strips use *relative* time within the window, so the x-axis never
        # scrolls -- the data slides through a fixed frame instead.
        for ax in (self.ax_speed, self.ax_thrust, self.ax_vent):
            ax.set_xlim(0.0, self.window)
        vref = getattr(env, "target_speed", 2.0)
        self.ax_speed.set_ylim(min(-1.0, vref - 3.0), vref + 3.0)
        fm = v.thrust_max * 1.18
        self.ax_thrust.set_ylim(-fm, fm)
        self.ax_vent.set_ylim(-0.05, 1.08)

        eta_ref = max(2.1 * float(np.sqrt(env.sim.field.m0())), 0.05)
        box = self.ax_scope.get_position()
        size = self.fig.get_size_inches()
        aspect = (box.height * size[1]) / (box.width * size[0])
        self._v_half = max(1.45 * eta_ref, 1.15 * v.length * aspect, 0.7)
        self.span = 2.0 * self._v_half / aspect
        self._xs = np.linspace(-self.span * 0.42, self.span * 0.58,
                               self._n_surf)
        self.ax_scope.set_xlim(self._xs[0], self._xs[-1])
        self.ax_scope.set_ylim(-self._v_half, self._v_half)
        self.ax_scope.set_ylabel("elevation [m]", fontsize=8.5)
        self._fmax_hi.set_data([0.0, self.window], [v.thrust_max] * 2)
        self._fmax_lo.set_data([0.0, self.window], [-v.thrust_max] * 2)

    # ------------------------------------------------------------------
    def update(self, env, h, frame_time: float = 0.0) -> None:
        if not h["t"]:
            return
        st = env.sim.state
        t, x, z, th = st.t, st.x, st.z, st.theta
        eta = np.asarray(env.sim.field.query(x + self._xs, t).eta)
        self._line.set_data(self._xs, eta)
        self._fill.set_xy(_surface_verts(self._xs, eta))

        v = env.sim.vessel
        L, H = v.length, max(v.length * 0.16, 0.22)
        body = np.array([[-0.5 * L, -0.18 * H], [0.42 * L, -0.18 * H],
                         [0.5 * L, 0.62 * H], [-0.5 * L, 0.62 * H]])
        c, s = np.cos(th), np.sin(th)
        rot = np.array([[c, -s], [s, c]])
        self._hull.set_xy(body @ rot.T + [0.0, z])
        pr = np.array([v.prop_x, -v.prop_depth]) @ rot.T + [0.0, z]
        self._prop.set_data([pr[0]], [pr[1]])
        self._prop.set_color(SIGNAL if st.ventilation > 0.85 else
                             (WARN if st.ventilation > 0.3 else DANGER))
        sc = 1.9 * L / max(v.thrust_max, 1e-9)
        f = st.thrust_delivered
        self._arrow.set_data([pr[0], pr[0] + f * sc * c],
                             [pr[1], pr[1] + f * sc * s])
        self._arrow.set_color(
            WARN if abs(st.thrust_cmd) >= v.thrust_max - 1.0 else SIGNAL)

        self._hud.set_text(
            f"t     {t:7.1f} s\nslope {st.wave_slope:+6.3f}\n"
            f"pitch {np.degrees(th):+6.1f} deg\nspeed {st.u:6.2f} m/s\n"
            f"immer {st.ventilation:6.2f}\nsat   {env.sim.saturation():6.2f}")
        if frame_time > 0:
            self._fps.set_text(f"{1.0 / frame_time:4.0f} fps  "
                              f"{frame_time * 1e3:4.1f} ms/frame")

        # relative time inside a fixed window
        ta = np.fromiter(h["t"], dtype=float, count=len(h["t"]))
        lo = max(0.0, t - self.window)
        m = ta >= lo
        tw = ta[m] - lo

        def series(key):
            return np.fromiter(h[key], dtype=float, count=len(h[key]))[m]

        self._vref.set_data(tw, series("target"))
        self._v.set_data(tw, series("u"))
        self._fc.set_data(tw, series("thrust"))
        self._fd.set_data(tw, series("thrust_delivered"))
        self._vent.set_data(tw, series("ventilation"))

__all__ = ["SeaScope", "animate_rollout", "animate_comparison", "save_frame",
           "LiveViewer"]