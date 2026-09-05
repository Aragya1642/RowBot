"""Static telemetry plots -- the offline replacement for the browser scope.

Matplotlib only; no interactive dependency. Use ``save`` to write PNGs from a
headless box.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .rollout import RolloutResult

_PALETTE = ["#33e6cf", "#ffb454", "#8fb8ff", "#ff6b6b", "#a6e26a", "#d98cff"]


def _style(ax) -> None:
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_rollout(res: RolloutResult, save: Optional[str] = None,
                 title: Optional[str] = None):
    """Five-panel telemetry stack for a single episode."""
    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    fig.suptitle(title or f"{res.name}  |  {res.sea_summary}", fontsize=11)

    ax = axes[0]
    if res.target_kind == "speed":
        ax.plot(res.t, res.u, color=_PALETTE[0], lw=1.6, label="speed u")
        ax.plot(res.t, res.target, "--", color=_PALETTE[2], lw=1.2,
                label="setpoint")
        ax.set_ylabel("speed [m/s]")
    else:
        ax.plot(res.t, res.x, color=_PALETTE[0], lw=1.6, label="position x")
        ax.plot(res.t, res.target, "--", color=_PALETTE[2], lw=1.2,
                label="target")
        ax.set_ylabel("position [m]")
        axb = ax.twinx()
        axb.plot(res.t, res.u, color=_PALETTE[4], lw=1.0, alpha=0.7)
        axb.set_ylabel("speed [m/s]")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.3)
    _style(ax)

    ax = axes[1]
    ax.plot(res.t, res.error, color=_PALETTE[3], lw=1.4)
    ax.axhline(0.0, color="grey", lw=0.8, ls=":")
    ax.set_ylabel("error [m/s]" if res.target_kind == "speed" else "error [m]")
    _style(ax)

    ax = axes[2]
    ax.plot(res.t, res.thrust, color=_PALETTE[1], lw=1.4, label="commanded")
    ax.plot(res.t, res.thrust_delivered, color=_PALETTE[5], lw=1.0,
            alpha=0.85, label="delivered")
    ax.set_ylabel("thrust [N]")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.3)
    _style(ax)

    ax = axes[3]
    ax.plot(res.t, res.ventilation, color=_PALETTE[4], lw=1.3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("prop immersion")
    _style(ax)

    ax = axes[4]
    ax.plot(res.t, res.wave_eta, color="#5ad1ff", lw=1.0, label="wave eta")
    ax.plot(res.t, res.z, color=_PALETTE[0], lw=1.3, label="heave z")
    ax2 = ax.twinx()
    ax2.plot(res.t, np.degrees(res.theta), color=_PALETTE[1], lw=1.0,
             alpha=0.7, label="pitch")
    ax2.set_ylabel("pitch [deg]")
    ax.set_ylabel("elevation [m]")
    ax.set_xlabel("time [s]")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.3)
    _style(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    if save:
        fig.savefig(save, dpi=130)
        plt.close(fig)
        return save
    return fig


def plot_comparison(results: Sequence[RolloutResult], save: Optional[str] = None,
                    title: str = "Controller comparison"):
    """Overlay several controllers run on the same wave realisation."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(title, fontsize=11)
    kind = results[0].target_kind if results else "speed"
    for i, res in enumerate(results):
        c = _PALETTE[i % len(_PALETTE)]
        primary = res.u if kind == "speed" else res.x
        axes[0].plot(res.t, primary, color=c, lw=1.4, label=res.name)
        axes[1].plot(res.t, res.error, color=c, lw=1.3, label=res.name)
        axes[2].plot(res.t, res.action, color=c, lw=1.1, label=res.name)
    if results:
        axes[0].plot(results[0].t, results[0].target, "--", color="grey",
                     lw=1.1, label="target")
    axes[0].set_ylabel("speed [m/s]" if kind == "speed" else "position [m]")
    axes[1].set_ylabel("error")
    axes[1].axhline(0.0, color="grey", lw=0.8, ls=":")
    axes[2].set_ylabel("action [-1,1]")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        _style(ax)
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.3, ncols=2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if save:
        fig.savefig(save, dpi=130)
        plt.close(fig)
        return save
    return fig


def plot_spectrum(field, save: Optional[str] = None):
    """Component amplitudes and the resulting slope spectrum contribution."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    w = field.w
    axes[0].stem(w, field.A, basefmt=" ", linefmt="-", markerfmt="o")
    axes[0].set_xlabel("omega [rad/s]")
    axes[0].set_ylabel("amplitude [m]")
    axes[0].set_title(f"components (Hs={field.hs_realised():.2f} m)")
    axes[1].stem(w, field.A * np.abs(field.k), basefmt=" ")
    axes[1].set_xlabel("omega [rad/s]")
    axes[1].set_ylabel("A*k [-]")
    axes[1].set_title(f"slope contribution (RMS={field.slope_rms():.3f})")
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130)
        plt.close(fig)
        return save
    return fig


def plot_learning_curve(timesteps, values, save: Optional[str] = None,
                        ylabel: str = "episode return"):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(timesteps, values, color=_PALETTE[0], lw=1.5)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(ylabel)
    _style(ax)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130)
        plt.close(fig)
        return save
    return fig


__all__ = ["plot_rollout", "plot_comparison", "plot_spectrum",
           "plot_learning_curve"]
