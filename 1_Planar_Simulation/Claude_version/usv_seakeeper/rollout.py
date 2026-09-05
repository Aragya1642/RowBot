"""Rollout harness and evaluation metrics.

The same function runs a hand-written PID and a trained RL policy, which is
the point: any comparison you produce is on identical wave realisations,
identical actuator limits, and identical scoring.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .controllers import Controller
from .envs import SpeedHoldEnv, StationKeepEnv


# --------------------------------------------------------------------------
@dataclass
class RolloutResult:
    """Time series and summary metrics from one episode."""
    name: str
    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    z: np.ndarray
    theta: np.ndarray
    q: np.ndarray
    action: np.ndarray
    thrust: np.ndarray
    thrust_delivered: np.ndarray
    ventilation: np.ndarray
    wave_eta: np.ndarray
    wave_slope: np.ndarray
    error: np.ndarray
    reward: np.ndarray
    target: np.ndarray
    target_kind: str = "speed"      # "speed" or "position"
    # Kept so the animated scope can redraw the exact wave surface the
    # controller actually saw. Excluded from CSV export.
    wave_field: Any = None
    vessel: Any = None
    thrust_max: float = 1.0
    metrics: Dict[str, float] = field(default_factory=dict)
    terminated: bool = False
    truncated: bool = False
    sea_summary: str = ""

    # ------------------------------------------------------------------
    def to_csv(self, path: str) -> None:
        cols = ["t", "x", "u", "z", "theta", "q", "action", "thrust",
                "thrust_delivered", "ventilation", "wave_eta", "wave_slope",
                "error", "reward", "target"]
        with open(path, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(cols)
            data = [getattr(self, c) for c in cols]
            for row in zip(*data):
                wr.writerow([f"{v:.6g}" for v in row])

    def summary(self) -> str:
        m = self.metrics
        order = ["rmse", "rmse_steady", "mae", "max_abs_error", "settling_time",
                 "in_tolerance_frac", "saturated_frac", "mean_ventilation",
                 "min_ventilation", "control_effort", "action_rate_rms",
                 "mean_speed", "distance", "arrival_time", "return"]
        parts = []
        for k in order:
            if k in m and m[k] is not None:
                parts.append(f"{k}={m[k]:.4g}")
        flag = " [TERMINATED EARLY]" if self.terminated else ""
        return f"{self.name}: " + "  ".join(parts) + flag


# --------------------------------------------------------------------------
def _compute_metrics(res: RolloutResult, tolerance: float,
                     dt: float) -> Dict[str, float]:
    err = res.error
    abs_err = np.abs(err)
    m: Dict[str, Any] = {
        "rmse": float(np.sqrt(np.mean(err ** 2))) if err.size else float("nan"),
        "mae": float(np.mean(abs_err)) if err.size else float("nan"),
        "max_abs_error": float(np.max(abs_err)) if err.size else float("nan"),
        "in_tolerance_frac": float(np.mean(abs_err < tolerance)) if err.size else 0.0,
        "saturated_frac": float(np.mean(np.abs(res.action) > 0.99)),
        "mean_ventilation": float(np.mean(res.ventilation)),
        "min_ventilation": float(np.min(res.ventilation)) if res.ventilation.size else 1.0,
        "control_effort": float(np.sum(np.abs(res.thrust)) * dt),
        "action_rate_rms": float(np.sqrt(np.mean(np.diff(res.action, prepend=0.0) ** 2))),
        "mean_speed": float(np.mean(res.u)),
        "distance": float(res.x[-1] - res.x[0]) if res.x.size else 0.0,
        "return": float(np.sum(res.reward)),
    }
    # RMSE over the second half of the episode. For station keeping the
    # whole-episode RMSE is dominated by the transit from the start point,
    # which says nothing about how well the vessel actually holds station;
    # this is the number to compare controllers on.
    if err.size > 4:
        half = err.size // 2
        m["rmse_steady"] = float(np.sqrt(np.mean(err[half:] ** 2)))
    else:
        m["rmse_steady"] = m["rmse"]

    # settling time: last time the error left the tolerance band
    inside = abs_err < tolerance
    if inside.any() and res.t.size:
        outside_idx = np.flatnonzero(~inside)
        if outside_idx.size == 0:
            m["settling_time"] = 0.0
        elif outside_idx[-1] < inside.size - 1:
            m["settling_time"] = float(res.t[outside_idx[-1] + 1])
        else:
            m["settling_time"] = float("nan")
    else:
        m["settling_time"] = float("nan")
    return m


# --------------------------------------------------------------------------
def rollout(env, controller: Controller, seed: Optional[int] = 0,
            max_steps: Optional[int] = None,
            name: Optional[str] = None) -> RolloutResult:
    """Run one episode of ``env`` under ``controller``.

    The controller receives a :class:`~usv_seakeeper.controllers.ControlObs`
    (named physical quantities); an RL policy wrapped in ``PolicyController``
    internally rebuilds the flat training observation from the same simulator
    state, so both see consistent information.
    """
    controller.reset()
    obs, info = env.reset(seed=seed)
    dt = env.dt
    if max_steps is None:
        max_steps = int(round(env.sim.cfg.max_time / dt)) + 2

    rec: Dict[str, List[float]] = {k: [] for k in (
        "t", "x", "u", "z", "theta", "q", "action", "thrust",
        "thrust_delivered", "ventilation", "wave_eta", "wave_slope",
        "error", "reward", "target")}

    terminated = truncated = False
    for _ in range(max_steps):
        cobs = env.control_obs()
        action = controller(cobs, dt)

        _, reward, terminated, truncated, info = env.step(action)
        st = env.sim.state

        rec["t"].append(st.t)
        rec["x"].append(st.x)
        rec["u"].append(st.u)
        rec["z"].append(st.z)
        rec["theta"].append(st.theta)
        rec["q"].append(st.q)
        rec["action"].append(action)
        rec["thrust"].append(st.thrust_cmd)
        rec["thrust_delivered"].append(st.thrust_delivered)
        rec["ventilation"].append(st.ventilation)
        rec["wave_eta"].append(st.wave_eta)
        rec["wave_slope"].append(st.wave_slope)
        rec["error"].append(info["error"])
        rec["reward"].append(reward)
        rec["target"].append(getattr(env, "target_speed",
                                     getattr(env, "target_position", 0.0)))

        if terminated or truncated:
            break

    arrays = {k: np.asarray(v, dtype=float) for k, v in rec.items()}
    kind = "speed" if hasattr(env, "target_speed") else "position"
    res = RolloutResult(name=name or getattr(controller, "name", "controller"),
                        terminated=terminated, truncated=truncated,
                        target_kind=kind, wave_field=env.sim.field,
                        vessel=env.sim.vessel,
                        thrust_max=env.sim.vessel.thrust_max,
                        sea_summary=env.sim.field.summary(), **arrays)
    res.metrics = _compute_metrics(res, env._tolerance(), dt)
    if "arrival_time" in info and info["arrival_time"] is not None:
        res.metrics["arrival_time"] = float(info["arrival_time"])
    return res


# --------------------------------------------------------------------------
def evaluate(env, controller: Controller, seeds: Sequence[int] = range(10),
             verbose: bool = False) -> Dict[str, float]:
    """Average metrics over several wave realisations.

    Single-episode numbers in an irregular sea are close to meaningless --
    the variance between phase realisations is large. Always evaluate over a
    seed set, and use the same seed set for every controller you compare.
    """
    results = [rollout(env, controller, seed=int(s)) for s in seeds]
    keys = set().union(*[set(r.metrics) for r in results])
    agg: Dict[str, float] = {}
    for k in keys:
        vals = [r.metrics[k] for r in results
                if k in r.metrics and r.metrics[k] is not None
                and np.isfinite(r.metrics[k])]
        if vals:
            agg[k] = float(np.mean(vals))
            agg[k + "_std"] = float(np.std(vals))
    agg["episodes"] = len(results)
    agg["early_termination_frac"] = float(np.mean([r.terminated for r in results]))
    if verbose:
        name = getattr(controller, "name", "controller")
        print(f"{name}: rmse={agg.get('rmse', float('nan')):.4f} "
              f"+-{agg.get('rmse_std', 0.0):.4f}  "
              f"effort={agg.get('control_effort', float('nan')):.0f}  "
              f"sat={agg.get('saturated_frac', 0.0):.2%}  "
              f"crash={agg['early_termination_frac']:.0%}")
    return agg


def compare(env_factory: Callable[[], Any],
            controllers: Sequence[Controller],
            seeds: Sequence[int] = range(10)) -> Dict[str, Dict[str, float]]:
    """Score several controllers on identical wave realisations."""
    out: Dict[str, Dict[str, float]] = {}
    for ctrl in controllers:
        env = env_factory()
        out[getattr(ctrl, "name", str(ctrl))] = evaluate(env, ctrl, seeds=seeds,
                                                         verbose=True)
    return out


def comparison_table(results: Dict[str, Dict[str, float]],
                     keys: Sequence[str] = ("rmse", "max_abs_error",
                                            "in_tolerance_frac",
                                            "saturated_frac",
                                            "min_ventilation",
                                            "control_effort",
                                            "action_rate_rms")) -> str:
    """Render a plain-text comparison table."""
    names = list(results)
    w = max(len(n) for n in names) + 2
    header = "controller".ljust(w) + "".join(k.rjust(16) for k in keys)
    lines = [header, "-" * len(header)]
    for n in names:
        row = n.ljust(w)
        for k in keys:
            v = results[n].get(k)
            row += ("--" if v is None else f"{v:.4g}").rjust(16)
        lines.append(row)
    return "\n".join(lines)


__all__ = ["rollout", "evaluate", "compare", "comparison_table",
           "RolloutResult"]
