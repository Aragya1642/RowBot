"""Plugging in your own controller -- three ways.

    python examples/02_custom_controller.py

The contract is one method::

    act(obs: ControlObs, dt: float) -> float    # normalised thrust, [-1, 1]

``obs`` carries named physical quantities (see ``ControlObs``): error, x, u,
z, w, theta, q, current thrust, prop immersion, local wave slope, the
look-ahead preview array, and the plant parameters. The return value is
clipped for you, so an unbounded output will not break the sim.
"""
import numpy as np

from usv_seakeeper import (SpeedHoldEnv, SpeedTaskConfig, SimConfig,
                           Controller, CallableController, PIDGains,
                           PIDSpeedController, PreviewFeedforwardController,
                           rollout, evaluate, comparison_table, compare)


# ---------------------------------------------------------------- 1. subclass
class SlidingModeController(Controller):
    """Boundary-layer sliding mode on the speed error.

    s = e, with a saturating switching term instead of a pure sign() so the
    thruster is not asked to chatter at the slew limit.
    """
    name = "sliding-mode"

    def __init__(self, k: float = 1.2, boundary: float = 0.15):
        self.k = k
        self.boundary = boundary

    def reset(self):
        pass

    def act(self, obs, dt):
        s = obs.error / self.boundary
        switching = np.clip(s, -1.0, 1.0)
        # feedforward the known drag so the switching term only handles
        # the wave disturbance and model error
        drag_ff = obs.drag_coeff * obs.u * abs(obs.u) / obs.thrust_max
        return drag_ff + self.k * switching


# ------------------------------------------------- 2. model-based (uses plant)
class InverseDynamicsController(Controller):
    """Feedback linearisation: ask for an acceleration, invert the model.

    Shows why full state access in ``ControlObs`` is useful -- this needs
    mass, drag and the local wave slope, all of which are provided.
    """
    name = "inverse-dynamics"

    def __init__(self, lam: float = 1.5):
        self.lam = lam

    def act(self, obs, dt):
        from usv_seakeeper import G
        a_desired = self.lam * obs.error              # first-order error decay
        m_eff = obs.mass * 1.15                       # include added mass
        f_drag = obs.drag_coeff * obs.u * abs(obs.u)
        theta = np.arctan(obs.wave_slope)
        f_grav = obs.mass * G * np.sin(theta) * np.cos(theta)
        f = m_eff * a_desired + f_drag + f_grav
        # invert the known thrust loss, floored so we never divide by ~0
        return f / obs.thrust_max / max(obs.ventilation, 0.3)


# ------------------------------------------------------------ 3. bare function
def lag_compensated_p(obs, dt):
    """A plain function, adapted with CallableController."""
    return 0.9 * obs.error + 0.25 * obs.wave_slope * 10.0


def main():
    def make(preview=0):
        return SpeedHoldEnv(
            preset="coastal_chop",
            task=SpeedTaskConfig(target_speed=2.0, randomize_target=None),
            sim=SimConfig(preview_points=preview, preview_distance=25.0))

    controllers = [
        PIDSpeedController(PIDGains(260, 50, 30), slope_feedforward=True,
                           name="PID+FF (baseline)"),
        SlidingModeController(),
        InverseDynamicsController(),
        CallableController(lag_compensated_p, name="function-P"),
    ]

    print("Single episode, identical sea (seed=7):\n")
    for c in controllers:
        print(rollout(make(), c, seed=7).summary())

    print("\nAveraged over 12 wave realisations:\n")
    table = {}
    for c in controllers:
        table[c.name] = evaluate(make(), c, seeds=range(12))
    print(comparison_table(table))

    # the preview controller needs a wave-radar scan, so it gets its own env
    print("\nWith wave preview enabled (preview_points=12):\n")
    prev = PreviewFeedforwardController(PIDGains(260, 50, 30))
    base = PIDSpeedController(PIDGains(260, 50, 30), slope_feedforward=True,
                             name="PID+FF")
    t2 = {c.name: evaluate(make(preview=12), c, seeds=range(12))
          for c in (base, prev)}
    print(comparison_table(t2))


if __name__ == "__main__":
    main()
