"""Train an RL policy and benchmark it against the PID baselines.

    pip install gymnasium "stable-baselines3[extra]"
    python examples/03_train_rl.py --task speed --steps 300000
    python examples/03_train_rl.py --task station --steps 500000
    python examples/03_train_rl.py --task speed --eval-only --model sac_speed

Notes on why this is set up the way it is:

* **Domain randomisation is on by default.** Wave phases are reseeded every
  episode; Hs, Tp, mass, thrust and drag are drawn from ranges. A policy
  trained on one fixed wave realisation will score beautifully and transfer
  to nothing, because the sea is a deterministic function of time that it can
  memorise.

* **SAC over PPO** for a 1-D continuous action with cheap-ish simulation:
  it is far more sample efficient here, and the entropy term keeps it from
  collapsing onto a bang-bang policy early.

* **Action-rate penalty in the reward.** Without ``w_rate``, the optimal
  policy against a pure tracking cost is to slam the thruster between limits
  -- the sim's slew limit means this partly works, and it is useless on real
  hardware. The bang-bang baseline in ``02_custom_controller.py`` beats a
  tuned PID on RMSE for exactly this reason.

* **Evaluate on held-out seeds.** Training seeds and evaluation seeds must be
  disjoint, or you are reporting memorisation.
"""
from __future__ import annotations

import argparse
import numpy as np

from usv_seakeeper import (SpeedHoldEnv, StationKeepEnv, SpeedTaskConfig,
                           StationTaskConfig, RewardConfig, RandomizeConfig,
                           SimConfig, ObsConfig, PIDGains, PIDSpeedController,
                           CascadePositionController, PolicyController,
                           rollout, evaluate, comparison_table)
from usv_seakeeper.plotting import plot_comparison

TRAIN_SEEDS = range(0, 200)
EVAL_SEEDS = range(1000, 1020)      # disjoint from training


# --------------------------------------------------------------------------
def make_env(task: str, randomize: bool = True, preview: int = 0):
    """Environment factory. Keep training and evaluation configs identical
    apart from the randomisation ranges."""
    rand = RandomizeConfig(
        reseed_waves=True,
        hs=(0.6, 3.0) if randomize else None,
        tp=(4.0, 9.0) if randomize else None,
        mass=(280.0, 520.0) if randomize else None,
        thrust_max=(450.0, 900.0) if randomize else None,
        drag_coeff=(30.0, 65.0) if randomize else None,
        initial_speed=(0.0, 2.5) if randomize else None,
    )
    sim = SimConfig(dof="surge_heave_pitch", dt=0.05, dt_physics=0.01,
                    ventilation=True, max_time=60.0, preview_points=preview)
    obs = ObsConfig(include_integral=True, include_preview=preview > 0)

    if task == "speed":
        return SpeedHoldEnv(
            task=SpeedTaskConfig(target_speed=2.0,
                                 randomize_target=(0.8, 3.0) if randomize else None,
                                 tolerance=0.15, err_scale=1.0),
            reward=RewardConfig(w_error=1.0, w_effort=0.02, w_rate=0.15,
                                bonus_in_tolerance=0.5, crash_penalty=50.0),
            randomize=rand, sim=sim, obs=obs, preset="coastal_chop")

    return StationKeepEnv(
        task=StationTaskConfig(target_position=40.0,
                               randomize_target=(20.0, 70.0) if randomize else None,
                               tolerance=1.0, err_scale=10.0, hold=True),
        reward=RewardConfig(w_error=1.0, w_effort=0.02, w_rate=0.10,
                            w_velocity=0.10, bonus_in_tolerance=0.5,
                            crash_penalty=50.0),
        randomize=rand, sim=sim, obs=obs, preset="coastal_chop")


# --------------------------------------------------------------------------
_DEPS_MSG = (
    "This example needs the RL extras, which are optional:\n\n"
    "    pip install gymnasium \"stable-baselines3[extra]\"\n\n"
    "Everything else in the package (physics, PID baselines, rollout,\n"
    "plotting, examples 01/02/04) runs without them."
)


def _require_rl_deps():
    try:
        import gymnasium  # noqa: F401
        import stable_baselines3  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"{_DEPS_MSG}\n\nmissing: {exc.name}") from None


def train(task: str, steps: int, n_envs: int, out: str, preview: int):
    _require_rl_deps()
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor

    def thunk(rank: int):
        def _init():
            env = make_env(task, randomize=True, preview=preview)
            env.reset(seed=int(list(TRAIN_SEEDS)[rank % len(TRAIN_SEEDS)]))
            return Monitor(env)
        return _init

    venv = SubprocVecEnv([thunk(i) for i in range(n_envs)])
    # observations are hand-normalised already, but reward scale varies with
    # the randomised sea state, so normalise returns
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=20.0)

    model = SAC(
        "MlpPolicy", venv,
        learning_rate=3e-4,
        buffer_size=400_000,
        batch_size=512,
        tau=0.01,
        gamma=0.995,          # 60 s episode at 20 Hz = 1200 steps; look far
        train_freq=(1, "step"),
        gradient_steps=n_envs,
        learning_starts=10_000,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
    )
    model.learn(total_timesteps=steps, progress_bar=False)
    model.save(out)
    venv.save(out + "_vecnorm.pkl")
    venv.close()
    print(f"saved {out}.zip")
    return model


# --------------------------------------------------------------------------
def benchmark(task: str, model_path: str, preview: int):
    _require_rl_deps()
    from stable_baselines3 import SAC
    model = SAC.load(model_path, device="cpu")

    # Fixed evaluation conditions -- no randomisation, held-out wave seeds.
    def make_eval():
        return make_env(task, randomize=False, preview=preview)

    if task == "speed":
        baselines = [
            PIDSpeedController(PIDGains(260, 50, 30), name="PID"),
            PIDSpeedController(PIDGains(260, 50, 30), slope_feedforward=True,
                               name="PID+FF"),
        ]
    else:
        baselines = [
            CascadePositionController(kp_pos=0.35, max_speed=2.5, name="cascade-PID"),
        ]

    table, traces = {}, []
    for ctrl in baselines:
        env = make_eval()
        table[ctrl.name] = evaluate(env, ctrl, seeds=EVAL_SEEDS, verbose=True)
        traces.append(rollout(make_eval(), ctrl, seed=list(EVAL_SEEDS)[0]))

    # The policy is wrapped so it runs through the *same* harness. Its
    # ObservationBuilder must match training, so we take it from an env built
    # by the same factory.
    env = make_eval()
    policy = PolicyController(model, env.obs_builder, name="SAC")
    table["SAC"] = evaluate(env, policy, seeds=EVAL_SEEDS, verbose=True)
    traces.append(rollout(make_eval(),
                          PolicyController(model, make_eval().obs_builder,
                                           name="SAC"),
                          seed=list(EVAL_SEEDS)[0]))

    print("\n" + comparison_table(table))
    plot_comparison(traces, save=f"rl_vs_pid_{task}.png",
                    title=f"{task}: SAC vs PID (held-out sea, seed "
                          f"{list(EVAL_SEEDS)[0]})")
    print(f"wrote rl_vs_pid_{task}.png")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["speed", "station"], default="speed")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--preview", type=int, default=0,
                    help="wave-radar preview points in the observation")
    ap.add_argument("--model", default=None)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    out = args.model or f"sac_{args.task}"
    if not args.eval_only:
        train(args.task, args.steps, args.n_envs, out, args.preview)
    benchmark(args.task, out, args.preview)


if __name__ == "__main__":
    main()
