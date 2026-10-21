"""
bang_bang.py - runs every speed controller one after another on the same waves,
saves the recorded data, and plots all of them together for comparison.

Plots one histogram per controller (as subplots) showing where on the wave
each controller spent its time.

nested_lead_dynamic_velocity_control adds an outer loop to the lead controller that
keeps the AVERAGE speed at FORWARD_TARGET_VELOCITY.

Also prints a summary table (average velocity, time on top of waves) and saves all
the data to controller_runs.npz.

Put this file in the same folder as skate_waves.py and run:  python bang_bang.py
"""

import numpy as np
from skate_waves import SkateWavesEnv, SkateConfig
import matplotlib.pyplot as plt

#######################################################
# Boat Control parameters
#######################################################
FORWARD_TARGET_VELOCITY = 2 # m/s
K_GAIN = 1
I_GAIN = 0.05
D_GAIN = 0.05

#######################################################
# Dynamic velocity control parameters
#######################################################
DELTA = 2.0          # m/s - target is this much slower on crests and faster in troughs
LEAD = 0.7           # rad - how far ahead in the wave cycle to look (1 rad is about 57 degrees)
K_OUTER = 0.5        # nested control: how quickly the base speed corrects the AVERAGE speed

#######################################################
# Run parameters
#######################################################
CONTROLLERS = [      # every controller in this list is run, in this order
    "bang_bang_control",
    "p_control",
    "p_i_control",
    "p_i_d_control",
    "dynamic_velocity_control",
    "lead_dynamic_velocity_control",
    "nested_lead_dynamic_velocity_control",
]
STEPS_PER_RUN = 3000          # 50 s at 60 Hz
SEED = 42                     # same seed for every run, so every controller sees the same waves
RENDER = False                # True = watch each run (slow: real time), False = run as fast as possible
TOP_THRESHOLD = 0.7           # sin_phase above this counts as "on top of the wave"
SAVE_FILE = "controller_runs.npz"

#######################################################
# Environment Control Parameters
#######################################################
# Boat
MASS = 4.0                          # kg
MAX_FORCE = 20.0                    # N
FORCE_MODE = "deck"                 # "deck" or "horizontal"
FORCE_ONLY_WHEN_GROUNDED = True
# Waves
WAVE_AMPLITUDE = 0.15               # m
WAVE_LENGTH = 3.0                   # m
WAVE_SPEED = 1.5                    # m/s - positive = waves move toward the board
# Episodes
MAX_EPISODE_STEPS = 3600           # 60s at 60Hz

#######################################################
# Indices for observations
#######################################################
X_INDEX = 0
HEIGHT_INDEX = 1
PITCH_INDEX = 2      # obs[2] is pitch in radians, counter-clockwise (nose up) positive
VX_INDEX = 3
VY_INDEX = 4
PITCH_RATE_INDEX = 5
WAVE_SLOPE_INDEX = 6
SIN_PHASE_INDEX = 7
COS_PHASE_INDEX = 8

class Controller:
    def __init__(self, env):
        self.dt = env.cfg.dt
        self.reset()

    def reset(self):
        self.integrated_error = 0
        self.last_error = 0
        self.target_velocity = FORWARD_TARGET_VELOCITY   # latest target, for plotting
        self.base_velocity = FORWARD_TARGET_VELOCITY     # nested control: centre of the dynamic target

    def bang_bang_control(self, obs):
        vx_velocity = obs[VX_INDEX]
        if vx_velocity < FORWARD_TARGET_VELOCITY:
            action = 1.0    # full forward
        else:
            action = -1.0   # full backward
        return action
    
    def p_control(self, obs):
        vx_velocity = obs[VX_INDEX]
        error = FORWARD_TARGET_VELOCITY - vx_velocity
        action = K_GAIN*error
        return action
    
    def p_i_control(self, obs):
        vx_velocity = obs[VX_INDEX]
        error = FORWARD_TARGET_VELOCITY - vx_velocity
        action = (K_GAIN*error) + (I_GAIN*self.integrated_error)
        self.integrated_error += error * self.dt
        return action

    def p_i_d_control(self, obs):
        vx_velocity = obs[VX_INDEX]
        error = FORWARD_TARGET_VELOCITY - vx_velocity
        derivative = (error - self.last_error) / self.dt
        action = (K_GAIN*error) + (I_GAIN*self.integrated_error) + (D_GAIN*derivative)
        self.integrated_error += error * self.dt
        self.last_error = error
        return action

    def pitch_p_control(self, obs):
        pitch = obs[PITCH_INDEX]
        if pitch > 0:
            vx_velocity = obs[VX_INDEX]
            error = FORWARD_TARGET_VELOCITY - vx_velocity
            action = K_GAIN*error
        else:
            action = 0.0
        return action

    def energy_pitch_p_control(self, obs):
        pitch = obs[PITCH_INDEX]
        if pitch > 0:
            vx_velocity = obs[VX_INDEX]
            error = FORWARD_TARGET_VELOCITY - vx_velocity
            action = max(0.0, K_GAIN*error)
        else:
            action = 0.0
        return action

    def dynamic_velocity_control(self, obs):
        vx_velocity = obs[VX_INDEX]
        sin_phase = obs[SIN_PHASE_INDEX]     # +1 on a crest, -1 in a trough

        # Target speed changes with the wave: slower on crests, faster in troughs
        self.target_velocity = FORWARD_TARGET_VELOCITY - DELTA*sin_phase

        # Same PID as p_i_d_control, but tracking the changing target
        error = self.target_velocity - vx_velocity
        derivative = (error - self.last_error) / self.dt
        action = (K_GAIN*error) + (I_GAIN*self.integrated_error) + (D_GAIN*derivative)
        self.integrated_error += error * self.dt
        self.last_error = error
        return action

    def lead_dynamic_velocity_control(self, obs):
        vx_velocity = obs[VX_INDEX]
        sin_phase = obs[SIN_PHASE_INDEX]     # +1 on a crest, -1 in a trough
        cos_phase = obs[COS_PHASE_INDEX]     # + climbing toward a crest, - falling toward a trough

        # Look ahead: sin(phase + LEAD) = sin(phase)cos(LEAD) + cos(phase)sin(LEAD)
        # This is the wave LEAD radians further along, so it hits +1 BEFORE the board reaches the crest
        lead_sin_phase = sin_phase*np.cos(LEAD) + cos_phase*np.sin(LEAD)

        # Target speed uses the look-ahead signal: start slowing before the crest
        self.target_velocity = FORWARD_TARGET_VELOCITY - DELTA*lead_sin_phase

        # Same PID as p_i_d_control, but tracking the changing target
        error = self.target_velocity - vx_velocity
        derivative = (error - self.last_error) / self.dt
        action = (K_GAIN*error) + (I_GAIN*self.integrated_error) + (D_GAIN*derivative)
        self.integrated_error += error * self.dt
        self.last_error = error
        return action

    def nested_lead_dynamic_velocity_control(self, obs):
        vx_velocity = obs[VX_INDEX]
        sin_phase = obs[SIN_PHASE_INDEX]     # +1 on a crest, -1 in a trough
        cos_phase = obs[COS_PHASE_INDEX]     # + climbing toward a crest, - falling toward a trough

        # Outer loop: slowly move the centre of the target so the AVERAGE speed stays at
        # FORWARD_TARGET_VELOCITY. Below target on average -> base creeps up, and vice versa.
        self.base_velocity += K_OUTER * (FORWARD_TARGET_VELOCITY - vx_velocity) * self.dt

        # Look ahead, exactly as in lead_dynamic_velocity_control
        lead_sin_phase = sin_phase*np.cos(LEAD) + cos_phase*np.sin(LEAD)

        # Inner target: same wave-shaped target, but centred on the adjusted base speed
        self.target_velocity = self.base_velocity - DELTA*lead_sin_phase

        # Inner loop: same PID as before, tracking the changing target
        error = self.target_velocity - vx_velocity
        derivative = (error - self.last_error) / self.dt
        action = (K_GAIN*error) + (I_GAIN*self.integrated_error) + (D_GAIN*derivative)
        self.integrated_error += error * self.dt
        self.last_error = error
        return action

class Plotter:
    def __init__(self):
        self.time = []
        self.velocity = []
        self.target_velocity = []
        self.sin_phase = []
        self.wave_position = []

    def record(self, obs, info, target_velocity):
        self.time.append(info["t"])
        self.velocity.append(obs[VX_INDEX])
        self.target_velocity.append(target_velocity)
        self.sin_phase.append(obs[SIN_PHASE_INDEX])

        # Where in the wave cycle the board is, in degrees from 0 to 360:
        # 90 = crest (top), 270 = trough (bottom)
        phase = np.degrees(np.arctan2(obs[SIN_PHASE_INDEX], obs[COS_PHASE_INDEX]))
        self.wave_position.append(phase % 360)

def run_controller(env, name):
    """Run one controller for STEPS_PER_RUN steps and return its recorded data."""
    controller = Controller(env)
    control = getattr(controller, name)       # e.g. controller.p_i_control
    plotter = Plotter()

    obs, info = env.reset(seed=SEED)
    crashes = 0
    for _ in range(STEPS_PER_RUN):
        action = control(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        plotter.record(obs, info, controller.target_velocity)

        if env.window_closed:
            break
        if terminated or truncated:
            crashes += terminated
            obs, info = env.reset()
            controller.reset()

    plotter.crashes = crashes
    return plotter

def plot_all(results):
    """One histogram subplot per controller: where on the wave each one spent its time."""
    n = len(results)
    n_cols = 2
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    bins = np.linspace(0, 360, 19)            # 18 bins of 20 degrees (narrower bins show sampling artefacts)
    wave_x = np.linspace(0, 360, 361)

    for ax, (name, plotter) in zip(axes, results.items()):
        wave_position = np.array(plotter.wave_position)
        on_top = 100 * np.mean(np.array(plotter.sin_phase) > TOP_THRESHOLD)
        avg_speed = np.mean(np.array(plotter.velocity))

        # Each timestep counts once, so bar height = percentage of time spent there
        weights = np.full(len(wave_position), 100 / len(wave_position))
        ax.hist(wave_position, bins=bins, weights=weights, edgecolor="black")
        ax.axhline(100 / 18, color="red", linestyle="--")   # even level (constant speed)
        ax.set_title(name, fontsize=10)
        ax.grid(True)

        # Text box with the run's average speed and time on top
        ax.text(0.98, 0.95, f"avg speed: {avg_speed:.2f} m/s\non top: {on_top:.1f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, zorder=5,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))

        # Wave shape drawn on top so you can see which bars are the crest and trough
        ax_wave = ax.twinx()
        ax_wave.plot(wave_x, np.sin(np.radians(wave_x)), color="darkorange", linewidth=2.5)
        ax_wave.set_yticks([])

    for ax in axes[n:]:                       # hide unused subplots if the count is odd
        ax.set_visible(False)

    for ax in axes[:n]:
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xticklabels(["0\nrising", "90\ncrest", "180\nfalling", "270\ntrough", "360\nrising"])
    for ax in axes[::n_cols]:
        ax.set_ylabel("time spent (%)")
    for col in range(n_cols):                 # show x labels on the lowest plot in each column
        column = [i for i in range(n) if i % n_cols == col]
        axes[column[-1]].tick_params(labelbottom=True)

    fig.suptitle("Where on the wave each controller spends its time "
                 "(red dashed line = even, as at constant speed)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])   # leave room for the title
    plt.show()

def print_summary(results):
    print(f"{'controller':<40}{'avg velocity (m/s)':>20}{'time on top (%)':>18}{'crashes':>10}")
    for name, plotter in results.items():
        velocity = np.array(plotter.velocity)
        on_top = 100 * np.mean(np.array(plotter.sin_phase) > TOP_THRESHOLD)
        print(f"{name:<40}{velocity.mean():>20.2f}{on_top:>18.1f}{plotter.crashes:>10}")

def save_results(results, path):
    """Save every recorded array, named like 'p_i_control_velocity'.
    Load later with: data = np.load(path); data["p_i_control_velocity"]"""
    arrays = {}
    for name, plotter in results.items():
        arrays[f"{name}_time"] = np.array(plotter.time)
        arrays[f"{name}_velocity"] = np.array(plotter.velocity)
        arrays[f"{name}_target_velocity"] = np.array(plotter.target_velocity)
        arrays[f"{name}_sin_phase"] = np.array(plotter.sin_phase)
        arrays[f"{name}_wave_position"] = np.array(plotter.wave_position)
    np.savez(path, **arrays)
    print(f"saved data to {path}")

config = SkateConfig(
    # Board
    mass=MASS,
    max_force=MAX_FORCE,
    force_mode=FORCE_MODE,
    force_only_when_grounded=FORCE_ONLY_WHEN_GROUNDED,
    # Waves
    wave_amplitude=WAVE_AMPLITUDE,
    wave_length=WAVE_LENGTH,
    wave_speed=WAVE_SPEED,
    # Episode
    max_episode_steps=MAX_EPISODE_STEPS
)

env = SkateWavesEnv(config=config, render_mode="human" if RENDER else None)

results = {}
for name in CONTROLLERS:
    print(f"running {name}...")
    results[name] = run_controller(env, name)
    if env.window_closed:
        break

env.close()
print_summary(results)
save_results(results, SAVE_FILE)
plot_all(results)