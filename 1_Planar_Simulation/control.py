"""
control.py - runs every speed controller one after another on the same Gerstner waves,
saves the recorded data, and plots all of them together for comparison.

Plots one histogram per controller (as subplots) showing where on the wave
each controller spent its time. The orange curve on each subplot is the real wave
shape (sharp crests and flat troughs for a Gerstner wave, a plain sine if ORBIT_RATIO = 0).

nested_lead_dynamic_velocity_control adds an outer loop to the lead controller that
keeps the AVERAGE speed at FORWARD_TARGET_VELOCITY.

"On top of the wave" is measured from the actual wave height under the board:
height / amplitude > TOP_THRESHOLD. For a Gerstner wave the crests are narrow, so even a
board moving at constant speed spends less time on top than on a sine wave. That
constant-speed level is printed and plotted as the baseline to beat.

Also prints a summary table (average velocity, time on top of waves, crashes) and saves all
the data to controller_runs.npz.

The waves are a mesh of water particles. Each particle always goes round its own orbit
(period ORBIT_PERIOD, or deep-water physics if None), and WAVE_SPEED is a drift that carries
every particle toward the board. The crests therefore move toward the board at
phase speed + WAVE_SPEED (printed in the summary).

The board spawns sitting still on the surface (SPAWN_MODE = "rest"): wheels touching with
the springs holding its weight and its motion matched to the water, so the first steps are
not a drop and bounce.

Put this file in the same folder as skate_waves.py and run:  python control.py
Set ORBIT_RATIO = 0.0 to rerun everything on sinusoidal waves for comparison.
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
    # "zero_control",
    # "bang_bang_control",
    # "p_control",
    # "p_i_control",
    # "p_i_d_control",
    "dynamic_velocity_control",
    "lead_dynamic_velocity_control",
    # "nested_lead_dynamic_velocity_control",
]
STEPS_PER_RUN = 3600          # 60 s at 60 Hz
SEED = 42                     # same seed for every run, so every controller sees the same waves
RENDER = True                 # True = watch each run (slow: real time), False = run as fast as possible
TOP_THRESHOLD = 0.7           # wave height / amplitude above this counts as "on top of the wave"
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
WAVE_AMPLITUDE = 0.3               # m
WAVE_LENGTH = 3.0                   # m
WAVE_SPEED = 0.0                    # m/s - drift of every water particle toward the board
                                    #       (0 = particles orbit in place). Crest speed = phase speed + this
ORBIT_PERIOD = None                 # s per particle orbit. None = deep-water physics, 2 pi / sqrt(g k)
ORBIT_RATIO = 1.0                   # 1 = Gerstner (circular orbits), 0 = sinusoid
MAX_STEEPNESS = 0.9                 # cap on steepness Q = k * ORBIT_RATIO * WAVE_AMPLITUDE
CONTACT_SURFACE = "smooth"          # "smooth" = exact surface, "mesh" = straight lines between particles
# Spawn
SPAWN_MODE = "rest"                 # "rest" = placed still on the surface, "drop" = fall from above
SPAWN_VX = 0.0                      # m/s - initial forward speed
SPAWN_PHASE_DEG = None              # where on the wave to start: 90 = crest, 270 = trough,
                                    #       None = random (same for every controller, from SEED)
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
SIN_PHASE_INDEX = 7  # sin of the position phase: crest at 90 deg, trough at 270 deg
COS_PHASE_INDEX = 8  # cos of the position phase: + climbing toward a crest, - falling

class Controller:
    def __init__(self, env):
        self.dt = env.cfg.dt
        self.reset()

    def reset(self):
        self.integrated_error = 0
        self.last_error = 0
        self.target_velocity = FORWARD_TARGET_VELOCITY   # latest target, for plotting
        self.base_velocity = FORWARD_TARGET_VELOCITY     # nested control: centre of the dynamic target

    def zero_control(self, obs):
        """No force at all: shows how the board moves on the waves by itself."""
        return 0.0

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
        # This is the wave LEAD radians further along, so it hits +1 BEFORE the board reaches the crest.
        # The phase is uniform in x, so LEAD is a fixed distance ahead: LEAD / (2 pi) wavelengths.
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
        self.wave_height = []          # actual wave height under the board / amplitude (-1 to +1)

    def record(self, obs, info, target_velocity):
        self.time.append(info["t"])
        self.velocity.append(obs[VX_INDEX])
        self.target_velocity.append(target_velocity)
        self.sin_phase.append(obs[SIN_PHASE_INDEX])
        self.wave_height.append(info["wave_height"] / WAVE_AMPLITUDE)

        # Where in the wave cycle the board is, in degrees from 0 to 360:
        # 90 = crest (top), 270 = trough (bottom)
        phase = np.degrees(np.arctan2(obs[SIN_PHASE_INDEX], obs[COS_PHASE_INDEX]))
        self.wave_position.append(phase % 360)

def wave_shape(steepness, n=3601):
    """Wave height / amplitude against position phase (degrees, 0 to 360) for steepness Q.

    A particle with phase phi sits at position phase psi = phi + Q cos(phi) and height sin(phi),
    so sweeping phi traces the real wave shape. Resampled onto an even psi grid."""
    phi = np.linspace(0.0, 2.0*np.pi, 20001)
    psi = np.degrees(phi + steepness*np.cos(phi))
    height = np.sin(phi)
    psi_grid = np.linspace(0.0, 360.0, n)
    return psi_grid, np.interp(psi_grid, psi, height, period=360.0)

def constant_speed_on_top(steepness):
    """Percent of time on top for a board at constant speed (psi is uniform in x)."""
    _, height = wave_shape(steepness)
    return 100 * np.mean(height[:-1] > TOP_THRESHOLD)

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

def plot_all(results, steepness, crest_speed):
    """One histogram subplot per controller: where on the wave each one spent its time."""
    n = len(results)
    n_cols = min(2, n)                        # one column if only one controller is run
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    bins = np.linspace(0, 360, 19)            # 18 bins of 20 degrees (narrower bins show sampling artefacts)
    wave_x, wave_y = wave_shape(steepness)
    baseline_on_top = constant_speed_on_top(steepness)

    for ax, (name, plotter) in zip(axes, results.items()):
        wave_position = np.array(plotter.wave_position)
        on_top = 100 * np.mean(np.array(plotter.wave_height) > TOP_THRESHOLD)
        avg_speed = np.mean(np.array(plotter.velocity))

        # Each timestep counts once, so bar height = percentage of time spent there
        weights = np.full(len(wave_position), 100 / len(wave_position))
        ax.hist(wave_position, bins=bins, weights=weights, edgecolor="black")
        ax.axhline(100 / 18, color="red", linestyle="--")   # even level (constant speed)
        ax.set_title(name, fontsize=10)
        ax.grid(True)

        # Text box with the run's average speed and time on top (and the constant-speed level)
        ax.text(0.98, 0.95, f"avg speed: {avg_speed:.2f} m/s\non top: {on_top:.1f}% "
                            f"(const: {baseline_on_top:.1f}%)",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, zorder=5,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))

        # Real wave shape drawn on top so you can see which bars are the crest and trough
        ax_wave = ax.twinx()
        ax_wave.plot(wave_x, wave_y, color="darkorange", linewidth=2.5)
        ax_wave.axhline(TOP_THRESHOLD, color="darkorange", linestyle=":", linewidth=1)
        ax_wave.set_ylim(-1.1, 1.1)
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

    model = "Gerstner" if steepness > 0 else "sinusoidal"
    fig.suptitle(f"Where on the wave each controller spends its time - {model} waves, "
                 f"steepness Q = {steepness:.2f}, crest speed {crest_speed:.2f} m/s\n(red dashed line = even, as at constant speed; "
                 f"orange dotted line = 'on top' threshold)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])   # leave room for the title
    plt.show()

def print_summary(results, env):
    model = "Gerstner" if env.steepness > 0 else "sinusoidal"
    period = 2*np.pi/env.omega if env.omega > 0 else np.inf
    print(f"\n{model} waves ({CONTACT_SURFACE} surface): amplitude {WAVE_AMPLITUDE} m, "
          f"wavelength {WAVE_LENGTH} m, steepness Q {env.steepness:.2f}")
    print(f"orbit period {period:.2f} s, phase speed {env.phase_speed:.2f} m/s, "
          f"drift {env.drift_speed:.2f} m/s -> crest speed {env.wave_c:.2f} m/s toward the board")
    print(f"constant-speed time on top: {constant_speed_on_top(env.steepness):.1f}%\n")
    print(f"{'controller':<40}{'avg velocity (m/s)':>20}{'time on top (%)':>18}{'crashes':>10}")
    for name, plotter in results.items():
        velocity = np.array(plotter.velocity)
        on_top = 100 * np.mean(np.array(plotter.wave_height) > TOP_THRESHOLD)
        print(f"{name:<40}{velocity.mean():>20.2f}{on_top:>18.1f}{plotter.crashes:>10}")

def save_results(results, path, env):
    """Save every recorded array, named like 'p_i_control_velocity', plus the wave settings.
    Load later with: data = np.load(path); data["p_i_control_velocity"]"""
    arrays = {
        "wave_amplitude": WAVE_AMPLITUDE,
        "wave_length": WAVE_LENGTH,
        "drift_speed": env.drift_speed,
        "orbit_omega": env.omega,
        "phase_speed": env.phase_speed,
        "crest_speed": env.wave_c,
        "orbit_ratio": ORBIT_RATIO,
        "contact_surface": CONTACT_SURFACE,
        "wave_steepness": env.steepness,
        "spawn_mode": SPAWN_MODE,
        "spawn_vx": SPAWN_VX,
    }
    for name, plotter in results.items():
        arrays[f"{name}_time"] = np.array(plotter.time)
        arrays[f"{name}_velocity"] = np.array(plotter.velocity)
        arrays[f"{name}_target_velocity"] = np.array(plotter.target_velocity)
        arrays[f"{name}_sin_phase"] = np.array(plotter.sin_phase)
        arrays[f"{name}_wave_position"] = np.array(plotter.wave_position)
        arrays[f"{name}_wave_height"] = np.array(plotter.wave_height)
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
    orbit_ratio=ORBIT_RATIO,
    orbit_period=ORBIT_PERIOD,
    max_steepness=MAX_STEEPNESS,
    contact_surface=CONTACT_SURFACE,
    # Spawn
    spawn_mode=SPAWN_MODE,
    spawn_vx=SPAWN_VX,
    spawn_phase_deg=SPAWN_PHASE_DEG,
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
print_summary(results, env)
save_results(results, SAVE_FILE, env)
plot_all(results, env.steepness, env.wave_c)