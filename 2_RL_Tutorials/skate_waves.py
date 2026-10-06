"""
skate_waves.py - a skateboard riding over rigid sinusoidal waves, as a Gymnasium env.

The board is a 2D rigid body (x, y, pitch) with two wheels. The ground is a rigid
sinusoid  h(x, t) = A sin(k x + k c t)  that travels toward the board (in the -x
direction) at the wave speed c. Each wheel touches the surface through a stiff
spring-damper normal contact and rolls freely along it (with a small rolling
resistance), so the board can ride the waves, get bumped back, catch air and crash.

You control one thing: a force. action in [-1, 1] is scaled by max_force and applied
at the board's centre, along the deck (or horizontally, see SkateConfig.force_mode).

Install:  pip install gymnasium pygame numpy
Run:      python skate_waves.py                   # drive it yourself with the arrow keys
          python skate_waves.py --mode random     # random actions, like the LunarLander example
          python skate_waves.py --mode policy     # your own controller, written in my_policy()
          python skate_waves.py --amplitude 0.25 --wave-speed 2.5   # tweak the waves

Keyboard controls (keyboard mode):
  RIGHT / LEFT   push forward / backward (full force)
  [ / ]          wave amplitude down / up
  - / =          wave speed down / up
  P  pause       R  reset       ESC  quit

Observation (9 floats):
  0 x                    board centre position along the track (m)
  1 height_above_wave    board centre height above the wave surface directly below it (m)
  2 pitch                board angle, counter-clockwise positive (rad)
  3 vx, 4 vy             board centre velocity (m/s)
  5 pitch_rate           (rad/s)
  6 wave_slope           dh/dx of the surface under the board
  7 sin(phase), 8 cos(phase)   where in the wave cycle the board currently sits
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import pygame
except ImportError:  # physics works without pygame; only rendering needs it
    pygame = None


# ----------------------------------------------------------------------------------
# Configuration - every tunable number lives here
# ----------------------------------------------------------------------------------
@dataclass
class SkateConfig:
    # --- Board ---
    mass: float = 4.0                  # kg
    deck_length: float = 0.80          # m
    deck_thickness: float = 0.03       # m
    wheelbase: float = 0.55            # m, distance between the two wheel centres
    truck_height: float = 0.07         # m, deck centre down to wheel centre
    wheel_radius: float = 0.035        # m
    max_force: float = 20.0            # N, force applied when action = +/-1
    force_mode: str = "deck"           # "deck": push along the deck, "horizontal": push along world x
    force_only_when_grounded: bool = False  # True = no pushing while airborne (more realistic)

    # --- Waves: h(x, t) = A sin(k x + k c t), travelling toward -x (at the board) ---
    wave_amplitude: float = 0.15       # m
    wave_length: float = 3.0           # m
    wave_speed: float = 1.5            # m/s, positive = waves move toward the board

    # --- Physics ---
    gravity: float = 9.81              # m/s^2
    contact_stiffness: float = 2.0e4   # N/m per wheel
    contact_damping: float = 150.0     # N s/m per wheel
    rolling_resistance: float = 0.02   # rolling force = coefficient * normal force
    linear_drag: float = 0.05          # N per m/s
    angular_drag: float = 0.01         # N m per rad/s
    dt: float = 1.0 / 60.0             # control step (s); one env.step() = dt seconds
    substeps: int = 20                 # physics substeps per control step (contact is stiff)

    # --- Episode ---
    max_episode_steps: int = 3600      # 60 s at 60 Hz
    crash_tilt_deg: float = 75.0       # board tilted more than this = crash

    # --- Rendering ---
    screen_width: int = 1000
    screen_height: int = 500
    pixels_per_metre: float = 160.0


# Colours
SKY = (205, 228, 245)
WATER = (40, 110, 170)
FOAM = (235, 248, 255)
MARK = (20, 60, 100)
DECK = (125, 75, 35)
TRUCK = (90, 90, 95)
WHEEL = (240, 200, 60)
DARK = (40, 40, 40)
ARROW = (220, 40, 40)
TEXT = (20, 20, 20)


# ----------------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------------
class SkateWavesEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str | None = None, config: SkateConfig | None = None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.cfg = replace(config) if config is not None else SkateConfig()
        c = self.cfg

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        high = np.full(9, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Moment of inertia of the deck about its centre (thin plank)
        self.inertia = c.mass * (c.deck_length ** 2 + c.deck_thickness ** 2) / 12.0

        # Simulation state: [x, y, pitch, vx, vy, pitch_rate]
        self.state = np.zeros(6)
        self.phase = 0.0          # wave phase, advances by k*c per second
        self.t = 0.0
        self.steps = 0
        self.last_action = 0.0
        self.n_contacts = 0
        self.wheel_spin = [0.0, 0.0]       # for drawing only
        self.wheel_spin_rate = [0.0, 0.0]

        # Rendering state
        self.screen = None
        self.clock = None
        self.font = None
        self.big_font = None
        self._cam_y = 0.0
        self._events = []
        self.window_closed = False
        self.overlay_text = ""
        self.show_controls = False

    # ------------------------------------------------------------------ waves
    @property
    def k(self) -> float:
        return 2.0 * math.pi / self.cfg.wave_length

    def surface_height(self, x):
        """Wave height at x. Works on floats and numpy arrays."""
        return self.cfg.wave_amplitude * np.sin(self.k * x + self.phase)

    def surface(self, x: float):
        """Height, slope dh/dx, and vertical velocity dh/dt of the wave at x."""
        c = self.cfg
        arg = self.k * x + self.phase
        ca = math.cos(arg)
        h = c.wave_amplitude * math.sin(arg)
        hx = c.wave_amplitude * self.k * ca
        ht = c.wave_amplitude * self.k * c.wave_speed * ca
        return h, hx, ht

    # ------------------------------------------------------------------ physics
    def _substep(self, u: float, h: float):
        c = self.cfg
        x, y, th, vx, vy, om = self.state
        cth, sth = math.cos(th), math.sin(th)

        Fx, Fy, tau = 0.0, -c.mass * c.gravity, 0.0
        contacts = 0

        # Wheel contacts: wheel i sits at body offset (+/- wheelbase/2, -truck_height)
        for i, side in enumerate((1.0, -1.0)):
            bx, by = side * c.wheelbase / 2.0, -c.truck_height
            rx = cth * bx - sth * by          # offset rotated into the world frame
            ry = sth * bx + cth * by
            wx, wy = x + rx, y + ry           # wheel centre
            wvx = vx - om * ry                # wheel centre velocity
            wvy = vy + om * rx

            hs, hx, ht = self.surface(wx)
            norm = math.sqrt(1.0 + hx * hx)
            nx, ny = -hx / norm, 1.0 / norm   # surface normal (pointing up)
            tx, ty = 1.0 / norm, hx / norm    # surface tangent (pointing +x)

            # Distance from wheel centre to surface (along the normal), minus radius
            penetration = c.wheel_radius - (wy - hs) / norm
            if penetration > 0.0:
                # Surface points move vertically at dh/dt (the shape travels, like water)
                vrel_x, vrel_y = wvx, wvy - ht
                vn = vrel_x * nx + vrel_y * ny
                vt = vrel_x * tx + vrel_y * ty

                N = c.contact_stiffness * penetration - c.contact_damping * vn
                N = max(N, 0.0)               # surface can push, never pull
                Ft = -c.rolling_resistance * N * math.tanh(vt / 0.05)

                fx = N * nx + Ft * tx
                fy = N * ny + Ft * ty
                Fx += fx
                Fy += fy
                tau += rx * fy - ry * fx
                contacts += 1
                self.wheel_spin_rate[i] = vt / c.wheel_radius

            self.wheel_spin[i] -= self.wheel_spin_rate[i] * h

        # Your control force
        if not (c.force_only_when_grounded and contacts == 0):
            F = u * c.max_force
            if c.force_mode == "deck":
                Fx += F * cth
                Fy += F * sth
            else:
                Fx += F

        # Drag
        Fx -= c.linear_drag * vx
        Fy -= c.linear_drag * vy
        tau -= c.angular_drag * om

        # Semi-implicit Euler integration
        vx += Fx / c.mass * h
        vy += Fy / c.mass * h
        om += tau / self.inertia * h
        x += vx * h
        y += vy * h
        th += om * h

        self.state[:] = (x, y, th, vx, vy, om)
        self.phase += self.k * c.wave_speed * h
        self.n_contacts = contacts

    def _crashed(self) -> bool:
        x, y, th = self.state[:3]
        too_tilted = math.cos(th) < math.cos(math.radians(self.cfg.crash_tilt_deg))
        below_surface = y < self.surface_height(x)
        return bool(too_tilted or below_surface)

    def _get_obs(self) -> np.ndarray:
        x, y, th, vx, vy, om = self.state
        h, hx, _ = self.surface(x)
        arg = self.k * x + self.phase
        return np.array([x, y - h, th, vx, vy, om, hx, math.sin(arg), math.cos(arg)],
                        dtype=np.float32)

    def _get_info(self) -> dict:
        return {"t": self.t, "wheels_in_contact": self.n_contacts,
                "force_N": self.last_action * self.cfg.max_force}

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c = self.cfg
        self.phase = float(self.np_random.uniform(0.0, 2.0 * math.pi))
        y0 = c.wave_amplitude + c.wheel_radius + c.truck_height + 0.05  # drop onto the waves
        self.state = np.array([0.0, y0, 0.0, 0.0, 0.0, 0.0])
        self.t = 0.0
        self.steps = 0
        self.last_action = 0.0
        self.n_contacts = 0
        self.wheel_spin = [0.0, 0.0]
        self.wheel_spin_rate = [0.0, 0.0]
        self._cam_y = 0.0

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._get_info()

    def step(self, action):
        c = self.cfg
        u = float(np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], -1.0, 1.0))
        self.last_action = u

        h = c.dt / c.substeps
        for _ in range(c.substeps):
            self._substep(u, h)
        self.t += c.dt
        self.steps += 1

        crashed = self._crashed()
        terminated = crashed
        truncated = self.steps >= c.max_episode_steps

        # Placeholder reward: forward progress, small effort cost, crash penalty.
        # Change this to whatever your control objective is.
        reward = self.state[3] * c.dt - 0.001 * u * u - (10.0 if crashed else 0.0)

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    # ------------------------------------------------------------------ rendering
    def render(self):
        if self.render_mode is None:
            gym.logger.warn("render() called without a render_mode, "
                            "e.g. SkateWavesEnv(render_mode='human').")
            return None
        if pygame is None:
            raise gym.error.DependencyNotInstalled("pygame is needed to render: pip install pygame")

        c = self.cfg
        W, H, ppm = c.screen_width, c.screen_height, c.pixels_per_metre

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((W, H))
                pygame.display.set_caption("Skateboard on rigid waves")
            else:
                self.screen = pygame.Surface((W, H))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("monospace", 15)
            self.big_font = pygame.font.SysFont("monospace", 40, bold=True)

        x, y, th, vx, vy, om = self.state
        s = self.screen

        # Camera: board stays at 35% across the screen; view rises if the board flies high
        self._cam_y += (max(0.0, y - 1.2) - self._cam_y) * 0.1
        board_px, ground_px = 0.35 * W, 0.65 * H

        def to_screen(wx, wy):
            return (board_px + (wx - x) * ppm, ground_px - (wy - self._cam_y) * ppm)

        s.fill(SKY)

        # Waves
        px = np.arange(0, W + 4, 4, dtype=np.float64)
        wx = x + (px - board_px) / ppm
        sy = ground_px - (self.surface_height(wx) - self._cam_y) * ppm
        pts = list(zip(px.tolist(), sy.tolist()))
        pygame.draw.polygon(s, WATER, pts + [(W, H), (0, H)])
        pygame.draw.lines(s, FOAM, False, pts, 3)

        # Distance markers every metre (labelled every 5 m) so you can see the board move
        for m in range(math.floor(wx[0]), math.ceil(wx[-1]) + 1):
            mx, my = to_screen(m, float(self.surface_height(m)))
            length = 14 if m % 5 == 0 else 7
            pygame.draw.line(s, MARK, (mx, my), (mx, my + length), 2)
            if m % 5 == 0:
                s.blit(self.font.render(f"{m} m", True, FOAM), (mx - 12, my + 16))

        # Board
        cth, sth = math.cos(th), math.sin(th)

        def body(bx, by):
            return to_screen(x + cth * bx - sth * by, y + sth * bx + cth * by)

        L, T = c.deck_length / 2.0, c.deck_thickness / 2.0
        r_px = max(4, int(c.wheel_radius * ppm))
        for i, side in enumerate((1.0, -1.0)):
            bx = side * c.wheelbase / 2.0
            top, centre = body(bx, -T), body(bx, -c.truck_height)
            pygame.draw.line(s, TRUCK, top, centre, 4)
            pygame.draw.circle(s, WHEEL, centre, r_px)
            a = self.wheel_spin[i]
            spoke = (centre[0] + r_px * math.cos(a), centre[1] - r_px * math.sin(a))
            pygame.draw.line(s, DARK, centre, spoke, 2)
        pygame.draw.polygon(s, DECK, [body(-L, -T), body(L, -T), body(L, T), body(-L, T)])

        # Force arrow
        u = self.last_action
        if abs(u) > 1e-3:
            dx, dy = (cth, sth) if c.force_mode == "deck" else (1.0, 0.0)
            sx0, sy0 = body(0.0, 0.15)
            ex, ey = sx0 + dx * u * 90.0, sy0 - dy * u * 90.0
            pygame.draw.line(s, ARROW, (sx0, sy0), (ex, ey), 4)
            ang = math.atan2(ey - sy0, ex - sx0)
            head = [(ex, ey),
                    (ex - 12 * math.cos(ang - 0.45), ey - 12 * math.sin(ang - 0.45)),
                    (ex - 12 * math.cos(ang + 0.45), ey - 12 * math.sin(ang + 0.45))]
            pygame.draw.polygon(s, ARROW, head)

        # HUD
        status = "GROUNDED" if self.n_contacts else "AIRBORNE"
        lines = [
            f"t {self.t:6.2f} s   x {x:7.2f} m   vx {vx:+5.2f} m/s",
            f"pitch {math.degrees(th):+6.1f} deg   force {u * c.max_force:+6.1f} N   "
            f"{status} ({self.n_contacts}/2 wheels)",
            f"waves: A {c.wave_amplitude:.2f} m   wavelength {c.wave_length:.1f} m   "
            f"speed {c.wave_speed:+.2f} m/s",
        ]
        if self.show_controls:
            lines.append("<- -> push   [ ] amplitude   - = wave speed   P pause   R reset   ESC quit")
        for i, line in enumerate(lines):
            s.blit(self.font.render(line, True, TEXT), (10, 8 + 18 * i))

        if self.overlay_text:
            img = self.big_font.render(self.overlay_text, True, ARROW)
            s.blit(img, img.get_rect(center=(W // 2, H // 3)))

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.window_closed = True
                else:
                    self._events.append(event)
            pygame.display.flip()
            self.clock.tick(round(1.0 / c.dt))
            return None
        return np.transpose(pygame.surfarray.array3d(s), (1, 0, 2))

    def pop_events(self):
        """Keyboard/window events collected during rendering (used by keyboard mode)."""
        events, self._events = self._events, []
        return events

    def close(self):
        if self.screen is not None and pygame is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None


# Lets you also do: gym.make("SkateWaves-v0", render_mode="human")
if "SkateWaves-v0" not in gym.registry:
    gym.register(id="SkateWaves-v0", entry_point=SkateWavesEnv)


# ----------------------------------------------------------------------------------
# Your controller
# ----------------------------------------------------------------------------------
def my_policy(obs: np.ndarray) -> np.ndarray:
    """Put your control law here. Must return an array of shape (1,) in [-1, 1].

    Example: proportional speed controller holding vx at 1 m/s.
    """
    vx = obs[3]
    target_speed = 1.0
    u = 0.5 * (target_speed - vx)
    return np.array([np.clip(u, -1.0, 1.0)], dtype=np.float32)


# ----------------------------------------------------------------------------------
# Runners
# ----------------------------------------------------------------------------------
def run_keyboard(env: SkateWavesEnv):
    env.show_controls = True
    obs, info = env.reset(seed=42)
    paused = False

    while not env.window_closed:
        for ev in env.pop_events():
            if ev.type != pygame.KEYDOWN:
                continue
            if ev.key == pygame.K_ESCAPE:
                return
            elif ev.key == pygame.K_r:
                obs, info = env.reset()
            elif ev.key == pygame.K_p:
                paused = not paused
            elif ev.key == pygame.K_LEFTBRACKET:
                env.cfg.wave_amplitude = max(0.0, env.cfg.wave_amplitude - 0.02)
            elif ev.key == pygame.K_RIGHTBRACKET:
                env.cfg.wave_amplitude = min(0.6, env.cfg.wave_amplitude + 0.02)
            elif ev.key == pygame.K_MINUS:
                env.cfg.wave_speed -= 0.25
            elif ev.key == pygame.K_EQUALS:
                env.cfg.wave_speed += 0.25

        if paused:
            env.overlay_text = "PAUSED"
            env.render()
            continue
        env.overlay_text = ""

        keys = pygame.key.get_pressed()
        u = float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
        obs, reward, terminated, truncated, info = env.step(np.array([u], dtype=np.float32))

        if terminated or truncated:
            env.overlay_text = "CRASHED!" if terminated else "TIME'S UP"
            for _ in range(60):  # hold the message for about a second
                env.render()
                if env.window_closed:
                    return
            env.overlay_text = ""
            obs, info = env.reset()


def run_policy(env: SkateWavesEnv, policy, steps: int):
    # Same shape as the Gymnasium LunarLander example
    obs, info = env.reset(seed=42)
    for _ in range(steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if env.window_closed:
            break
        if terminated or truncated:
            obs, info = env.reset()


def main():
    parser = argparse.ArgumentParser(description="Skateboard on rigid sinusoidal waves")
    parser.add_argument("--mode", choices=["keyboard", "random", "policy"], default="keyboard")
    parser.add_argument("--steps", type=int, default=3000, help="steps to run in random/policy mode")
    parser.add_argument("--amplitude", type=float, default=None, help="wave amplitude (m)")
    parser.add_argument("--wavelength", type=float, default=None, help="wavelength (m)")
    parser.add_argument("--wave-speed", type=float, default=None, help="wave speed toward board (m/s)")
    parser.add_argument("--force", type=float, default=None, help="max push force (N)")
    parser.add_argument("--grounded-push", action="store_true",
                        help="only allow pushing when a wheel touches the surface")
    args = parser.parse_args()

    cfg = SkateConfig()
    if args.amplitude is not None:
        cfg.wave_amplitude = args.amplitude
    if args.wavelength is not None:
        cfg.wave_length = args.wavelength
    if args.wave_speed is not None:
        cfg.wave_speed = args.wave_speed
    if args.force is not None:
        cfg.max_force = args.force
    if args.grounded_push:
        cfg.force_only_when_grounded = True

    env = SkateWavesEnv(render_mode="human", config=cfg)
    try:
        if args.mode == "keyboard":
            run_keyboard(env)
        elif args.mode == "random":
            run_policy(env, lambda obs: env.action_space.sample(), args.steps)
        else:
            run_policy(env, my_policy, args.steps)
    finally:
        env.close()


if __name__ == "__main__":
    main()