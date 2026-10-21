"""
skate_waves.py - a skateboard riding over rigid Gerstner (trochoidal) waves, as a Gymnasium env.

The board is a 2D rigid body (x, y, pitch) with two wheels. The ground is a Gerstner wave,
modelled as a mesh of water particles that each go round their own orbit.

Gerstner wave as a particle mesh (2D, deep water)
-------------------------------------------------
Each surface particle i has a fixed label a_i (its rest position). At time t it sits at

    phi_i(t) = k a_i + omega t + phi_0                (orbit phase)
    x_i(t)   = a_i - V t + R cos(phi_i)              (drift + horizontal orbit, radius R)
    z_i(t)   = A sin(phi_i)                          (vertical orbit, radius A)

Two separate motions:
  * ORBIT   - every particle circles at angular frequency omega, always, even with V = 0.
              omega comes from deep-water physics, omega = sqrt(g k), unless you set
              SkateConfig.orbit_period. The phase lag k a_i between neighbouring particles is
              what makes the crest shape travel toward the board at the phase speed
              c_p = omega / k, even though no particle travels anywhere.
  * DRIFT   - SkateConfig.wave_speed (V) translates every particle toward the board, like a
              current carrying the whole wave field. V = 0 means the particles go round closed
              circles with no net motion.

The crest shape therefore moves toward the board at  c_crest = c_p + V.
(Set V = -c_p to hold the crests still in the world: the water then streams away from the
board while the particles keep orbiting.)

R = orbit_ratio * A. With orbit_ratio = 1 the orbits are circles (the classic Gerstner /
trochoidal wave): particles bunch up under the crests and spread out in the troughs, giving
sharp crests and flat troughs, like real water. With orbit_ratio = 0 the surface is a sinusoid.
Steepness Q = k R must stay below 1 (Q = 1 is a cusp), so it is clamped to max_steepness.

The mesh covers one wavelength (mesh_points_per_wavelength particles) and repeats, since
particle a + L moves exactly like particle a shifted by L.

How the wheels find the surface (SkateConfig.contact_surface):
  "smooth" - exact surface between the particles: solve x = b + R cos(k b + phase) for the
             mean position b with Newton's method (the left side always increases with b, so
             there is one answer). Smooth slope, so the stiff contact stays well behaved.
  "mesh"   - straight lines between neighbouring mesh particles (piecewise-linear surface).
             This is literally the surface of the point mesh, but the slope jumps at every
             particle, which shows up as small bumps in the contact force.
Either way the wheel feels the velocity of the water particle under it (orbit + drift).

Each wheel touches the surface through a stiff spring-damper normal contact and rolls freely
along it (with a small rolling resistance), so the board can ride the waves, get bumped back,
catch air and crash.

Spawning (SkateConfig.spawn_mode):
  "rest" - the board is placed sitting on the surface: both wheels touching with the springs
           already compressed to hold its weight, the deck tilted to match the surface, and its
           vertical and pitch motion matched to the water under the wheels, so nothing slams
           into anything on the first step. It still feels gravity along the slope and the
           passing waves from then on.
  "drop" - the old behaviour: dropped level from just above the highest crest.
spawn_phase_deg picks where on the wave it starts (270 = trough, the calmest spot);
None = random, from the reset seed.

You control one thing: a force. action in [-1, 1] is scaled by max_force and applied at the
board's centre, along the deck (or horizontally, see SkateConfig.force_mode).

Install:  pip install gymnasium pygame numpy
Run:      python skate_waves.py                   # drive it yourself with the arrow keys
          python skate_waves.py --mode random     # random actions, like the LunarLander example
          python skate_waves.py --mode policy     # your own controller, written in my_policy()
          python skate_waves.py --amplitude 0.25 --wave-speed 1.0   # tweak the waves
          python skate_waves.py --orbit-ratio 0   # sinusoidal surface, for comparison
          python skate_waves.py --orbit-period 2  # particles take 2 s per orbit
          python skate_waves.py --contact-surface mesh   # wheels ride the piecewise-linear mesh

Keyboard controls (keyboard mode):
  RIGHT / LEFT   push forward / backward (full force)
  [ / ]          wave amplitude down / up
  - / =          drift speed down / up
  G              toggle Gerstner / sinusoid
  O              show / hide particle orbits
  P  pause       R  reset       ESC  quit

Observation (9 floats):
  0 x                    board centre position along the track (m)
  1 height_above_wave    board centre height above the wave surface directly below it (m)
  2 pitch                board angle, counter-clockwise positive (rad)
  3 vx, 4 vy             board centre velocity (m/s)
  5 pitch_rate           (rad/s)
  6 wave_slope           dh/dx of the surface under the board
  7 sin(psi), 8 cos(psi) where in the wave cycle the board currently sits, with
                         psi = k x + phase (position phase, phase advances at k * c_crest).
                         The crest is always at psi = 90 deg and the trough at 270 deg, and
                         cos(psi) > 0 while climbing toward a crest. psi is uniform in x, so at
                         constant speed the board spends equal time at every psi.
                         (For a Gerstner wave sin(psi) is not exactly height / A.)
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

    # --- Waves: Gerstner particle mesh ---
    wave_amplitude: float = 0.15       # m, vertical orbit radius A (crest height above mean level)
    wave_length: float = 3.0           # m
    wave_speed: float = 0.0            # m/s, drift of ALL particles toward the board (a current).
                                       #      0 = particles orbit in place, no translation
    orbit_period: float | None = None  # s per orbit. None = deep-water physics, T = 2 pi / sqrt(g k)
                                       #      math.inf = no orbiting (shape just drifts at wave_speed)
    orbit_ratio: float = 1.0           # horizontal orbit radius / A. 1 = Gerstner, 0 = sinusoid
    max_steepness: float = 0.9         # cap on Q = k R (Q = 1 is a cusp at the crest)
    mesh_points_per_wavelength: int = 64   # particles in the mesh (it repeats every wavelength)
    contact_surface: str = "smooth"    # "smooth": exact surface, "mesh": straight lines between particles
    newton_iterations: int = 8         # max Newton steps for the smooth surface
    newton_tolerance: float = 1e-10    # m, stop early once the position error is this small

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

    # --- Spawn ---
    spawn_mode: str = "rest"           # "rest": sit still on the surface, "drop": fall from above (old)
    spawn_vx: float = 0.0              # m/s, initial forward speed
    spawn_phase_deg: float | None = None   # position phase under the board at spawn
                                       #      (90 = crest, 270 = trough), None = random
    spawn_iterations: int = 20         # Newton steps when fitting the board onto the surface

    # --- Rendering ---
    screen_width: int = 1000
    screen_height: int = 500
    pixels_per_metre: float = 160.0
    show_orbits: bool = True           # draw the orbit of every few particles
    orbit_draw_every: int = 8          # draw the orbit of every Nth mesh particle


# Colours
SKY = (205, 228, 245)
WATER = (40, 110, 170)
FOAM = (235, 248, 255)
ORBIT = (120, 175, 220)
PARTICLE = (255, 255, 255)
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
        assert c.contact_surface in ("smooth", "mesh")
        assert c.spawn_mode in ("rest", "drop")

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        high = np.full(9, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Moment of inertia of the deck about its centre (thin plank)
        self.inertia = c.mass * (c.deck_length ** 2 + c.deck_thickness ** 2) / 12.0

        # Simulation state: [x, y, pitch, vx, vy, pitch_rate]
        self.state = np.zeros(6)
        self.orbit_phase = 0.0    # omega t + phi_0: how far round its orbit particle a = 0 is
        self.drift = 0.0          # -V t: how far every particle has been carried (m)
        self.phase = 0.0          # shape phase = orbit_phase - k * drift (advances at k * c_crest)
        self.t = 0.0
        self.steps = 0
        self.last_action = 0.0
        self.n_contacts = 0
        self.wheel_spin = [0.0, 0.0]       # for drawing only
        self.wheel_spin_rate = [0.0, 0.0]

        # Particle mesh: labels a_i over one wavelength, plus their current state
        n = c.mesh_points_per_wavelength
        self.mesh_a = np.arange(n) * (c.wave_length / n)
        self.mesh_x = np.zeros(n)
        self.mesh_z = np.zeros(n)
        self.mesh_u = np.zeros(n)
        self.mesh_w = np.zeros(n)
        self._mesh_stale = True            # mesh is only recomputed when something reads it

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

    # ------------------------------------------------------------------ wave parameters
    @property
    def k(self) -> float:
        """Wavenumber (rad/m)."""
        return 2.0 * math.pi / self.cfg.wave_length

    @property
    def omega(self) -> float:
        """Orbit angular frequency (rad/s). Independent of the drift speed."""
        T = self.cfg.orbit_period
        if T is None:
            return math.sqrt(self.cfg.gravity * self.k)      # deep-water dispersion
        return 0.0 if math.isinf(T) else 2.0 * math.pi / T

    @property
    def phase_speed(self) -> float:
        """Speed of the crest shape relative to the water, c_p = omega / k (m/s)."""
        return self.omega / self.k

    @property
    def drift_speed(self) -> float:
        """Speed every particle is carried toward the board (m/s)."""
        return self.cfg.wave_speed

    @property
    def wave_c(self) -> float:
        """Speed of the crests toward the board in the world, c_p + drift (m/s)."""
        return self.phase_speed + self.drift_speed

    @property
    def steepness(self) -> float:
        """Q = k R, clamped below 1 so the surface never folds over itself."""
        c = self.cfg
        q = self.k * c.orbit_ratio * c.wave_amplitude
        return min(max(q, 0.0), c.max_steepness)

    @property
    def orbit_radius(self) -> float:
        """Horizontal orbit radius R (m)."""
        return self.steepness / self.k

    # ------------------------------------------------------------------ particle mesh
    def _update_mesh(self):
        """Positions and velocities of the mesh particles right now (closed form, no drift error).
        Only does work when the waves have moved since the last call."""
        if not self._mesh_stale:
            return
        self._mesh_stale = False
        A, R, om, V = self.cfg.wave_amplitude, self.orbit_radius, self.omega, self.drift_speed
        phi = self.k * self.mesh_a + self.orbit_phase
        s, co = np.sin(phi), np.cos(phi)
        self.mesh_x = self.mesh_a + self.drift + R * co
        self.mesh_z = A * s
        self.mesh_u = -V - R * om * s
        self.mesh_w = A * om * co

    def mesh_tiles(self, x_min: float, x_max: float):
        """Mesh particles (x, z, orbit-centre x) repeated to cover x_min..x_max, for drawing."""
        self._update_mesh()
        L = self.cfg.wave_length
        x0 = self.mesh_x[0]
        n_lo = math.floor((x_min - x0) / L) - 1
        n_hi = math.ceil((x_max - x0) / L) + 1
        shifts = np.arange(n_lo, n_hi + 1) * L
        xs = (self.mesh_x[None, :] + shifts[:, None]).ravel()
        zs = np.tile(self.mesh_z, len(shifts))
        centres = (self.mesh_a[None, :] + self.drift + shifts[:, None]).ravel()
        index = np.tile(np.arange(len(self.mesh_a)), len(shifts))
        return xs, zs, centres, index

    def _mesh_ext(self):
        """Mesh arrays with the first particle repeated one wavelength on, closing the period."""
        self._update_mesh()
        L = self.cfg.wave_length
        return (np.append(self.mesh_x, self.mesh_x[0] + L), np.append(self.mesh_z, self.mesh_z[0]),
                np.append(self.mesh_u, self.mesh_u[0]), np.append(self.mesh_w, self.mesh_w[0]))

    def _mesh_reduce(self, x):
        """Shift x by whole wavelengths into the span of the base mesh tile."""
        self._update_mesh()
        L = self.cfg.wave_length
        x0 = self.mesh_x[0]
        return x0 + np.mod(np.asarray(x, dtype=np.float64) - x0, L)

    def _surface_mesh(self, x: float):
        """Straight-line surface between the two mesh particles either side of x."""
        xs, zs, us, ws = self._mesh_ext()
        xr = float(self._mesh_reduce(x))
        i = int(np.searchsorted(xs, xr, side="right")) - 1
        i = min(max(i, 0), len(xs) - 2)
        dx = xs[i + 1] - xs[i]
        f = (xr - xs[i]) / dx
        h = zs[i] + f * (zs[i + 1] - zs[i])
        hx = (zs[i + 1] - zs[i]) / dx
        u = us[i] + f * (us[i + 1] - us[i])
        w = ws[i] + f * (ws[i + 1] - ws[i])
        return float(h), float(hx), float(u), float(w)

    # ------------------------------------------------------------------ smooth surface
    def _shape_phase(self, x: float) -> float:
        """Orbit phase phi of the water particle currently at world position x (scalar).
        Solves b + R cos(k b + phase) = x for the particle's mean position b with Newton's
        method; returns k b + phase (= its orbit phase)."""
        c = self.cfg
        k, R, ph = self.k, self.orbit_radius, self.phase
        if R == 0.0:
            return k * x + ph
        b = x - R * math.cos(k * x + ph)          # good first guess
        for _ in range(c.newton_iterations):
            phi = k * b + ph
            err = b + R * math.cos(phi) - x
            if abs(err) < c.newton_tolerance:
                break
            b -= err / (1.0 - k * R * math.sin(phi))
        return k * b + ph

    def _shape_phase_array(self, x: np.ndarray) -> np.ndarray:
        """Same as _shape_phase, for numpy arrays."""
        c = self.cfg
        k, R, ph = self.k, self.orbit_radius, self.phase
        x = np.asarray(x, dtype=np.float64)
        if R == 0.0:
            return k * x + ph
        b = x - R * np.cos(k * x + ph)
        for _ in range(c.newton_iterations):
            phi = k * b + ph
            b = b - (b + R * np.cos(phi) - x) / (1.0 - k * R * np.sin(phi))
        return k * b + ph

    def _surface_smooth(self, x: float):
        """Exact surface: height, slope and particle velocity at x.

        h     = A sin(phi)
        dh/dx = A k cos(phi) / (1 - Q sin(phi))
        u     = -V - R omega sin(phi)       (drift + orbit; with the wave under a crest)
        w     =      A omega cos(phi)
        """
        A, k, R, om, V = (self.cfg.wave_amplitude, self.k, self.orbit_radius,
                          self.omega, self.drift_speed)
        phi = self._shape_phase(x)
        s, co = math.sin(phi), math.cos(phi)
        h = A * s
        hx = A * k * co / (1.0 - k * R * s)
        u = -V - R * om * s
        w = A * om * co
        return h, hx, u, w

    # ------------------------------------------------------------------ surface queries
    def surface(self, x: float):
        """Height h, slope dh/dx, and the velocity (u, w) of the water particle at x."""
        if self.cfg.contact_surface == "mesh":
            return self._surface_mesh(x)
        return self._surface_smooth(x)

    def surface_height(self, x):
        """Wave height at x. Works on floats and numpy arrays."""
        if self.cfg.contact_surface == "mesh":
            xs, zs, _, _ = self._mesh_ext()
            h = np.interp(self._mesh_reduce(x), xs, zs)
            return float(h) if np.ndim(x) == 0 else h
        A = self.cfg.wave_amplitude
        if np.ndim(x) == 0:
            return A * math.sin(self._shape_phase(float(x)))
        return A * np.sin(self._shape_phase_array(x))

    def wave_profile(self, x_min: float, x_max: float, n: int):
        """Dense points (x, z) on the exact surface between x_min and x_max, for drawing."""
        k, R, ph, A = self.k, self.orbit_radius, self.phase, self.cfg.wave_amplitude
        b = np.linspace(x_min - R, x_max + R, n)
        phi = k * b + ph
        return b + R * np.cos(phi), A * np.sin(phi)

    # ------------------------------------------------------------------ physics
    def _advance_waves(self, h: float):
        """Move the wave field forward by h seconds: orbit, drift, then update the mesh."""
        self.orbit_phase += self.omega * h
        self.drift -= self.drift_speed * h
        self.phase = self.orbit_phase - self.k * self.drift
        self._mesh_stale = True

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

            hs, hx, us, ws = self.surface(wx)
            norm = math.sqrt(1.0 + hx * hx)
            nx, ny = -hx / norm, 1.0 / norm   # surface normal (pointing up)
            tx, ty = 1.0 / norm, hx / norm    # surface tangent (pointing +x)

            # Distance from wheel centre to surface (along the normal), minus radius
            penetration = c.wheel_radius - (wy - hs) / norm
            if penetration > 0.0:
                # Velocity relative to the water particle under the wheel (orbit + drift)
                vrel_x, vrel_y = wvx - us, wvy - ws
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
        self._advance_waves(h)
        self.n_contacts = contacts

    # ------------------------------------------------------------------ spawning
    def _wheel_contacts(self, x: float, y: float, th: float):
        """For each wheel: offset (rx, ry), penetration, surface normal (nx, ny) and water
        particle velocity (u, w) under it. Same geometry as _substep."""
        c = self.cfg
        cth, sth = math.cos(th), math.sin(th)
        out = []
        for side in (1.0, -1.0):
            bx, by = side * c.wheelbase / 2.0, -c.truck_height
            rx = cth * bx - sth * by
            ry = sth * bx + cth * by
            hs, hx, us, ws = self.surface(x + rx)
            norm = math.sqrt(1.0 + hx * hx)
            penetration = c.wheel_radius - (y + ry - hs) / norm
            out.append((rx, ry, penetration, -hx / norm, 1.0 / norm, us, ws))
        return out

    def _rest_state(self, vx0: float) -> np.ndarray:
        """Board sitting still on the surface at x = 0.

        Position: solve for height y and pitch th so that BOTH wheels sink in by the static
        sag, the spring compression that carries half the weight each (m g / 2k).
        Velocity: forward speed vx0, and vy and pitch rate chosen so both wheels move with the
        water along the surface normal (zero normal speed), so the dampers feel no impact."""
        c = self.cfg
        sag = c.mass * c.gravity / (2.0 * c.contact_stiffness)

        # First guess: deck parallel to the local slope, wheels just touching
        _, hx0, _, _ = self.surface(0.0)
        th = math.atan(hx0)
        y = self.surface_height(0.0) + (c.wheel_radius + c.truck_height) / math.cos(th)

        # Newton on (y, th): penetration of each wheel = sag
        def residual(y_, th_):
            return np.array([wc[2] - sag for wc in self._wheel_contacts(0.0, y_, th_)])

        eps = 1e-7
        for _ in range(c.spawn_iterations):
            r = residual(y, th)
            if np.max(np.abs(r)) < 1e-9:
                break
            J = np.column_stack([(residual(y + eps, th) - r) / eps,
                                 (residual(y, th + eps) - r) / eps])
            dy, dth = np.linalg.solve(J, -r)
            y, th = y + dy, th + dth

        # Velocities: normal speed relative to the water = 0 at both wheels.
        # vn_i = (vx - om ry_i - u_i) nx_i + (vy + om rx_i - w_i) ny_i = 0, linear in (vy, om)
        rows, rhs = [], []
        for rx, ry, _, nx, ny, us, ws in self._wheel_contacts(0.0, y, th):
            rows.append([ny, rx * ny - ry * nx])
            rhs.append(-(vx0 - us) * nx + ws * ny)
        vy, om = np.linalg.solve(np.array(rows), np.array(rhs))

        return np.array([0.0, y, th, vx0, vy, om])

    def _crashed(self) -> bool:
        x, y, th = self.state[:3]
        too_tilted = math.cos(th) < math.cos(math.radians(self.cfg.crash_tilt_deg))
        below_surface = y < self.surface_height(x)
        return bool(too_tilted or below_surface)

    def _get_obs(self) -> np.ndarray:
        x, y, th, vx, vy, om = self.state
        h, hx, _, _ = self.surface(x)
        psi = self.k * x + self.phase        # position phase: crest at 90 deg, trough at 270 deg
        return np.array([x, y - h, th, vx, vy, om, hx, math.sin(psi), math.cos(psi)],
                        dtype=np.float32)

    def _get_info(self) -> dict:
        return {"t": self.t, "wheels_in_contact": self.n_contacts,
                "force_N": self.last_action * self.cfg.max_force,
                "wave_height": self.surface_height(self.state[0]),
                "wave_steepness": self.steepness,
                "crest_speed": self.wave_c}

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c = self.cfg
        random_phase = float(self.np_random.uniform(0.0, 2.0 * math.pi))
        if c.spawn_phase_deg is None:
            self.orbit_phase = random_phase
        else:
            self.orbit_phase = math.radians(c.spawn_phase_deg)   # board spawns at x = 0, so psi = phase
        self.drift = 0.0
        self.phase = self.orbit_phase
        self._mesh_stale = True
        if c.spawn_mode == "rest":
            self.state = self._rest_state(c.spawn_vx)
        else:
            y0 = c.wave_amplitude + c.wheel_radius + c.truck_height + 0.05  # drop onto the waves
            self.state = np.array([0.0, y0, 0.0, c.spawn_vx, 0.0, 0.0])
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
                pygame.display.set_caption("Skateboard on Gerstner waves")
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

        # Water surface: the exact curve, or straight lines through the mesh (what the wheels feel)
        x_left = x - board_px / ppm
        x_right = x + (W - board_px) / ppm
        self._mesh_stale = True            # settings may have changed from the keyboard
        px_all, pz_all, centres, index = self.mesh_tiles(x_left, x_right)
        if c.contact_surface == "mesh":
            wxs, wzs = px_all, pz_all
        else:
            wxs, wzs = self.wave_profile(x_left, x_right, 600)
        spx = board_px + (wxs - x) * ppm
        spy = ground_px - (wzs - self._cam_y) * ppm
        pts = list(zip(spx.tolist(), spy.tolist()))
        pygame.draw.polygon(s, WATER, pts + [(pts[-1][0], H), (pts[0][0], H)])

        # Particle orbits: circle round each orbit centre, with a spoke to the particle
        R, A = self.orbit_radius, c.wave_amplitude
        if c.show_orbits:
            for xp, zp, xc, i in zip(px_all, pz_all, centres, index):
                if i % c.orbit_draw_every or not (x_left - 1 < xc < x_right + 1):
                    continue
                cx, cy = to_screen(xc, 0.0)
                rect = pygame.Rect(0, 0, max(2, int(2 * R * ppm)), max(2, int(2 * A * ppm)))
                rect.center = (int(cx), int(cy))
                pygame.draw.ellipse(s, ORBIT, rect, 1)
                pygame.draw.line(s, ORBIT, (cx, cy), to_screen(xp, zp), 1)

        pygame.draw.lines(s, FOAM, False, pts, 3)

        # Mesh particles
        for xp, zp in zip(px_all, pz_all):
            if x_left - 0.1 < xp < x_right + 0.1:
                sx, sy = to_screen(xp, zp)
                pygame.draw.circle(s, PARTICLE, (int(sx), int(sy)), 2)

        # Distance markers every metre (labelled every 5 m) so you can see the board move
        for m in range(math.floor(x_left), math.ceil(x_right) + 1):
            mx, my = to_screen(m, float(self.surface_height(float(m))))
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
        model = "Gerstner" if self.orbit_radius > 0.0 else "sinusoid"
        period = 2.0 * math.pi / self.omega if self.omega > 0.0 else math.inf
        lines = [
            f"t {self.t:6.2f} s   x {x:7.2f} m   vx {vx:+5.2f} m/s",
            f"pitch {math.degrees(th):+6.1f} deg   force {u * c.max_force:+6.1f} N   "
            f"{status} ({self.n_contacts}/2 wheels)",
            f"waves ({model}, {c.contact_surface}): A {c.wave_amplitude:.2f} m   "
            f"wavelength {c.wave_length:.1f} m   Q {self.steepness:.2f}",
            f"orbit period {period:.2f} s   drift {self.drift_speed:+.2f} m/s   "
            f"crest speed {self.wave_c:+.2f} m/s",
        ]
        if self.show_controls:
            lines.append("<- -> push   [ ] amplitude   - = drift   G Gerstner/sine   O orbits   "
                         "P pause   R reset   ESC quit")
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
    gerstner_ratio = env.cfg.orbit_ratio if env.cfg.orbit_ratio > 0.0 else 1.0

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
            elif ev.key == pygame.K_g:
                env.cfg.orbit_ratio = 0.0 if env.cfg.orbit_ratio > 0.0 else gerstner_ratio
            elif ev.key == pygame.K_o:
                env.cfg.show_orbits = not env.cfg.show_orbits
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
    parser = argparse.ArgumentParser(description="Skateboard on rigid Gerstner waves")
    parser.add_argument("--mode", choices=["keyboard", "random", "policy"], default="keyboard")
    parser.add_argument("--steps", type=int, default=3000, help="steps to run in random/policy mode")
    parser.add_argument("--amplitude", type=float, default=None, help="wave amplitude (m)")
    parser.add_argument("--wavelength", type=float, default=None, help="wavelength (m)")
    parser.add_argument("--wave-speed", type=float, default=None,
                        help="drift of all particles toward the board (m/s), 0 = orbit in place")
    parser.add_argument("--orbit-period", type=float, default=None,
                        help="seconds per particle orbit (default: deep-water physics)")
    parser.add_argument("--orbit-ratio", type=float, default=None,
                        help="horizontal/vertical orbit ratio: 1 = Gerstner, 0 = sinusoid")
    parser.add_argument("--contact-surface", choices=["smooth", "mesh"], default=None,
                        help="smooth = exact surface, mesh = straight lines between particles")
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
    if args.orbit_period is not None:
        cfg.orbit_period = args.orbit_period
    if args.orbit_ratio is not None:
        cfg.orbit_ratio = args.orbit_ratio
    if args.contact_surface is not None:
        cfg.contact_surface = args.contact_surface
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