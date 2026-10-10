"""
bang_bang.py - bang-bang control of the skateboard based on its pitch.

Rule: nose up (climbing a wave face) -> full push forward
      nose down (going down the back) -> full push backward

Put this file in the same folder as skate_waves.py and run:  python bang_bang.py
"""

import numpy as np
from skate_waves import SkateWavesEnv, SkateConfig

#######################################################
# Boat Control parameters
#######################################################
FORWARD_TARGET_VELOCITY = 2 # m/s
K_GAIN = 1

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
WAVE_SPEED = -1.5                    # m/s - positive = waves move toward the board
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

def bang_bang(obs):
    vx_velocity = obs[VX_INDEX]
    if vx_velocity < FORWARD_TARGET_VELOCITY:
        return 1.0    # full forward
    else:
        return -1.0   # full backward

def p_control(obs):
    vx_velocity = obs[VX_INDEX]
    error = FORWARD_TARGET_VELOCITY - vx_velocity
    action = K_GAIN*error
    # print(action)
    return action

def pitch_p_control(obs):
    pitch = obs[PITCH_INDEX]
    if pitch > 0:
        vx_velocity = obs[VX_INDEX]
        error = FORWARD_TARGET_VELOCITY - vx_velocity
        action = K_GAIN*error
    else:
        action = 0.0
    return action

def positive_pitch_p_control(obs):
    pitch = obs[PITCH_INDEX]
    if pitch > 0:
        vx_velocity = obs[VX_INDEX]
        error = FORWARD_TARGET_VELOCITY - vx_velocity
        action = max(0.0, K_GAIN*error)
    else:
        action = 0.0
    return action

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

env = SkateWavesEnv(config=config, render_mode="human")
obs, info = env.reset(seed=42)

for _ in range(3000):
    action = p_control(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    if env.window_closed:
        break
    if terminated or truncated:
        obs, info = env.reset()

env.close()