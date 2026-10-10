"""
bang_bang.py - bang-bang control of the skateboard based on its pitch.

Rule: nose up (climbing a wave face) -> full push forward
      nose down (going down the back) -> full push backward

Put this file in the same folder as skate_waves.py and run:  python bang_bang.py
"""

import numpy as np
from skate_waves import SkateWavesEnv

# Boat Control parameters
FORWARD_TARGET_VELOCITY = 2 # m/s
K_GAIN = 1

# Indices for observations
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
    vx_velocity = obs[3]
    error = FORWARD_TARGET_VELOCITY - vx_velocity
    action = K_GAIN*error
    # print(action)
    return action

env = SkateWavesEnv(render_mode="human")
obs, info = env.reset(seed=42)

for _ in range(3000):
    action = bang_bang(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    if env.window_closed:
        break
    if terminated or truncated:
        obs, info = env.reset()

env.close()