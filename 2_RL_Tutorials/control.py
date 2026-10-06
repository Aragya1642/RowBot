"""
bang_bang.py - bang-bang control of the skateboard based on its pitch.

Rule: nose up (climbing a wave face) -> full push forward
      nose down (going down the back) -> full push backward

Put this file in the same folder as skate_waves.py and run:  python bang_bang.py
"""

import numpy as np
from skate_waves import SkateWavesEnv

PITCH_INDEX = 2      # obs[2] is pitch in radians, counter-clockwise (nose up) positive


def bang_bang(obs):
    pitch = obs[PITCH_INDEX]
    if pitch > 0:
        return np.array([1.0])    # full forward
    else:
        return np.array([-1.0])   # full backward


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