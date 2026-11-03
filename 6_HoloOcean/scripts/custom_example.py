import os
from datetime import datetime
import cv2
import holoocean
import numpy as np
from pynput import keyboard

DRONE_HEIGHT = 20.0   # meters
# Each run saves to 6_HoloOcean/data/<script>_<timestamp>/frames
HOLOOCEAN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
RUN_DIR = os.path.join(HOLOOCEAN_DIR, "data", f"{SCRIPT_NAME}_{datetime.now():%Y-%m-%d_%H-%M-%S}")
OUT_DIR = os.path.join(RUN_DIR, "frames")
os.makedirs(OUT_DIR, exist_ok=True)

cfg = {
    "name": "waves_teleop_uav",
    "world": "OpenWater",
    "package_name": "Ocean",
    "main_agent": "sv0",
    "ticks_per_sec": 60,
    "fft_waves": {"sea_state": 0},
    "agents": [
        {
            "agent_name": "sv0",
            "agent_type": "SurfaceVessel",
            "control_scheme": 0,
            "location": [0, 0, 1],
            "rotation": [0, 0, 0],
            "sensors": [{"sensor_type": "PoseSensor"}]
        },
        {
            "agent_name": "drone",
            "agent_type": "UavAgent",
            "control_scheme": 1,   # [pitch, roll, yaw rate, altitude] targets
            "location": [0, 0, DRONE_HEIGHT],
            "rotation": [0, 0, 0],
            "sensors": [
                {"sensor_type": "PoseSensor"},
                {
                    "sensor_type": "RGBCamera",
                    "sensor_name": "DownCamera",
                    "socket": "CameraSocket",
                    "rotation": [0, 0, 0],  # remove or flip sign if not looking down
                    "configuration": {"CaptureWidth": 512, "CaptureHeight": 512}
                }
            ]
        }
    ]
}

# ---------- keyboard ----------
pressed, events = set(), []

def on_press(key):
    try:
        if key.char not in pressed:
            events.append(key.char)
        pressed.add(key.char)
    except AttributeError:
        pass

def on_release(key):
    try:
        pressed.discard(key.char)
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

THRUST = 1500

def boat_command():
    left, right = 0.0, 0.0
    if 'w' in pressed: left += THRUST; right += THRUST
    if 's' in pressed: left -= THRUST; right -= THRUST
    if 'a' in pressed: left -= THRUST / 2; right += THRUST / 2
    if 'd' in pressed: left += THRUST / 2; right -= THRUST / 2
    return np.array([left, right])

def drone_command():
    # Level, no yaw rotation, fixed height (altitude target likely in cm)
    return np.array([0.0, 0.0, 0.0, DRONE_HEIGHT])

print("W/S/A/D: drive | 0-9: sea state | P: pause/resume | N: step one tick | Q: quit")

# ---------- sim loop ----------
with holoocean.make(scenario_cfg=cfg) as env:
    env.agents["drone"].set_control_scheme(1)
    paused, tick = False, 0

    while 'q' not in pressed:
        step_once = False
        while events:
            k = events.pop(0)
            if k == 'p':
                paused = not paused
                print("Paused" if paused else "Running")
            elif k == 'n' and paused:
                step_once = True
            elif k.isdigit():
                env.fftwaves.fft_sea_state(int(k))

        if paused and not step_once:
            cv2.waitKey(30)
            continue

        env.act("sv0", boat_command())
        env.act("drone", drone_command())

        state = env.tick()
        tick += 1

        boat_pos = state["sv0"]["PoseSensor"][:3, 3]
        drone_pos = state["drone"]["PoseSensor"][:3, 3]

        if tick % 60 == 0:
            print(f"tick {tick}: drone height {drone_pos[2]:.2f} m")

        if "DownCamera" in state["drone"]:
            img = state["drone"]["DownCamera"][:, :, 0:3]
            cv2.imwrite(os.path.join(OUT_DIR, f"tick_{tick:06d}.png"), img)
            cv2.imshow("Drone Down Camera", img)
            cv2.waitKey(1)

        if step_once:
            print(f"tick {tick}: boat {np.round(boat_pos, 2)}, drone {np.round(drone_pos, 2)}")

listener.stop()
cv2.destroyAllWindows()