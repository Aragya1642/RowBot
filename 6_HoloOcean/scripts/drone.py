import os
import cv2
import holoocean
import numpy as np
from holoocean import packagemanager as pm
from pynput import keyboard

WORLD = "OpenWater"         # change to any world with land
SPAWN = [0, 0, 5]            # drone spawn, meters
USV_SPAWN = [-11.5, 0, 0]      # boat spawn, meters
ALTITUDE_TARGET = 3000.0
INITIAL_SEA_STATE = 0
USV_THRUST = 1500            # tune for your vessel
SAVE_FRAMES = True
OUT_DIR = "drone_frames"

if SAVE_FRAMES:
    os.makedirs(OUT_DIR, exist_ok=True)

cfg = {
    "name": "uav_usv_waves",
    "world": WORLD,
    "package_name": "Ocean",
    "main_agent": "drone",
    "ticks_per_sec": 60,
    "fft_waves": {"sea_state": INITIAL_SEA_STATE},
    "agents": [
        {
            "agent_name": "drone",
            "agent_type": "UavAgent",
            "control_scheme": 1,   # [pitch, roll, yaw rate, altitude] targets
            "location": SPAWN,
            "rotation": [0, 0, 0],
            "sensors": [
                {"sensor_type": "PoseSensor"},
                {
                    "sensor_type": "RGBCamera",
                    "sensor_name": "DownCamera",
                    "socket": "CameraSocket",
                    "rotation": [0, 90, 0],  # remove or flip sign if not looking down
                    "configuration": {"CaptureWidth": 1024, "CaptureHeight": 512}
                }
            ]
        },
        {
            "agent_name": "sv0",
            "agent_type": "SurfaceVessel",
            "control_scheme": 0,   # [left_thrust, right_thrust]
            "location": USV_SPAWN,
            "rotation": [0, 0, 0],
            "sensors": [{"sensor_type": "PoseSensor"}]
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

def usv_command():
    left, right = 0.0, 0.0
    if 'w' in pressed: left += USV_THRUST; right += USV_THRUST
    if 's' in pressed: left -= USV_THRUST; right -= USV_THRUST
    if 'a' in pressed: left -= USV_THRUST / 2; right += USV_THRUST / 2   # turn left
    if 'd' in pressed: left += USV_THRUST / 2; right -= USV_THRUST / 2   # turn right
    return np.array([left, right])

print("W/S: boat forward/back | A/D: boat turn | 0-9: sea state | Q: quit")

# ---------- sim loop ----------
with holoocean.make(scenario_cfg=cfg) as env:
    env.agents["drone"].set_control_scheme(1)
    drone_command = np.array([0.0, 0.0, 0.0, ALTITUDE_TARGET])
    sea_state = INITIAL_SEA_STATE

    tick = 0
    while 'q' not in pressed:
        while events:
            k = events.pop(0)
            if k is not None and k.isdigit() and int(k) != sea_state:
                sea_state = int(k)
                env.fftwaves.fft_sea_state(sea_state)
                print(f"sea state -> {sea_state}")

        env.act("drone", drone_command)
        env.act("sv0", usv_command())
        state = env.tick()
        tick += 1

        # Multi-agent: state is keyed by agent name, then sensor name
        if "DownCamera" in state["drone"]:
            img = state["drone"]["DownCamera"][:, :, 0:3]   # drop alpha channel
            cv2.imshow("Drone Down Camera", img)
            cv2.waitKey(1)
            if SAVE_FRAMES:
                cv2.imwrite(os.path.join(OUT_DIR, f"tick_{tick:06d}.png"), img)

        if tick % 60 == 0:
            drone_pos = state["drone"]["PoseSensor"][:3, 3]
            usv_pos = state["sv0"]["PoseSensor"][:3, 3]
            print(f"tick {tick}: sea state {sea_state} | "
                  f"drone height {drone_pos[2]:.2f} m | "
                  f"usv {np.round(usv_pos, 2)}")

listener.stop()
cv2.destroyAllWindows()