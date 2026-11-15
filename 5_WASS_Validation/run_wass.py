#!/usr/bin/env python3
"""Run the whole WASS pipeline on a folder of stereo images, non-interactively, and record how long each step takes.

    images + calibration  ->  prepare -> match -> autocalibrate -> stereo -> grid -> video
                                                                    gridding/gridded.nc, reconstruction.mp4, gridded.mp4

Does what `wasscli` + `wassgridsurface` do, without prompts:
  - lays out a WASS sequence folder (config/, output/, gridding/, logs/) in --out (images are not copied)
  - copies the calibration files, and copies or generates prepare/matcher/stereo_config.txt
  - works around the WASS DENSE_SCALE=1.0 bug (sets DENSE_SCALE=0.9999 unless you set another value)
  - matches a random subset of frames, autocalibrates the extrinsics, reconstructs every frame in parallel
  - picks the grid area automatically (largest square covered by points in several frames) unless given
  - renders gridding/reconstruction.mp4 (surface overlaid on the undistorted camera images + top view, view_stereo.py)
    and gridding/gridded.mp4 (top view only, view_nc.py); skip with --no-video
  - writes timings.json (stages + per-frame), timings_frames.csv and a summary table

Image folder layouts that are recognised:
  IMAGES/cam0 + IMAGES/cam1             (WASS)
  IMAGES/cam1 + IMAGES/cam2             (Ifremer AA / YS)
  IMAGES/*_01.ext + IMAGES/*_02.ext     (Ifremer BS / LJ, flat)
or give the two folders explicitly with --cam0 / --cam1.

The calibration folder needs intrinsics_00.xml, intrinsics_01.xml, distortion_00.xml and distortion_01.xml.
Any prepare/matcher/stereo_config.txt found there are used too. Default: IMAGES/../config.

Run inside `nix-shell` (5_WASS_Validation/shell.nix), which puts the WASS tools on PATH.

Examples:
    python3 run_wass.py data/wass_quickstart/seq/input --baseline 3.0 --out data/wass_quickstart/run1
    python3 run_wass.py data/wass_quickstart/seq/input --baseline 3.0 --out /tmp/test --frames 6 --no-video
    python3 run_wass.py ... --steps grid --grid-center 0 -35 --grid-size 40    # redo only the gridding
    python3 run_wass.py --out data/wass_quickstart/run1 --plot-only            # redraw timings.png
"""

import argparse
import concurrent.futures as cf
import csv
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STEPS = ["prepare", "match", "autocalibrate", "stereo", "grid", "video"]
WASS_TOOLS = ["wass_prepare", "wass_match", "wass_autocalibrate", "wass_stereo"]
IMAGE_EXTS = ["tif", "tiff", "png", "jpg", "jpeg", "bmp"]
CALIB_FILES = ["intrinsics_00.xml", "intrinsics_01.xml", "distortion_00.xml", "distortion_01.xml"]
CONFIG_FILES = {"prepare_config.txt": "wass_prepare", "matcher_config.txt": "wass_match", "stereo_config.txt": "wass_stereo"}


def log(msg=""):
    print(msg, flush=True)


def die(msg):
    sys.exit(f"ERROR: {msg}")


# ----------------------------------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------------------------------

def list_images(folder, suffix=""):
    folder = Path(folder).resolve()  # the WASS tools run from inside the sequence folder
    for ext in IMAGE_EXTS:
        files = sorted(p for p in folder.iterdir() if p.is_file() and re.search(rf"{suffix}\.{ext}$", p.name, re.I))
        if files:
            return files
    return []


def find_pairs(images, cam0=None, cam1=None):
    """Return (cam0 files, cam1 files, description of the layout)."""
    if cam0 or cam1:
        if not (cam0 and cam1):
            die("--cam0 and --cam1 must be given together")
        return list_images(Path(cam0)), list_images(Path(cam1)), f"{cam0} + {cam1}"
    images = Path(images)
    if not images.is_dir():
        die(f"{images} is not a folder")
    for a, b in (("cam0", "cam1"), ("cam1", "cam2")):
        if (images / a).is_dir() and (images / b).is_dir():
            return list_images(images / a), list_images(images / b), f"{images}/{a} + {images}/{b}"
    left, right = list_images(images, "_01"), list_images(images, "_02")
    if left or right:
        return left, right, f"{images}/*_01 + {images}/*_02"
    die(f"no stereo images found in {images} (expected cam0/+cam1/, cam1/+cam2/ or *_01/*_02 files)")


def infer_fps(files):
    """From WASS-style names NNNNNN_<timestamp in ms>_01.ext, e.g. 000001_0000000000199_01.jpg."""
    ts = [int(m.group(1)) for f in files if (m := re.match(r"^\d+_(\d+)_0[12]\.", f.name))]
    if len(ts) < 2:
        return None
    dt = np.median(np.diff(ts))
    return round(1000.0 / dt, 3) if dt > 0 else None


def check_tools():
    missing = [t for t in WASS_TOOLS + ["wassgridsurface"] if shutil.which(t) is None]
    if missing:
        die(f"not found on PATH: {', '.join(missing)}. Run this inside `nix-shell` in {HERE} (build first with ./build_wass.sh).")


def wass_version():
    try:
        out = subprocess.run(["wass_stereo"], capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"v\.?\s*([\d.]+)", out)
        return m.group(1) if m else None
    except Exception:
        return None


# ----------------------------------------------------------------------------------------------------------------------
# Sequence folder
# ----------------------------------------------------------------------------------------------------------------------

def setup_config(calib, config):
    config.mkdir(parents=True, exist_ok=True)
    for name in CALIB_FILES:
        if not (calib / name).exists():
            die(f"{calib / name} not found (need {', '.join(CALIB_FILES)})")
        shutil.copy2(calib / name, config / name)
    for name, tool in CONFIG_FILES.items():
        if (calib / name).exists():
            if (calib / name).resolve() != (config / name).resolve():
                shutil.copy2(calib / name, config / name)
            log(f"  {name}: from {calib}")
        else:
            subprocess.run([tool, "--genconfig"], cwd=config, capture_output=True)
            if not (config / name).exists():
                die(f"`{tool} --genconfig` did not create {name}")
            log(f"  {name}: WASS defaults ({tool} --genconfig)")
    fix_dense_scale(config / "stereo_config.txt")


def fix_dense_scale(path):
    """WASS commit 66f43a2: with DENSE_SCALE == 1.0 (the default) wass_stereo gets empty images and aborts."""
    lines = path.read_text().splitlines()
    active = [i for i, l in enumerate(lines) if re.match(r"\s*DENSE_SCALE\s*=", l)]
    if active:
        value = float(lines[active[-1]].split("=", 1)[1].split("#")[0])
        if value != 1.0:
            log(f"  stereo_config.txt: DENSE_SCALE={value:g} (kept)")
            return
        lines[active[-1]] = "DENSE_SCALE=0.9999"
    else:
        lines.append("DENSE_SCALE=0.9999")
    path.write_text("\n".join(lines) + "\n")
    log("  stereo_config.txt: DENSE_SCALE=0.9999 (workaround for the WASS DENSE_SCALE=1.0 bug)")


# ----------------------------------------------------------------------------------------------------------------------
# Running tools, with timing
# ----------------------------------------------------------------------------------------------------------------------

def children_cpu():
    """CPU seconds (user + system) used so far by finished child processes, and their peak RSS in MB.

    The tools run as children of this script, so the difference across a stage is that stage's CPU time,
    whatever the number of workers (a hardware-neutral cost, unlike wall time)."""
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    return ru.ru_utime + ru.ru_stime, ru.ru_maxrss / 1024


class Timer:
    def __init__(self):
        self.stages = {}   # name -> dict(seconds, cpu_seconds, started, ...)
        self.frames = []   # dict(stage, frame, seconds, ok)

    def stage(self, name):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t0 = time.perf_counter()
                self.cpu0 = children_cpu()[0]
                self.started = datetime.now().isoformat(timespec="seconds")
                log(f"\n=== {name} ===")
                return self

            def __exit__(self, exc_type, *_):
                sec = time.perf_counter() - self.t0
                cpu, peak_mb = children_cpu()
                timer.stages[name] = {"seconds": round(sec, 3), "cpu_seconds": round(cpu - self.cpu0, 3),
                                      "started": self.started, "ok": exc_type is None}
                timer.peak_rss_mb = round(peak_mb, 1)
                log(f"--- {name}: {fmt(sec)}  (CPU {fmt(cpu - self.cpu0)})")

        return _Ctx()


def fmt(sec):
    return f"{sec:.1f} s" if sec < 120 else f"{sec / 60:.1f} min"


def run(cmd, cwd, logfile, timer=None, name=None):
    """Run one command; with timer and name, also record it as a (sub)stage with its CPU time."""
    t0, cpu0 = time.perf_counter(), children_cpu()[0]
    with open(logfile, "w") as f:
        ret = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    sec, ok = time.perf_counter() - t0, ret.returncode == 0
    if timer is not None:
        timer.stages[name] = {"seconds": round(sec, 3), "cpu_seconds": round(children_cpu()[0] - cpu0, 3), "ok": ok}
    return ok, sec


def run_frames(timer, stage, frames, make_cmd, seq, workers, expect=None):
    """Run make_cmd(frame) for each frame in parallel; record per-frame times. Returns the frames that failed.

    expect: file that must exist in the frame's workdir afterwards (some WASS tools exit 0 on errors)."""
    logs = seq / "logs" / stage
    logs.mkdir(parents=True, exist_ok=True)
    failed = []

    def task(t):
        if expect:  # so a leftover from an earlier run is not mistaken for success
            (seq / "output" / f"{t:06d}_wd" / expect).unlink(missing_ok=True)
        ok, sec = run(make_cmd(t), seq, logs / f"{t:06d}.log")
        if expect and not (seq / "output" / f"{t:06d}_wd" / expect).exists():
            ok = False
        return t, ok, sec

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for done, (t, ok, sec) in enumerate(ex.map(task, frames), 1):
            timer.frames.append({"stage": stage, "frame": t, "seconds": round(sec, 3), "ok": ok})
            if not ok:
                failed.append(t)
            log(f"  [{done:4d}/{len(frames)}] frame {t:06d}  {sec:6.1f} s  {'ok' if ok else 'FAILED, see ' + str(logs / f'{t:06d}.log')}")
    return failed


def workdirs(seq):
    return sorted((seq / "output").glob("*_wd"))


# ----------------------------------------------------------------------------------------------------------------------
# Grid area
# ----------------------------------------------------------------------------------------------------------------------

def auto_grid_area(seq, baseline, n_frames=5, margin=0.05):
    """Largest axis-aligned square (in sea-plane coordinates, m) that has points in all of a few sample frames."""
    import cv2
    from wassgridsurface.wass_utils import align_on_sea_plane, load_camera_mesh

    plane = np.nanmean(np.atleast_2d(np.loadtxt(seq / "output" / "planes.txt")), axis=0)
    wds = workdirs(seq)
    sample = [wds[i] for i in np.unique(np.linspace(0, len(wds) - 1, min(n_frames, len(wds))).astype(int))]
    clouds = []
    for wd in sample:
        pts = align_on_sea_plane(load_camera_mesh(str(wd / "mesh_cam.xyzC")), plane) * baseline
        clouds.append(pts[:2, ::10])
    allpts = np.hstack(clouds)
    lo, hi = np.percentile(allpts, 0.5, axis=1), np.percentile(allpts, 99.5, axis=1)
    cell = max(hi - lo) / 200
    nx, ny = [int(np.ceil(v)) + 1 for v in (hi - lo) / cell]

    covered = np.ones((ny, nx), bool)
    for c in clouds:
        ix, iy = ((c - lo[:, None]) / cell).astype(int)
        keep = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        occ = np.zeros((ny, nx), np.uint8)
        occ[iy[keep], ix[keep]] = 1
        occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # fill small gaps
        covered &= occ.astype(bool)

    # Largest all-True square (dynamic programming).
    dp = np.zeros((ny + 1, nx + 1), int)
    best, bi, bj = 0, 0, 0
    for i in range(ny):
        for j in range(nx):
            if covered[i, j]:
                dp[i + 1, j + 1] = 1 + min(dp[i, j + 1], dp[i + 1, j], dp[i, j])
                if dp[i + 1, j + 1] > best:
                    best, bi, bj = dp[i + 1, j + 1], i, j
    if best < 4:
        die("could not find a grid area covered by points; give --grid-center and --grid-size")
    size = best * cell * (1 - 2 * margin)
    cx = lo[0] + (bj + 1 - best / 2) * cell
    cy = lo[1] + (bi + 1 - best / 2) * cell
    return (round(float(cx), 2), round(float(cy), 2)), round(float(size), 2)


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="?", help="folder with the stereo images (see layouts above)")
    p.add_argument("--cam0", help="folder with the cam0 (left, _01) images; use with --cam1 instead of IMAGES")
    p.add_argument("--cam1", help="folder with the cam1 (right, _02) images")
    p.add_argument("--calib", help="folder with intrinsics/distortion xml files (default: IMAGES/../config)")
    p.add_argument("--out", required=True, help="sequence folder to create / use (config/, output/, gridding/, logs/)")
    p.add_argument("--baseline", type=float, help="camera baseline in m (scales the output to metres; needed for the grid step)")
    p.add_argument("--fps", type=float, help="frame rate in Hz (default: from WASS-style filename timestamps)")
    p.add_argument("--start", type=int, default=0, help="first stereo pair to use (default 0)")
    p.add_argument("--frames", type=int, default=0, help="number of stereo pairs to process (default 0 = all)")
    p.add_argument("--workers", type=int, default=os.cpu_count() // 2 or 1, help="parallel processes (default: half the CPUs)")
    p.add_argument("--match-frames", type=int, default=50, help="frames used for extrinsic calibration (default 50, as wasscli)")
    p.add_argument("--seed", type=int, default=0, help="random seed for choosing match frames (default 0)")
    p.add_argument("--grid-center", type=float, nargs=2, metavar=("X", "Y"), help="grid centre in m (default: automatic)")
    p.add_argument("--grid-size", type=float, help="grid side in m (default: automatic)")
    p.add_argument("--grid-n", type=int, default=256, help="grid points per side, even (default 256)")
    p.add_argument("--no-video", action="store_true",
                   help="skip the videos (gridding/reconstruction.mp4: surface over the camera images; gridded.mp4: top view)")
    p.add_argument("--steps", default=",".join(STEPS),
                   help=f"comma-separated steps to run, from: {','.join(STEPS)} (default: all)")
    p.add_argument("--plot-only", action="store_true", help="only redraw timings.png from the timings files in --out")
    p.add_argument("--overwrite", action="store_true", help="delete existing output/, gridding/ and logs/ in --out before prepare")
    args = p.parse_args()
    if args.plot_only:
        plot_timings(Path(args.out).resolve())
        return

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    if args.no_video and "video" in steps:
        steps.remove("video")
    if bad := set(steps) - set(STEPS):
        die(f"unknown step(s): {', '.join(sorted(bad))}")
    steps = [s for s in STEPS if s in steps]
    if args.grid_n % 2:
        die("--grid-n must be even")
    if (args.grid_center is None) != (args.grid_size is None):
        die("give both --grid-center and --grid-size, or neither")
    if "grid" in steps and args.baseline is None:
        die("--baseline is needed for the grid step")
    check_tools()

    seq = Path(args.out).resolve()
    config, output, gridding = seq / "config", seq / "output", seq / "gridding"
    timer = Timer()
    t_start = time.perf_counter()
    info = {"started": datetime.now().isoformat(timespec="seconds"), "command": sys.argv, "steps": steps,
            "machine": {"host": platform.node(), "cpus": os.cpu_count(), "platform": platform.platform()},
            "wass_version": wass_version(), "workers": args.workers, "baseline_m": args.baseline}

    if "prepare" in steps:
        if not args.images and not args.cam0:
            die("IMAGES (or --cam0/--cam1) is needed for the prepare step")
        cam0, cam1, layout = find_pairs(args.images, args.cam0, args.cam1)
        if len(cam0) != len(cam1):
            die(f"{len(cam0)} cam0 images but {len(cam1)} cam1 images")
        end = args.start + args.frames if args.frames > 0 else None
        cam0, cam1 = cam0[args.start:end], cam1[args.start:end]
        if len(cam0) < 3:
            die(f"only {len(cam0)} stereo pairs selected; WASS needs at least 3")
        fps = args.fps or infer_fps(cam0)
        calib = Path(args.calib) if args.calib else (Path(args.images or args.cam0) / ".." / "config")
        log(f"Images:   {layout}: {len(cam0)} pairs ({cam0[0].name} ... {cam0[-1].name})")
        log(f"Calib:    {calib.resolve()}")
        log(f"Output:   {seq}")
        log(f"Baseline: {args.baseline} m, fps: {fps if fps else 'unknown (pass --fps)'}, workers: {args.workers}")

        if output.exists() and any(output.iterdir()):
            if not args.overwrite:
                die(f"{output} is not empty; pass --overwrite to replace it, or --steps to skip prepare")
            for d in (output, gridding, seq / "logs"):
                shutil.rmtree(d, ignore_errors=True)
        output.mkdir(parents=True, exist_ok=True)
        log("Config:")
        setup_config(calib, config)
        (seq / "pairs.txt").write_text("".join(f"{t:06d} {a} {b}\n" for t, (a, b) in enumerate(zip(cam0, cam1))))
        info.update(images=layout, n_pairs=len(cam0), fps=fps, calib=str(calib.resolve()))
    else:
        old = json.loads((seq / "timings.json").read_text()) if (seq / "timings.json").exists() else {}
        fps = args.fps or old.get("fps")
        info.update(n_pairs=len(workdirs(seq)), fps=fps)
        # Keep the timings of steps that are not re-run now.
        for k, v in old.get("stages", {}).items():
            if k.split("_")[0] not in steps:
                timer.stages[k] = v
        if (seq / "timings_frames.csv").exists():
            with open(seq / "timings_frames.csv") as f:
                timer.frames = [{"stage": r["stage"], "frame": int(r["frame"]), "seconds": float(r["seconds"]),
                                 "ok": r["ok"] == "True"} for r in csv.DictReader(f) if r["stage"] not in steps]
        for k in ("images", "calib", "match_frames", "stereo_failed", "grid"):
            if k in old:
                info[k] = old[k]
    info["fps"] = fps

    # --- prepare: undistort + copy each pair into output/NNNNNN_wd ---
    if "prepare" in steps:
        with timer.stage("prepare"):
            failed = run_frames(timer, "prepare", list(range(len(cam0))),
                                lambda t: ["wass_prepare", "--workdir", f"output/{t:06d}_wd", "--calibdir", "config/",
                                           "--c0", cam0[t], "--c1", cam1[t]], seq, args.workers,
                                expect="undistorted/00000001.png")
            if failed:
                die(f"wass_prepare failed on {len(failed)} frame(s): {failed[:10]}")

    wds = workdirs(seq)
    if not wds:
        die(f"no frames in {output}; run the prepare step first")
    n = len(wds)

    # --- match: features on a random subset of frames ---
    if "match" in steps:
        with timer.stage("match"):
            k = min(args.match_frames, n)
            chosen = sorted(np.random.default_rng(args.seed).permutation(n)[:k].tolist())
            log(f"  matching {k} of {n} frames: {chosen}")
            failed = run_frames(timer, "match", chosen,
                                lambda t: ["wass_match", "config/matcher_config.txt", f"output/{t:06d}_wd"], seq, args.workers,
                                expect="matches.txt")
            if len(failed) == k:
                die("wass_match failed on every frame")
            info["match_frames"] = chosen

    # --- autocalibrate: extrinsics from all matched frames ---
    if "autocalibrate" in steps:
        with timer.stage("autocalibrate"):
            (output / "workspaces.txt").write_text("\n".join(f"output/{wd.name}" for wd in wds))
            (seq / "logs").mkdir(exist_ok=True)
            ok, sec = run(["wass_autocalibrate", "output/workspaces.txt"], seq, seq / "logs" / "autocalibrate.log")
            if not ok or not (output / "000000_wd" / "ext_R.xml").exists():
                die(f"wass_autocalibrate failed, see {seq / 'logs' / 'autocalibrate.log'}")

    # --- stereo: dense reconstruction of every frame ---
    if "stereo" in steps:
        with timer.stage("stereo"):
            frames = [int(wd.name[:6]) for wd in wds]
            failed = run_frames(timer, "stereo", frames,
                                lambda t: ["wass_stereo", "config/stereo_config.txt", f"output/{t:06d}_wd"], seq, args.workers,
                                expect="mesh_cam.xyzC")
            if len(failed) == len(frames):
                die("wass_stereo failed on every frame (first one: see logs/stereo/)")
            for t in failed:  # keep them out of the gridding (which globs *_wd)
                (output / f"{t:06d}_wd").rename(output / f"{t:06d}_wd_failed")
            planes = [" ".join((output / f"{t:06d}_wd" / "plane.txt").read_text().split())
                      for t in frames if t not in failed]
            (output / "planes.txt").write_text("\n".join(planes) + "\n")
            info["stereo_failed"] = failed
            if failed:
                log(f"  WARNING: {len(failed)} frame(s) failed and were renamed *_wd_failed: {failed}")

    # --- grid: interpolate the point clouds onto a regular grid -> gridded.nc ---
    if "grid" in steps:
        with timer.stage("grid"):
            gridding.mkdir(exist_ok=True)
            if args.grid_center:
                center, size = tuple(args.grid_center), args.grid_size
                log(f"  grid area (given): centre {center} m, size {size} m")
            else:
                center, size = auto_grid_area(seq, args.baseline)
                log(f"  grid area (automatic): centre ({center[0]}, {center[1]}) m, size {size} m "
                    f"-> override with --grid-center X Y --grid-size S")
            (gridding / "gridconfig.txt").write_text(
                f"[Area]\narea_center_x={center[0]}\narea_center_y={center[1]}\narea_size={size}\nN={args.grid_n}\n")
            info["grid"] = {"center_m": center, "size_m": size, "N": args.grid_n, "spacing_m": round(size / (args.grid_n - 1), 4)}
            setup_cmd = ["wassgridsurface", "--action", "setup", "./output", "./gridding",
                         "--gridconfig", "./gridding/gridconfig.txt", "--baseline", args.baseline, "--fps", fps or 0]
            ok, sec = run(setup_cmd, seq, seq / "logs" / "grid_setup.log", timer, "grid_setup")
            if not ok:
                die(f"wassgridsurface setup failed, see {seq / 'logs' / 'grid_setup.log'}")
            log(f"  setup done ({fmt(sec)}); check {gridding / 'area_grid.png'} and grid_projected_cam0.jpg")
            ok, sec = run(["wassgridsurface", "--action", "grid", "--gridsetup", "./gridding/config.mat",
                           "./output", "./gridding", "-p", args.workers], seq, seq / "logs" / "grid.log",
                          timer, "grid_interpolate")
            if not ok or not (gridding / "gridded.nc").exists():
                die(f"wassgridsurface grid failed, see {seq / 'logs' / 'grid.log'}")
            log(f"  wrote {gridding / 'gridded.nc'} ({fmt(sec)})")

    # --- video: reconstruction overlaid on the camera images (view_stereo.py) + top view (view_nc.py) ---
    if "video" in steps:
        with timer.stage("video"):
            rate = ["--fps", fps] if fps else []
            videos = {
                "reconstruction.mp4": [sys.executable, HERE / "view_stereo.py", seq, "--topview",
                                       "--nc", gridding / "gridded.nc",
                                       "--left", "output/*_wd/undistorted/00000000.png",
                                       "--right", "output/*_wd/undistorted/00000001.png",
                                       "--save", gridding / "reconstruction.mp4"] + rate,
                "gridded.mp4": [sys.executable, HERE / "view_nc.py", gridding / "gridded.nc",
                                "--save", gridding / "gridded.mp4"] + rate,
            }
            for name, cmd in videos.items():
                logfile = seq / "logs" / f"video_{Path(name).stem}.log"
                ok, sec = run(cmd, seq, logfile, timer, f"video_{Path(name).stem}")
                log(f"  wrote {gridding / name} ({fmt(sec)})" if ok else f"  WARNING: {name} failed, see {logfile}")

    total = time.perf_counter() - t_start
    write_timings(seq, info, timer, total)


def write_timings(seq, info, timer, total):
    per_stage = {}
    for s in ("prepare", "match", "stereo"):
        secs = [f["seconds"] for f in timer.frames if f["stage"] == s and f["ok"]]
        if secs:
            per_stage[s] = {"frames": len(secs), "mean_s": round(float(np.mean(secs)), 3),
                            "median_s": round(float(np.median(secs)), 3), "min_s": round(min(secs), 3),
                            "max_s": round(max(secs), 3), "sum_s": round(float(np.sum(secs)), 3)}
    processing = sum(v["seconds"] for k, v in timer.stages.items() if k in STEPS and k != "video")
    processing_cpu = sum(v.get("cpu_seconds") or 0 for k, v in timer.stages.items() if k in STEPS and k != "video")
    info.update(finished=datetime.now().isoformat(timespec="seconds"), total_seconds=round(total, 3),
                processing_seconds=round(processing, 3), processing_cpu_seconds=round(processing_cpu, 3),
                stages=timer.stages, per_frame_stats=per_stage)
    if getattr(timer, "peak_rss_mb", None):
        info["peak_rss_mb_single_process"] = timer.peak_rss_mb
    n = info.get("n_pairs") or 0
    if n and processing:
        info["seconds_per_frame_overall"] = round(processing / n, 3)

    (seq / "timings.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    with open(seq / "timings_frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "frame", "seconds", "ok"])
        w.writeheader()
        w.writerows(timer.frames)

    lines = [f"WASS run {info['started']}  ({n} pairs, {info['workers']} workers, {info['machine']['cpus']} CPUs, "
             f"WASS {info.get('wass_version') or '?'})", "",
             f"{'stage':<24}{'wall time':>10}{'CPU time':>10}   per frame (median, with {info['workers']} running at once)"]
    order = ["prepare", "match", "autocalibrate", "stereo", "grid", "grid_setup", "grid_interpolate", "video",
             "video_reconstruction", "video_gridded"]
    for k in sorted(timer.stages, key=order.index):
        v = timer.stages[k]
        pf = per_stage.get(k)
        extra = f"{pf['median_s']:.2f} s  x {pf['frames']} frames" if pf else ""
        indent = "  " if k.startswith(("grid_", "video_")) else ""
        cpu = fmt(v["cpu_seconds"]) if v.get("cpu_seconds") is not None else "-"
        lines.append(f"{indent + k:<24}{fmt(v['seconds']):>10}{cpu:>10}   {extra}".rstrip())
    lines += ["", f"{'processing':<24}{fmt(processing):>10}{fmt(processing_cpu):>10}"
              + (f"   = {fmt(processing / n)} wall, {fmt(processing_cpu / n)} CPU per stereo pair" if n else ""),
              f"{'total (this run)':<24}{fmt(total):>10}"]
    summary = "\n".join(lines)
    (seq / "timings_summary.txt").write_text(summary + "\n")
    log("\n" + summary)
    plot_timings(seq)
    log(f"\nTimings: {seq / 'timings.json'}, {seq / 'timings_frames.csv'}, {seq / 'timings_summary.txt'}, "
        f"{seq / 'timings.png'}")


def plot_timings(seq):
    """timings.png: wall time per step (left) and time per frame for each per-frame step (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not (seq / "timings.json").exists():
        die(f"{seq / 'timings.json'} not found")
    info = json.loads((seq / "timings.json").read_text())
    frames = []
    if (seq / "timings_frames.csv").exists():
        with open(seq / "timings_frames.csv") as f:
            frames = list(csv.DictReader(f))

    BLUE, RED, INK, INK2, GRID, SURFACE = "#2a78d6", "#d03b3b", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})

    order = ["prepare", "match", "autocalibrate", "stereo", "grid", "video"]
    stages = [k for k in order if k in info["stages"]]
    per_frame = [k for k in ("prepare", "match", "stereo") if any(r["stage"] == k for r in frames)]
    rows = max(len(per_frame), 1)

    fig = plt.figure(figsize=(12, max(4.5, 2.2 * rows + 1)), layout="constrained")
    gs = fig.add_gridspec(rows, 2, width_ratios=[1, 1.4])
    n, workers = info.get("n_pairs") or 0, info.get("workers")
    proc = info.get("processing_seconds", 0)
    fig.suptitle(f"WASS processing times: {n} stereo pairs, {workers} workers, {info['machine']['cpus']} CPUs   "
                 f"(processing {fmt(proc)}" + (f", {fmt(proc / n)} per pair)" if n else ")"),
                 x=0.01, ha="left", fontsize=12, fontweight="bold")

    # Left: wall time per step (pipeline order, top to bottom).
    ax = fig.add_subplot(gs[:, 0])
    secs = [info["stages"][k]["seconds"] for k in stages]
    y = np.arange(len(stages))[::-1]
    ax.barh(y, secs, height=0.55, color=BLUE)
    for yi, sec in zip(y, secs):
        ax.text(sec, yi, f"  {fmt(sec)}", va="center", ha="left", color=INK, fontsize=9)
    ax.set_yticks(y, stages)
    ax.set_xlim(0, max(secs) * 1.25 if secs else 1)
    ax.set_xlabel("wall time (s)")
    ax.set_title("Wall time per step", loc="left", fontsize=11)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    # Right: time per frame, one panel per step (each with its own scale).
    for i, k in enumerate(per_frame):
        ax = fig.add_subplot(gs[i, 1])
        rs = [r for r in frames if r["stage"] == k]
        t = np.array([int(r["frame"]) for r in rs])
        sec = np.array([float(r["seconds"]) for r in rs])
        ok = np.array([r["ok"] == "True" for r in rs])
        ax.scatter(t[ok], sec[ok], s=22, color=BLUE, edgecolors=SURFACE, linewidths=1, zorder=3)
        if (~ok).any():
            ax.scatter(t[~ok], sec[~ok], s=40, marker="x", color=RED, zorder=3, label="failed")
            ax.legend(frameon=False, loc="upper right")
        if ok.any():
            med = float(np.median(sec[ok]))
            ax.axhline(med, color=INK2, linewidth=1, linestyle="--", zorder=2)
            ax.text(1.0, med, f" median {med:.1f} s", transform=ax.get_yaxis_transform(), va="center", ha="left",
                    color=INK2, fontsize=9)
        ax.set_ylim(0, max(sec.max() * 1.15, 1e-3))
        ax.set_ylabel("s")
        ax.set_title(f"{k}: time per frame ({len(rs)} frames)", loc="left", fontsize=11)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if i == len(per_frame) - 1:
            ax.set_xlabel(f"frame number   (times measured with {workers} frames running at once)")

    fig.savefig(seq / "timings.png", dpi=130)
    plt.close(fig)
    log(f"Plot: {seq / 'timings.png'}")


if __name__ == "__main__":
    main()
