#!/usr/bin/env python3
"""Run WASSfast (CNN mode) on the same stereo pairs and grid as a run_wass.py run, and record how long it takes.

    run_wass.py output (pairs.txt, config/, gridding/config.mat)  ->  wassfast  ->  OUT/wassfast.nc + timings

WASSfast does not compute extrinsics or the sea plane itself: it reads them from the WASS grid setup
(gridding/config.mat, made by `wassgridsurface --action setup`, which run_wass.py runs in its grid step with N=256).
So run run_wass.py first (at least its prepare/match/autocalibrate/stereo/grid steps) with --grid-n 256.
That also makes the two outputs directly comparable: same image pairs, same grid, same sea plane.

What this does:
  - links the exact image pairs of the WASS run (pairs.txt) into OUT/input/cam0|cam1 with WASS-style names
    NNNNNN_<ms>_01.ext, which WASSfast needs (it parses the timestamp from the name)
  - writes WASSfast's default settings (wassfast --generate_settings) to OUT/settings.cfg unless --settings is given
  - runs `wassfast ... CNN` and timestamps its output lines, to split the wall time into
      startup      (Python + TensorFlow import, model load, first two frames)
      reconstruct  (all frames: sparse matching in a process pool + CNN, per batch)
      report       (WASSfast's own sea-state HTML report; not part of reconstruction)
  - writes OUT/timings.json, timings_frames.csv and timings_summary.txt in the same style as run_wass.py

Known WASSfast quirks handled here:
  - only 256x256 grids are supported in CNN mode (checked)
  - when (frames - 1) is a multiple of --batchsize, WASSfast's batching drops 15 frames; one frame is left out instead
  - only one WASSfast can run at a time (it opens a status server on 127.0.0.1:18098)

Run inside `nix-shell` (5_WASS_Validation/shell.nix), which provides the `wassfast` command.

Examples:
    python3 run_wassfast.py --wass data/compare/quickstart/wass --out data/compare/quickstart/wassfast
    python3 run_wassfast.py --wass data/compare/quickstart/wass --out /tmp/wf15 --fps 15     # override frame rate
"""

import argparse
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


def log(msg=""):
    print(msg, flush=True)


def die(msg):
    sys.exit(f"ERROR: {msg}")


def fmt(sec):
    return f"{sec:.1f} s" if sec < 120 else f"{sec / 60:.1f} min"


def children_cpu():
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    return ru.ru_utime + ru.ru_stime, ru.ru_maxrss / 1024


def link_inputs(pairs, fps, dest):
    """Symlink the pairs into dest/cam0, dest/cam1 as NNNNNN_<ms>_0[12].ext (WASSfast reads the time from the name)."""
    if dest.exists():
        shutil.rmtree(dest)
    for cam in ("cam0", "cam1"):
        (dest / cam).mkdir(parents=True)
    for i, (a, b) in enumerate(pairs):
        ms = round(i * 1000 / fps)
        for cam, src, n in (("cam0", a, "01"), ("cam1", b, "02")):
            (dest / cam / f"{i:06d}_{ms:013d}_{n}{src.suffix.lower()}").symlink_to(src)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wass", required=True, help="run_wass.py output folder (pairs.txt, config/, gridding/config.mat)")
    p.add_argument("--out", help="output folder (default: WASS/../wassfast)")
    p.add_argument("--fps", type=float, help="frame rate in Hz (default: the WASS run's, from timings.json)")
    p.add_argument("--direction", default="AUTO",
                   help="wave direction as seen in the images: LRTB RLTB LRBT RLBT AUTO NONE (default AUTO)")
    p.add_argument("--batchsize", type=int, default=16, help="CNN batch size (default 16, as in the WASS guide)")
    p.add_argument("--settings", help="WASSfast settings file (default: generated with wassfast --generate_settings)")
    p.add_argument("--frames", type=int, default=0, help="use only the first N pairs of the WASS run (default all)")
    args = p.parse_args()

    if shutil.which("wassfast") is None:
        die(f"`wassfast` not on PATH; run this inside `nix-shell` in {HERE}")
    wass = Path(args.wass).resolve()
    out = Path(args.out).resolve() if args.out else wass.parent / "wassfast"
    gridsetup, calib = wass / "gridding" / "config.mat", wass / "config"
    for f in (wass / "pairs.txt", gridsetup, calib / "intrinsics_00.xml"):
        if not f.exists():
            die(f"{f} not found; run run_wass.py (with its grid step) first")

    import scipy.io as sio
    n_grid = int(np.ravel(sio.loadmat(gridsetup)["N"])[0])
    if n_grid != 256:
        die(f"{gridsetup} has N={n_grid}; WASSfast's CNN mode needs a 256x256 grid (re-run run_wass.py --steps grid --grid-n 256)")

    wass_info = json.loads((wass / "timings.json").read_text()) if (wass / "timings.json").exists() else {}
    fps = args.fps or wass_info.get("fps")
    if not fps:
        die("frame rate unknown; pass --fps")

    pairs = [tuple(Path(x) for x in line.split()[1:3]) for line in (wass / "pairs.txt").read_text().splitlines() if line.strip()]
    if args.frames:
        pairs = pairs[: args.frames]
    if (len(pairs) - 1) % args.batchsize == 0:
        log(f"Note: dropping the last pair ({len(pairs)} -> {len(pairs) - 1}): with (frames-1) a multiple of the batch size "
            f"WASSfast loses 15 frames.")
        pairs = pairs[:-1]
    n = len(pairs)

    out.mkdir(parents=True, exist_ok=True)
    link_inputs(pairs, fps, out / "input")
    if args.settings:
        settings = Path(args.settings).resolve()
    else:
        subprocess.run(["wassfast", "--generate_settings"], cwd=out, capture_output=True)
        generated = [f for f in out.glob("*settings*") if f.is_file() and f.name != "settings.cfg"]
        if not generated:
            die("`wassfast --generate_settings` did not write a settings file")
        settings = out / "settings.cfg"
        generated[0].replace(settings)
    nc, dbg = out / "wassfast.nc", out / "dbg"
    dbg.mkdir(exist_ok=True)
    for f in (nc, nc.with_suffix(".html")):
        f.unlink(missing_ok=True)

    cmd = ["wassfast", "input", gridsetup, calib, settings, args.direction, "CNN", "--batchsize", args.batchsize,
           "-r", fps, "--nographics", "--debug_stats", "-dd", dbg, "-o", nc]
    cmd = [str(c) for c in cmd]
    log(f"Pairs:    {n} (from {wass / 'pairs.txt'}), fps {fps}, batch size {args.batchsize}, direction {args.direction}")
    log(f"Output:   {out}")
    log("Command:  " + " ".join(cmd))

    # Run, timestamping each output line, to split the run into phases.
    marks, batches = {}, []
    t0, cpu0 = time.perf_counter(), children_cpu()[0]
    started = datetime.now().isoformat(timespec="seconds")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(out / "wassfast.log", "w") as logf:
        proc = subprocess.Popen(cmd, cwd=out, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                                bufsize=1)
        for line in proc.stdout:
            t = time.perf_counter() - t0
            logf.write(f"[{t:9.3f}] {line}")
            s = line.strip()
            if s.startswith("Loading CNN model"):
                marks.setdefault("model_load", t)
            elif s.startswith("Reading frames 0 and 1 first"):
                marks.setdefault("first_frames", t)
            elif s.startswith("Loading remaining frames into batches"):
                marks.setdefault("batches_start", t)
            elif s.startswith("Running CNN-based interpolator"):
                batches.append(t)
                log(f"  [{t:7.1f} s] CNN batch {len(batches)}")
            elif m := re.match(r"All done in ([\d.]+) seconds", s):
                marks["reconstruct_end"] = t
                marks["wassfast_reported_s"] = float(m.group(1))
            elif s.startswith("- Generating report") or s.startswith("Generating report"):
                marks.setdefault("report", t)
        ret = proc.wait()
    total = time.perf_counter() - t0
    cpu, peak_mb = children_cpu()
    cpu -= cpu0
    if ret != 0 or not nc.exists():
        die(f"wassfast failed (exit {ret}), see {out / 'wassfast.log'}")
    if "reconstruct_end" not in marks:
        die(f"could not find 'All done in' in {out / 'wassfast.log'}")

    # Check the frame count of the output.
    import netCDF4
    with netCDF4.Dataset(nc) as ds:
        n_out = ds["Z"].shape[0]
    if n_out != n:
        log(f"WARNING: {n} pairs in, {n_out} frames in {nc}")

    # Phases. Startup ends when the first frame starts processing (TF import + model load + settings).
    startup = marks.get("first_frames", 0.0)
    reconstruct = marks["reconstruct_end"] - startup
    report = total - marks["reconstruct_end"]
    stages = {
        "startup": {"seconds": round(startup, 3)},
        "reconstruct": {"seconds": round(reconstruct, 3)},
        "report": {"seconds": round(report, 3)},
    }

    # Per-frame times: each CNN batch line marks the end of a batch's frame processing; frames in batch k get
    # the batch duration divided by the batch size (WASSfast processes a batch's frames in parallel).
    frames = []
    edges = [marks.get("batches_start", startup)] + batches
    sizes = []
    remaining = n
    for k in range(len(batches)):
        size = min(args.batchsize, remaining)
        if k == len(batches) - 1:
            size = remaining
        sizes.append(size)
        remaining -= size
    # The first batch also includes frames 0 and 1, which are processed before the pool starts.
    first = 0
    for k, size in enumerate(sizes):
        dur = edges[k + 1] - edges[k] + (edges[0] - startup if k == 0 else 0)
        for f in range(first, first + size):
            frames.append({"stage": "reconstruct", "frame": f, "seconds": round(dur / size, 3), "ok": True,
                           "batch": k})
        first += size

    batch_seconds = [round(edges[k + 1] - edges[k], 3) for k in range(len(batches))]
    # Marginal cost per pair: batches after the first (the first also pays for starting the worker pool).
    later = sum(sizes[1:])
    marginal = sum(batch_seconds[1:]) / later if later >= args.batchsize else None

    info = {
        "pipeline": "wassfast",
        "batch_seconds": batch_seconds, "batch_sizes": sizes,
        "marginal_seconds_per_pair": round(marginal, 4) if marginal else None, "started": started, "finished": datetime.now().isoformat(timespec="seconds"),
        "command": cmd, "wass_run": str(wass), "n_pairs": n, "n_output_frames": n_out, "fps": fps,
        "direction": args.direction, "batchsize": args.batchsize, "settings": str(settings),
        "machine": {"host": platform.node(), "cpus": os.cpu_count(), "platform": platform.platform()},
        "wassfast_version": wassfast_version(), "tensorflow": tf_info(),
        "stages": stages, "total_seconds": round(total, 3),
        "processing_seconds": round(startup + reconstruct, 3),
        "processing_cpu_seconds": round(cpu, 3),  # whole run incl. report; WASSfast is one process tree
        "reconstruct_seconds_per_frame": round(reconstruct / n, 4),
        "processing_seconds_per_frame": round((startup + reconstruct) / n, 4),
        "wassfast_reported_seconds": marks.get("wassfast_reported_s"),
        "peak_rss_mb_single_process": round(peak_mb, 1),
        "marks": {k: round(v, 3) for k, v in marks.items()},
    }
    (out / "timings.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    with open(out / "timings_frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "frame", "seconds", "ok", "batch"])
        w.writeheader()
        w.writerows(frames)

    lines = [f"WASSfast run {started}  ({n} pairs, {info['machine']['cpus']} CPUs, WASSfast {info['wassfast_version']}, "
             f"TensorFlow {info['tensorflow']})", "",
             f"{'stage':<24}{'wall time':>10}",
             f"{'startup':<24}{fmt(startup):>10}   TensorFlow import, model load",
             f"{'reconstruct':<24}{fmt(reconstruct):>10}   {fmt(reconstruct / n)} per pair"
             + (f"; {fmt(marginal)} per pair after the first batch (worker pool start-up excluded)" if marginal else ""),
             f"{'report':<24}{fmt(report):>10}   (WASSfast's HTML sea-state report, not counted)", "",
             f"{'processing':<24}{fmt(startup + reconstruct):>10}   = {fmt((startup + reconstruct) / n)} per stereo pair",
             f"{'CPU time (whole run)':<24}{fmt(cpu):>10}   = {fmt(cpu / n)} per stereo pair",
             f"{'total (this run)':<24}{fmt(total):>10}"]
    summary = "\n".join(lines)
    (out / "timings_summary.txt").write_text(summary + "\n")
    log("\n" + summary)
    log(f"\nOutput: {nc}\nTimings: {out / 'timings.json'}")


def wassfast_version():
    try:
        out = subprocess.run(["wassfast", "--help"], capture_output=True, text=True, timeout=120).stdout
        m = re.search(r"v\.\s*([\d.]+)", out)
        return m.group(1) if m else None
    except Exception:
        return None


def tf_info():
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        return f"{tf.__version__} ({'GPU' if gpus else 'CPU only'})"
    except Exception:
        return None


if __name__ == "__main__":
    main()
