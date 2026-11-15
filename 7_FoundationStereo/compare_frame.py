#!/usr/bin/env python3
"""Reconstruct one WASS frame with FoundationStereo (or Fast-FoundationStereo) and compare it with WASS.

    python compare_frame.py --wass ../5_WASS_Validation/data/compare/quickstart/wass --frame 0 --scale 0.35
    python compare_frame.py --wass ... --frame 0 --model vitl --scale 0.4 --out output/q0_vitl
    python compare_frame.py --wass ... --frame 0 --method ffs --scale 0.5       # Fast-FoundationStereo, 20-30-48

--method fs (default) is FoundationStereo; --method ffs is Fast-FoundationStereo. Both go through exactly the same
rectification, filtering and comparison. Below, "FoundationStereo" means whichever was chosen.

FoundationStereo replaces only WASS's dense-stereo step (SGBM). Everything else comes from the WASS run, so both
reconstructions use the same cameras and end up in the same sea-plane coordinates:

  WASS frame folder (output/NNNNNN_wd/)                 this script
  ------------------------------------                  -----------
  undistorted/00000000.png, 00000001.png  ──────────►  rectify (cv2.stereoRectify with WASS's ext_R, ext_T)
  intrinsics_0000000{0,1}.xml, ext_R/T.xml               │  scale down by --scale, FoundationStereo -> disparity
                                                         ▼
                                                       triangulate (Q) -> cam0 -> WASS world -> sea plane (m)
  Cam0_poseR/T.txt, output/planes.txt (mean plane) ──►  same transform WASS's gridding uses (align_on_sea_plane)
  mesh_cam.xyzC (WASS point cloud)  ─────────────────►  same transform -> compared point for point

Filtering of the FoundationStereo cloud, mirroring wass_stereo: pixels must be inside both rectified images,
then a RANSAC sea-plane fit on FoundationStereo's own points, keeping points within PLANE_MAX_DISTANCE (1.5 baseline
units, WASS's default) of that plane. WASS's plane is used only to express both clouds in the same coordinates.

Comparisons (WASS = reference):
  1. image space: WASS's points re-projected into the rectified left image as disparity, vs FoundationStereo's
  2. sea-plane space: both clouds binned on the same horizontal grid (--cell m) over the WASS grid area; elevation
     difference, RMSE, correlation, Hs = 4 std(Z), coverage
  3. sea-plane fit: angle and offset between FoundationStereo's plane and WASS's plane for this frame

Outputs in --out: comparison.png, summary.json, summary.md, rectified.jpg, disparity_fs.npy,
clouds.html (both clouds in an interactive 3D viewer, sea-plane frame), fs_cloud_sea.ply.

Run inside 7_FoundationStereo's nix-shell. Needs only the WASS run folder, not the WASS tools.
"""

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
FS_DIR = HERE / "FoundationStereo"
FFS_DIR = HERE / "Fast-FoundationStereo"
METHODS = {
    # name, repo, checkpoints, defaults
    "fs": {"name": "FoundationStereo", "dir": FS_DIR, "default_model": "vits", "valid_iters": 32, "max_disp": 416,
           "models": {"vits": FS_DIR / "pretrained_models/11-33-40/model_best_bp2.pth",
                      "vitl": FS_DIR / "pretrained_models/23-51-11/model_best_bp2.pth"}},
    "ffs": {"name": "Fast-FoundationStereo", "dir": FFS_DIR, "default_model": "20-30-48", "valid_iters": 8, "max_disp": 192,
            "models": {k: FFS_DIR / f"weights/{k}/model_best_bp2_serialize.pth"
                       for k in ["23-36-37", "20-26-39", "20-30-48", "15-44-51"]}},
}
PLANE_MAX_DISTANCE = 1.5  # baseline units, wass_stereo default


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# WASS files (same maths as wass/gridding/wassgridsurface/wass_utils.py, copied so the WASS Python isn't needed)
# ----------------------------------------------------------------------------------------------------------------------

def read_cv_mat(path, key):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    m = fs.getNode(key).mat()
    fs.release()
    return m


def load_camera_mesh(path):
    """WASS mesh_cam.xyzC -> 3xN points in WASS world coordinates (baseline units)."""
    with open(path, "rb") as f:
        npts = struct.unpack("I", f.read(4))[0]
        limits = np.array(struct.unpack("dddddd", f.read(48)))
        Rinv = np.reshape(np.array(struct.unpack("d" * 9, f.read(72))), (3, 3))
        Tinv = np.reshape(np.array(struct.unpack("ddd", f.read(24))), (3, 1))
        data = np.frombuffer(f.read(npts * 6), dtype=np.uint16).reshape((3, npts), order="F")
    mesh = data.astype(np.float64) / limits[:3, None] + limits[3:, None]
    return Rinv @ mesh + Tinv


def sea_plane_RT(plane):
    a, b, c, d = plane
    q = (1 - c) / (a * a + b * b)
    R = np.array([[1 - a * a * q, -a * b * q, -a], [-a * b * q, 1 - b * b * q, -b], [a, b, c]])
    T = np.array([[0.0], [0.0], [d]])
    return R, T


def to_sea(pts_world, plane, baseline):
    """3xN WASS-world points (baseline units) -> 3xN sea-plane coordinates in metres, z up (wassgridsurface)."""
    R, T = sea_plane_RT(plane)
    p = R @ pts_world + T
    p[2] *= -1
    return p * baseline


def fit_plane_ransac(pts, rounds=400, thresh=0.1, refine_dist=1.0, seed=0):
    """pts 3xN. Returns plane (a,b,c,d) with unit normal, a*x+b*y+c*z+d=0, refined by least squares on inliers."""
    rng = np.random.default_rng(seed)
    n = pts.shape[1]
    sub = pts[:, rng.choice(n, min(n, 200_000), replace=False)]
    best, best_in = None, -1
    for _ in range(rounds):
        p = sub[:, rng.choice(sub.shape[1], 3, replace=False)]
        nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        if np.linalg.norm(nrm) < 1e-12:
            continue
        nrm /= np.linalg.norm(nrm)
        d = -nrm @ p[:, 0]
        inl = np.count_nonzero(np.abs(nrm @ sub + d) < thresh)
        if inl > best_in:
            best, best_in = (nrm, d), inl
    # Refine by least squares on all points near the RANSAC plane (not just the tight inliers, which would cut
    # off crests and troughs and bias the fit), twice.
    nrm, d = best
    for _ in range(2):
        inl = sub[:, np.abs(nrm @ sub + d) < refine_dist]
        c = inl.mean(axis=1, keepdims=True)
        nrm = np.linalg.eigh(np.cov(inl - c))[1][:, 0]  # direction of least variance
        d = -nrm @ c[:, 0]
    plane = np.r_[nrm, d]
    return plane if plane[3] < 0 else -plane  # same sign convention as WASS's planes.txt (d < 0)


# ----------------------------------------------------------------------------------------------------------------------
# FoundationStereo
# ----------------------------------------------------------------------------------------------------------------------

def load_model(method, model, valid_iters, max_disp, amp=True):
    """Both repos have a top-level `core` package and `Utils` module, so only one method can be loaded per process."""
    import torch
    m = METHODS[method]
    sys.path.insert(0, str(m["dir"]))
    ckpt_path = m["models"][model]
    if not ckpt_path.exists():
        sys.exit(f"ERROR: {ckpt_path} not found (see README: model weights)")
    if method == "ffs":
        # The checkpoint is the whole pickled module (needs the repo's `core` on sys.path).
        net = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        net.args.valid_iters = valid_iters
        net.args.max_disp = max_disp
        net.args.mixed_precision = amp
        return net.cuda().eval()
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo
    cfg = OmegaConf.load(ckpt_path.parent / "cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg["valid_iters"] = valid_iters
    cfg["max_disp"] = max_disp
    cfg["mixed_precision"] = amp  # the model also autocasts internally from this flag
    net = FoundationStereo(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model"])
    return net.cuda().eval()


def run_model(net, left, right, valid_iters, hiera, amp, method="fs"):
    """left/right HxWx3 uint8 (already scaled). Returns disparity HxW float32, seconds, peak GPU MB."""
    import torch
    from core.utils.utils import InputPadder

    torch.cuda.reset_peak_memory_stats()
    H, W = left.shape[:2]
    t = lambda im: torch.as_tensor(im).cuda().float()[None].permute(0, 3, 1, 2)
    img0, img1 = t(left), t(right)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", enabled=amp):
        if hiera:
            disp = net.run_hierachical(img0, img1, iters=valid_iters, test_mode=True, small_ratio=0.5)
        else:
            kw = {"optimize_build_volume": "pytorch1"} if method == "ffs" else {}  # Triton kernel: not on Pascal
            disp = net.forward(img0, img1, iters=valid_iters, test_mode=True, **kw)
    torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    disp = padder.unpad(disp.float()).cpu().numpy().reshape(H, W)
    return disp, sec, torch.cuda.max_memory_allocated() / 2**20


# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------

def bin_mean(x, y, v, x0, y0, cell, nx, ny):
    """Mean of v per grid cell. Returns (ny, nx) array, NaN where empty, and the count per cell."""
    ix = np.floor((x - x0) / cell).astype(int)
    iy = np.floor((y - y0) / cell).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & np.isfinite(v)
    idx = iy[ok] * nx + ix[ok]
    s = np.bincount(idx, v[ok], nx * ny)
    c = np.bincount(idx, minlength=nx * ny)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / c
    return m.reshape(ny, nx), c.reshape(ny, nx)


def stats(a, b):
    """a = candidate, b = reference, same shape, NaN = missing."""
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    return {
        "n": int(ok.sum()),
        "bias": float(d.mean()),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "mae": float(np.abs(d).mean()),
        "median_abs": float(np.median(np.abs(d))),
        "p95_abs": float(np.percentile(np.abs(d), 95)),
        "corr": float(np.corrcoef(a[ok], b[ok])[0, 1]),
    }


def sample_gray(img, u, v):
    ui = np.clip(np.round(u).astype(int), 0, img.shape[1] - 1)
    vi = np.clip(np.round(v).astype(int), 0, img.shape[0] - 1)
    g = img[vi, ui].astype(np.float32) / 255
    return np.repeat(g[:, None], 3, axis=1)


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wass", required=True, type=Path, help="run_wass.py output folder (has output/, config/, gridding/)")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    help="default: output/<wass run name>_frame<NNNNNN>_<model>_s<scale> (ffs: _ffs-<model>_s<scale>)")
    ap.add_argument("--method", choices=list(METHODS), default="fs",
                    help="fs = FoundationStereo, ffs = Fast-FoundationStereo")
    ap.add_argument("--model", help="fs: vits (default) or vitl; ffs: 20-30-48 (default), 23-36-37, 20-26-39, 15-44-51")
    ap.add_argument("--scale", type=float, default=0.5, help="image scale fed to the network (GPU memory!)")
    ap.add_argument("--valid_iters", type=int, help="refinement iterations (default: fs 32, ffs 8)")
    ap.add_argument("--max_disp", type=int,
                    help="max disparity at the network's scale. Default: fs 416; ffs 192, raised to cover 1.2x the "
                         "largest WASS disparity of this frame (rounded up to 32)")
    ap.add_argument("--hiera", action="store_true", help="hierarchical inference (for high resolution)")
    ap.add_argument("--fp32", action="store_true", help="disable mixed precision (upstream default is fp16 autocast)")
    ap.add_argument("--repeat", type=int, default=2, help="inference runs; the first includes CUDA warm-up")
    ap.add_argument("--cell", type=float, default=0.5, help="sea-plane binning cell for the elevation comparison (m)")
    ap.add_argument("--baseline", type=float, help="metres; default from the WASS run's timings.json")
    ap.add_argument("--no-html", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="skip inference and reuse --out's disparity_fs.npy + summary.json timings (analysis only)")
    args = ap.parse_args()
    meth = METHODS[args.method]
    args.model = args.model or meth["default_model"]
    if args.model not in meth["models"]:
        sys.exit(f"ERROR: --model for --method {args.method} must be one of {list(meth['models'])}")
    args.valid_iters = args.valid_iters or meth["valid_iters"]
    args.name = meth["name"]
    if args.method == "ffs":
        # Fast-FoundationStereo uses @torch.compile, whose backend needs Triton (compute capability >= 7.0). On the
        # GTX 1060 (6.1) run those functions eagerly. Must be set before torch is imported.
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    seq = args.wass.resolve()
    wd = seq / "output" / f"{args.frame:06d}_wd"
    if not wd.is_dir():
        sys.exit(f"ERROR: {wd} not found")
    info = json.loads((seq / "timings.json").read_text()) if (seq / "timings.json").exists() else {}
    B = args.baseline or info.get("baseline_m")
    if not B:
        sys.exit("ERROR: baseline unknown; pass --baseline")
    tag = args.model if args.method == "fs" else f"ffs-{args.model}"
    out = args.out or HERE / "output" / f"{seq.name if seq.name != 'wass' else seq.parent.name}_frame{args.frame:06d}_{tag}_s{args.scale:g}"
    out = out.resolve()  # absolute: the script later chdirs into the model's repo
    if HERE not in out.parents:
        sys.exit(f"ERROR: --out must be inside {HERE}")
    out.mkdir(parents=True, exist_ok=True)
    log(f"WASS frame: {wd}\nOutput:     {out}")

    # --- Calibration from WASS -----------------------------------------------------------------------------------
    K0 = read_cv_mat(wd / "intrinsics_00000000.xml", "intr")
    K1 = read_cv_mat(wd / "intrinsics_00000001.xml", "intr")
    R = read_cv_mat(wd / "ext_R.xml", "ext_R")         # x_cam1 = R x_cam0 + T   (baseline units, |T| = 1)
    T = read_cv_mat(wd / "ext_T.xml", "ext_T").reshape(3, 1)
    R0 = np.loadtxt(wd / "Cam0_poseR.txt")             # x_cam0 = R0 X_world + T0
    T0 = np.loadtxt(wd / "Cam0_poseT.txt").reshape(3, 1)
    planes = np.atleast_2d(np.loadtxt(seq / "output" / "planes.txt"))
    mean_plane = np.nanmean(planes, axis=0)            # what wassgridsurface uses
    frame_plane = np.loadtxt(wd / "plane.txt")         # WASS's plane for this frame
    grid = {}
    gc = seq / "gridding" / "gridconfig.txt"
    if gc.exists():
        for line in gc.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                grid[k.strip()] = float(v)
    cx, cy, size = grid.get("area_center_x", 0.0), grid.get("area_center_y", -70.0), grid.get("area_size", 90.0)

    I0 = cv2.imread(str(wd / "undistorted/00000000.png"), cv2.IMREAD_GRAYSCALE)
    I1 = cv2.imread(str(wd / "undistorted/00000001.png"), cv2.IMREAD_GRAYSCALE)
    H, W = I0.shape
    log(f"Images {W}x{H}, baseline {B} m, |T| = {np.linalg.norm(T):.4f}, grid area {size} m at ({cx}, {cy})")

    # --- Rectify (undistorted images: zero distortion) -------------------------------------------------------------
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K0, np.zeros(5), K1, np.zeros(5), (W, H), R, T,
                                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    m0 = cv2.initUndistortRectifyMap(K0, np.zeros(5), R1, P1, (W, H), cv2.CV_32FC1)
    m1 = cv2.initUndistortRectifyMap(K1, np.zeros(5), R2, P2, (W, H), cv2.CV_32FC1)
    L = cv2.remap(I0, *m0, cv2.INTER_LINEAR)
    Rr = cv2.remap(I1, *m1, cv2.INTER_LINEAR)
    ones = np.full((H, W), 255, np.uint8)
    Lmask = cv2.remap(ones, *m0, cv2.INTER_NEAREST, borderValue=0) > 0
    Rmask = cv2.remap(ones, *m1, cv2.INTER_NEAREST, borderValue=0) > 0
    if P2[0, 3] > 0:
        sys.exit("ERROR: cam1 is to the left of cam0 after rectification; swapping is not implemented")
    log(f"Rectified: f = {P1[0, 0]:.1f} px, Tx = {P2[0, 3] / P2[0, 0]:.4f} baseline units")

    s = args.scale
    # Largest disparity WASS found in this frame, at the network's scale: the network can't output more than max_disp
    w_rect = R1 @ (R0 @ load_camera_mesh(wd / "mesh_cam.xyzC") + T0)
    wass_dmax = float(np.percentile(-P2[0, 3] / w_rect[2], 99.9)) * s
    del w_rect
    if not args.max_disp:
        args.max_disp = meth["max_disp"]
        if args.method == "ffs":
            args.max_disp = max(args.max_disp, int(np.ceil(1.2 * wass_dmax / 32)) * 32)
    log(f"WASS disparity up to {wass_dmax:.0f} px at scale {s}; max_disp {args.max_disp}")
    if args.max_disp < 1.05 * wass_dmax:
        log(f"WARNING: max_disp {args.max_disp} leaves no margin over WASS's disparities; near-field values will clip")
    Ls = cv2.resize(L, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    Rs = cv2.resize(Rr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    hs, ws = Ls.shape
    prev = np.hstack([Ls, Rs])
    prev = cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR)
    for y in range(0, hs, max(1, hs // 16)):
        cv2.line(prev, (0, y), (2 * ws, y), (0, 200, 255), 1)
    cv2.imwrite(str(out / "rectified.jpg"), prev, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # --- Network ----------------------------------------------------------------------------------------------------
    os.chdir(meth["dir"])  # torch.hub cache / relative paths inside the repo
    if args.reuse:
        disp_s = np.load(out / "disparity_fs.npy")
        if disp_s.shape != (hs, ws):
            sys.exit(f"ERROR: saved disparity is {disp_s.shape}, expected {(hs, ws)} for --scale {s}")
        old = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
        times, peak, load_sec = old.get("inference_seconds", [np.nan]), old.get("peak_gpu_mb", np.nan), old.get("model_load_seconds", np.nan)
        log(f"Reusing {out / 'disparity_fs.npy'} (no inference)")
    else:
        disp_s, times, peak, load_sec = infer(args, Ls, Rs, ws, hs)
        np.save(out / "disparity_fs.npy", disp_s.astype(np.float32))
    analyse(args, seq, wd, out, B, K0, R0, T0, R1, P1, P2, Q, I0, Ls, Lmask, Rmask, W, H, ws, hs, s,
            frame_plane, mean_plane, cx, cy, size, disp_s, times, peak, load_sec)


def infer(args, Ls, Rs, ws, hs):
    import torch
    t0 = time.perf_counter()
    net = load_model(args.method, args.model, args.valid_iters, args.max_disp, amp=not args.fp32)
    load_sec = time.perf_counter() - t0
    log(f"{args.name} {args.model} loaded in {load_sec:.1f} s; input {ws}x{hs} (scale {args.scale}), "
        f"{'fp32' if args.fp32 else 'fp16 autocast'}, {args.valid_iters} iters, max_disp {args.max_disp}"
        f"{', hierarchical' if args.hiera else ''}")
    to3 = lambda g: np.repeat(g[..., None], 3, axis=2)
    times = []
    for k in range(args.repeat):
        try:
            disp_s, sec, peak = run_model(net, to3(Ls), to3(Rs), args.valid_iters, args.hiera, not args.fp32, args.method)
        except torch.cuda.OutOfMemoryError:
            sys.exit(f"ERROR: out of GPU memory at {ws}x{hs}. Lower --scale (or use --model vits / --hiera).")
        times.append(sec)
        log(f"  inference run {k + 1}: {sec:.2f} s, peak GPU memory {peak:.0f} MB")
    return disp_s, times, peak, load_sec


def analyse(args, seq, wd, out, B, K0, R0, T0, R1, P1, P2, Q, I0, Ls, Lmask, Rmask, W, H, ws, hs, s,
            frame_plane, mean_plane, cx, cy, size, disp_s, times, peak, load_sec):
    import torch

    # --- FoundationStereo disparity -> 3D --------------------------------------------------------------------------
    vs, us = np.mgrid[0:hs, 0:ws]
    u = (us + 0.5) / s - 0.5                           # full-resolution rectified pixel coordinates
    v = (vs + 0.5) / s - 0.5
    d = disp_s / s                                     # full-resolution disparity
    ui, vi = np.clip(np.round(u).astype(int), 0, W - 1), np.clip(np.round(v).astype(int), 0, H - 1)
    ur = np.round(u - d).astype(int)
    valid = (d > 0.5) & Lmask[vi, ui] & (ur >= 0) & (ur < W)
    valid[valid] &= Rmask[vi[valid], ur[valid]]
    hom = Q @ np.stack([u[valid], v[valid], d[valid], np.ones(valid.sum())])
    p_rect = hom[:3] / hom[3]
    p_cam0 = R1.T @ p_rect
    p_world = R0.T @ (p_cam0 - T0)
    n_valid = int(valid.sum())

    fs_plane = fit_plane_ransac(p_world)
    keep = np.abs(fs_plane[:3] @ p_world + fs_plane[3]) < PLANE_MAX_DISTANCE
    p_world, p_cam0 = p_world[:, keep], p_cam0[:, keep]
    fs_pix = np.flatnonzero(valid.ravel())[keep]       # scaled-image pixel index of each kept point
    log(f"{args.name}: {n_valid:,} valid pixels, {keep.sum():,} points within {PLANE_MAX_DISTANCE} of its plane")

    # --- WASS cloud ----------------------------------------------------------------------------------------------
    w_world = load_camera_mesh(wd / "mesh_cam.xyzC")
    w_cam0 = R0 @ w_world + T0

    # 3. sea plane: the same fitter on both clouds (WASS's own plane uses only the image's central third, so it is
    # not directly comparable with this fit; its angle is reported too, for reference)
    w_plane = fit_plane_ransac(w_world)

    def plane_diff(a, b):
        na, nb = np.linalg.norm(a[:3]), np.linalg.norm(b[:3])
        ang = float(np.degrees(np.arccos(np.clip(abs(a[:3] @ b[:3]) / na / nb, -1, 1))))
        return ang, float((a[3] / na - b[3] / nb) * B)
    plane_angle, plane_offset = plane_diff(fs_plane, w_plane)
    ref_angle, _ = plane_diff(w_plane, frame_plane)
    log(f"Sea plane, same fit on both clouds: {plane_angle:.3f} deg, offset {plane_offset * 100:.1f} cm "
        f"(this fit vs WASS's own plane, on WASS's points: {ref_angle:.3f} deg)")
    log(f"WASS: {w_world.shape[1]:,} points")

    # 1. image space: WASS points as disparity in the (scaled) rectified left image
    w_rect = R1 @ w_cam0
    wu = (P1[0, 0] * w_rect[0] / w_rect[2] + P1[0, 2] + 0.5) * s - 0.5
    wv = (P1[1, 1] * w_rect[1] / w_rect[2] + P1[1, 2] + 0.5) * s - 0.5
    wd_s = -P2[0, 3] / w_rect[2] * s
    disp_w, _ = bin_mean(wu + 0.5, wv + 0.5, wd_s, 0, 0, 1.0, ws, hs)
    disp_f = np.full(hs * ws, np.nan)
    disp_f[fs_pix] = disp_s.ravel()[fs_pix]
    disp_f = disp_f.reshape(hs, ws)
    dstat = stats(disp_f, disp_w)
    log(f"Disparity vs WASS: median |diff| {dstat['median_abs']:.3f} px, RMSE {dstat['rmse']:.3f} px, "
        f"bias {dstat['bias']:+.3f} px over {dstat['n']:,} pixels")

    # 2. sea-plane space
    fs_sea = to_sea(p_world, mean_plane, B)
    w_sea = to_sea(w_world, mean_plane, B)
    x0, y0 = cx - size / 2, cy - size / 2
    nb = int(round(size / args.cell))
    zf, cf = bin_mean(fs_sea[0], fs_sea[1], fs_sea[2], x0, y0, args.cell, nb, nb)
    zw, cw = bin_mean(w_sea[0], w_sea[1], w_sea[2], x0, y0, args.cell, nb, nb)
    zstat = stats(zf, zw)
    both = np.isfinite(zf) & np.isfinite(zw)
    hs_w, hs_f = 4 * np.nanstd(zw[both]), 4 * np.nanstd(zf[both])
    cov_f, cov_w = float(np.isfinite(zf).mean()), float(np.isfinite(zw).mean())
    log(f"Elevation vs WASS ({args.cell} m cells, {size} m area): RMSE {zstat['rmse'] * 100:.1f} cm, "
        f"corr {zstat['corr']:.3f}, bias {zstat['bias'] * 100:+.1f} cm, Hs net/WASS {hs_f:.3f}/{hs_w:.3f} m, "
        f"coverage net {cov_f:.0%} WASS {cov_w:.0%}")

    # --- Summary files -------------------------------------------------------------------------------------------
    summary = {
        "wass_frame": str(wd), "method": args.method, "method_name": args.name, "model": args.model,
        "max_disp": args.max_disp, "scale": s, "input_size": [ws, hs], "full_size": [W, H],
        "valid_iters": args.valid_iters, "hiera": args.hiera, "precision": "fp32" if args.fp32 else "fp16 autocast",
        "gpu": torch.cuda.get_device_name(0), "model_load_seconds": load_sec,
        "inference_seconds": times, "peak_gpu_mb": peak, "baseline_m": B,
        "grid_area": {"center": [cx, cy], "size": size, "cell": args.cell},
        "points": {"fs_valid_pixels": n_valid, "fs_kept": int(keep.sum()), "wass": int(w_world.shape[1])},
        "plane": {"fs": fs_plane.tolist(), "wass_same_fit": w_plane.tolist(), "wass_frame": frame_plane.tolist(),
                  "wass_mean": mean_plane.tolist(), "angle_deg": plane_angle, "offset_m": plane_offset,
                  "same_fit_vs_wass_plane_deg": ref_angle},
        "disparity_px": dstat,
        "elevation_m": {**zstat, "hs_fs": hs_f, "hs_wass": hs_w, "coverage_fs": cov_f, "coverage_wass": cov_w},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    wass_stereo_sec = None
    logf = wd / "wass_stereo_log.txt"
    if logf.exists():
        for line in logf.read_text().splitlines():
            if "Dense Stereo" in line and "|" in line:
                wass_stereo_sec = float(line.split("|")[-2])
    md = [
        f"# {args.name} vs WASS, frame {args.frame:06d}",
        "",
        f"WASS run `{seq}`, model **{args.model}**, input {ws}×{hs} (scale {s}), {summary['precision']}, "
        f"{args.valid_iters} iterations, {summary['gpu']}.",
        "",
        "| | value |", "|---|---:|",
        f"| inference time (warm) | {min(times):.2f} s |",
        f"| WASS dense stereo (SGBM, CPU, same frame) | {wass_stereo_sec:.1f} s |" if wass_stereo_sec else "| WASS dense stereo | n/a |",
        f"| peak GPU memory | {peak:.0f} MB |",
        f"| points: {args.name} / WASS | {keep.sum():,} / {w_world.shape[1]:,} |",
        f"| disparity vs WASS: median abs / RMSE / bias | {dstat['median_abs']:.3f} / {dstat['rmse']:.3f} / {dstat['bias']:+.3f} px |",
        f"| elevation vs WASS: RMSE / bias | {zstat['rmse'] * 100:.1f} / {zstat['bias'] * 100:+.1f} cm |",
        f"| elevation correlation | {zstat['corr']:.3f} |",
        f"| Hs (4·std) {args.name} / WASS | {hs_f:.3f} / {hs_w:.3f} m |",
        f"| grid coverage {args.name} / WASS | {cov_f:.0%} / {cov_w:.0%} |",
        f"| sea plane, same fit on both clouds: angle / offset | {plane_angle:.3f}° / {plane_offset * 100:.1f} cm |",
        "",
    ]
    (out / "summary.md").write_text("\n".join(md))

    # --- Figure ------------------------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 4, figsize=(24, 11), constrained_layout=True)
    dlo, dhi = np.nanpercentile(disp_w, [1, 99])
    ax = axs[0, 0]; ax.imshow(Ls, cmap="gray"); ax.set_title(f"Rectified left, {ws}x{hs}"); ax.axis("off")
    ax = axs[0, 1]
    full = np.where(valid, disp_s, np.nan)
    im = ax.imshow(full, cmap="turbo", vmin=dlo, vmax=dhi)
    ax.imshow(np.where(np.isfinite(disp_f) | ~valid, np.nan, 1.0), cmap="gray", vmin=0, vmax=2, alpha=0.6)
    fig.colorbar(im, ax=ax, shrink=0.8, label="disparity [px at this scale]")
    ax.set_title(f"{args.name} disparity\n(grey = removed by the sea-plane filter)"); ax.axis("off")
    ax = axs[0, 2]
    im = ax.imshow(disp_w, cmap="turbo", vmin=dlo, vmax=dhi)
    fig.colorbar(im, ax=ax, shrink=0.8, label="disparity [px]"); ax.set_title("WASS points as disparity"); ax.axis("off")
    ax = axs[0, 3]
    im = ax.imshow(disp_f - disp_w, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8, label="network − WASS [px]")
    ax.set_title(f"Disparity difference: median |d| {dstat['median_abs']:.2f} px, bias {dstat['bias']:+.2f} px")
    ax.axis("off")

    ext = [x0, x0 + size, y0, y0 + size]
    zl = np.nanpercentile(np.abs(zw), 99)
    for ax, z, title in [(axs[1, 0], zw, "WASS elevation"), (axs[1, 1], zf, f"{args.name} elevation")]:
        im = ax.imshow(z, origin="lower", extent=ext, cmap="RdBu_r", vmin=-zl, vmax=zl)
        fig.colorbar(im, ax=ax, shrink=0.8, label="z [m]")
        ax.set_title(f"{title} ({args.cell} m cells), Hs {4 * np.nanstd(z[both]):.2f} m")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax = axs[1, 2]
    im = ax.imshow(zf - zw, origin="lower", extent=ext, cmap="RdBu_r", vmin=-zl / 2, vmax=zl / 2)
    fig.colorbar(im, ax=ax, shrink=0.8, label="network − WASS [m]")
    ax.set_title(f"Elevation difference: RMSE {zstat['rmse'] * 100:.1f} cm, bias {zstat['bias'] * 100:+.1f} cm")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax = axs[1, 3]
    ax.hexbin(zw[both], zf[both], gridsize=80, bins="log", cmap="viridis", extent=(-zl, zl, -zl, zl))
    ax.plot([-zl, zl], [-zl, zl], "w--", lw=1)
    ax.set_xlabel("WASS z [m]"); ax.set_ylabel(f"{args.name} z [m]"); ax.set_aspect("equal")
    ax.set_title(f"Per cell: correlation {zstat['corr']:.3f}, {zstat['n']:,} cells")
    fig.suptitle(f"{args.name} ({args.model}, scale {s}, {min(times):.1f} s on {summary['gpu']}) vs WASS "
                 f"— frame {args.frame:06d}", fontsize=15)
    fig.savefig(out / "comparison.png", dpi=80)
    plt.close(fig)

    # --- Point clouds ---------------------------------------------------------------------------------------------
    import open3d as o3d
    gray = Ls.ravel()[fs_pix].astype(np.float64) / 255
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fs_sea.T))
    pcd.colors = o3d.utility.Vector3dVector(np.repeat(gray[:, None], 3, axis=1))
    o3d.io.write_point_cloud(str(out / "fs_cloud_sea.ply"), pcd)
    if not args.no_html:
        from cloud_to_html import write_viewer
        margin = 0.25 * size
        def crop(p):
            return (p[0] > x0 - margin) & (p[0] < x0 + size + margin) & (p[1] > y0 - margin) & (p[1] < y0 + size + margin)
        wcol = sample_gray(I0, K0[0, 0] * w_cam0[0] / w_cam0[2] + K0[0, 2], K0[1, 1] * w_cam0[1] / w_cam0[2] + K0[1, 2])
        cf_, cw_ = crop(fs_sea), crop(w_sea)
        eye = to_sea(R0.T @ (-T0), mean_plane, B)[:, 0]  # cam0 centre
        write_viewer(out / "clouds.html",
                     [(args.name, fs_sea[:, cf_].T, np.asarray(pcd.colors)[cf_]),
                      ("WASS", w_sea[:, cw_].T, wcol[cw_])],
                     frame="zup", eye=eye, mode="height", vex=3.0, max_points=350_000, point_size=1.5,
                     title=f"Frame {args.frame:06d}: {args.name} vs WASS",
                     subtitle="Sea-plane coordinates (m), grid area ±25 %. Yellow dot = camera.")
    log(f"Wrote {out}/comparison.png, summary.md, summary.json" + ("" if args.no_html else ", clouds.html"))


if __name__ == "__main__":
    main()
