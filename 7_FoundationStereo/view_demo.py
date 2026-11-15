#!/usr/bin/env python3
"""Render FoundationStereo's run_demo.py outputs into one PNG, without needing a display.

    python view_demo.py --out_dir output/demo_vits

Reads from OUT_DIR (as written by FoundationStereo/scripts/run_demo.py, or Fast-FoundationStereo's, which
writes disp_vis.png = left | right | disparity instead; pass its --intrinsic_file):
  vis.png              left image | colourised disparity, side by side
  depth_meter.npy      depth per pixel, metres (inf where the point isn't visible in the right image)
  cloud_denoise.ply    point cloud (falls back to cloud.ply)
and writes OUT_DIR/summary.png: left image, disparity, depth, and the point cloud seen from two viewpoints.

For an interactive 3D view instead (needs WSLg / a display):
    python view_demo.py --out_dir output/demo_vits --interactive
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--intrinsic_file", type=Path, default=Path(__file__).parent / "FoundationStereo/assets/K.txt")
    ap.add_argument("--scale", type=float, default=1.0, help="the --scale run_demo.py was run with")
    ap.add_argument("--max_points", type=int, default=150_000, help="points drawn in the 3D views")
    ap.add_argument("--interactive", action="store_true", help="open an Open3D window instead")
    args = ap.parse_args()

    ply = args.out_dir / "cloud_denoise.ply"
    if not ply.exists():
        ply = args.out_dir / "cloud.ply"
    pcd = o3d.io.read_point_cloud(str(ply))
    if args.interactive:
        o3d.visualization.draw_geometries([pcd], window_name=str(ply))
        return

    if (args.out_dir / "vis.png").exists():  # FoundationStereo: left | disparity
        vis = imageio.imread(args.out_dir / "vis.png")
        w = vis.shape[1] // 2
        left, disp_vis = vis[:, :w], vis[:, w:]
    else:  # Fast-FoundationStereo: disp_vis.png = left | right | disparity
        vis = imageio.imread(args.out_dir / "disp_vis.png")
        w = vis.shape[1] // 3
        left, disp_vis = vis[:, :w], vis[:, 2 * w:]
    depth = np.load(args.out_dir / "depth_meter.npy")

    # Disparity back from depth (d = fx * B / Z), so the colour bar shows real pixel values.
    lines = args.intrinsic_file.read_text().split("\n")
    fx = float(lines[0].split()[0]) * args.scale
    baseline = float(lines[1])
    finite = np.isfinite(depth) & (depth > 0)
    disp = np.full(depth.shape, np.nan)
    disp[finite] = fx * baseline / depth[finite]

    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    if len(pts) > args.max_points:
        idx = np.random.default_rng(0).choice(len(pts), args.max_points, replace=False)
        pts, cols = pts[idx], cols[idx]

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0]); ax.imshow(left); ax.set_title("Left input"); ax.axis("off")
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(disp_vis); ax.set_title("Disparity (run_demo.py colouring)"); ax.axis("off")
    ax = fig.add_subplot(gs[0, 2])
    lo, hi = np.nanpercentile(depth[finite], [1, 99])
    im = ax.imshow(np.where(finite, depth, np.nan), cmap="turbo", vmin=lo, vmax=hi)
    fig.colorbar(im, ax=ax, label="depth [m]", shrink=0.8)
    ax.set_title(f"Depth (valid {finite.mean():.0%} of pixels)"); ax.axis("off")

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(disp, cmap="turbo")
    fig.colorbar(im, ax=ax, label="disparity [px]", shrink=0.8)
    ax.set_title(f"Disparity {np.nanmin(disp):.1f}–{np.nanmax(disp):.1f} px"); ax.axis("off")

    # Camera frame: x right, y down, z forward. Plot with y flipped so "up" is up.
    for k, (elev, azim, title) in enumerate([(15, -100, "Point cloud, from behind the camera"),
                                             (80, -90, "Point cloud, from above")]):
        ax = fig.add_subplot(gs[1, 1 + k], projection="3d")
        ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], c=cols, s=0.3, linewidths=0)
        ax.set_xlabel("x [m]"); ax.set_ylabel("z (forward) [m]"); ax.set_zlabel("-y (up) [m]")
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect(np.ptp(pts[:, [0, 2, 1]], axis=0))
        ax.set_title(f"{title}\n{len(np.asarray(pcd.points)):,} points ({ply.name})")

    out = args.out_dir / "summary.png"
    fig.savefig(out, dpi=90)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
