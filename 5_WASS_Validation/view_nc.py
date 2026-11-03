#!/usr/bin/env python3
"""Animate a gridded sea-surface NetCDF (Z over time) seen from above.

For reconstructions that come without camera images, e.g. the S.A. Agulhas II shipborne
WASS data. Reads both formats used in this project:
  - WASS / wassgridsurface style: Z(count, X, Y) in mm, X_grid / Y_grid in mm, fps variable
  - Ifremer style:                Z(time, Y, X) in m,  X / Y in m, "Acquisition rate [Hz]" attribute
Units are taken from the variables' `units` attributes. The local file or https:// URL both work.

The view is drawn with x to the right and y up (for Agulhas II: y = distance from the ship's
port side, so the ship is below the bottom edge). Colour = elevation relative to the record
mean (red = crest, blue = trough, grey = no data).

Controls: space = pause, left/right = step (while paused), up/down = faster/slower, q / Esc = quit.

Examples:
    python3 view_nc.py data/SA_Agulhas_II_Winter_2017_Cruise__WASS/wass__20170704_0805.nc
    python3 view_nc.py data/SA_Agulhas_II_Winter_2017_Cruise__WASS/wass__20170704_0805.nc --fps 8 --save agulhas_0805.mp4
"""

import argparse
import sys

import cv2
import numpy as np

from view_stereo import LUT, draw_colorbar, play, save_video, z_levels

UNIT_SCALE = {"mm": 1e-3, "millimeter": 1e-3, "millimeters": 1e-3, "cm": 1e-2, "m": 1.0, "meter": 1.0, "meters": 1.0}


def to_m(var):
    return UNIT_SCALE.get(getattr(var, "units", "m").strip().lower(), 1.0)


class NcGrid:
    def __init__(self, src):
        import netCDF4

        if src.startswith(("http://", "https://")) and "#" not in src:
            src += "#mode=bytes"
        self.ds = ds = netCDF4.Dataset(src)
        if "Z" not in ds.variables:
            sys.exit(f"{src} has no Z variable")
        self.zvar = ds["Z"]
        self.zscale = to_m(self.zvar)
        self.n = self.zvar.shape[0]

        xname, yname = ("X_grid", "Y_grid") if "X_grid" in ds.variables else ("X", "Y")
        X = np.asarray(ds[xname][:], float) * to_m(ds[xname])
        Y = np.asarray(ds[yname][:], float) * to_m(ds[yname])
        # Arrange the grid so x runs along columns (left -> right) and y along rows (top = largest y).
        self.transpose = np.ptp(X, axis=0).max() > np.ptp(X, axis=1).max()
        if self.transpose:
            X, Y = X.T, Y.T
        self.flip_cols = X[0, 0] > X[0, -1]
        self.flip_rows = Y[0, 0] < Y[-1, 0]
        self.xr, self.yr = (np.nanmin(X), np.nanmax(X)), (np.nanmin(Y), np.nanmax(Y))

        if "fps" in ds.variables:
            self.fps = float(np.ravel(ds["fps"][:])[0])
        elif "Acquisition rate [Hz]" in ds.ncattrs():
            self.fps = float(ds.getncattr("Acquisition rate [Hz]"))
        else:
            t = np.ravel(ds["time"][:2])
            self.fps = 1 / float(t[1] - t[0])

        # Reference level and colour range from a sample of frames (robust to occasional spikes).
        sample = np.array([self.raw(t) for t in np.linspace(0, self.n - 1, 20).astype(int)])
        self.zmean = float(np.nanmedian(sample))
        robust_std = 1.4826 * np.nanmedian(np.abs(sample - self.zmean))  # MAD-based, ignores spikes
        self.zlim = float(np.ceil(3 * robust_std * 10) / 10)

    def raw(self, t):
        return np.ma.filled(self.zvar[t].astype(float), np.nan) * self.zscale

    def frame(self, t):
        """Elevation (m, relative to record mean) oriented x right / y up."""
        z = self.raw(t) - self.zmean
        if self.transpose:
            z = z.T
        if self.flip_rows:
            z = z[::-1]
        if self.flip_cols:
            z = z[:, ::-1]
        return z


def render(g, t, args, info=""):
    z = g.frame(t)
    img = LUT[z_levels(np.nan_to_num(z), args.zlim)]
    img[~np.isfinite(z)] = 64

    # Scale so the longer side is args.size px, keeping the grid's metric aspect ratio.
    w_m, h_m = np.ptp(g.xr), np.ptp(g.yr)
    s = args.size / max(w_m, h_m)
    w, h = round(w_m * s), round(h_m * s)
    img = cv2.resize(np.ascontiguousarray(img), (w, h), interpolation=cv2.INTER_NEAREST)

    left, right, top, bottom = 60, 120, 40, 50
    canvas = np.full((h + top + bottom, w + left + right, 3), 30, np.uint8)
    canvas[top : top + h, left : left + w] = img
    draw_colorbar(canvas, args.zlim)

    font, white = cv2.FONT_HERSHEY_SIMPLEX, (230, 230, 230)
    cv2.putText(canvas, f"{g.xr[0]:.0f} m", (left, top + h + 20), font, 0.5, white, 1)
    cv2.putText(canvas, f"{g.xr[1]:.0f} m", (left + w - 50, top + h + 20), font, 0.5, white, 1)
    cv2.putText(canvas, "x", (left + w // 2, top + h + 20), font, 0.5, white, 1)
    cv2.putText(canvas, f"{g.yr[1]:.0f}", (5, top + 12), font, 0.5, white, 1)
    cv2.putText(canvas, f"{g.yr[0]:.0f}", (5, top + h), font, 0.5, white, 1)
    cv2.putText(canvas, "y (m)", (5, top + h // 2), font, 0.5, white, 1)

    valid = np.isfinite(z)
    hs = 4 * np.nanstd(z) if valid.any() else float("nan")
    cv2.putText(canvas, f"{args.title}   t = {t / g.fps:7.1f} s  (step {t})", (left, 26), font, 0.6, white, 1)
    cv2.putText(canvas, f"{info}   valid {valid.mean():.0%}   4*std(Z) = {hs:.1f} m",
                (left, top + h + 42), font, 0.5, (0, 255, 0), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nc", help="NetCDF file or https:// URL")
    parser.add_argument("--fps", type=float, help="playback / output frame rate (default: the file's acquisition rate)")
    parser.add_argument("--size", type=int, default=900, help="longer side of the map in px (default 900)")
    parser.add_argument("--start", type=int, default=0, help="first time step")
    parser.add_argument("--count", type=int, default=0, help="show at most COUNT steps (0 = all)")
    parser.add_argument("--step", type=int, default=1, help="show every STEP-th time step")
    parser.add_argument("--zlim", type=float, help="colour scale +/-ZLIM in m (default: 3 x robust std of Z over 20 sample frames)")
    parser.add_argument("--loop", action="store_true", help="loop playback")
    parser.add_argument("--save", metavar="FILE.mp4", help="write an mp4 instead of opening a window")
    args = parser.parse_args()

    g = NcGrid(args.nc)
    args.fps = args.fps or g.fps
    args.zlim = args.zlim or g.zlim
    args.title = args.nc.rstrip("/").split("/")[-1]
    steps = list(range(g.n))[args.start :: args.step]
    if args.count > 0:
        steps = steps[: args.count]
    if not steps:
        sys.exit("No time steps left after --start/--step/--count.")
    print(f"{args.title}: {g.n} steps @ {g.fps:g} Hz ({g.n / g.fps / 60:.1f} min), grid x {g.xr[0]:.0f}..{g.xr[1]:.0f} m, "
          f"y {g.yr[0]:.0f}..{g.yr[1]:.0f} m, colour scale +/-{args.zlim:g} m; showing {len(steps)} steps", file=sys.stderr)

    render_fn = lambda i, info: render(g, steps[i], args, info)
    if args.save:
        save_video(len(steps), render_fn, args)
    else:
        play(len(steps), render_fn, args, lambda i: f"{args.title}  step {steps[i]}")


if __name__ == "__main__":
    main()
