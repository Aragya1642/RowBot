#!/usr/bin/env python3
"""Compare WASS, WASSfast, FoundationStereo and Fast-FoundationStereo on one frame, on WASS's own sea-surface grid.

    python compare_methods.py --wass ../5_WASS_Validation/data/compare/quickstart/wass \\
        --wassfast ../5_WASS_Validation/data/compare/quickstart/wassfast --frame 0 \\
        --runs output/quickstart_frame000000_vits_s0.5 output/quickstart_frame000000_ffs-20-30-48_s0.5

--runs are compare_frame.py output folders (FoundationStereo or Fast-FoundationStereo, any model/scale). Each
contributes its sea-plane point cloud (fs_cloud_sea.ply) and timing (summary.json).

Everything is put on the 256x256 grid of WASS's gridded.nc (the grid WASSfast also uses):
  WASS        gridded.nc, this frame (WASS's own gridding: wassgridsurface)
  WASSfast    wassfast.nc, this frame (WASSfast's own output on the same grid)
  each run    its cloud, gridded here: mean of the points in each grid cell, then holes up to --fill m from data
              filled by linear interpolation (no extrapolation), inside WASS's maskZ
  WASS points WASS's own cloud (mesh_cam.xyzC) gridded the same way as the runs. It is not a method: it shows how much of
              a run's difference from gridded.nc comes from this simpler gridding rather than from the stereo itself.

There is no ground truth: WASS is the reference only in the sense that it is the established method. The pairwise
table (every method against every other) and the wavenumber spectra say more than "difference from WASS" alone.
All statistics use the grid cells where every method has data ("common area"), so they compare like with like.

Outputs (--out, inside 7_FoundationStereo/): methods.png, methods.json, methods.md, and clouds_all.html: every point
cloud overlaid in one interactive 3D viewer (WASS's cloud, each run's cloud, and WASSfast's grid as points, since
WASSfast has no point cloud), in sea-plane metres over the grid area + 25 %. In "Layer colour" mode the layers use
the same colours as methods.png; tick layers on and off to compare them.
Run inside 7_FoundationStereo's nix-shell.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_frame import bin_mean, load_camera_mesh, read_cv_mat, sample_gray, to_sea  # noqa: E402

UNIT = {"millimeter": 1e-3, "mm": 1e-3, "meter": 1.0, "m": 1.0}
COLORS = {"WASS": "#2a78d6", "WASSfast": "#e8862a", "WASS points": "#7a7a7a"}
RUN_COLORS = ["#2a9d5c", "#c23b7a", "#7b52c7", "#b58b00"]


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------------------------------

def read_nc(path):
    import netCDF4
    ds = netCDF4.Dataset(path)
    u = lambda v: UNIT.get(str(getattr(v, "units", "m")).strip().lower(), 1.0)
    X = np.asarray(ds["X_grid"][:], float) * u(ds["X_grid"])
    Y = np.asarray(ds["Y_grid"][:], float) * u(ds["Y_grid"])
    Z = np.ma.filled(ds["Z"][:].astype(np.float32), np.nan) * np.float32(u(ds["Z"]))
    mask = np.asarray(ds["maskZ"][:], float) > 0 if "maskZ" in ds.variables else np.ones(X.shape, bool)
    ds.close()
    return X, Y, Z, mask


def read_ply_points(path):
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud(str(path)).points).T  # 3xN


def grid_cloud(p, X, Y, mask, fill):
    """3xN sea-plane points (m) -> elevation on the grid (X[i, j] = x_j, Y[i, j] = y_i)."""
    from scipy.interpolate import griddata
    from scipy.ndimage import distance_transform_edt
    dx = X[0, 1] - X[0, 0]
    ny, nx = X.shape
    z, _ = bin_mean(p[0], p[1], p[2], X[0, 0] - dx / 2, Y[0, 0] - dx / 2, dx, nx, ny)
    have = np.isfinite(z)
    if fill > 0 and have.any():
        near = distance_transform_edt(~have) * dx <= fill
        todo = near & ~have
        if todo.any():
            z[todo] = griddata((X[have], Y[have]), z[have], (X[todo], Y[todo]), method="linear")
    z[~mask] = np.nan
    return z


# ----------------------------------------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------------------------------------

def pair_stats(a, b, area):
    """a vs b (b = reference) on area."""
    da, db = a[area], b[area]
    d = da - db
    dm = (da - da.mean()) - (db - db.mean())
    return {"bias_m": float(d.mean()), "rmse_m": float(np.sqrt(np.mean(d ** 2))),
            "rmse_demeaned_m": float(np.sqrt(np.mean(dm ** 2))), "corr": float(np.corrcoef(da, db)[0, 1]),
            "slope": float(np.polyfit(db - db.mean(), da - da.mean(), 1)[0])}


def largest_square(good, min_n=32):
    """Largest all-True square in a boolean grid (for the spectrum). Returns (rows slice, cols slice) or None."""
    n0, n1 = good.shape
    dp = np.zeros((n0 + 1, n1 + 1), int)
    best, bi, bj = 0, 0, 0
    for i in range(n0):
        row = good[i]
        for j in range(n1):
            if row[j]:
                dp[i + 1, j + 1] = 1 + min(dp[i, j + 1], dp[i + 1, j], dp[i, j])
                if dp[i + 1, j + 1] > best:
                    best, bi, bj = dp[i + 1, j + 1], i, j
    if best < min_n:
        return None
    best -= best % 2
    return slice(bi + 1 - best, bi + 1), slice(bj + 1 - best, bj + 1)


def k_spectrum(z, dx):
    """Omnidirectional wavenumber spectrum S(k) (m^3/rad) of a square patch (Hann window, mean removed)."""
    n = z.shape[0]
    win = np.outer(np.hanning(n), np.hanning(n))
    zz = (z - z.mean()) * win
    P = np.abs(np.fft.fftshift(np.fft.fft2(zz))) ** 2 / (win ** 2).sum() * dx ** 2 / (2 * np.pi) ** 2
    k = np.fft.fftshift(np.fft.fftfreq(n, dx)) * 2 * np.pi
    K = np.hypot(*np.meshgrid(k, k))
    dk = 2 * np.pi / (n * dx)
    edges = np.arange(dk / 2, np.abs(k).max(), dk)
    which = np.digitize(K.ravel(), edges)
    S = np.bincount(which, P.ravel(), minlength=len(edges) + 1)[1:len(edges)]
    cnt = np.bincount(which, minlength=len(edges) + 1)[1:len(edges)]
    kc = (edges[:-1] + edges[1:]) / 2
    ok = cnt > 0
    # sum over the annulus, times dk^2 per bin / dk -> density in k (2D -> 1D: S(k) = sum P(kx,ky) dkx dky / dk)
    return kc[ok], (S[ok] * dk ** 2 / dk)


# ----------------------------------------------------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------------------------------------------------

def wass_times(seq, frame):
    """WASS per-frame seconds: dense stereo (SGBM) from the frame's log, gridding per frame from timings.json."""
    out = {}
    logf = seq / "output" / f"{frame:06d}_wd" / "wass_stereo_log.txt"
    if logf.exists():
        for line in logf.read_text().splitlines():
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                try:
                    if "Dense Stereo" in line:
                        out["dense_stereo_s"] = float(parts[-2])
                    elif "Total" in line or "TOTAL" in line:
                        out["stereo_total_s"] = float(parts[-2])
                except ValueError:
                    pass
    tj = seq / "timings.json"
    if tj.exists():
        t = json.loads(tj.read_text())
        g = t.get("stages", {}).get("grid", {}).get("seconds")
        if g and t.get("n_pairs"):
            out["grid_per_frame_s"] = g / t["n_pairs"]
        out["workers"] = t.get("workers")
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wass", required=True, type=Path, help="run_wass.py output folder")
    ap.add_argument("--wassfast", type=Path, help="run_wassfast.py output folder (wassfast.nc) of the same pairs")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--runs", type=Path, nargs="*", default=[], help="compare_frame.py output folders")
    ap.add_argument("--fill", type=float, default=1.5, help="fill holes up to this far from data (m) when gridding clouds")
    ap.add_argument("--out", type=Path, help="default: output/<run name>_frame<NNNNNN>_methods")
    ap.add_argument("--no-html", action="store_true", help="skip clouds_all.html")
    ap.add_argument("--max_points", type=int, default=250_000, help="per layer in clouds_all.html")
    args = ap.parse_args()

    seq = args.wass.resolve()
    name = seq.name if seq.name != "wass" else seq.parent.name
    out = (args.out or HERE / "output" / f"{name}_frame{args.frame:06d}_methods").resolve()
    if HERE not in out.parents:
        sys.exit(f"ERROR: --out must be inside {HERE}")
    out.mkdir(parents=True, exist_ok=True)

    # --- WASS gridded surface ---------------------------------------------------------------------------------------
    import netCDF4
    X, Y, Zw_all, mask = read_nc(seq / "gridding" / "gridded.nc")
    ds = netCDF4.Dataset(seq / "gridding" / "gridded.nc")
    frames = np.asarray(ds["workdir"][:], int)
    ds.close()
    if args.frame not in frames:
        sys.exit(f"ERROR: frame {args.frame} not in gridded.nc (has {frames.min()}..{frames.max()})")
    ti = int(np.flatnonzero(frames == args.frame)[0])  # time index; WASSfast frame i = WASS pair i
    dx = X[0, 1] - X[0, 0]
    log(f"Grid {X.shape[1]}x{X.shape[0]}, {dx:.3f} m, x {X.min():.0f}..{X.max():.0f}, y {Y.min():.0f}..{Y.max():.0f} m; "
        f"frame {args.frame} = time index {ti}")
    info = json.loads((seq / "timings.json").read_text()) if (seq / "timings.json").exists() else {}
    B = info.get("baseline_m")

    surf = {}   # label -> elevation grid
    meta = {}   # label -> info (kind, timing, colour)
    surf["WASS"] = np.where(mask, Zw_all[ti], np.nan)
    wt = wass_times(seq, args.frame)
    meta["WASS"] = {"kind": "reference", "color": COLORS["WASS"], "hardware": f"CPU ({info.get('machine', {}).get('cpus', '?')} cores)",
                    "seconds": wt.get("dense_stereo_s"), "seconds_what": "SGBM dense stereo, this frame",
                    "detail": wt}

    if args.wassfast:
        Xf, Yf, Zf_all, maskf = read_nc(args.wassfast / "wassfast.nc")
        if not np.allclose(X, Xf, atol=1e-3) and np.allclose(X, Xf.T, atol=1e-3):
            Xf, Yf, maskf, Zf_all = Xf.T, Yf.T, maskf.T, Zf_all.transpose(0, 2, 1)
        if not (np.allclose(X, Xf, atol=1e-3) and np.allclose(Y, Yf, atol=1e-3)):
            sys.exit("ERROR: wassfast.nc is not on gridded.nc's grid")
        surf["WASSfast"] = np.where(mask & maskf, Zf_all[ti], np.nan)
        ft = json.loads((args.wassfast / "timings.json").read_text()) if (args.wassfast / "timings.json").exists() else {}
        meta["WASSfast"] = {"kind": "method", "color": COLORS["WASSfast"],
                            "hardware": f"CPU, TensorFlow {ft.get('tensorflow', '')}".replace(" (CPU only)", ""),
                            "seconds": ft.get("marginal_seconds_per_pair"),
                            "seconds_what": "marginal time per pair (batches of %s)" % ft.get("batchsize", "?")}

    # --- Clouds: WASS points (gridding check) and each run -------------------------------------------------------
    planes = np.atleast_2d(np.loadtxt(seq / "output" / "planes.txt"))
    mean_plane = np.nanmean(planes, axis=0)
    wd = seq / "output" / f"{args.frame:06d}_wd"
    w_world = load_camera_mesh(wd / "mesh_cam.xyzC")
    w_sea = to_sea(w_world, mean_plane, B)
    surf["WASS points"] = grid_cloud(w_sea, X, Y, mask, args.fill)
    meta["WASS points"] = {"kind": "check", "color": COLORS["WASS points"], "hardware": "", "seconds": None,
                           "seconds_what": "WASS's cloud, gridded like the runs"}

    names = {}
    for k, run in enumerate(args.runs):
        run = run.resolve()
        sj = json.loads((run / "summary.json").read_text())
        if Path(sj["wass_frame"]).resolve() != (seq / "output" / f"{args.frame:06d}_wd").resolve():
            sys.exit(f"ERROR: {run} was made from {sj['wass_frame']}, not this WASS frame")
        method_name = sj.get("method_name", "FoundationStereo")
        short = "FS" if method_name == "FoundationStereo" else "FFS"
        label = f"{short} {sj['model']} s{sj['scale']:g}"
        if label in names:
            label += f" ({run.name})"
        names[label] = run
        surf[label] = grid_cloud(read_ply_points(run / "fs_cloud_sea.ply"), X, Y, mask, args.fill)
        meta[label] = {"kind": "method", "color": RUN_COLORS[k % len(RUN_COLORS)], "hardware": sj.get("gpu", "GPU"),
                       "seconds": float(min(sj["inference_seconds"])),
                       "seconds_what": f"network forward pass ({sj['input_size'][0]}x{sj['input_size'][1]}, "
                                       f"{sj['precision']}, {sj['valid_iters']} iters)",
                       "peak_gpu_mb": sj.get("peak_gpu_mb"), "run": str(run), "method_name": method_name}
        log(f"{label}: {run}")

    labels = list(surf)
    for lab in labels:
        surf[lab] = surf[lab].astype(np.float64)
    coverage = {lab: float(np.isfinite(surf[lab]).sum() / mask.sum()) for lab in labels}
    area = mask.copy()
    for lab in labels:
        area &= np.isfinite(surf[lab])
    log(f"Common area: {area.sum():,} cells ({area.sum() / mask.sum():.0%} of the masked grid)")

    # --- Statistics ------------------------------------------------------------------------------------------------
    hs = {lab: float(4 * surf[lab][area].std()) for lab in labels}
    vs_wass = {lab: pair_stats(surf[lab], surf["WASS"], area) for lab in labels if lab != "WASS"}
    pairwise = {a: {b: pair_stats(surf[a], surf[b], area) for b in labels} for a in labels}

    box = largest_square(area)
    spectra = {}
    if box is not None:
        for lab in labels:
            k, S = k_spectrum(surf[lab][box], dx)
            spectra[lab] = (k, S)
        nbox = box[0].stop - box[0].start
        log(f"Spectrum box: {nbox}x{nbox} cells = {nbox * dx:.1f} m square")
    # Spectral energy relative to WASS in two wavelength bands: 5-20 m (where WASS follows the expected k^-3 tail, so
    # mostly real waves) and 2-5 m (where WASS flattens into a plateau above k^-3, so partly matching noise). Plus the
    # longest wavelength at which a method has dropped below half of WASS's energy.
    spec_summary = {}
    if spectra:
        kw, Sw = spectra["WASS"]
        for lab in labels:
            k, S = spectra[lab]
            band = lambda lo, hi: (k >= 2 * np.pi / hi) & (k < 2 * np.pi / lo)
            ratio = lambda b: float(S[b].sum() / Sw[b].sum()) if b.any() else None
            low = np.flatnonzero(S / Sw < 0.5)
            spec_summary[lab] = {
                "energy_ratio_5_20m": ratio(band(5, 20)), "energy_ratio_2_5m": ratio(band(2, 5)),
                "half_energy_wavelength_m": float(2 * np.pi / k[low[0]]) if len(low) else None,
            }

    for lab in labels:
        if lab == "WASS":
            continue
        s = vs_wass[lab]
        log(f"{lab:>22}: vs WASS RMSE {s['rmse_m'] * 100:5.1f} cm, demeaned {s['rmse_demeaned_m'] * 100:5.1f} cm, "
            f"corr {s['corr']:.3f}, bias {s['bias_m'] * 100:+.1f} cm, slope {s['slope']:.2f}; Hs {hs[lab]:.2f} m "
            f"(WASS {hs['WASS']:.2f}); coverage {coverage[lab]:.0%}")

    res = {"wass": str(seq), "wassfast": str(args.wassfast.resolve()) if args.wassfast else None, "frame": args.frame,
           "grid_spacing_m": dx, "common_area_cells": int(area.sum()), "common_area_fraction": float(area.sum() / mask.sum()),
           "fill_m": args.fill, "methods": {lab: {**meta[lab], "hs_m": hs[lab], "coverage": coverage[lab],
                                                  "vs_wass": vs_wass.get(lab), "spectrum": spec_summary.get(lab)}
                                            for lab in labels},
           "pairwise": pairwise}
    (out / "methods.json").write_text(json.dumps(res, indent=2, default=float))
    write_md(out, res, labels, seq, args)
    plot(out, X, Y, surf, labels, meta, area, hs, vs_wass, pairwise, spectra, box, dx, args.frame)
    if not args.no_html:
        write_clouds(out / "clouds_all.html", args, wd, w_world, w_sea, mean_plane, B, X, Y, surf, meta, names)
    log(f"Wrote {out}/methods.png, methods.md, methods.json" + ("" if args.no_html else ", clouds_all.html"))


def write_clouds(path, args, wd, w_world, w_sea, mean_plane, B, X, Y, surf, meta, names):
    """All clouds in one viewer: WASS's cloud, WASSfast's grid (as points), each run's cloud."""
    import cv2
    from cloud_to_html import write_viewer
    size = X.max() - X.min()
    x0, x1 = X.min() - 0.25 * size, X.max() + 0.25 * size
    y0, y1 = Y.min() - 0.25 * size, Y.max() + 0.25 * size
    crop = lambda p: (p[0] > x0) & (p[0] < x1) & (p[1] > y0) & (p[1] < y1)
    K0 = read_cv_mat(wd / "intrinsics_00000000.xml", "intr")
    R0 = np.loadtxt(wd / "Cam0_poseR.txt")
    T0 = np.loadtxt(wd / "Cam0_poseT.txt").reshape(3, 1)
    I0 = cv2.imread(str(wd / "undistorted/00000000.png"), cv2.IMREAD_GRAYSCALE)

    layers = []
    c = crop(w_sea)
    w_cam0 = R0 @ w_world[:, c] + T0
    layers.append(("WASS (point cloud)", w_sea[:, c].T,
                   sample_gray(I0, K0[0, 0] * w_cam0[0] / w_cam0[2] + K0[0, 2], K0[1, 1] * w_cam0[1] / w_cam0[2] + K0[1, 2]),
                   meta["WASS"]["color"]))
    if "WASSfast" in surf:
        z = surf["WASSfast"]
        ok = np.isfinite(z)
        layers.append(("WASSfast (grid points)", np.stack([X[ok], Y[ok], z[ok]], axis=1), None, meta["WASSfast"]["color"]))
    import open3d as o3d
    for label, run in names.items():
        pcd = o3d.io.read_point_cloud(str(run / "fs_cloud_sea.ply"))
        p = np.asarray(pcd.points).T
        cols = np.asarray(pcd.colors) if pcd.has_colors() else None
        c = crop(p)
        layers.append((label, p[:, c].T, None if cols is None else cols[c], meta[label]["color"]))
    eye = to_sea(R0.T @ (-T0), mean_plane, B)[:, 0]  # cam0 centre
    write_viewer(path, layers, frame="zup", eye=eye, mode="layer", vex=3.0, max_points=args.max_points, point_size=1.5,
                 title=f"Frame {args.frame:06d}: all methods",
                 subtitle="Sea-plane coordinates (m), grid area ±25 %. Tick layers on/off; colour by layer or height.")


def write_md(out, res, labels, seq, args):
    M = res["methods"]
    L = [f"# WASS vs WASSfast vs FoundationStereo models, frame {args.frame:06d}", "",
         f"WASS run `{seq}`. Everything on gridded.nc's grid ({res['grid_spacing_m']:.3f} m). Statistics over the "
         f"{res['common_area_cells']:,} cells where every method has data ({res['common_area_fraction']:.0%} of the grid). "
         "No ground truth: WASS is only the established reference. *WASS points* is WASS's own cloud gridded the way "
         "the network clouds are: it shows what the simpler gridding alone costs.", "",
         "| method | time / frame | hardware | Hs | vs WASS: RMSE | RMSE w/o offset | corr | bias | slope | energy 5–20 m vs WASS | energy 2–5 m vs WASS | coverage |",
         "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    pct = lambda x: "–" if x is None else f"{x:.0%}"
    for lab in labels:
        m = M[lab]
        t = f"{m['seconds']:.2f} s" if m.get("seconds") else "–"
        v = m.get("vs_wass")
        sp = m.get("spectrum") or {}
        e1, e2 = pct(sp.get("energy_ratio_5_20m")), pct(sp.get("energy_ratio_2_5m"))
        if v:
            L.append(f"| {lab} | {t} | {m['hardware']} | {m['hs_m']:.2f} m | {v['rmse_m'] * 100:.1f} cm | "
                     f"{v['rmse_demeaned_m'] * 100:.1f} cm | {v['corr']:.3f} | {v['bias_m'] * 100:+.1f} cm | {v['slope']:.2f} | "
                     f"{e1} | {e2} | {m['coverage']:.0%} |")
        else:
            L.append(f"| **{lab}** (reference) | {t} | {m['hardware']} | {m['hs_m']:.2f} m | – | – | – | – | – | 100% | 100% | {m['coverage']:.0%} |")
    L += ["", "Times: " + "; ".join(f"{lab}: {M[lab]['seconds_what']}" for lab in labels if M[lab].get("seconds")) + ".",
          "Energy columns: spectral energy in that wavelength band relative to WASS (below 100 % = smoother, above = "
          "rougher or noisier). In 5–20 m WASS follows the expected k⁻³ tail; in 2–5 m it flattens above it, which "
          "suggests matching noise, so less than WASS there is not necessarily worse.", "", "## Pairwise RMSE without offset (cm) / correlation", "",
          "| | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for a in labels:
        L.append(f"| {a} | " + " | ".join("–" if a == b else
                                          f"{res['pairwise'][a][b]['rmse_demeaned_m'] * 100:.1f} / {res['pairwise'][a][b]['corr']:.3f}"
                                          for b in labels) + " |")
    (out / "methods.md").write_text("\n".join(L) + "\n")


def plot(out, X, Y, surf, labels, meta, area, hs, vs_wass, pairwise, spectra, box, dx, frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(labels)
    fig = plt.figure(figsize=(4.6 * n, 14.5), constrained_layout=True)
    gs = fig.add_gridspec(3, n, height_ratios=[1, 1, 1.05])
    ext = [X.min() - dx / 2, X.max() + dx / 2, Y.min() - dx / 2, Y.max() + dx / 2]
    zl = np.nanpercentile(np.abs(surf["WASS"][area]), 99)
    for j, lab in enumerate(labels):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(surf[lab], origin="lower", extent=ext, cmap="RdBu_r", vmin=-zl, vmax=zl)
        ax.contour(X, Y, area.astype(float), levels=[0.5], colors="k", linewidths=0.6)
        t = meta[lab].get("seconds")
        ax.set_title(f"{lab}\nHs {hs[lab]:.2f} m" + (f", {t:.2f} s" if t else ""), color=meta[lab]["color"], fontsize=11)
        ax.set_xticks([]) if j else ax.set_ylabel("y [m]")
        if j:
            ax.set_yticks([])
        ax = fig.add_subplot(gs[1, j])
        if lab == "WASS":
            ax.axis("off")
            ax.text(0.5, 0.5, "elevation z [m]\n(colour scale of row 1)\n\nrow 2: method − WASS\n(black line: common area)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
            fig.colorbar(im, ax=ax, location="left", shrink=0.7)
            continue
        im2 = ax.imshow(surf[lab] - surf["WASS"], origin="lower", extent=ext, cmap="RdBu_r", vmin=-zl / 2, vmax=zl / 2)
        ax.contour(X, Y, area.astype(float), levels=[0.5], colors="k", linewidths=0.6)
        s = vs_wass[lab]
        ax.set_title(f"− WASS: RMSE {s['rmse_m'] * 100:.1f} cm (w/o offset {s['rmse_demeaned_m'] * 100:.1f})\n"
                     f"corr {s['corr']:.3f}, bias {s['bias_m'] * 100:+.1f} cm", fontsize=10)
        ax.set_xlabel("x [m]")
        if j > 1:
            ax.set_yticks([])
    fig.colorbar(im2, ax=fig.axes[-1], shrink=0.8, label="difference [m]")

    # Spectra
    split = max(2, n - 3)
    ax = fig.add_subplot(gs[2, :split])
    if spectra:
        for lab in labels:
            k, S = spectra[lab]
            ax.loglog(k, S, color=meta[lab]["color"], lw=2.2 if lab == "WASS" else 1.5,
                      ls=":" if lab == "WASS points" else "-", label=lab)
        k = spectra["WASS"][0]
        kk = k[(k > 0.3) & (k < 3)]
        if len(kk):
            S0 = np.interp(kk[0], *spectra["WASS"])
            ax.loglog(kk, S0 * (kk / kk[0]) ** -3, "k--", lw=0.8, label="k⁻³ (saturation range)")
        for lam in [20, 5, 2]:
            ax.axvline(2 * np.pi / lam, color="0.8", lw=0.8)
            ax.text(2 * np.pi / lam, ax.get_ylim()[1], f" {lam} m", va="top", fontsize=9, color="0.4")
        lo = min(np.percentile(spectra[lab][1], 5) for lab in labels if lab != "WASS")
        ax.set_ylim(bottom=lo / 10)  # WASS's own gridding cuts off sharply above ~7 rad/m; don't let it set the scale
        nb = box[0].stop - box[0].start
        ax.set_title(f"Wavenumber spectrum, {nb * dx:.0f} m square inside the common area")
        ax.set_xlabel("k [rad/m]"); ax.set_ylabel("S(k) [m³/rad]"); ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    # Pairwise matrix
    ax = fig.add_subplot(gs[2, split:])
    M = np.array([[pairwise[a][b]["rmse_demeaned_m"] * 100 if a != b else np.nan for b in labels] for a in labels])
    im = ax.imshow(M, cmap="magma_r", vmin=0)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{M[i, j]:.1f}\n{pairwise[labels[i]][labels[j]]['corr']:.2f}", ha="center", va="center",
                        fontsize=9, color="w" if M[i, j] > np.nanmax(M) * 0.6 else "k")
    ax.set_xticks(range(n), labels, rotation=30, ha="right"); ax.set_yticks(range(n), labels)
    fig.colorbar(im, ax=ax, shrink=0.8, label="RMSE without offset [cm]")
    ax.set_title("Every pair: RMSE without offset [cm] / correlation (common area)")

    fig.suptitle(f"Frame {frame:06d}: WASS vs WASSfast vs FoundationStereo models, on WASS's grid", fontsize=16)
    fig.savefig(out / "methods.png", dpi=75)
    plt.close(fig)


if __name__ == "__main__":
    main()
