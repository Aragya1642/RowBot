#!/usr/bin/env python3
"""Compare a WASS run (run_wass.py) with a WASSfast run (run_wassfast.py) of the same stereo pairs: speed, accuracy, videos.

    python3 compare_runs.py --wass data/compare/quickstart/wass --wassfast data/compare/quickstart/wassfast \\
        --out data/compare/quickstart/report

Both runs must share the grid (run_wassfast.py uses the WASS run's gridding/config.mat), so every grid point is the
same place on the sea in both NetCDF files and the surfaces can be compared point by point. WASS is the reference.

Extra WASSfast runs of the same pairs (e.g. another --fps) can be given with --also LABEL=DIR; they are added to the
accuracy table only.

Writes to --out:
  timing_overview.png      wall time per stage (from scratch / calibrated cameras / WASSfast), time and CPU per pair
  timing_scaling.png       projected processing time vs record length, against real time
  accuracy_timeseries.png  Hs per frame, RMSE + correlation per frame, elevation at the grid centre
  accuracy_maps.png        one frame side by side + difference, and where the differences are over the whole record
  accuracy_summary.png     one frame (both maps, difference, scatter) above Hs per frame + elevation at one grid point
  accuracy_spectrum.png    wavenumber spectrum of both (what scales WASSfast keeps) + point-by-point scatter
  wass.mp4, wassfast.mp4   the same layout for each: surface over the undistorted cam0 image + top view, same colours
  comparison.mp4           both overlays, both top views and their difference, frame by frame
  comparison.json, comparison_summary.md
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from view_stereo import LUT, N_LEVELS, draw_colorbar, draw_overlay, to_bgr8  # noqa: E402

UNIT_SCALE = {"mm": 1e-3, "millimeter": 1e-3, "millimeters": 1e-3, "cm": 1e-2, "m": 1.0, "meter": 1.0, "meters": 1.0}
WASS_C, FAST_C, INK, INK2, GRID_C, SURFACE = "#2a78d6", "#e8862a", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def to_m(var):
    return UNIT_SCALE.get(str(getattr(var, "units", "m")).strip().lower(), 1.0)


def fmt(sec):
    if sec < 120:
        return f"{sec:.1f} s"
    if sec < 7200:
        return f"{sec / 60:.1f} min"
    return f"{sec / 3600:.1f} h"


# ----------------------------------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------------------------------

class Surfaces:
    """Z (m) of one NetCDF file on its grid, plus frame numbers."""

    def __init__(self, path):
        import netCDF4
        self.path = Path(path)
        ds = netCDF4.Dataset(path)
        self.X = np.asarray(ds["X_grid"][:], float) * to_m(ds["X_grid"])
        self.Y = np.asarray(ds["Y_grid"][:], float) * to_m(ds["Y_grid"])
        self.Z = np.ma.filled(ds["Z"][:].astype(np.float32), np.nan) * np.float32(to_m(ds["Z"]))
        self.mask = np.asarray(ds["maskZ"][:], float) > 0 if "maskZ" in ds.variables else np.ones(self.X.shape, bool)
        # Frame number of each time step: WASS writes the WASS frame (workdir) number. WASSfast 1.6.3 writes the
        # index within its CNN batch there (0..15, 0..15, ...), so use workdir only when it is strictly increasing.
        wd = np.asarray(ds["workdir"][:], int) if "workdir" in ds.variables else None
        if wd is not None and len(wd) == self.Z.shape[0] and np.all(np.diff(wd) > 0):
            self.frames = wd
        else:
            self.frames = np.arange(self.Z.shape[0])
        meta = ds.groups.get("meta")
        self.meta = {k: np.asarray(v[:], float) for k, v in meta.variables.items()} if meta is not None else {}
        ds.close()

    def transpose(self):
        self.X, self.Y, self.mask = self.X.T, self.Y.T, self.mask.T
        self.Z = np.ascontiguousarray(self.Z.transpose(0, 2, 1))


def load_pair(wass_dir, fast_dir):
    w = Surfaces(wass_dir / "gridding" / "gridded.nc")
    f = Surfaces(fast_dir / "wassfast.nc")
    if not np.allclose(w.X, f.X, atol=1e-3) and np.allclose(w.X, f.X.T, atol=1e-3):
        f.transpose()
    if not (np.allclose(w.X, f.X, atol=1e-3) and np.allclose(w.Y, f.Y, atol=1e-3)):
        sys.exit("ERROR: the two NetCDF files are not on the same grid (was WASSfast run with this WASS run's config.mat?)")
    # Frames present in both. WASSfast frame i is WASS pair i (run_wassfast.py links pairs.txt in order).
    common = np.intersect1d(w.frames, f.frames)
    wi = np.searchsorted(w.frames, common)
    fi = np.searchsorted(f.frames, common)
    return w, f, common, wi, fi


# ----------------------------------------------------------------------------------------------------------------------
# Accuracy
# ----------------------------------------------------------------------------------------------------------------------

def accuracy(w, f, wi, fi):
    """Per-frame metrics on the grid points where both have data."""
    rows = []
    sum_absdiff = np.zeros(w.X.shape)
    n_valid = np.zeros(w.X.shape)
    for a, b in zip(wi, fi):
        zw, zf = w.Z[a], f.Z[b]
        valid = np.isfinite(zw) & np.isfinite(zf) & w.mask & f.mask
        if valid.sum() < 100:
            rows.append(None)
            continue
        dw, df = zw[valid], zf[valid]
        d = df - dw
        # Shape error: each surface relative to its own mean over the common area (removes a constant offset).
        ds = (df - df.mean()) - (dw - dw.mean())
        rows.append({
            "valid_points": int(valid.sum()), "coverage": float(valid.mean()),
            "bias_m": float(d.mean()), "rmse_m": float(np.sqrt(np.mean(d ** 2))),
            "rmse_demeaned_m": float(np.sqrt(np.mean(ds ** 2))), "mae_m": float(np.mean(np.abs(d))),
            "corr": float(np.corrcoef(dw, df)[0, 1]),
            "hs_wass_m": float(4 * dw.std()), "hs_wassfast_m": float(4 * df.std()),
        })
        diff = np.where(valid, np.abs(zf - zw), 0)
        sum_absdiff += diff
        n_valid += valid
    with np.errstate(invalid="ignore"):
        mean_absdiff = np.where(n_valid > 0, sum_absdiff / n_valid, np.nan)
    return rows, mean_absdiff, n_valid


def summarize(rows):
    ok = [r for r in rows if r]
    if not ok:
        return {}
    get = lambda k: np.array([r[k] for r in ok])
    hs_w, hs_f = get("hs_wass_m"), get("hs_wassfast_m")
    return {
        "frames_compared": len(ok),
        "rmse_m_median": float(np.median(get("rmse_m"))), "rmse_m_mean": float(np.mean(get("rmse_m"))),
        "rmse_demeaned_m_median": float(np.median(get("rmse_demeaned_m"))),
        "bias_m_mean": float(np.mean(get("bias_m"))), "corr_median": float(np.median(get("corr"))),
        "corr_mean": float(np.mean(get("corr"))),
        "hs_wass_m_mean": float(hs_w.mean()), "hs_wassfast_m_mean": float(hs_f.mean()),
        "hs_ratio_mean": float(np.mean(hs_f / hs_w)), "coverage_mean": float(np.mean(get("coverage"))),
        "rmse_over_hs": float(np.median(get("rmse_m")) / hs_w.mean()),
    }


def best_box(valid_frac, min_frac=0.98, min_half=8):
    """Largest square anywhere with WASS data in >= min_frac of frames (WASS rarely covers the grid centre fully)."""
    good = (valid_frac >= min_frac).astype(np.uint8)
    n = good.shape[0]
    dp = np.zeros((n + 1, n + 1), int)
    best, bi, bj = 0, 0, 0
    for i in range(n):
        for j in range(n):
            if good[i, j]:
                dp[i + 1, j + 1] = 1 + min(dp[i, j + 1], dp[i + 1, j], dp[i, j])
                if dp[i + 1, j + 1] > best:
                    best, bi, bj = dp[i + 1, j + 1], i, j
    if best < 2 * min_half:
        return None
    best -= best % 2
    return slice(bi + 1 - best, bi + 1), slice(bj + 1 - best, bj + 1)


def k_spectrum(Z, idx, box, dx):
    """Omnidirectional wavenumber spectrum (averaged over frames) in a square box; NaNs filled with the mean."""
    sl_i, sl_j = box
    n = sl_i.stop - sl_i.start
    win = np.outer(np.hanning(n), np.hanning(n))
    acc = None
    for t in idx:
        z = Z[t][sl_i, sl_j].astype(float)
        if np.isnan(z).mean() > 0.05:
            continue
        z = np.where(np.isfinite(z), z, np.nanmean(z))
        z = (z - z.mean()) * win
        P = np.abs(np.fft.fftshift(np.fft.fft2(z))) ** 2
        acc = P if acc is None else acc + P
    if acc is None:
        return None, None
    k = np.fft.fftshift(np.fft.fftfreq(n, dx)) * 2 * np.pi
    KX, KY = np.meshgrid(k, k)
    K = np.hypot(KX, KY)
    dk = 2 * np.pi / (n * dx)
    bins = np.arange(dk / 2, K.max(), dk)
    which = np.digitize(K.ravel(), bins)
    S = np.bincount(which, acc.ravel(), minlength=len(bins) + 1)[1:len(bins)]
    cnt = np.bincount(which, minlength=len(bins) + 1)[1:len(bins)]
    kc = (bins[:-1] + bins[1:]) / 2
    keep = (cnt > 0) & (kc <= np.abs(k).max())
    return kc[keep], (S[keep] / cnt[keep]) * kc[keep]


# ----------------------------------------------------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------------------------------------------------

def timing(wass_t, fast_t):
    st = wass_t["stages"]
    sec = lambda k: (st.get(k) or {}).get("seconds", 0.0)
    cpu = lambda k: (st.get(k) or {}).get("cpu_seconds")
    n = wass_t["n_pairs"]
    nf = fast_t["n_pairs"]
    wass_calib = {"match": sec("match"), "autocalibrate": sec("autocalibrate"), "grid setup": sec("grid_setup")}
    wass_seq = {"prepare": sec("prepare"), "stereo": sec("stereo"), "grid": sec("grid_interpolate")}
    fast_seq = {"startup": fast_t["stages"]["startup"]["seconds"], "reconstruct": fast_t["stages"]["reconstruct"]["seconds"]}
    cpu_parts = [cpu(k) for k in ("prepare", "match", "autocalibrate", "stereo", "grid")]
    seq_cpu_parts = [cpu("prepare"), cpu("stereo"), cpu("grid_interpolate")]
    out = {
        "n_pairs_wass": n, "n_pairs_wassfast": nf, "fps": wass_t.get("fps"), "wass_workers": wass_t.get("workers"),
        "cpus": wass_t["machine"]["cpus"],
        "wass_calibration_stages_s": wass_calib, "wass_sequence_stages_s": wass_seq, "wassfast_stages_s": fast_seq,
        "wass_full_s": sum(wass_calib.values()) + sum(wass_seq.values()),
        "wass_calibrated_s": sum(wass_seq.values()),
        "wassfast_s": sum(fast_seq.values()),
        "wass_full_s_per_pair": (sum(wass_calib.values()) + sum(wass_seq.values())) / n,
        "wass_calibrated_s_per_pair": sum(wass_seq.values()) / n,
        "wassfast_s_per_pair": sum(fast_seq.values()) / nf,
        "wassfast_reconstruct_s_per_pair": fast_seq["reconstruct"] / nf,
        "wass_full_cpu_s_per_pair": sum(cpu_parts) / n if None not in cpu_parts else None,
        "wass_calibrated_cpu_s_per_pair": sum(seq_cpu_parts) / n if None not in seq_cpu_parts else None,
        "wassfast_cpu_s_per_pair": fast_t["processing_cpu_seconds"] / nf,
        # Fixed (per run) and marginal (per pair) costs, for projecting to longer records.
        "wass_fixed_s": sum(wass_calib.values()), "wass_per_pair_s": sum(wass_seq.values()) / n,
    }
    # WASSfast: marginal cost from the batches after the first (the first also starts the worker pool); the rest is fixed.
    marginal = fast_t.get("marginal_seconds_per_pair")
    if marginal:
        out["wassfast_per_pair_s"] = marginal
        out["wassfast_fixed_s"] = out["wassfast_s"] - marginal * nf
    else:
        out["wassfast_per_pair_s"] = fast_seq["reconstruct"] / nf
        out["wassfast_fixed_s"] = fast_seq["startup"]
    out["speedup_vs_wass_full"] = out["wass_full_s_per_pair"] / out["wassfast_s_per_pair"]
    out["speedup_vs_wass_calibrated"] = out["wass_calibrated_s_per_pair"] / out["wassfast_s_per_pair"]
    out["speedup_marginal"] = out["wass_per_pair_s"] / out["wassfast_per_pair_s"]
    if out["wass_full_cpu_s_per_pair"]:
        out["cpu_speedup_vs_wass_full"] = out["wass_full_cpu_s_per_pair"] / out["wassfast_cpu_s_per_pair"]
    if out["wass_calibrated_cpu_s_per_pair"]:
        out["cpu_speedup_vs_wass_calibrated"] = out["wass_calibrated_cpu_s_per_pair"] / out["wassfast_cpu_s_per_pair"]
    if out["fps"]:
        out["realtime_factor_wass_calibrated"] = out["wass_per_pair_s"] * out["fps"]
        out["realtime_factor_wassfast"] = out["wassfast_per_pair_s"] * out["fps"]
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------------------------

def style(plt):
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.edgecolor": GRID_C, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "axes.titlesize": 11, "axes.titlelocation": "left",
                         "legend.frameon": False})


def clean(ax, grid="x"):
    if grid:
        ax.grid(axis=grid, color=GRID_C, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_timing(T, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    fig = plt.figure(figsize=(13, 5.6), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1])
    fig.suptitle(f"{title}: WASS vs WASSfast processing time\n({T['n_pairs_wass']} stereo pairs, "
                 f"{T['cpus']} CPUs, WASS with {T['wass_workers']} parallel workers, CPU only)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")

    # Left: stacked stage bars.
    ax = fig.add_subplot(gs[0])
    shades_w = ["#0d366b", "#1c5cab", "#2a78d6", "#5598e6", "#86b6ee", "#b7d3f5"]
    shades_f = ["#a85200", "#e8862a"]
    bars = [
        ("WASS, from scratch", {**T["wass_calibration_stages_s"], **T["wass_sequence_stages_s"]}, shades_w),
        ("WASS, cameras\nalready calibrated", T["wass_sequence_stages_s"], shades_w[3:]),
        ("WASSfast", T["wassfast_stages_s"], shades_f),
    ]
    ymax = 0
    for row, (label, stages, shades) in enumerate(bars):
        y = len(bars) - 1 - row
        left, small = 0, []
        for (name, s), colour in zip(stages.items(), shades):
            if s <= 0:
                continue
            ax.barh(y, s, left=left, height=0.6, color=colour, edgecolor=SURFACE, linewidth=1)
            if s > 0.06 * max(T["wass_full_s"], 1):
                ax.text(left + s / 2, y, f"{name}\n{fmt(s)}", ha="center", va="center", fontsize=8.5,
                        color="white" if colour in shades_w[:3] + shades_f[:1] else INK)
            else:
                small.append(f"{name} {fmt(s)}")
            left += s
        if small:  # segments too thin to label inside the bar
            ax.text(0, y - 0.36, " + ".join(small), ha="left", va="top", fontsize=8, color=INK2)
        ymax = max(ymax, left)
        ax.text(left, y, f"  {fmt(left)}", va="center", ha="left", fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(bars)), [b[0] for b in bars][::-1])
    ax.set_xlim(0, ymax * 1.18)
    ax.set_xlabel("wall time (s)")
    ax.set_title("Wall time for the whole record, by stage")
    ax.tick_params(axis="y", length=0)
    clean(ax)
    ax.spines["left"].set_visible(False)

    # Right: seconds per pair, wall and CPU, log scale.
    ax = fig.add_subplot(gs[1])
    labels = ["WASS\nfrom scratch", "WASS\ncalibrated", "WASSfast"]
    wall = [T["wass_full_s_per_pair"], T["wass_calibrated_s_per_pair"], T["wassfast_s_per_pair"]]
    cpu = [T["wass_full_cpu_s_per_pair"], T["wass_calibrated_cpu_s_per_pair"], T["wassfast_cpu_s_per_pair"]]
    x = np.arange(3)
    colours = [WASS_C, WASS_C, FAST_C]
    ax.bar(x - 0.19, wall, 0.36, color=colours, label="wall time")
    if None not in cpu:
        ax.bar(x + 0.19, cpu, 0.36, color=colours, alpha=0.45, hatch="//", edgecolor=SURFACE, label="CPU time")
        for xi, v in zip(x, cpu):
            ax.text(xi + 0.19, v * 1.08, fmt(v), ha="center", va="bottom", fontsize=8.5, color=INK2)
    for xi, v in zip(x, wall):
        ax.text(xi - 0.19, v * 1.08, fmt(v), ha="center", va="bottom", fontsize=8.5)
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("seconds per stereo pair (log)")
    ax.set_ylim(min(wall + [c for c in cpu if c]) / 3, max(wall + [c for c in cpu if c]) * 4)
    ax.set_title(f"Per stereo pair: WASSfast is {T['speedup_vs_wass_calibrated']:.0f}x faster (wall)"
                 + (f", {T['cpu_speedup_vs_wass_calibrated']:.0f}x (CPU)" if T.get("cpu_speedup_vs_wass_calibrated") else "")
                 + "\nthan calibrated WASS", fontsize=10.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=INK2, label="wall time"), Patch(facecolor=INK2, alpha=0.45, hatch="//", label="CPU time (all cores)")],
              loc="upper right", fontsize=8.5)
    clean(ax, "y")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_scaling(T, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    fps = T["fps"]
    fig, ax = plt.subplots(figsize=(9, 5.6), layout="constrained")
    minutes = np.logspace(np.log10(1 / 60), np.log10(60), 200)  # record length, 1 s .. 1 h
    pairs = minutes * 60 * fps
    wass_full = T["wass_fixed_s"] + T["wass_per_pair_s"] * pairs
    wass_cal = T["wass_per_pair_s"] * pairs
    fast = T["wassfast_fixed_s"] + T["wassfast_per_pair_s"] * pairs
    ax.loglog(minutes, wass_full / 60, color=WASS_C, linewidth=2, label="WASS, from scratch (calibration each time)")
    ax.loglog(minutes, wass_cal / 60, color=WASS_C, linewidth=2, linestyle="--", label="WASS, cameras already calibrated")
    ax.loglog(minutes, fast / 60, color=FAST_C, linewidth=2.4, label="WASSfast")
    ax.loglog(minutes, minutes, color=INK2, linewidth=1, linestyle=":", label="real time (processing = record length)")
    rec_min = T["n_pairs_wass"] / fps / 60
    ax.plot([rec_min] * 2, [(T["wass_full_s"]) / 60, T["wassfast_s"] / 60], "o", color=INK, markersize=5, zorder=5)
    ax.annotate("measured", (rec_min, T["wassfast_s"] / 60), textcoords="offset points", xytext=(8, -12), fontsize=8.5,
                color=INK2)
    for m in (10, 30):
        if m <= 60:
            p = m * 60 * fps
            ax.axvline(m, color=GRID_C, linewidth=0.8)
            ax.text(m, ax.get_ylim()[1] if False else (T["wass_fixed_s"] + T["wass_per_pair_s"] * p) / 60 * 1.25,
                    f"{m}-min record:\nWASS {fmt(T['wass_per_pair_s'] * p)}\nWASSfast {fmt(T['wassfast_fixed_s'] + T['wassfast_per_pair_s'] * p)}",
                    fontsize=8, ha="right", va="bottom", color=INK2)
    ax.set_xlabel(f"record length (minutes at {fps:g} Hz)")
    ax.set_ylabel("processing time (minutes)")
    ax.set_title(f"{title}: projected processing time (linear in pairs, from the measured per-pair cost)\n"
                 f"Per extra pair: WASS {T['realtime_factor_wass_calibrated']:.0f}x slower than real time, "
                 f"WASSfast {T['realtime_factor_wassfast']:.1f}x (start-up costs excluded)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    clean(ax, "both")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def panel_hs(ax, t, rows):
    """Significant wave height estimate (4 std Z over the common area) per frame, WASS and WASSfast."""
    get = lambda k: np.array([r[k] if r else np.nan for r in rows])
    ax.plot(t, get("hs_wass_m"), color=WASS_C, linewidth=1.8, label="WASS")
    ax.plot(t, get("hs_wassfast_m"), color=FAST_C, linewidth=1.8, label="WASSfast")
    ax.set_ylabel("4·std(Z) (m)")
    ax.set_title("Significant wave height estimate per frame (4·std of elevation over the common area)")
    ax.legend(loc="upper right", ncol=2)
    clean(ax, "y")


def panel_point(ax, t, w, f, wi, fi, centre):
    """Elevation time series at one grid point, WASS and WASSfast."""
    i, j = centre
    zw = np.array([w.Z[a][i, j] for a in wi])
    zf = np.array([f.Z[b][i, j] for b in fi])
    ax.plot(t, zw, color=WASS_C, linewidth=1.8, marker="o", markersize=2.5, label="WASS")
    ax.plot(t, zf, color=FAST_C, linewidth=1.8, marker="o", markersize=2.5, label="WASSfast")
    ax.axhline(0, color=GRID_C, linewidth=0.8)
    ax.set_ylabel("Z (m)")
    ax.set_xlabel("time (s)")
    ax.set_title(f"Elevation time series at one grid point (x = {w.X[i, j]:.1f} m, y = {w.Y[i, j]:.1f} m)")
    ax.legend(loc="upper right", ncol=2)
    clean(ax, "y")


def panel_scatter(ax, w, f, wi, fi, frame_k, zlim, title="Same frame, point by point (line: equal)"):
    """WASSfast vs WASS elevation at every grid point of one frame (log-density hexbin)."""
    zw, zf = w.Z[wi[frame_k]], f.Z[fi[frame_k]]
    valid = np.isfinite(zw) & np.isfinite(zf)
    if valid.any():
        lim = zlim * 1.2
        ax.hexbin(zw[valid], zf[valid], gridsize=60, extent=(-lim, lim, -lim, lim), cmap="Greys", mincnt=1, bins="log")
        ax.plot([-lim, lim], [-lim, lim], color=FAST_C, linewidth=1.2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
    ax.set_xlabel("Z WASS (m)")
    ax.set_ylabel("Z WASSfast (m)")
    ax.set_title(title, fontsize=10.5)
    clean(ax, None)


def fig_timeseries(w, f, wi, fi, common, rows, fps, centre, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    t = common / fps
    ok = np.array([r is not None for r in rows])
    get = lambda k: np.array([r[k] if r else np.nan for r in rows])
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True, layout="constrained")
    fig.suptitle(f"{title}: WASSfast vs WASS (reference), frame by frame, on grid points where both have data",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    panel_hs(axs[0], t, rows)
    ax = axs[1]
    ax.plot(t, get("rmse_m"), color=INK, linewidth=1.6, label="RMSE")
    ax.plot(t, get("rmse_demeaned_m"), color=INK2, linewidth=1.2, linestyle="--", label="RMSE without mean offset")
    ax.set_ylabel("RMSE (m)")
    ax.set_ylim(0, np.nanmax(get("rmse_m")) * 1.3)
    ax2 = ax.twinx()
    ax2.plot(t, get("corr"), color="#3c9a5f", linewidth=1.4, label="correlation")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("correlation", color="#3c9a5f")
    ax2.tick_params(axis="y", colors="#3c9a5f")
    for side in ("top",):
        ax2.spines[side].set_visible(False)
    ax.set_title("Point-by-point difference per frame")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", ncol=3)
    clean(ax, "y")
    panel_point(axs[2], t, w, f, wi, fi, centre)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_summary(w, f, wi, fi, common, rows, fps, centre, zlim, frame_k, title, path):
    """One figure. Top row, one frame: WASS map, WASSfast map, difference map, point-by-point scatter.
    Below, full width with a shared time axis: Hs per frame and elevation at one grid point."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    t = common / fps
    fig = plt.figure(figsize=(19, 12), layout="constrained")
    fig.suptitle(f"{title}: WASSfast vs WASS (reference); maps in top view, cameras at y = 0 beyond the top edge",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.35])
    axs = top.subplots(1, 4)
    for ax, spec in zip(axs, frame_maps(w, f, wi, fi, common, zlim, frame_k)):
        panel_map(top, ax, w, *spec)
    panel_scatter(axs[3], w, f, wi, fi, frame_k, zlim,
                  title=f"Same frame (t = {t[frame_k]:.1f} s), every grid point\n(line: equal)")
    ax_hs, ax_pt = bottom.subplots(2, 1, sharex=True)
    panel_hs(ax_hs, t, rows)
    panel_point(ax_pt, t, w, f, wi, fi, centre)
    for ax in (ax_hs, ax_pt):  # mark the frame shown in the top row; legends where they cover least data
        ax.axvline(t[frame_k], color=INK2, linewidth=1, linestyle=":", zorder=0)
        ax.legend(loc="best", ncol=2)
    ax_hs.annotate(f"frame {common[frame_k]} (top row)", (t[frame_k], 1), xycoords=("data", "axes fraction"),
                   xytext=(4, -4), textcoords="offset points", va="top", fontsize=9, color=INK2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def oriented(Z, X, Y):
    """Array oriented for imshow with x to the right and y up, and its extent."""
    transpose = np.ptp(X, axis=0).max() > np.ptp(X, axis=1).max()
    if transpose:
        Z, X, Y = Z.T, X.T, Y.T
    if X[0, 0] > X[0, -1]:
        Z, X, Y = Z[:, ::-1], X[:, ::-1], Y[:, ::-1]
    if Y[0, 0] < Y[-1, 0]:
        Z, X, Y = Z[::-1], X[::-1], Y[::-1]
    return Z, (X.min(), X.max(), Y.min(), Y.max())


def frame_maps(w, f, wi, fi, common, zlim, frame_k):
    """Specs (Z, title, cmap, limits, unit) for the WASS, WASSfast and difference maps of one frame."""
    zw, zf = w.Z[wi[frame_k]], f.Z[fi[frame_k]]
    both = np.isfinite(zw) & np.isfinite(zf)
    diff = np.where(both, zf - zw, np.nan)
    dlim = max(0.05, float(np.ceil(np.nanpercentile(np.abs(diff), 98) * 20) / 20)) if both.any() else zlim
    return [
        (zw, f"WASS, frame {common[frame_k]}", "jet", (-zlim, zlim), "Z (m)"),
        (zf, f"WASSfast, frame {common[frame_k]}", "jet", (-zlim, zlim), "Z (m)"),
        (diff, "WASSfast − WASS (same frame)", "RdBu_r", (-dlim, dlim), "ΔZ (m)"),
    ]


def panel_map(fig, ax, w, Z, ttl, cmap, lim, unit):
    """Top-view map of one field on the shared grid, with colour bar."""
    import matplotlib.pyplot as plt
    Zo, ext = oriented(Z, w.X, w.Y)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#bdbcb8")
    im = ax.imshow(Zo, extent=ext, cmap=cm, vmin=lim[0], vmax=lim[1], interpolation="nearest")
    ax.set_title(ttl, fontsize=10.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.colorbar(im, ax=ax, shrink=0.85, label=unit)


def fig_maps(w, f, wi, fi, common, mean_absdiff, n_valid, zlim, frame_k, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    fig, axs = plt.subplots(2, 3, figsize=(14, 9.4), layout="constrained")
    fig.suptitle(f"{title}: surfaces on the shared grid (top view; cameras at y = 0, beyond the top edge)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    panels = frame_maps(w, f, wi, fi, common, zlim, frame_k) + [
        (n_valid / len(wi), "Frames with data in both", "Greys", (0, 1), "fraction"),
        (mean_absdiff, "Mean |WASSfast − WASS| over all frames", "magma_r",
         (0, float(np.nanpercentile(mean_absdiff, 98)) if np.isfinite(mean_absdiff).any() else 1), "m"),
    ]
    for ax, spec in zip(axs.flat, panels):
        panel_map(fig, ax, w, *spec)
    panel_scatter(axs[1, 2], w, f, wi, fi, frame_k, zlim)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_spectrum(w, f, wi, fi, rows, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    # Square where *both* have data in nearly every frame (WASS grids the whole area; WASSfast only its camera mask).
    valid_frac = np.mean([np.isfinite(w.Z[a]) & np.isfinite(f.Z[b]) & w.mask & f.mask for a, b in zip(wi, fi)], axis=0)
    box = best_box(valid_frac)
    dx = float(np.median(np.abs(np.diff(w.X, axis=1)))) or float(np.median(np.abs(np.diff(w.X, axis=0))))
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    fig.suptitle(f"{title}: which wave scales WASSfast reproduces", x=0.01, ha="left", fontsize=12, fontweight="bold")
    ax = axs[0]
    result = {}
    if box is not None:
        kw, Sw = k_spectrum(w.Z, wi, box, dx)
        kf, Sf = k_spectrum(f.Z, fi, box, dx)
        if kw is not None and kf is not None:
            side = (box[0].stop - box[0].start) * dx
            ax.loglog(kw, Sw, color=WASS_C, linewidth=2, label="WASS")
            ax.loglog(kf, Sf, color=FAST_C, linewidth=2, label="WASSfast")
            ratio = Sf / Sw
            # Smallest wavenumber above the spectral peak where WASSfast keeps less than half the energy.
            peak = int(np.argmax(Sw))
            below = np.where(ratio[peak:] < 0.5)[0]
            if below.size:
                kc = float(kw[peak + below[0]])
                ax.axvline(kc, color=INK2, linestyle="--", linewidth=1)
                ax.text(kc, ax.get_ylim()[0] * 3, f"  half the energy\n  at k = {kc:.2f} rad/m\n  (λ ≈ {2 * np.pi / kc:.1f} m)",
                        fontsize=8.5, color=INK2)
                result["k_half_energy_rad_m"] = kc
                result["wavelength_half_energy_m"] = 2 * np.pi / kc
            result["k_peak_wass_rad_m"] = float(kw[peak])
            result["wavelength_peak_wass_m"] = float(2 * np.pi / kw[peak])
            result["spectrum_box_m"] = side
            ax.set_title(f"Wavenumber spectrum (k·S(k), {side:.0f} m square with data from both in ≥98% of frames)",
                         fontsize=10)
            ax.set_xlabel("wavenumber k (rad/m)")
            ax.set_ylabel("k·S(k) (relative)")
            ax.legend()
            sx = ax.secondary_xaxis("top", functions=(lambda k: 2 * np.pi / np.maximum(k, 1e-9),
                                                       lambda l: 2 * np.pi / np.maximum(l, 1e-9)))
            sx.set_xlabel("wavelength (m)")
    else:
        ax.text(0.5, 0.5, "no square with continuous WASS coverage", ha="center", transform=ax.transAxes)
    clean(ax, "both")
    ax = axs[1]
    allw, allf = [], []
    for a, b in zip(wi, fi):
        v = np.isfinite(w.Z[a]) & np.isfinite(f.Z[b]) & w.mask & f.mask
        allw.append(w.Z[a][v][::7])
        allf.append(f.Z[b][v][::7])
    allw, allf = np.concatenate(allw), np.concatenate(allf)
    lim = float(np.percentile(np.abs(allw), 99.5)) * 1.1
    ax.hexbin(allw, allf, gridsize=70, extent=(-lim, lim, -lim, lim), cmap="Greys", mincnt=1, bins="log")
    ax.plot([-lim, lim], [-lim, lim], color=FAST_C, linewidth=1.2, label="equal")
    slope = float(np.polyfit(allw, allf, 1)[0])
    ax.plot([-lim, lim], [-lim * slope, lim * slope], color=INK2, linewidth=1, linestyle="--", label=f"fit, slope {slope:.2f}")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Z WASS (m)")
    ax.set_ylabel("Z WASSfast (m)")
    ax.set_title(f"All frames, point by point (r = {np.corrcoef(allw, allf)[0, 1]:.2f})", fontsize=10)
    ax.legend(loc="upper left")
    clean(ax, None)
    result["scatter_slope"] = slope
    result["pooled_corr"] = float(np.corrcoef(allw, allf)[0, 1])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return result


# ----------------------------------------------------------------------------------------------------------------------
# Videos
# ----------------------------------------------------------------------------------------------------------------------

def diverging_lut():
    import matplotlib
    cm = matplotlib.colormaps["RdBu_r"]
    rgb = (np.array([cm(x)[:3] for x in np.linspace(0, 1, N_LEVELS)]) * 255).astype(np.uint8)
    return rgb[:, ::-1].copy()  # BGR


class Projector:
    """Grid point (X, Y, Z) -> pixel in the undistorted cam0 image, with the WASS grid's camera matrices."""

    def __init__(self, w):
        if not {"Cam0toGrid", "intr0"} <= set(w.meta):
            sys.exit("ERROR: WASS gridded.nc has no meta/Cam0toGrid + intr0 (needed for the camera overlay)")
        self.G = np.linalg.inv(w.meta["Cam0toGrid"])
        self.K = w.meta["intr0"]
        self.X, self.Y = w.X, w.Y

    def __call__(self, Z):
        pts = np.stack([self.X, self.Y, np.nan_to_num(Z), np.ones_like(self.X)]).reshape(4, -1)
        cam = (self.G @ pts)[:3]
        px = self.K @ cam
        with np.errstate(divide="ignore", invalid="ignore"):
            iR, jR = (px[:2] / px[2]).reshape(2, *Z.shape)
        bad = (cam[2] <= 0).reshape(Z.shape) | ~np.isfinite(Z)
        iR[bad] = jR[bad] = np.nan
        return iR, jR


def label(img, text, org=(12, 34), scale=0.9, colour=(255, 255, 255)):
    for c, th in (((0, 0, 0), 4), (colour, 2)):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, c, th, cv2.LINE_AA)


def topview(Z, w, size, zlim, lut, title, unit="m"):
    Zo, ext = oriented(Z, w.X, w.Y)
    lv = np.clip((np.nan_to_num(Zo) + zlim) / (2 * zlim) * (N_LEVELS - 1), 0, N_LEVELS - 1).round().astype(int)
    img = lut[lv]
    img[~np.isfinite(Zo)] = 70
    img = cv2.resize(np.ascontiguousarray(img), (size, size), interpolation=cv2.INTER_NEAREST)
    label(img, title, scale=0.75)
    label(img, f"{ext[1] - ext[0]:.0f} x {ext[3] - ext[2]:.0f} m", (12, size - 14), 0.55)
    # Colour bar.
    bh, x0, y0 = size // 2, size - 26, size // 4
    bar = lut[::-1][np.linspace(0, N_LEVELS - 1, bh).astype(int)]
    img[y0:y0 + bh, x0:x0 + 12] = bar[:, None, :]
    for frac, txt in ((0, f"+{zlim:g}"), (1, f"-{zlim:g} {unit}")):
        y = y0 + int(frac * (bh - 1))
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label(img, txt, (x0 - tw - 6, y + 5), 0.45)
    return img


def overlay_tile(img_path, Z, proj, width, zlim, alpha, title):
    img = to_bgr8(cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED))
    scale = width / img.shape[1]
    tile = cv2.resize(img, (width, round(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    iR, jR = proj(Z)
    draw_overlay(tile, Z, iR, jR, scale, zlim, alpha)
    draw_colorbar(tile, zlim)
    label(tile, title, scale=0.8)
    return tile


class Writer:
    """H.264 mp4 through ffmpeg (plays in browsers, PowerPoint, Keynote), frames as BGR arrays."""

    def __init__(self, path, fps, size):
        w, h = size
        self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                   "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "medium",
                                   "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                                  stdin=subprocess.PIPE)
        self.size = size

    def write(self, frame):
        h, w = frame.shape[:2]
        if (w, h) != self.size:
            frame = cv2.copyMakeBorder(frame, 0, self.size[1] - h, 0, self.size[0] - w, cv2.BORDER_CONSTANT)
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def even(v):
    return v + (v % 2)


def make_videos(w, f, wi, fi, common, rows, wass_dir, out, zlim, fps, video_fps, title, fast_label):
    proj = Projector(w)
    dlut = diverging_lut()
    imgs = [wass_dir / "output" / f"{c:06d}_wd" / "undistorted" / "00000000.png" for c in common]
    missing = [p for p in imgs if not p.exists()]
    if missing:
        print(f"  WARNING: {len(missing)} undistorted images missing (e.g. {missing[0]}); skipping videos")
        return
    diffs = [f.Z[b] - w.Z[a] for a, b in zip(wi, fi)]
    dlim = max(0.05, float(np.ceil(np.nanpercentile(np.abs(np.stack(diffs)), 98) * 20) / 20))

    # Layouts. Single: [overlay 1000 px | top view, same height]. Comparison: two overlays on top, three top views below.
    H0 = cv2.imread(str(imgs[0]), cv2.IMREAD_UNCHANGED).shape
    ow = 1000
    oh = round(H0[0] * ow / H0[1])
    single_size = (even(ow + 6 + oh), even(oh + 56))
    cw = 960
    ch = round(H0[0] * cw / H0[1])
    tv = 640
    comp_size = (1920, even(56 + ch + 6 + tv))

    vw = {name: Writer(out / f"{name}.mp4", video_fps, single_size) for name in ("wass", "wassfast")}
    vc = Writer(out / "comparison.mp4", video_fps, comp_size)
    names = {"wass": "WASS", "wassfast": fast_label}
    for k, (a, b) in enumerate(zip(wi, fi)):
        t = common[k] / fps
        r = rows[k]
        Zs = {"wass": w.Z[a], "wassfast": f.Z[b]}
        for name, Z in Zs.items():
            ov = overlay_tile(imgs[k], Z, proj, ow, zlim, 0.55, f"{names[name]}: surface on cam0 (undistorted)")
            top = topview(Z, w, oh, zlim, LUT, f"{names[name]}: top view")
            canvas = np.zeros((single_size[1], single_size[0], 3), np.uint8)
            canvas[56:56 + oh, :ow] = ov
            canvas[56:56 + oh, ow + 6:ow + 6 + oh] = top
            hs = 4 * np.nanstd(Z[np.isfinite(Z) & w.mask])
            label(canvas, f"{title} | {names[name]} | frame {common[k]}  t = {t:6.2f} s | 4*std(Z) = {hs:.2f} m "
                          f"(whole grid)", (12, 36), 0.75)
            vw[name].write(canvas)

        canvas = np.zeros((comp_size[1], comp_size[0], 3), np.uint8)
        y = 56
        for col, name in enumerate(("wass", "wassfast")):
            ov = overlay_tile(imgs[k], Zs[name], proj, cw - 3, zlim, 0.55, names[name])
            canvas[y:y + ov.shape[0], col * (cw + 3):col * (cw + 3) + ov.shape[1]] = ov
        y += ch + 6
        tops = [topview(Zs["wass"], w, tv - 4, zlim, LUT, "WASS"),
                topview(Zs["wassfast"], w, tv - 4, zlim, LUT, names["wassfast"]),
                topview(diffs[k], w, tv - 4, dlim, dlut, f"{names['wassfast']} - WASS")]
        for col, tile in enumerate(tops):
            canvas[y:y + tile.shape[0], col * tv:col * tv + tile.shape[1]] = tile
        stats = (f"RMSE {r['rmse_m']:.2f} m  corr {r['corr']:.2f}  4*std: WASS {r['hs_wass_m']:.2f} / "
                 f"WASSfast {r['hs_wassfast_m']:.2f} m (common area)") if r else "no common data"
        label(canvas, f"{title} | frame {common[k]}  t = {t:6.2f} s | {stats}", (12, 36), 0.75)
        vc.write(canvas)
        if k % 25 == 0:
            print(f"  video frame {k + 1}/{len(wi)}", flush=True)
    for v in (*vw.values(), vc):
        v.close()


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wass", required=True, type=Path, help="run_wass.py output folder")
    p.add_argument("--wassfast", required=True, type=Path, help="run_wassfast.py output folder")
    p.add_argument("--also", action="append", default=[], metavar="LABEL=DIR",
                   help="other WASSfast runs of the same pairs to add to the accuracy table (repeatable)")
    p.add_argument("--out", required=True, type=Path, help="folder for figures, videos and the summary")
    p.add_argument("--title", default=None, help="dataset name for titles (default: name of the WASS folder's parent)")
    p.add_argument("--zlim", type=float, help="colour scale +/-ZLIM m (default: 3 x robust std of WASS Z)")
    p.add_argument("--video-fps", type=float, help="video frame rate (default: the acquisition rate, i.e. real time)")
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    wass_dir, fast_dir, out = args.wass.resolve(), args.wassfast.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    title = args.title or wass_dir.parent.name
    wass_t = json.loads((wass_dir / "timings.json").read_text())
    fast_t = json.loads((fast_dir / "timings.json").read_text())
    fps = fast_t["fps"]

    print("Loading surfaces...")
    w, f, common, wi, fi = load_pair(wass_dir, fast_dir)
    print(f"  {len(common)} frames in both (WASS {w.Z.shape[0]}, WASSfast {f.Z.shape[0]})")
    rows, mean_absdiff, n_valid = accuracy(w, f, wi, fi)
    acc = summarize(rows)
    allz = w.Z[wi][np.isfinite(w.Z[wi])]
    zlim = args.zlim or float(np.ceil(3 * 1.4826 * np.median(np.abs(allz - np.median(allz))) * 10) / 10)

    T = timing(wass_t, fast_t)
    print("Figures...")
    fig_timing(T, title, out / "timing_overview.png")
    if T.get("fps"):
        fig_scaling(T, title, out / "timing_scaling.png")
    # Grid point for the time series: the one compared in most frames, nearest the grid centre among those.
    score = n_valid - 1e-3 * np.hypot(*np.indices(n_valid.shape) - np.array(n_valid.shape)[:, None, None] / 2)
    centre = np.unravel_index(np.argmax(score), score.shape)
    fig_timeseries(w, f, wi, fi, common, rows, fps, centre, title, out / "accuracy_timeseries.png")
    best_k = int(np.argmax([r["coverage"] if r else -1 for r in rows]))
    mid_k = len(rows) // 2 if rows[len(rows) // 2] else best_k
    fig_maps(w, f, wi, fi, common, mean_absdiff, n_valid, zlim, mid_k, title, out / "accuracy_maps.png")
    fig_summary(w, f, wi, fi, common, rows, fps, centre, zlim, mid_k, title, out / "accuracy_summary.png")
    spec = fig_spectrum(w, f, wi, fi, rows, title, out / "accuracy_spectrum.png")
    acc.update(spec)

    variants = {}
    for item in args.also:
        lab, d = item.split("=", 1)
        try:
            w2, f2, c2, wi2, fi2 = load_pair(wass_dir, Path(d).resolve())
            r2, _, _ = accuracy(w2, f2, wi2, fi2)
            t2 = json.loads((Path(d) / "timings.json").read_text())
            variants[lab] = {**summarize(r2), "fps": t2.get("fps"), "direction": t2.get("direction"),
                             "s_per_pair": t2.get("processing_seconds_per_frame")}
        except SystemExit as e:
            print(f"  skipping {lab}: {e}")

    result = {"title": title, "wass_run": str(wass_dir), "wassfast_run": str(fast_dir), "zlim_m": zlim,
              "timing": T, "accuracy": acc, "variants": variants,
              "per_frame": [dict(frame=int(c), **(r or {})) for c, r in zip(common, rows)]}
    (out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    write_summary(result, fast_t, wass_t, out / "comparison_summary.md")
    print((out / "comparison_summary.md").read_text())

    if not args.no_video:
        print("Videos...")
        make_videos(w, f, wi, fi, common, rows, wass_dir, out, zlim, fps, args.video_fps or fps, title,
                    "WASSfast")
    print(f"Done: {out}")


def write_summary(R, fast_t, wass_t, path):
    T, A = R["timing"], R["accuracy"]
    L = [f"# WASS vs WASSfast: {R['title']}", "",
         f"{T['n_pairs_wass']} stereo pairs at {T['fps']:g} Hz ({T['n_pairs_wass'] / T['fps']:.0f} s of record), "
         f"{T['cpus']} CPUs, CPU only. WASS with {T['wass_workers']} parallel workers; WASSfast "
         f"{fast_t.get('wassfast_version')} (TensorFlow {fast_t.get('tensorflow')}), batch size {fast_t.get('batchsize')}, "
         f"wave direction {fast_t.get('direction')}.", "",
         "## Speed", "",
         "| | wall time | per pair (wall) | per pair (CPU) | vs real time |",
         "|---|---:|---:|---:|---:|",
         f"| WASS, from scratch | {fmt(T['wass_full_s'])} | {fmt(T['wass_full_s_per_pair'])} | "
         f"{fmt(T['wass_full_cpu_s_per_pair']) if T['wass_full_cpu_s_per_pair'] else '-'} | "
         f"{T['wass_full_s_per_pair'] * T['fps']:.0f}x slower |",
         f"| WASS, cameras already calibrated | {fmt(T['wass_calibrated_s'])} | {fmt(T['wass_calibrated_s_per_pair'])} | "
         f"{fmt(T['wass_calibrated_cpu_s_per_pair']) if T['wass_calibrated_cpu_s_per_pair'] else '-'} | "
         f"{T['wass_calibrated_s_per_pair'] * T['fps']:.0f}x slower |",
         f"| **WASSfast** | **{fmt(T['wassfast_s'])}** | **{fmt(T['wassfast_s_per_pair'])}** | "
         f"**{fmt(T['wassfast_cpu_s_per_pair'])}** | {T['wassfast_s_per_pair'] * T['fps']:.1f}x slower |", "",
         f"- WASSfast is **{T['speedup_vs_wass_calibrated']:.0f}x faster** than WASS on calibrated cameras "
         f"({T['speedup_vs_wass_full']:.0f}x vs WASS from scratch) in wall time"
         + (f", **{T['cpu_speedup_vs_wass_calibrated']:.0f}x** in CPU time." if T.get("cpu_speedup_vs_wass_calibrated") else "."),
         f"- Fixed costs: WASS calibration (match + autocalibrate + grid setup) {fmt(T['wass_fixed_s'])}, once per camera "
         f"installation; WASSfast start-up (TensorFlow + model) {fmt(T['wassfast_fixed_s'])} per run.",
         f"- Marginal cost per extra pair: WASS {fmt(T['wass_per_pair_s'])}, WASSfast {fmt(T['wassfast_per_pair_s'])} "
         f"({T['speedup_marginal']:.0f}x).", "",
         "## Accuracy (WASS as reference, grid points where both have data)", "",
         "| | value |", "|---|---:|",
         f"| frames compared | {A.get('frames_compared')} |",
         f"| RMSE, median over frames | {A.get('rmse_m_median', float('nan')):.3f} m |",
         f"| RMSE without mean offset | {A.get('rmse_demeaned_m_median', float('nan')):.3f} m |",
         f"| RMSE / Hs | {A.get('rmse_over_hs', float('nan')):.0%} |",
         f"| mean offset (WASSfast − WASS) | {A.get('bias_m_mean', float('nan')):+.3f} m |",
         f"| correlation, median over frames | {A.get('corr_median', float('nan')):.2f} |",
         f"| Hs = 4·std(Z): WASS / WASSfast | {A.get('hs_wass_m_mean', float('nan')):.2f} / {A.get('hs_wassfast_m_mean', float('nan')):.2f} m "
         f"(ratio {A.get('hs_ratio_mean', float('nan')):.2f}) |",
         f"| scatter slope (WASSfast vs WASS) | {A.get('scatter_slope', float('nan')):.2f} |",
         f"| common coverage of the grid | {A.get('coverage_mean', float('nan')):.0%} |"]
    if "wavelength_half_energy_m" in A:
        L.append(f"| shortest wavelength with ≥ half the WASS energy | {A['wavelength_half_energy_m']:.1f} m "
                 f"(WASS spectral peak at {A['wavelength_peak_wass_m']:.1f} m) |")
    if R["variants"]:
        L += ["", "## Other WASSfast settings on the same pairs", "",
              "| run | fps | direction | RMSE (median) | corr | Hs ratio | s/pair |", "|---|---:|---|---:|---:|---:|---:|",
              f"| main | {fast_t.get('fps')} | {fast_t.get('direction')} | {A.get('rmse_m_median', float('nan')):.3f} m | "
              f"{A.get('corr_median', float('nan')):.2f} | {A.get('hs_ratio_mean', float('nan')):.2f} | "
              f"{fast_t.get('processing_seconds_per_frame', float('nan')):.2f} |"]
        for lab, v in R["variants"].items():
            L.append(f"| {lab} | {v.get('fps')} | {v.get('direction')} | {v.get('rmse_m_median', float('nan')):.3f} m | "
                     f"{v.get('corr_median', float('nan')):.2f} | {v.get('hs_ratio_mean', float('nan')):.2f} | "
                     f"{(v.get('s_per_pair') or float('nan')):.2f} |")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
