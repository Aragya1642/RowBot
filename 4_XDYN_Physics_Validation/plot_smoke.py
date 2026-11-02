"""One-page overview of the DTMB smoke tests.

Compares the first run (hull free to drift and turn) with the second run
(surge, sway and yaw held, bow into the waves), with the flat-sea run as the
grey baseline.

Usage (from ~/rowbot_smoke):
    python3 plot_smoke.py free=out_free held=out
    python3 plot_smoke.py held=out                 # a single run works too
Each folder must hold calm.csv and waves.csv. Writes smoke_overview.png.
Needs matplotlib, e.g.:
    nix-shell -p 'python3.withPackages (p: [p.matplotlib])' --run "python3 plot_smoke.py free=out_free held=out"
"""
import csv
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OMEGA = 0.66                       # rad/s, must match waves_regular.snippet
WAVE_T = 2 * math.pi / OMEGA       # 9.52 s
T_SKIP = 100.0                     # s, start-up transient
ZOOM = (100.0, 150.0)              # s, window shown for heave and pitch

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e8e7e3", "#fcfcfb"
FLAT = "#a3a29c"                                       # neutral baseline
STYLE = {"free": ("#eb6834", "hull free (drifts, turns)"),
         "held": ("#2a78d6", "hull held head-on")}     # validated categorical pair


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    names = [h.split("(")[0].strip() for h in rows[0]]
    cols = {n: [] for n in names}
    for r in rows[1:]:
        if len(r) == len(names):
            for n, v in zip(names, r):
                cols[n].append(float(v))
    return cols


def upcross_period(t, x):
    w = [(a, b) for a, b in zip(t, x) if a >= T_SKIP]
    m = sum(b for _, b in w) / len(w)
    ups = [w[k][0] for k in range(1, len(w)) if w[k - 1][1] - m < 0 <= w[k][1] - m]
    return (ups[-1] - ups[0]) / (len(ups) - 1) if len(ups) >= 3 else float("nan")


def style_axis(ax, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=6)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def label_end(ax, x, y, text, color):
    ax.annotate(text, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=INK2,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec=color, lw=1))


def main(argv):
    runs = []
    for arg in argv:
        key, folder = arg.split("=", 1)
        if key not in STYLE:
            sys.exit(f"Run label must be 'free' or 'held', got '{key}'")
        runs.append((key, load(f"{folder}/waves.csv"), load(f"{folder}/calm.csv")))
    if not runs:
        sys.exit(__doc__)

    deg = 180 / math.pi
    fig = plt.figure(figsize=(12, 10), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8])
    a_heave, a_pitch = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    a_roll, a_head = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    a_period = fig.add_subplot(gs[2, :])

    calm = runs[0][2]
    for ax, key, scale in [(a_heave, "z", 1), (a_pitch, "theta", deg),
                           (a_roll, "phi", deg), (a_head, "psi", deg)]:
        ax.plot(calm["t"], [v * scale for v in calm[key]], color=FLAT, lw=1.4, ls="--")
        for name, waves, _ in runs:
            color, _ = STYLE[name]
            ax.plot(waves["t"], [v * scale for v in waves[key]], color=color, lw=1.6)
        if key in ("z", "theta"):
            ax.set_xlim(*ZOOM)
        else:
            ax.axvspan(0, T_SKIP, color="#f0efec", zorder=0, lw=0)
            ax.set_xlim(0, calm["t"][-1])
        ax.set_xlabel("time (s)", color=INK2, fontsize=9)

    style_axis(a_heave, "heave z, NED (m, + down)", f"Heave follows the {WAVE_T:.1f} s wave (1 m amplitude), t = {ZOOM[0]:.0f}–{ZOOM[1]:.0f} s")
    style_axis(a_pitch, "pitch (deg)", f"Pitch responds at the same period, t = {ZOOM[0]:.0f}–{ZOOM[1]:.0f} s")
    has_free = any(n == "free" for n, _, _ in runs)
    style_axis(a_roll, "roll (deg)", "Free hull's roll grows once it turns side-on" if has_free
               else "Head seas give almost no roll")
    style_axis(a_head, "heading (deg)", "Unsteered hull swings toward beam-on" if has_free
               else "Heading held at 0°")
    for ax, half in [(a_roll, 2.0), (a_head, 5.0)]:     # keep a readable span when lines are flat
        lo, hi = ax.get_ylim()
        if hi - lo < 2 * half:
            mid = (lo + hi) / 2
            ax.set_ylim(mid - half, mid + half)
    for name, waves, _ in runs:
        color, _ = STYLE[name]
        peak = max(abs(v) for t, v in zip(waves["t"], waves["phi"]) if t >= waves["t"][-1] - 60) * deg
        label_end(a_roll, waves["t"], [v * deg for v in waves["phi"]], f"peak {peak:.1f}°", color)
        label_end(a_head, waves["t"], [v * deg for v in waves["psi"]], f"{waves['psi'][-1] * deg:.0f}°", color)

    # Periods: measured heave period vs the wave and the Doppler-shifted prediction.
    labels, measured, colors = ["wave itself"], [WAVE_T], [FLAT]
    for name, waves, _ in runs:
        color, text = STYLE[name]
        labels.append(text.split(" (")[0])
        measured.append(upcross_period(waves["t"], waves["z"]))
        colors.append(color)
    xs = range(len(labels))
    a_period.axhline(WAVE_T, color=FLAT, lw=1, ls="--", zorder=1)
    for x, m, c in zip(xs, measured, colors):
        a_period.plot(x, m, "o", ms=11, color=c, mec=SURFACE, mew=2, zorder=3)
        a_period.text(x, m + 0.07, f"{m:.2f} s", ha="center", va="bottom", fontsize=9, color=INK)
    a_period.set_xlim(-0.6, len(labels) - 0.4)
    a_period.set_xticks(list(xs), labels)
    a_period.set_ylim(WAVE_T - 0.5, max(measured) + 0.4)
    style_axis(a_period, "heave period (s)",
               "Bobbing period: the free hull feels a slightly longer period because it drifted with the waves"
               if has_free else "Bobbing period matches the wave period")

    handles = [plt.Line2D([], [], color=FLAT, lw=1.4, ls="--", label="flat sea (control)")]
    handles += [plt.Line2D([], [], color=STYLE[n][0], lw=1.6, label=STYLE[n][1]) for n, _, _ in runs]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.962), ncol=len(handles),
               frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("DTMB 5415 smoke test in xdyn: the waves move the hull", x=0.012, y=0.985,
                 ha="left", fontsize=14, color=INK, fontweight="bold")
    fig.text(0.012, 0.012, f"Regular head wave, Hs 2 m, T {WAVE_T:.2f} s · shaded = start-up transient "
             f"(excluded from stats) · dt 0.05 s", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94), h_pad=2.2, w_pad=2.5)
    fig.savefig("smoke_overview.png", dpi=150, facecolor=SURFACE)
    print("Wrote smoke_overview.png")


if __name__ == "__main__":
    main(sys.argv[1:])
