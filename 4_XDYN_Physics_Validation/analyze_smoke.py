"""Check the DTMB smoke test: does the hull respond to xdyn's waves?

Usage: python3 analyze_smoke.py calm.csv waves.csv
Standard library only. Draws smoke_test.png too if matplotlib is importable.
"""
import csv
import math
import sys

WAVE_PERIOD = 2 * math.pi / 0.66   # s, must match omega0 in waves_regular.snippet
WAVE_AMP = 1.0                     # m, Hs/2 for the regular wave
T_SKIP = 100.0                     # s, ignore start-up transient


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header = [h.split("(")[0].strip() for h in rows[0]]
    cols = {h: [] for h in header}
    for r in rows[1:]:
        if not r or len(r) < len(header):
            continue
        for h, v in zip(header, r):
            cols[h].append(float(v))
    return cols


def window(cols, key):
    return [v for t, v in zip(cols["t"], cols[key]) if t >= T_SKIP]


def finite(cols):
    return all(math.isfinite(v) for vals in cols.values() for v in vals)


def amp(x):
    return (max(x) - min(x)) / 2 if x else float("nan")


def period(x, dt):
    """Mean period from upward zero crossings of the mean-removed signal."""
    if len(x) < 3:
        return float("nan")
    m = sum(x) / len(x)
    ups = [i for i in range(1, len(x)) if x[i - 1] - m < 0 <= x[i] - m]
    if len(ups) < 3:
        return float("nan")
    return (ups[-1] - ups[0]) * dt / (len(ups) - 1)


def summarize(name, cols):
    dt = cols["t"][1] - cols["t"][0]
    z, th, ph = window(cols, "z"), window(cols, "theta"), window(cols, "phi")
    s = {
        "finite": finite(cols),
        "heave_amp_m": amp(z),
        "heave_T_s": period(z, dt),
        "pitch_amp_deg": math.degrees(amp(th)),
        "pitch_T_s": period(th, dt),
        "roll_amp_deg": math.degrees(amp(ph)),
        "mean_z_m": sum(z) / len(z) if z else float("nan"),
        "x_drift_m": cols["x"][-1] - cols["x"][0],
        "max_heading_deg": max(abs(math.degrees(v)) for v in cols["psi"]),
        "t_end": cols["t"][-1],
    }
    print(f"\n[{name}]  t = 0..{s['t_end']:.0f} s, stats over t >= {T_SKIP:.0f} s")
    print(f"  all values finite : {s['finite']}")
    print(f"  heave amplitude   : {s['heave_amp_m']:.3f} m   (period {s['heave_T_s']:.2f} s)")
    print(f"  pitch amplitude   : {s['pitch_amp_deg']:.3f} deg (period {s['pitch_T_s']:.2f} s)")
    print(f"  roll amplitude    : {s['roll_amp_deg']:.3f} deg")
    print(f"  mean heave (NED z): {s['mean_z_m']:.3f} m,  surge drift {s['x_drift_m']:.2f} m")
    print(f"  max |heading|     : {s['max_heading_deg']:.2f} deg")
    return s


def check(label, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return ok


def near(value, target, tol=0.05):
    return math.isfinite(value) and abs(value - target) <= tol * target


def plot(calm, waves):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available, skipping smoke_test.png)")
        return
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    for ax, key, scale, unit in [(axes[0], "z", 1.0, "heave z, NED (m)"),
                                 (axes[1], "theta", 180 / math.pi, "pitch (deg)")]:
        ax.plot(calm["t"], [v * scale for v in calm[key]], color="#9a9a9a", lw=1.2, label="flat sea")
        ax.plot(waves["t"], [v * scale for v in waves[key]], color="#2a6fdb", lw=1.4, label="regular head wave")
        ax.axvspan(0, T_SKIP, color="#f0efec", zorder=0)
        ax.set_ylabel(unit)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].set_title(f"DTMB 5415 in xdyn: hull responds to a {WAVE_PERIOD:.1f} s head wave", loc="left")
    axes[1].set_xlabel("time (s)   (shaded = start-up transient, excluded from stats)")
    fig.tight_layout()
    fig.savefig("smoke_test.png", dpi=150)
    print("\nWrote smoke_test.png")


def main():
    calm_path, waves_path = sys.argv[1], sys.argv[2]
    calm, waves = load(calm_path), load(waves_path)
    c, w = summarize("flat sea", calm), summarize("regular head wave", waves)

    print(f"\nChecks (expected wave period {WAVE_PERIOD:.2f} s):")
    results = [
        check("no NaN/inf in either run", c["finite"] and w["finite"]),
        check("flat sea: hull settles (heave < 0.05 m, pitch < 0.1 deg)",
              c["heave_amp_m"] < 0.05 and c["pitch_amp_deg"] < 0.1),
        check("waves: heave amplitude > 0.1 m and > 5x flat sea",
              w["heave_amp_m"] > 0.1 and w["heave_amp_m"] > 5 * max(c["heave_amp_m"], 1e-6)),
        check("waves: heave period within 5% of wave period", near(w["heave_T_s"], WAVE_PERIOD)),
        check("waves: pitch amplitude > 0.3 deg and period within 5%",
              w["pitch_amp_deg"] > 0.3 and near(w["pitch_T_s"], WAVE_PERIOD)),
        check("hull held head-on: |heading| < 1 deg and surge drift < 1 m",
              w["max_heading_deg"] < 1 and abs(w["x_drift_m"]) < 1),
        check("head seas give (almost) no roll: < 0.5 deg", w["roll_amp_deg"] < 0.5),
    ]
    print(f"\n  heave RAO (heave amp / wave amp): {w['heave_amp_m'] / WAVE_AMP:.2f}"
          f"   pitch RAO: {w['pitch_amp_deg'] / WAVE_AMP:.2f} deg/m")
    plot(calm, waves)
    passed = all(results)
    print("\nSMOKE TEST", "PASSED: xdyn waves are moving the hull." if passed
          else "FAILED: paste this output back for debugging.")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
