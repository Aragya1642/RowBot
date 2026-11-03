#!/usr/bin/env python3
"""Download stereo wave datasets from the Ifremer WASS archive.

Source: https://data-dataref.ifremer.fr/stereo/

The local layout mirrors the server (<out> defaults to the data/ folder next to this script):
    <out>/<campaign>/<acquisition>/{config,input,nc}/...
    <out>/<campaign>/<campaign>_Meteo/...      (extras)
    <out>/<campaign>/<campaign>_WaveGauge/...  (extras)

Downloads are resumable and re-running a command skips files already on disk,
so the same command can be used to repeat or continue a download.

Examples:
    python3 download_data.py list
    python3 download_data.py download BS_2011 --acq 11-38 --frames 300 --nc short
    python3 download_data.py download all --frames 0 --nc none --dry-run
"""

import argparse
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "https://data-dataref.ifremer.fr/stereo/"
CAMPAIGNS = ["AA_2014", "AA_2015", "BS_2011", "BS_2013", "LJ_2018", "YS_2017"]
IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg")

# One row of an Apache directory index: link, last-modified, size.
ROW_RE = re.compile(
    r'<a href="([^"?/][^"]*)">.*?</a></td>'
    r'<td[^>]*>[^<]*</td>'
    r'<td[^>]*>\s*([^<]*?)\s*</td>'
)
SIZE_UNITS = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def parse_size(text):
    m = re.fullmatch(r"([\d.]+)([KMGT]?)", text.strip())
    return int(float(m.group(1)) * SIZE_UNITS[m.group(2)]) if m else 0


def human(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def list_dir(url):
    """Return [(name, url, size_bytes, is_dir)] for an Apache index page."""
    with urllib.request.urlopen(url, timeout=60) as resp:
        html = resp.read().decode("utf-8", "replace")
    entries = []
    for href, size in ROW_RE.findall(html):
        name = urllib.parse.unquote(href).removeprefix("./")
        is_dir = name.endswith("/")
        entries.append((name.rstrip("/"), urllib.parse.urljoin(url, href), parse_size(size), is_dir))
    return entries


def local_name(name):
    # Wave gauge files contain ':' which breaks on Windows filesystems (/mnt/c).
    return name.replace(":", "-")


def frame_index(name):
    m = re.match(r"(\d+)_", name)
    return int(m.group(1)) if m else None


def plan_acquisition(acq_url, dest, args):
    """Build a list of (url, path, size) jobs for one stereo acquisition."""
    jobs = []
    subdirs = {name: url for name, url, _, is_dir in list_dir(acq_url) if is_dir}

    if "config" in subdirs:
        for name, url, size, is_dir in list_dir(subdirs["config"]):
            if not is_dir:
                jobs.append((url, dest / "config" / local_name(name), size))

    if "input" in subdirs and args.frames != -1:
        # Images are either flat in input/ or split into input/cam1, input/cam2.
        images = []  # (rel_path, url, size)
        for name, url, size, is_dir in list_dir(subdirs["input"]):
            if is_dir:
                images += [(f"{name}/{n}", u, s) for n, u, s, d in list_dir(url) if not d]
            else:
                images.append((name, url, size))
        images = [i for i in images if i[0].lower().endswith(IMAGE_EXTS)]

        frames = sorted({frame_index(Path(p).name) for p, _, _ in images} - {None})
        frames = frames[args.start :: args.step]
        if args.frames > 0:
            frames = frames[: args.frames]
        wanted = set(frames)
        for rel, url, size in images:
            if frame_index(Path(rel).name) in wanted:
                jobs.append((url, dest / "input" / rel, size))

    if "nc" in subdirs and args.nc != "none":
        for name, url, size, is_dir in list_dir(subdirs["nc"]):
            short = "_short" in name
            if is_dir or (args.nc == "short" and not short) or (args.nc == "full" and short):
                continue
            jobs.append((url, dest / "nc" / local_name(name), size))

    return jobs


def plan(args):
    campaigns = CAMPAIGNS if "all" in args.campaigns else args.campaigns
    jobs = []
    for campaign in campaigns:
        for name, url, _, is_dir in list_dir(BASE_URL + campaign + "/"):
            if not is_dir:
                continue
            dest = args.out / campaign / name
            if name.endswith("Hz"):
                if args.acq and not any(a in name for a in args.acq):
                    continue
                print(f"  scanning {campaign}/{name}", file=sys.stderr)
                jobs += plan_acquisition(url, dest, args)
            elif args.extras:  # *_Meteo, *_WaveGauge
                jobs += [(u, dest / local_name(n), s) for n, u, s, d in list_dir(url) if not d]
    return jobs


def download(url, path):
    """Download url to path, resuming a partial .part file. Returns bytes fetched."""
    if path.exists():
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # .part already complete
            part.rename(path)
            return 0
        raise
    with resp, open(part, "ab" if resp.status == 206 else "wb") as f:
        fetched = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            fetched += len(chunk)
    part.rename(path)
    return fetched


def cmd_list(args):
    for campaign in CAMPAIGNS:
        print(campaign)
        for name, _, _, is_dir in list_dir(BASE_URL + campaign + "/"):
            if is_dir:
                print(f"    {name}")


def cmd_download(args):
    unknown = set(args.campaigns) - set(CAMPAIGNS) - {"all"}
    if unknown:
        sys.exit(f"Unknown campaign(s): {', '.join(sorted(unknown))}. Choose from {CAMPAIGNS} or 'all'.")

    jobs = plan(args)
    todo = [j for j in jobs if not j[1].exists()]
    total = sum(s for _, _, s in todo)
    print(f"{len(jobs)} files selected, {len(jobs) - len(todo)} already present, "
          f"{len(todo)} to download (~{human(total)}) -> {args.out}")
    if args.dry_run or not todo:
        return
    if not args.yes and total > 50 * 1024**3:
        if input(f"This will download ~{human(total)}. Continue? [y/N] ").lower() != "y":
            return

    done, fetched, failed = 0, 0, []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(download, url, path): path for url, path, _ in todo}
        for fut in as_completed(futures):
            done += 1
            try:
                fetched += fut.result()
            except Exception as e:
                failed.append((futures[fut], e))
            print(f"\r  [{done}/{len(todo)}] {human(fetched)} fetched", end="", file=sys.stderr)
    print(file=sys.stderr)
    for path, err in failed:
        print(f"FAILED {path}: {err}", file=sys.stderr)
    if failed:
        sys.exit(f"{len(failed)} file(s) failed; re-run the same command to retry.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list campaigns and acquisitions on the server").set_defaults(func=cmd_list)

    d = sub.add_parser("download", help="download one or more campaigns")
    d.set_defaults(func=cmd_download)
    d.add_argument("campaigns", nargs="+", metavar="CAMPAIGN", help=f"one or more of {CAMPAIGNS}, or 'all'")
    d.add_argument("--acq", action="append", metavar="SUBSTR",
                   help="only acquisitions whose name contains SUBSTR (repeatable), e.g. '11-38'")
    d.add_argument("--frames", type=int, default=100,
                   help="stereo pairs per acquisition: N>0 = first N, 0 = all, -1 = no images (default 100)")
    d.add_argument("--start", type=int, default=0, help="skip the first START pairs (default 0)")
    d.add_argument("--step", type=int, default=1, help="take every STEP-th pair (default 1)")
    d.add_argument("--nc", choices=["none", "short", "full", "all"], default="none",
                   help="reference WASS surface NetCDF files to fetch (default none)")
    d.add_argument("--no-extras", dest="extras", action="store_false",
                   help="skip the small Meteo / WaveGauge .mat files")
    d.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data",
                   help="output directory (default: data/ next to this script)")
    d.add_argument("--workers", type=int, default=4, help="parallel downloads (default 4)")
    d.add_argument("--dry-run", action="store_true", help="show what would be downloaded and exit")
    d.add_argument("-y", "--yes", action="store_true", help="don't prompt for downloads over 50 GB")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
