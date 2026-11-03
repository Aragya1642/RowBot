#!/usr/bin/env python3
"""Play a folder of stereo image pairs as a side-by-side video.

By default, pairs are found by filename: anything named <frame>_..._01.<ext> (left)
and <frame>_..._02.<ext> (right), anywhere under FOLDER. That covers both Ifremer
layouts (flat input/ and input/cam1 + input/cam2), so you can point it at an
acquisition's input/ folder or the acquisition folder itself.

For anything else (e.g. WASS output), give explicit glob patterns for each camera.
The two lists are sorted and paired in order:
    python3 view_stereo.py OUT --left '*_wd/undistorted/00000000.png' --right '*_wd/undistorted/00000001.png'

Reconstruction overlay (--nc): colours the right image by the surface elevation Z from an
Ifremer *full* NetCDF file, using its iR/jR variables (right-camera pixel of each grid point).
`_short` files lack iR/jR and cannot be overlaid. NetCDF time step t is matched to the t-th
image pair found in FOLDER (+ --nc-offset), so point FOLDER at a single acquisition.
The NetCDF can be a local path or an https:// URL (only the frames shown are fetched).

Controls: space = pause, left/right = step (while paused), up/down = faster/slower, q / Esc = quit.

Examples:
    python3 view_stereo.py data/BS_2013/2013-09-22_13-00-01_10Hz
    python3 view_stereo.py data/YS_2017/2017-05-13_05-00-00_10Hz/input --fps 5 --loop
    python3 view_stereo.py data/BS_2013/2013-09-22_13-00-01_10Hz --save bs2013.mp4
    python3 view_stereo.py data/BS_2013/2013-09-22_13-00-01_10Hz --topview \\
        --nc https://data-dataref.ifremer.fr/stereo/BS_2013/2013-09-22_13-00-01_10Hz/nc/Surfaces_20130922_130001.nc
"""

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
# <frame index>_<anything>_<camera>.<ext>, e.g. 000012_01.tif or 000012_0000999_02.png
PAIR_RE = re.compile(r"^(\d+)_(?:.*_)?(01|02)$")
GAP = 4
N_LEVELS = 64  # colour levels used for the overlay
LUT = cv2.applyColorMap(np.linspace(0, 255, N_LEVELS).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)[:, 0]


def find_pairs(folder, left_glob=None, right_glob=None):
    """Return [(label, left_path, right_path)] sorted by frame."""
    if left_glob or right_glob:
        if not (left_glob and right_glob):
            sys.exit("--left and --right must be given together.")
        lefts, rights = sorted(folder.glob(left_glob)), sorted(folder.glob(right_glob))
        if len(lefts) != len(rights):
            print(f"warning: {len(lefts)} left vs {len(rights)} right images; "
                  f"using the first {min(len(lefts), len(rights))}", file=sys.stderr)
        return [(left.parent.relative_to(folder).as_posix() or left.stem, left, right)
                for left, right in zip(lefts, rights)]

    # Key by (sequence folder, frame index) so several acquisitions under FOLDER don't collide.
    # Images in input/cam1 + input/cam2 belong to the same sequence (input/).
    frames = {}  # (sequence, frame index) -> {"01": path, "02": path}
    for path in folder.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTS and (m := PAIR_RE.match(path.stem)):
            seq = path.parent.parent if re.fullmatch(r"cam\d+", path.parent.name) else path.parent
            frames.setdefault((seq, int(m.group(1))), {})[m.group(2)] = path
    multi = len({seq for seq, _ in frames}) > 1
    pairs = [(f"{seq.relative_to(folder).as_posix()}  {idx:06d}" if multi else f"{idx:06d}", cams["01"], cams["02"])
             for (seq, idx), cams in sorted(frames.items()) if "01" in cams and "02" in cams]
    if not pairs:
        sys.exit(f"No *_01 / *_02 image pairs found under {folder}. Use --left/--right glob patterns.")
    return pairs


class NcSurface:
    """Per-frame access to an Ifremer WASS NetCDF file (Z, iR, jR)."""

    def __init__(self, src, offset=0, zlim=None):
        import netCDF4  # only needed for --nc

        if src.startswith(("http://", "https://")) and "#" not in src:
            src += "#mode=bytes"  # HTTP byte-range access: fetch only what is read
        self.ds = netCDF4.Dataset(src)
        missing = {"Z", "iR", "jR"} - set(self.ds.variables)
        if missing:
            sys.exit(f"{src} has no {', '.join(sorted(missing))}; overlay needs a full (not _short) file.")
        self.n = len(self.ds["time"])
        self.offset = offset
        self.extent = (float(np.ptp(self.ds["X"][:])), float(np.ptp(self.ds["Y"][:])))
        self.block, self._cache = 16, {}

        Z, iR, jR = self.frame_at(0)
        self.zlim = zlim or float(np.ceil(np.nanmax(np.abs(np.nanpercentile(Z, [1, 99]))) * 15) / 10)
        # Orient the top view like the camera sees it. Which grid axis runs along the image's
        # horizontal is taken from the data (in Ifremer files it is the first one, despite the names).
        rows, cols = np.mgrid[: Z.shape[0], : Z.shape[1]]
        ok = np.isfinite(iR)
        corr = lambda a: abs(np.corrcoef(iR[ok], a[ok])[0, 1])
        self.transpose = corr(rows) > corr(cols)
        if self.transpose:
            iR, jR = iR.T, jR.T
        self.flip_rows = np.nanmean(jR[0]) > np.nanmean(jR[-1])     # far (small jR) at the top
        self.flip_cols = np.nanmean(iR[:, 0]) > np.nanmean(iR[:, -1])  # image-left on the left

    def frame_at(self, t):
        # Read BLOCK time steps per request and cache them: remote byte-range reads are slow per call.
        b = t // self.block
        if b not in self._cache:
            sl = slice(b * self.block, min((b + 1) * self.block, self.n))
            get = lambda v: np.ma.filled(self.ds[v][sl].astype(np.float32), np.nan)
            self._cache = {b: (get("Z"), get("iR"), get("jR"))}  # keep only the current block
        return tuple(a[t - b * self.block] for a in self._cache[b])

    def frame(self, pos):
        """Surface for image pair number `pos`, or None if out of range."""
        t = pos + self.offset
        return (t, *self.frame_at(t)) if 0 <= t < self.n else None


def to_bgr8(img):
    """Convert any grayscale/colour, 8/16-bit image to 8-bit BGR."""
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def z_levels(z, zlim):
    return np.clip((z + zlim) / (2 * zlim) * (N_LEVELS - 1), 0, N_LEVELS - 1).round().astype(int)


def draw_overlay(tile, Z, iR, jR, scale, zlim, alpha):
    """Fill each grid cell's projection in `tile`, coloured by its mean Z."""
    valid = np.isfinite(Z) & np.isfinite(iR) & np.isfinite(jR)
    cells = valid[:-1, :-1] & valid[1:, :-1] & valid[1:, 1:] & valid[:-1, 1:]
    if not cells.any():
        return
    shift = 4  # sub-pixel precision for fillPoly
    px = np.stack([iR, jR], axis=-1) * scale * (1 << shift)
    quads = np.stack([px[:-1, :-1], px[:-1, 1:], px[1:, 1:], px[1:, :-1]], axis=2)[cells]
    quads = quads.round().astype(np.int32)
    zc = ((Z[:-1, :-1] + Z[:-1, 1:] + Z[1:, 1:] + Z[1:, :-1]) / 4)[cells]
    levels = z_levels(zc, zlim)

    layer, mask = tile.copy(), np.zeros(tile.shape[:2], np.uint8)
    for lvl in np.unique(levels):  # one fillPoly call per colour
        cv2.fillPoly(layer, quads[levels == lvl], LUT[lvl].tolist(), cv2.LINE_8, shift)
    cv2.fillPoly(mask, quads, 255, cv2.LINE_8, shift)
    blended = cv2.addWeighted(tile, 1 - alpha, layer, alpha, 0)
    tile[mask > 0] = blended[mask > 0]


def draw_colorbar(tile, zlim):
    h, w = tile.shape[:2]
    bh, x0, y0 = h // 2, w - 30, h // 4
    bar = LUT[::-1][np.linspace(0, N_LEVELS - 1, bh).astype(int)]
    tile[y0 : y0 + bh, x0 : x0 + 15] = bar[:, None, :]
    for frac, text in ((0, f"+{zlim:g} m"), (0.5, "0"), (1, f"-{zlim:g} m")):
        y = y0 + int(frac * (bh - 1))
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        for colour, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):  # outlined for contrast
            cv2.putText(tile, text, (x0 - tw - 4, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, thickness)


def top_view(Z, size, nc):
    img = LUT[z_levels(np.nan_to_num(Z), nc.zlim)]
    img[~np.isfinite(Z)] = 64
    if nc.transpose:
        img = img.transpose(1, 0, 2)
    if nc.flip_rows:
        img = img[::-1]
    if nc.flip_cols:
        img = img[:, ::-1]
    img = cv2.resize(np.ascontiguousarray(img), (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.putText(img, f"Z top view {nc.extent[0]:.0f}x{nc.extent[1]:.0f} m", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def render(item, args, nc=None, info=""):
    """Load one pair and return the composed BGR frame (about args.width px wide)."""
    pos, (label, left_path, right_path) = item
    imgs = []
    for path in (left_path, right_path):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"could not read {path}")
        imgs.append(to_bgr8(img))

    surface = nc.frame(pos) if nc else None
    panels = 3 if args.topview and nc else 2
    tile_w = (args.width - GAP * (panels - 1)) // panels
    tiles = []
    for k, (img, name) in enumerate(zip(imgs, ("left (01)", "right (02)"))):
        scale = tile_w / img.shape[1]
        tile = cv2.resize(img, (tile_w, round(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        if k == 1 and surface:
            draw_overlay(tile, *surface[1:], scale, nc.zlim, args.alpha)
            draw_colorbar(tile, nc.zlim)
        cv2.putText(tile, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(tile)
    h = max(t.shape[0] for t in tiles)
    if panels == 3:
        tiles.append(top_view(surface[1], h, nc) if surface else np.full((h, h, 3), 64, np.uint8))

    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    spacer = np.zeros((h, GAP, 3), np.uint8)
    frame = np.hstack([x for t in tiles for x in (t, spacer)][:-1])
    if nc:
        info += f"  nc t={surface[0]}" if surface else "  (no nc frame)"
    cv2.putText(frame, f"{label}  {info}", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


def save_video(n, render_fn, args):
    """Write frames render_fn(i, info) for i in range(n) to args.save at args.fps."""
    first = render_fn(0, "")
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    if not writer.isOpened():
        sys.exit(f"Could not open video writer for {args.save}")
    for i in range(n):
        writer.write(render_fn(i, f"[{i + 1}/{n}]"))
        print(f"\r  writing {i + 1}/{n}", end="", file=sys.stderr)
    writer.release()
    print(f"\nSaved {args.save} ({w}x{h}, {args.fps} fps)", file=sys.stderr)


def play(n, render_fn, args, caption_fn=lambda i: ""):
    """Interactive playback of frames render_fn(i, info) in a pygame window."""
    import pygame  # only needed for live playback

    pygame.init()
    first = render_fn(0, "")
    screen = pygame.display.set_mode((first.shape[1], first.shape[0]))
    clock = pygame.time.Clock()
    fps, i, paused, redraw = args.fps, 0, False, True

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    i, paused, redraw = min(i + 1, n - 1), True, True
                elif event.key == pygame.K_LEFT:
                    i, paused, redraw = max(i - 1, 0), True, True
                elif event.key == pygame.K_UP:
                    fps = min(fps * 2, 120)
                elif event.key == pygame.K_DOWN:
                    fps = max(fps / 2, 0.5)
                redraw = True

        if redraw or not paused:
            state = "PAUSED" if paused else f"{fps:g} fps"
            frame = render_fn(i, f"[{i + 1}/{n}]  {state}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            screen.blit(pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], "RGB"), (0, 0))
            pygame.display.flip()
            pygame.display.set_caption(caption_fn(i))
            redraw = False

        if not paused:
            if i + 1 < n:
                i += 1
            elif args.loop:
                i = 0
            else:
                paused = True  # hold on the last frame
        clock.tick(fps if not paused else 30)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder containing the stereo images (searched recursively)")
    parser.add_argument("--left", metavar="GLOB", help="glob for left-camera images, relative to FOLDER")
    parser.add_argument("--right", metavar="GLOB", help="glob for right-camera images, relative to FOLDER")
    parser.add_argument("--fps", type=float, default=10, help="playback / output frame rate (default 10)")
    parser.add_argument("--width", type=int, default=1600, help="width of the combined frame in px (default 1600)")
    parser.add_argument("--start", type=int, default=0, help="skip the first START pairs")
    parser.add_argument("--count", type=int, default=0, help="show at most COUNT pairs (0 = all)")
    parser.add_argument("--step", type=int, default=1, help="show every STEP-th pair")
    parser.add_argument("--loop", action="store_true", help="loop playback")
    parser.add_argument("--save", type=Path, metavar="FILE.mp4", help="write an mp4 instead of opening a window")
    ov = parser.add_argument_group("reconstruction overlay")
    ov.add_argument("--nc", metavar="FILE|URL", help="Ifremer full .nc file to overlay on the right image")
    ov.add_argument("--nc-offset", type=int, default=0,
                    help="NetCDF time step = image pair number + OFFSET (default 0)")
    ov.add_argument("--zlim", type=float, help="colour scale limit in m, +/-ZLIM (default: from first frame)")
    ov.add_argument("--alpha", type=float, default=0.5, help="overlay opacity 0-1 (default 0.5)")
    ov.add_argument("--topview", action="store_true", help="add a third panel with Z seen from above")
    args = parser.parse_args()

    if not args.folder.is_dir():
        sys.exit(f"{args.folder} is not a directory")
    pairs = find_pairs(args.folder, args.left, args.right)
    # Keep each pair's position in the full sequence; it is what maps to the NetCDF time step.
    items = list(enumerate(pairs))[args.start :: args.step]
    if args.count > 0:
        items = items[: args.count]
    if not items:
        sys.exit("No pairs left after --start/--step/--count.")
    print(f"{len(items)} stereo pairs from {args.folder}", file=sys.stderr)

    nc = None
    if args.nc:
        nc = NcSurface(args.nc, args.nc_offset, args.zlim)
        print(f"NetCDF: {nc.n} time steps, colour scale +/-{nc.zlim:g} m; "
              f"pair 0 ({pairs[0][0]}) -> t={args.nc_offset}", file=sys.stderr)

    render_fn = lambda i, info: render(items[i], args, nc, info)
    if args.save:
        save_video(len(items), render_fn, args)
    else:
        play(len(items), render_fn, args, lambda i: f"{items[i][1][1].name} | {items[i][1][2].name}")


if __name__ == "__main__":
    main()
