# WASS Validation

Tools for validating [WASS](https://sites.google.com/unive.it/wass) (Waves Acquisition Stereo System) against public stereo wave datasets:

| Dataset | Platform | Link | How to get it |
|---|---|---|---|
| Ifremer stereo wave dataset (Guimarães et al. 2020) | Fixed: platforms, towers, lighthouse | https://data-dataref.ifremer.fr/stereo/ | `download_data.py` |
| S.A. Agulhas II WASS, July 2017 (Alberello et al. 2021) | **Moving ship** (icebreaker, Antarctic) | https://data.aad.gov.au/eds/5241/download | Manual download (email form), see [Shipborne data](#shipborne-data-sa-agulhas-ii) |

Other Ifremer data is at https://data-dataref.ifremer.fr/, but it may be less relevant.

| File | Purpose |
|---|---|
| [`download_data.py`](download_data.py) | Download Ifremer campaigns, frames and reference reconstructions into `data/` |
| [`view_stereo.py`](view_stereo.py) | Play stereo pairs side by side, optionally with Ifremer's reconstructed surface overlaid |
| [`view_nc.py`](view_nc.py) | Animate a gridded surface NetCDF from above (for data without images, e.g. Agulhas II) |
| `data/` | Downloaded datasets only. **Not in git** (see `.gitignore`); recreate as described below |

```
5_WASS_Validation/
├── README.md
├── download_data.py
├── view_stereo.py
├── view_nc.py
└── data/                         ← git-ignored
    ├── <campaign>/               ← Ifremer, from download_data.py
    │   ├── <acquisition>/{config,input,nc}/
    │   ├── <campaign>_Meteo/
    │   └── <campaign>_WaveGauge/
    └── SA_Agulhas_II_Winter_2017_Cruise__WASS/   ← manual download
        └── wass__20170704_hhmm.nc  (x6)
```

## Setup

| Script | Needs |
|---|---|
| `download_data.py` | Python 3.9+ standard library only |
| `view_stereo.py` | `opencv-python`, `numpy`; plus `pygame-ce` for the live window and `netCDF4` for `--nc` (imported only when used) |

**NixOS (this machine).** `python3` is not on the PATH, and the nixpkgs OpenCV is built without GUI support
(`cv2.imshow` fails), which is why the viewer's window uses pygame. Run everything through nix:

```bash
cd 5_WASS_Validation
alias py='nix shell nixpkgs#python3 -c python3'                                                       # downloader
alias pyv='nix-shell -p "python3.withPackages(ps: [ps.opencv4 ps.numpy ps.pygame-ce ps.netcdf4])" --run'  # viewer
py download_data.py --help
pyv "python3 view_stereo.py --help"
```

The first run downloads the packages (~100 MB). Later runs start in a few seconds.

**Elsewhere (venv / pip):**

```bash
pip install opencv-python numpy pygame-ce netCDF4
```

The examples below use plain `python3 ...`. On NixOS, use `py download_data.py ...` and `pyv "python3 view_stereo.py ..."`.

## Quick start

```bash
cd 5_WASS_Validation

# 1. Get some data: 300 pairs from one BS_2013 acquisition, ~3 GB
python3 download_data.py download BS_2013 --acq 13-00 --frames 300 --dry-run   # check the size first
python3 download_data.py download BS_2013 --acq 13-00 --frames 300

# 2. Watch it
python3 view_stereo.py data/BS_2013/2013-09-22_13-00-01_10Hz

# 3. Watch it with Ifremer's reconstruction overlaid, saved to a video
python3 view_stereo.py data/BS_2013/2013-09-22_13-00-01_10Hz --topview --save bs2013_overlay.mp4 \
    --nc https://data-dataref.ifremer.fr/stereo/BS_2013/2013-09-22_13-00-01_10Hz/nc/Surfaces_20130922_130001.nc
```

---

# download_data.py

Downloads any subset of the Ifremer archive into `data/`, mirroring the server's folder structure.

```bash
python3 download_data.py list                                     # campaigns and acquisitions on the server
python3 download_data.py download BS_2011 --acq 11-38 --dry-run   # what would be downloaded, and how big
python3 download_data.py download BS_2011 --acq 11-38 --frames 300 --nc short
```

Re-running the same command is safe. Files already on disk are skipped and interrupted downloads resume,
so a command doubles as a record of exactly which data was used. If any files fail, re-run the command to retry them.

## Options (`download`)

| Option | Default | Meaning |
|---|---|---|
| `CAMPAIGN ...` | (required) | One or more of `AA_2014 AA_2015 BS_2011 BS_2013 LJ_2018 YS_2017`, or `all` |
| `--acq SUBSTR` | all | Only acquisitions whose folder name contains `SUBSTR` (repeatable, case-sensitive), e.g. `--acq 11-38` |
| `--frames N` | `100` | Stereo pairs per acquisition. `0` = all, `-1` = no images (config/nc only) |
| `--start N` / `--step N` | `0` / `1` | Skip the first N pairs / take every Nth pair (e.g. `--step 12` ≈ 1 pair/s at 12 Hz) |
| `--nc` | `none` | Reference WASS surfaces: `none`, `short`, `full`, or `all` |
| `--no-extras` | off | Skip the small Meteo / WaveGauge `.mat` files (fetched by default) |
| `--out DIR` | `data/` | Where to write data |
| `--workers N` | `4` | Parallel downloads |
| `--dry-run` | off | Print file count and total size, download nothing |
| `-y` | off | Skip the confirmation prompt for downloads over 50 GB |

`config/` (camera calibration) is always downloaded for each selected acquisition, because WASS needs it.
`--acq` only filters recording folders. A typo matches nothing and gives "0 files selected" rather than an error.

## Local layout

```
data/<campaign>/<acquisition>/config/   camera calibration + WASS configs
data/<campaign>/<acquisition>/input/    stereo image pairs (NNNNNN_01 = left, NNNNNN_02 = right)
data/<campaign>/<acquisition>/nc/       WASS-reconstructed surfaces (NetCDF), for comparison
data/<campaign>/<campaign>_Meteo/       weather data (.mat)
data/<campaign>/<campaign>_WaveGauge/   in-situ wave gauge records (.mat), BS campaigns only
```

Wave gauge filenames on the server contain `:`, which is replaced with `-` locally so they work on Windows drives (`/mnt/c`).

## What each campaign contains

Every acquisition has the same three folders:
- **`config/`** (a few KB): intrinsics and distortion for both cameras, plus `stereo_config.txt` and `matcher_config.txt` for WASS.
  BS and LJ also include the extrinsics (`ext_R.xml`, `ext_T.xml`). AA and YS do not, so WASS must estimate the extrinsics itself.
- **`input/`**: raw synchronized stereo pairs. This is the bulk of the data.
- **`nc/`**: the 3D sea-surface reconstruction already produced by WASS, which you can use as ground truth for your own run.
  Both the full and `_short` files cover **all time steps on the same grid** (e.g. BS_2013: 191×191 points at 0.1 m).
  `_short` contains only `Z` (elevation, m), `X`, `Y`, `mask_Z` and `time`. Full additionally has `count` (frame number)
  and `iR`/`jR` (right-camera pixel coordinates for each grid point, per time step), which is why it is ~3× larger.
  Use `short` for elevation comparisons and wave statistics. Use `full` if you need to map the surface back to image pixels,
  including for the `view_stereo.py --nc` overlay.
  To inspect a header without downloading the file: `ncdump -h "<url>.nc#mode=bytes"` (needs the `netcdf` package).

Sizes below are approximate. "Images" means the full `input/` folder for both cameras.

### AA_2014: 1 acquisition, ~200 GB images

| Acquisition | Rate | Pairs | Duration | Images | Format | nc |
|---|---|---|---|---|---|---|
| `2014-03-27_09-10-00_12Hz` | 12 Hz | 43,161 | ~60 min | ~205 GB | PNG, ~2.4 MB, `input/cam1`, `input/cam2` | full 5.6 GB (no short) |

Extras: `AA_2014_Meteo` (4 KB). There's no wave gauge and no extrinsics in config.
This is the longest continuous record in the archive.

### AA_2015: 2 acquisitions, ~170 GB images

| Acquisition | Rate | Pairs | Duration | Images | Format | nc |
|---|---|---|---|---|---|---|
| `2015-03-05_10-35-00_12Hz` | 12 Hz | 21,573 | ~30 min | ~85 GB | PNG, ~2 MB, `cam1`/`cam2` | full 3.9 GB |
| `2015-05-15_09-00-00_12Hz` | 12 Hz | 21,552 | ~30 min | ~85 GB | PNG, ~2 MB, `cam1`/`cam2` | full 1.7 GB |

Extras: `AA_2015_Meteo` (two small `.mat` files). There's no wave gauge and no extrinsics.

### BS_2011: 4 acquisitions, ~650 GB images (has wave gauge)

| Acquisition | Rate | Pairs | Duration | Images | nc (full / short) |
|---|---|---|---|---|---|
| `2011-10-01_16-18-00_15Hz` | 15 Hz | 10,829 | ~12 min | ~105 GB | 4.3 GB / 1.4 GB |
| `2011-10-04_11-38-00_12Hz` | 12 Hz | 14,376 | ~20 min | ~140 GB | 6.1 GB / 2.0 GB |
| `2011-10-04_13-07-00_12Hz` | 12 Hz | 21,575 | ~30 min | ~205 GB | 8.9 GB / 3.0 GB |
| `2011-10-04_15-30-00_12Hz` | 12 Hz | 21,576 | ~30 min | ~205 GB | 13 GB / 4.5 GB |

The images are flat in `input/` as TIF, ~4.8 MB each, numbered from `000000`. Config includes the extrinsics.
Extras: `BS_2011_Meteo` (41 KB), and one `WaveGauge_BS0[1-4]_*.mat` (~0.8 MB) per acquisition.

### BS_2013: 3 acquisitions, ~590 GB images (has wave gauge)

| Acquisition | Rate | Pairs | Duration | Images | nc (full / short) |
|---|---|---|---|---|---|
| `2013-09-22_13-00-01_10Hz` | 10 Hz | ~17,980 | ~30 min | ~170 GB | 3.7 GB / 1.2 GB |
| `2013-09-25_12-15-01_12Hz` | 12 Hz | 21,578 | ~30 min | ~205 GB | 4.4 GB / 1.5 GB |
| `2013-09-30_10-20-01_12Hz` | 12 Hz | 21,578 | ~30 min | ~205 GB | 3.5 GB / 1.2 GB |

The images are flat in `input/` as TIF, ~4.8 MB each, numbered from `000001`. Config includes the extrinsics.
Extras: `BS_2013_Meteo` (63 KB), and one `WaveGauge_BS0[5-7]_*.mat` (~1.5 MB) per acquisition.

### LJ_2018: 1 acquisition, ~345 GB images (higher resolution)

| Acquisition | Rate | Pairs | Duration | Images | nc |
|---|---|---|---|---|---|
| `2018-01-03_09-39-38_10Hz` | 10 Hz | 18,000 | 30 min | ~345 GB | Grid1 16 GB / 5.4 GB short, Grid2 30 GB / 10 GB short |

The images are flat in `input/` as TIF, **~9.6 MB each** (twice the resolution of BS). Config includes the extrinsics.
There are two reconstruction grids in `nc/`, and `--nc short` fetches both short files (~15 GB). There are no extras.

### YS_2017: 1 acquisition, ~58 GB images (smallest)

| Acquisition | Rate | Pairs | Duration | Images | Format | nc |
|---|---|---|---|---|---|---|
| `2017-05-13_05-00-00_10Hz` | 10 Hz | 6,000 | 10 min | ~58 GB | TIF, ~4.8 MB, `cam1`/`cam2`, from `000001` | full 458 MB (no short) |

There are no extras and no extrinsics.

## Choosing a dataset

- **Test that the WASS pipeline runs:** a few hundred frames from any acquisition with `--nc none`. For example, `--frames 300` on BS is ~3 GB.
- **Compare against Ifremer's WASS output:** add `--nc short` (or `full` for AA/YS, which have no short file).
  The smallest reference is YS_2017 at 458 MB. The BS_2013 short files are ~1.2 GB.
- **Overlay the reconstruction in `view_stereo.py`:** needs `--nc full` (or pass the URL directly; see [Local file vs URL](#local-file-vs-url)).
- **Validate against independent in-situ measurements:** BS_2011 or BS_2013 are the only campaigns with wave gauge data.
- **Get a long record for spectra/statistics:** use `--step` to subsample, e.g. `--frames 0 --step 12` on a 12 Hz record gives 1 pair/s (~1/12 the size).
- **Downloading everything:** `download all --frames 0 --nc all` is about **2.1 TB**. You almost certainly don't want this.

The numbers above were surveyed from the server on 2026-11-02. Run `--dry-run` for exact current sizes.

---

# Shipborne data: S.A. Agulhas II

The only public WASS dataset recorded **from a moving ship** that we found. Between 07:00 and 08:00 UTC on 4 July 2017, the icebreaker
S.A. Agulhas II entered the Antarctic marginal ice zone (62°S 30°E) during an explosive polar cyclone.
The cameras looked out from the port side.

- **Download:** https://data.aad.gov.au/eds/5241/download (asks for an email address, then gives a zip). 1.48 GB.
- **Metadata:** https://data.aad.gov.au/metadata/AAS_4434_AGULHAS_WASS_2017, DOI [10.26179/q9bd-5f74](https://doi.org/10.26179/q9bd-5f74)
- **Licence:** CC BY 4.0. Cite: Alberello, A., Bennetts, L., Onorato, M., et al. (2021), *Wave acquisition stereo-camera system
  measurements (WASS) from a voyage of the S.A. Agulhas II, July 2017*, Australian Antarctic Data Centre.
- **Paper:** Alberello et al. (2022), *Three-dimensional imaging of waves and floes in the marginal ice zone during a cyclone*,
  Nature Communications, [doi:10.1038/s41467-022-32036-2](https://doi.org/10.1038/s41467-022-32036-2)

**Install:** unzip it so the files sit in `data/SA_Agulhas_II_Winter_2017_Cruise__WASS/`.

**Contents:** six `wass__20170704_hhmm.nc` files (hhmm = UTC start time) plus `README`, `LICENSE` and metadata XML.
There are **no images and no camera calibration**, only the reconstructed surface. So `view_stereo.py` and wassncplot can't be used;
view it with [`view_nc.py`](#view_ncpy).

| File | Duration @ 2 Hz | Rough wave height (4·std Z, 60 sampled frames) | Valid grid points |
|---|---|---|---|
| `wass__20170704_0805.nc` | 29.8 min | ~8 m (largest) | 78% |
| `wass__20170704_0845.nc` | 30.2 min | ~6 m | 74% |
| `wass__20170704_0925.nc` | 24.4 min | ~6 m | 77% |
| `wass__20170704_1000.nc` | 57.6 min | ~5 m | 73% |
| `wass__20170704_1105.nc` | 40.6 min | ~5 m | 74% |
| `wass__20170704_1150.nc` | 35.0 min | ~5 m | 64% |

- **Grid:** 151 × 201 points at 1 m. x = −100…100 m across the view (roughly parallel to the ship; which end is the bow is not documented),
  y = 50…200 m out from the cameras. Only a wedge (the cameras' field of view) has data; the rest is NaN.
- **Cameras** (Alberello et al. 2022, Methods): on the monkey bridge ~34 m above the waterline, looking out over the **port side**
  (~90° field of view), tilted 20° below the horizon, 4 m baseline, with an IMU alongside. The ship was moving south at near-constant
  heading and speed during each record. The paper does not say whether the view was angled forward or aft.
- **Format:** WASS / wassgridsurface style. `Z(count, X, Y)` is a 16-bit integer in **mm**, with `0` = missing. `X_grid`/`Y_grid` are in mm.
  There's also `fps`, `scale` (stereo baseline 3.73 m) and `mask_Z`. `time` is a dummy; use `count / fps`.
- The wave heights above are my rough estimates, not the paper's values. The 08:05 record also contains some spikes (|Z| up to ~15 m),
  which is why `view_nc.py` sets its colour scale robustly.

Other shipborne data we looked at, but which has no raw images:
- **R/V Thomas G. Thompson**, North Pacific 2014–15 (Schwendeman & Thomson 2017). Gridded elevation + radiance `.mat` files plus ship motion data.
  Not processed with WASS. Raw frames on request. https://digital.lib.washington.edu/researchworks/items/970ce35d-41c7-4e7d-b5eb-6bc6a5cf053a
- For raw shipborne images to run WASS yourself, contact the dataset authors (Alberello for Agulhas II, or Benetazzo / Bergamasco via the WASS site).

---

# view_nc.py

Animates any gridded sea-surface NetCDF from above: elevation as colour (red = crest, blue = trough, grey = no data) with a colour bar,
axes in metres, time, and the frame's 4·std(Z). It's for reconstructions without camera images, like Agulhas II. It also reads Ifremer `.nc` files
(full or `_short`, local or URL). Same controls and `--save` as `view_stereo.py`. Needs `opencv-python`, `numpy`, `netCDF4`, and `pygame-ce` for the window.

```bash
A=data/SA_Agulhas_II_Winter_2017_Cruise__WASS
python3 view_nc.py $A/wass__20170704_0805.nc                                  # live, at the file's 2 Hz
python3 view_nc.py $A/wass__20170704_0805.nc --fps 8 --loop                     # 4x real time
python3 view_nc.py $A/wass__20170704_0805.nc --start 600 --count 240 --fps 8 --save agulhas_0805.mp4  # 2 min of record from t=300 s
python3 view_nc.py https://data-dataref.ifremer.fr/stereo/BS_2013/2013-09-22_13-00-01_10Hz/nc/Surfaces_20130922_130001_short.nc --count 100
```

| Option | Default | Meaning |
|---|---|---|
| `--fps N` | file's rate (2 Hz Agulhas, 10–15 Hz Ifremer) | Playback / mp4 frame rate. Higher than the file's rate = faster than real time |
| `--start N` / `--count N` / `--step N` | `0` / all / `1` | Which time steps to show (time = step / file fps) |
| `--zlim M` | 3 × robust std of Z | Colour scale ±M metres. Fix it to compare files |
| `--size PX` | `900` | Longer side of the map; the aspect ratio follows the grid in metres |
| `--loop`, `--save FILE.mp4` | off | As in `view_stereo.py` |

How it handles the different formats:
- **Units** come from each variable's `units` attribute (mm or m), and everything is shown in metres.
- **Orientation:** x runs to the right and y up, worked out from the grid coordinates. For Agulhas II the ship is below the bottom edge.
- **Elevation** is shown relative to the median over 20 sampled frames, so a nonzero mean plane doesn't shift the colours.

On NixOS: `pyv "python3 view_nc.py ..."`, with the `pyv` alias from [Setup](#setup).

---

# view_stereo.py

Plays a folder of stereo image pairs as a video, with the left and right cameras side by side.
It can open a live window or write an mp4. Optionally, it overlays a reconstructed sea surface (`.nc`) on the right camera image.

```
┌──────────────┬──────────────────────┬──────────────┐
│ left (01)    │ right (02)           │ Z top view   │   ← third panel only with --topview
│              │  ▞▀▀▀▀▀▀▚  ▐ +0.5 m  │ 19x19 m      │
│   raw image  │ ▞ overlay ▚ ▐ 0      │  (grid seen  │
│              │▞▄▄▄▄▄▄▄▄▄▄▚▐ -0.5 m  │   from above)│
│ 000001 [1/20] nc t=0                │              │   ← frame label, position, NetCDF step
└──────────────┴──────────────────────┴──────────────┘
```

## Choosing the folder (how pairs are found)

```
python3 view_stereo.py FOLDER [options]
```

FOLDER is searched recursively for files named `<frame>_01.<ext>` (left camera) and `<frame>_02.<ext>` (right camera).
Anything between the frame number and the camera suffix is ignored, e.g. AA's `000012_0000999_01.png`.
Supported extensions: tif, tiff, png, jpg, jpeg, bmp. 8- and 16-bit, grayscale or colour.

This covers both Ifremer layouts, so any of these work:

| FOLDER | Layout |
|---|---|
| `data/BS_2013/2013-09-22_13-00-01_10Hz` | acquisition folder (finds `input/` itself) |
| `data/BS_2013/2013-09-22_13-00-01_10Hz/input` | flat: `input/000001_01.tif`, `input/000001_02.tif` |
| `data/YS_2017/2017-05-13_05-00-00_10Hz/input` | split: `input/cam1/000001_01.tif`, `input/cam2/000001_02.tif` |
| `data/BS_2013` or `data` | several acquisitions, played one after another; the label shows which |

### Other file naming (e.g. your own WASS output)

If files aren't named `*_01` / `*_02`, give a glob pattern per camera, relative to FOLDER.
Both lists are sorted and paired in order:

```bash
python3 view_stereo.py path/to/wass_output \
    --left  '*_wd/undistorted/00000000.png' \
    --right '*_wd/undistorted/00000001.png'
```

The pattern above is a guess at WASS's output names. Check one of your `NNNNNN_wd/` folders and adjust it.
Quote the patterns so the shell doesn't expand them. If the two lists differ in length, the extra images are dropped with a warning.

## Options

### Playback and selection

| Option | Default | Meaning |
|---|---|---|
| `--fps N` | `10` | Playback speed, or the frame rate of the saved mp4. Match the acquisition rate (10, 12 or 15 Hz) for real time |
| `--width PX` | `1600` | Width of the combined frame. Smaller is faster to render |
| `--start N` | `0` | Skip the first N pairs |
| `--step N` | `1` | Show every Nth pair |
| `--count N` | `0` (all) | Show at most N pairs |
| `--loop` | off | Restart at the end instead of holding on the last frame |
| `--save FILE.mp4` | off | Write an mp4 (`mp4v` codec) instead of opening a window. Works without a display |

`--start`/`--step`/`--count` select from the pairs on disk. They are applied in that order: skip, subsample, then cap.

### Reconstruction overlay

| Option | Default | Meaning |
|---|---|---|
| `--nc FILE\|URL` | off | Ifremer **full** `.nc` file to overlay on the right image. Local path or `https://` URL |
| `--topview` | off | Add a third panel with `Z` seen from above, turned to match the camera's view |
| `--zlim M` | auto | Colour scale ±M metres. Default: 1.5 × the 99th-percentile \|Z\| of the first frame, rounded up |
| `--alpha A` | `0.5` | Overlay opacity, 0 (invisible) to 1 (opaque) |
| `--nc-offset N` | `0` | NetCDF time step = pair number + N (see [Frame matching](#frame-matching)) |

### Keyboard (live window)

| Key | Action |
|---|---|
| Space | Pause / resume |
| ← / → | Step one frame back / forward (pauses) |
| ↑ / ↓ | Double / halve playback speed (0.5–120 fps) |
| Q / Esc | Quit |

The window title shows the two filenames. The bottom-left label shows the frame number, position (`[i/N]`), speed
or `PAUSED`, and `nc t=` (the NetCDF time step drawn) when `--nc` is used.

## The reconstruction overlay in detail

### How it works

Ifremer's full `.nc` files store, for every grid point and time step:
- `Z`: elevation in metres
- `iR`, `jR`: the right-camera pixel (column, row) where that point appears in the **raw** image

The script fills each grid cell's projection on the right image, coloured by its mean `Z` (blue = trough, red = crest).
No camera calibration is involved, so what you see is exactly where Ifremer's pipeline placed the surface.

Things to note:
- **Only `full` files work.** `_short` files contain `Z` but not `iR`/`jR`. The script stops with
  `... has no iR, jR; overlay needs a full (not _short) file.` AA_2014, AA_2015 and YS_2017 have only full files anyway.
- The grid covers a fixed patch of sea (19 × 19 m at 0.1 m spacing for BS_2013). The rest of the image is left as is.
- Grid points with no data (`NaN`) are left uncoloured, and they show as grey in the top view.

### Frame matching

NetCDF time steps are matched to image pairs **by position**: time step `t` = the `t`-th pair found in FOLDER, plus `--nc-offset`.
Position counts from the first pair on disk and ignores `--start`/`--step`. So `--start 100` shows pair 100 with time step 100.

- Verified for **BS_2013 2013-09-22**: time step 0 = image `000001`. This was checked by correlating surface slope with image brightness over several frames.
  The file's `count` variable (30001, 30002, …) is the camera's internal frame counter, not the file number.
- **Not yet verified** for the other acquisitions. If the overlay looks right for one frame but drifts or jitters against the waves, try `--nc-offset ±1`.
- Point FOLDER at **one acquisition**. With several, positions run on across acquisitions and the matching is wrong.
- If you downloaded frames that don't start at the beginning of the record (`download_data.py --start N`), set `--nc-offset N`.
- Pairs past the end of the NetCDF are shown without an overlay and labelled `(no nc frame)`.

### Local file vs URL

`--nc` accepts a URL, and the script then reads only the time steps it shows, using HTTP byte-range requests.
That avoids downloading a 3.7–13 GB file, but **the Ifremer server is slow: about 0.5 s per frame.**

| Use case | Recommendation |
|---|---|
| Quick look, or making a video | URL + `--save out.mp4` (e.g. 300 frames ≈ 2.5 min) |
| Smooth live playback, scrubbing back and forth | Download the full file first, then pass the local path |

```bash
# Download only the full NetCDF for one acquisition (no images)
python3 download_data.py download BS_2013 --acq 13-00 --frames -1 --nc full
python3 view_stereo.py data/BS_2013/2013-09-22_13-00-01_10Hz --topview \
    --nc data/BS_2013/2013-09-22_13-00-01_10Hz/nc/Surfaces_20130922_130001.nc
```

### Your own WASS output

The overlay only understands Ifremer's format (`Z` + `iR`/`jR`). For NetCDF files from your own WASS runs (made with `wassgridsurface`),
use WASS's own [wassncplot](https://github.com/fbergama/wassncplot) (`pip install wassncplot`).
It renders the 3D surface over the original images using the camera geometry stored in the file.
It cannot read Ifremer's files because they lack that metadata, which is why this script exists.

To compare raw input with WASS's rectified images, play them separately with `--left`/`--right`.

## Recipes

```bash
# Real-time playback of a 12 Hz acquisition
python3 view_stereo.py data/BS_2011/2011-10-04_11-38-00_12Hz --fps 12

# Skim a long record: 1 pair per second, looped
python3 view_stereo.py data/BS_2011/2011-10-04_11-38-00_12Hz --step 12 --loop

# Look at a specific moment: pairs 600-699
python3 view_stereo.py data/BS_2013/2013-09-22_13-00-01_10Hz --start 600 --count 100

# Small preview video for slides / sharing
python3 view_stereo.py data/YS_2017 --width 960 --count 100 --save ys2017_preview.mp4

# Overlay with a fixed colour scale, so videos of different acquisitions are comparable
python3 view_stereo.py data/BS_2013/2013-09-22_13-00-01_10Hz --nc <file-or-url> --zlim 1.0 --alpha 0.6 --save out.mp4
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No *_01 / *_02 image pairs found` | Wrong folder, or a different naming scheme. Check with `ls`, or use `--left`/`--right` |
| `The function is not implemented. Rebuild the library with ... GTK` | Something called `cv2.imshow`. This script doesn't; make sure you're running the current `view_stereo.py` |
| `ModuleNotFoundError: pygame` / `netCDF4` | Add `ps.pygame-ce` / `ps.netcdf4` to the nix-shell packages, or pip install them |
| No window appears (WSL) | WSLg must be running (`echo $DISPLAY` should print `:0`). Otherwise use `--save` |
| Playback slower than `--fps` | Each frame is read from disk and rendered (~35 ms, ~110 ms with overlay). Lower `--width`, or use `--save` |
| Overlay playback very slow | `--nc` is a URL. Download the full file (see [Local file vs URL](#local-file-vs-url)) |
| `has no iR, jR` | You passed a `_short` file. Use the full one |
| Overlay doesn't follow the waves | Frame mismatch. See [Frame matching](#frame-matching) and try `--nc-offset` |
| `(no nc frame)` in the label | That pair is past the end of the NetCDF record (or before it, with a negative offset) |
