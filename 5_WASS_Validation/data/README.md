# WASS Validation Data

Stereo data is hosted at: https://data-dataref.ifremer.fr/stereo/
- This is what is used for WASS validation.

There is also other data at: https://data-dataref.ifremer.fr/
- But this may not be as related.

Downloaded data is **not** committed to git (see `.gitignore`). Use `download_data.py` to recreate it.

## Quick start

The script only uses the Python standard library (3.9+), so you don't need a venv or `pip install`.
On this NixOS machine `python3` is not on the PATH, so run it through nix:

```bash
cd 5_WASS_Validation/data
alias py="nix shell nixpkgs#python3 -c python3"   # or just use `python3` elsewhere

py download_data.py list                                   # what's on the server
py download_data.py download BS_2011 --acq 11-38 --dry-run # check size first
py download_data.py download BS_2011 --acq 11-38 --frames 300 --nc short
```

Re-running the same command is safe. Files already on disk are skipped and interrupted downloads resume,
so a command doubles as a record of exactly which data was used.

### Options (`download`)

| Option | Default | Meaning |
|---|---|---|
| `CAMPAIGN ...` | (required) | One or more of `AA_2014 AA_2015 BS_2011 BS_2013 LJ_2018 YS_2017`, or `all` |
| `--acq SUBSTR` | all | Only acquisitions whose folder name contains `SUBSTR` (repeatable), e.g. `--acq 11-38` |
| `--frames N` | `100` | Stereo pairs per acquisition. `0` = all, `-1` = no images (config/nc only) |
| `--start N` / `--step N` | `0` / `1` | Skip the first N pairs / take every Nth pair (e.g. `--step 12` ≈ 1 pair/s at 12 Hz) |
| `--nc` | `none` | Reference WASS surfaces: `none`, `short`, `full`, or `all` |
| `--no-extras` | off | Skip the small Meteo / WaveGauge `.mat` files (fetched by default) |
| `--out DIR` | this folder | Where to write data |
| `--workers N` | `4` | Parallel downloads |
| `--dry-run` | off | Print file count and total size, download nothing |
| `-y` | off | Skip the confirmation prompt for downloads over 50 GB |

### Local layout

The server structure is mirrored:

```
data/<campaign>/<acquisition>/config/   camera calibration + WASS configs
data/<campaign>/<acquisition>/input/    stereo image pairs (NNNNNN_01 = left, NNNNNN_02 = right)
data/<campaign>/<acquisition>/nc/       WASS-reconstructed surfaces (NetCDF), for comparison
data/<campaign>/<campaign>_Meteo/       weather data (.mat)
data/<campaign>/<campaign>_WaveGauge/   in-situ wave gauge records (.mat), BS campaigns only
```

## What each campaign contains

Every acquisition has the same three folders:
- **`config/`** (a few KB): intrinsics and distortion for both cameras, plus `stereo_config.txt` and `matcher_config.txt` for WASS.
  BS and LJ also include the extrinsics (`ext_R.xml`, `ext_T.xml`). AA and YS do not, so WASS must estimate the extrinsics itself.
- **`input/`**: raw synchronized stereo pairs. This is the bulk of the data.
- **`nc/`**: the 3D sea-surface reconstruction already produced by WASS, which you can use as ground truth for your own run.
  `_short` files are a reduced version, about 1/3 the size. Check with `ncdump -h` whether they cut time steps or variables.

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

### Choosing a dataset

- **Test that the WASS pipeline runs:** a few hundred frames from any acquisition with `--nc none`. For example, `--frames 300` on BS is ~3 GB.
- **Compare against Ifremer's WASS output:** add `--nc short` (or `full` for AA/YS, which have no short file).
  The smallest reference is YS_2017 at 458 MB. The BS_2013 short files are ~1.2 GB.
- **Validate against independent in-situ measurements:** BS_2011 or BS_2013 are the only campaigns with wave gauge data.
- **Get a long record for spectra/statistics:** use `--step` to subsample, e.g. `--frames 0 --step 12` on a 12 Hz record gives 1 pair/s (~1/12 the size).
- **Downloading everything:** `download all --frames 0 --nc all` is about **2.1 TB**. You almost certainly don't want this.

The numbers above were surveyed from the server on 2026-11-02. Run `--dry-run` for exact current sizes.
