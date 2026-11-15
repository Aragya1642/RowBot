# FoundationStereo for sea-surface reconstruction

This folder tests whether [FoundationStereo](https://github.com/NVlabs/FoundationStereo) (NVIDIA, CVPR 2025), a zero-shot
deep stereo model, can replace the dense-matching step of [WASS](../5_WASS_Validation/README.md).

WASS reconstructs the sea surface in four stages:

| WASS stage | What it does | FoundationStereo's role |
|---|---|---|
| `wass_prepare` | undistorts the images | none (WASS's undistorted images are reused) |
| `wass_match` + `wass_autocalibrate` | estimates the extrinsics (R, T) once per camera setup | none (reused) |
| `wass_stereo` | rectifies, runs **SGBM** dense disparity, triangulates, fits the sea plane | **replaces SGBM**. Rectification, triangulation and the plane filter are re-implemented around it |
| `wassgridsurface` | grids the point clouds into `gridded.nc` | not used yet (planned) |

So FoundationStereo needs a WASS run on the same cameras. It gets the calibration from that run, and the WASS
output is the reference it is compared with.

| File | Purpose |
|---|---|
| [`shell.nix`](shell.nix), [`requirements.txt`](requirements.txt) | NixOS environment: Python venv with CUDA PyTorch on the WSL GPU. See [Setup](#setup) |
| [`view_demo.py`](view_demo.py) | Renders FoundationStereo's own demo output into one PNG (no display needed) |
| [`cloud_to_html.py`](cloud_to_html.py) | Turns point clouds (`.ply`) into a self-contained interactive 3D viewer (HTML) |
| [`compare_frame.py`](compare_frame.py) | Runs FoundationStereo or Fast-FoundationStereo (`--method ffs`) on one WASS frame and compares it with WASS. See [One frame vs WASS](#one-frame-vs-wass-compare_framepy) |
| [`compare_methods.py`](compare_methods.py) | WASS vs WASSfast vs the network runs on one frame, all on WASS's grid, with spectra. See [All methods](#all-methods-on-one-frame-compare_methodspy) |
| [`make_index.py`](make_index.py) | Builds `output/index.html`: one page with every result (summaries, plots, links to 3D viewers). `--open` opens it in the Windows browser |
| [`headless_run.py`](headless_run.py) | Runs an upstream demo script with its display windows stubbed out (and `torch.compile` off for Pascal) |
| `FoundationStereo/` | Git submodule: [NVlabs/FoundationStereo](https://github.com/NVlabs/FoundationStereo) |
| `FoundationStereo/pretrained_models/` | Model weights. Downloaded by hand, git-ignored by the submodule |
| `Fast-FoundationStereo/` | Git submodule: [NVlabs/Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo), the real-time distilled version. See [Fast-FoundationStereo](#fast-foundationstereo) |
| `Fast-FoundationStereo/weights/` | Its weights. Downloaded by hand, git-ignored by the submodule |
| `output/` | Everything the scripts write. Git-ignored |
| `.venv/` | Python environment, created by `shell.nix`. Git-ignored |

```
7_FoundationStereo/
├── README.md
├── shell.nix                 ← nix-shell: venv + CUDA via /usr/lib/wsl/lib
├── requirements.txt          ← pip packages for the venv (PyTorch 2.4.1 + CUDA 12.1, ...)
├── view_demo.py              ← demo output -> summary.png
├── cloud_to_html.py          ← .ply -> interactive 3D viewer (.html)
├── compare_frame.py          ← one WASS frame: FoundationStereo / Fast-FoundationStereo vs WASS
├── compare_methods.py        ← one frame: WASS vs WASSfast vs network runs, on WASS's grid
├── headless_run.py           ← run upstream demo scripts without a display
├── make_index.py             ← output/index.html: every result on one page
├── FoundationStereo/         ← submodule
│   └── pretrained_models/    ← weights (git-ignored): 11-33-40 (ViT-S), 23-51-11 (ViT-L), onnx/
├── Fast-FoundationStereo/    ← submodule
│   └── weights/              ← weights (git-ignored): 23-36-37, 20-26-39, 20-30-48, 15-44-51, onnx/
├── output/                   ← git-ignored; all results go here
└── .venv/                    ← git-ignored
```

## Setup

### Machine

This was set up on NixOS under WSL2 with a **GTX 1060 6 GB** (Pascal, compute capability 6.1), 16 CPUs and 16 GB RAM.
Windows uses about 2 GB of the GPU memory, so about 4 GB is free for models. FoundationStereo was tested upstream on
RTX 3090/4090, A100, V100 and Jetson Orin, so this GPU is at the low end. See [Performance](#performance-on-the-gtx-1060).

### Environment

```bash
git submodule update --init 7_FoundationStereo/FoundationStereo 7_FoundationStereo/Fast-FoundationStereo   # if not checked out yet
cd 7_FoundationStereo
nix-shell     # first run: creates .venv and installs requirements.txt (~3 GB of wheels, several minutes)
```

How it works, and why:
- **PyTorch comes from pip wheels**, not nixpkgs. The wheels bundle CUDA, and they run on NixOS through **nix-ld**
  (`programs.nix-ld.enable = true` in `/etc/nixos/configuration.nix`, already on this machine). CUDA PyTorch from
  nixpkgs is a lot more trouble on WSL.
- **The GPU driver** comes from Windows, in `/usr/lib/wsl/lib`. `shell.nix` puts that on `LD_LIBRARY_PATH`. Without it,
  `nvidia-smi` and CUDA fail with "couldn't find libnvidia-ml.so". Inside the shell, `nvidia-smi` works.
- **PyTorch 2.4.1 + CUDA 12.1** (upstream's version). PyTorch's CUDA 12.8+ builds dropped Pascal GPUs, so don't upgrade
  this without checking `torch.cuda.get_arch_list()` still includes `sm_60`/`sm_61`.
- **Only what inference needs.** Upstream's `environment.yml` also has training, Jupyter and dataset packages.
  `requirements.txt` has just what the inference code imports. **xformers and flash-attn are not installed.** The model
  falls back to plain PyTorch attention (flash-attn doesn't support Pascal anyway).
- **Native libraries.** The opencv/open3d wheels need libGL, X11, glib, libudev, libusb, tbb and others that nix-ld doesn't
  provide. `shell.nix` lists them.
- **Reinstalling.** `shell.nix` reinstalls when `requirements.txt` is newer than `.venv/.installed`. To start over:
  `rm -rf .venv`.
- **First model load** fetches DINOv2's code (not weights) from GitHub through `torch.hub` into `~/.cache/torch/hub`.
  After that it works offline.

### Model weights

The weights are on Google Drive. They are not in git. Download them once:

```bash
cd 7_FoundationStereo/FoundationStereo
mkdir -p pretrained_models && cd pretrained_models
nix-shell -p python3Packages.gdown --run 'gdown --folder https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf'
mv pretrained_models/* . && rmdir pretrained_models    # gdown nests the Drive folder one level too deep
```

| Folder | Model | Size | Notes |
|---|---|---|---|
| `11-33-40/` | ViT-small | 0.8 GB | faster, slightly less accurate. **Used here** (fits the 6 GB GPU) |
| `23-51-11/` | ViT-large | 3.3 GB | upstream's best model |
| `onnx/` | ViT-large ONNX | 0.75 GB | for TensorRT (upstream supports this only in Docker) |

The licence is NVIDIA's (see `FoundationStereo/LICENSE`): **non-commercial research use**. NVIDIA sells a separate
commercial model.

## The upstream demo

This checks that everything works, using FoundationStereo's sample desk scene:

```bash
cd 7_FoundationStereo && nix-shell
cd FoundationStereo
env -u DISPLAY -u WAYLAND_DISPLAY python scripts/run_demo.py \
    --left_file assets/left.png --right_file assets/right.png --intrinsic_file assets/K.txt \
    --ckpt_dir pretrained_models/11-33-40/model_best_bp2.pth --out_dir ../output/demo_vits
cd .. && python view_demo.py --out_dir output/demo_vits                 # -> output/demo_vits/summary.png
python cloud_to_html.py output/demo_vits/cloud_denoise.ply --open        # -> interactive 3D viewer
```

- `run_demo.py` writes `vis.png` (left image | disparity), `depth_meter.npy`, `cloud.ply` and `cloud_denoise.ply`.
- At the end it tries to open an Open3D 3D window. **That doesn't work on this WSL setup**: there's no EGL access to
  `/dev/dri`. With `DISPLAY` unset it fails straight away with `AttributeError: 'NoneType' object has no attribute
  'point_size'`, *after* everything has been saved. Ignore that error. Without `env -u DISPLAY` it may hang instead.
- **Result:** ViT-S produced a clean, sharp disparity map (mug handle, keyboard edges) and 470k points, with depths of
  0.4–1.1 m and 91 % of pixels valid. The whole script took 55 s, most of it model loading and point-cloud denoising.

### view_demo.py

`python view_demo.py --out_dir DIR [--scale S]` → `DIR/summary.png`: left image, disparity (both colourings), depth, and
the point cloud from behind the camera and from above. `--interactive` tries the Open3D window instead, which needs a
working display.

### cloud_to_html.py: interactive 3D viewer

```bash
python cloud_to_html.py CLOUD.ply [MORE.ply ...] [--frame opencv|zup] [--max_points N] [--open]
```

This writes one self-contained `.html` file (the points are embedded, and three.js comes from a CDN). Open it in any browser.
From Windows it is at `\\wsl.localhost\NixOS\home\nixos\RowBot\7_FoundationStereo\...`, or use `--open`.

- **Mouse:** left-drag orbits, right/shift-drag pans, the wheel zooms, and double-click orbits around the clicked point.
- **Panel:**
  - one checkbox per input file (layer);
  - colour by image, layer, height or distance (with a colour bar);
  - point size and **vertical exaggeration** (to see waves);
  - Start/Top/Side views.
- **`--frame opencv`** (default) is the camera frame (x right, y down, z forward), as in FoundationStereo's output. It
  starts from the camera, shown as a yellow frustum.
- **`--frame zup`** is horizontal x, y with z up, such as WASS's sea-plane coordinates. `--eye X Y Z` sets the start view.
- **`--max_points`** (per layer, default 400k) subsamples randomly. Each point adds about 20 bytes to the file.
- It can also be imported: `write_viewer(out, [(name, points, colours), ...], frame=..., eye=...)`. `compare_frame.py`
  uses this.

## Fast-FoundationStereo

[Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo) (NVIDIA, CVPR 2026) is a distilled,
pruned version of FoundationStereo that upstream reports as more than 10× faster at nearly the same zero-shot
accuracy. It uses a single small backbone instead of ViT + DINOv2, fewer refinement iterations (4–8 instead of 32) and
`max_disp` 192. Upstream recommends it for real-time use and the original for best offline accuracy. Both are compared here.

**Environment.** It shares the same `.venv`. Upstream asks for PyTorch 2.6 + xformers, but it runs on the 2.4.1 +
CUDA 12.1 install that the GTX 1060 needs. It also needs `scikit-image` and `pyyaml`, which were added to
`requirements.txt`. Two things don't work on Pascal, and both are optional:
- **`@torch.compile`** on a few helper functions: the Inductor backend needs Triton, and Triton needs compute
  capability ≥ 7.0. `headless_run.py` sets `TORCHDYNAMO_DISABLE=1`, so those functions run eagerly with the same maths.
  Any script that imports Fast-FoundationStereo's `core` on this GPU must set this too.
- **The Triton cost-volume kernel** (`optimize_build_volume='triton'`), used only by the profiling and ONNX scripts. The
  demo uses the PyTorch version (`'pytorch1'`).

**Weights** (Google Drive, research licence, git-ignored):

```bash
cd 7_FoundationStereo && nix-shell
gdown --folder https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap -O Fast-FoundationStereo/weights
```

| Folder | Notes (upstream, RTX 3090, 640×480) |
|---|---|
| `23-36-37/` | most accurate, 49 ms at 8 iterations. **Default** |
| `20-26-39/` | 44 ms |
| `20-30-48/` | fastest, 38 ms |
| `15-44-51/` | not in upstream's table |
| `onnx/` | ONNX exports at 576×960 and 320×736, for TensorRT |

The licence is NVIDIA's research licence (`Fast-FoundationStereo/LICENSE.txt`). A commercial checkpoint (C-Fast-FoundationStereo)
is published separately.

**Demo** (same desk scene and intrinsics as FoundationStereo's):

```bash
cd 7_FoundationStereo && nix-shell
python headless_run.py Fast-FoundationStereo/scripts/run_demo.py \
    --model_dir Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth \
    --left_file Fast-FoundationStereo/demo_data/left.png --right_file Fast-FoundationStereo/demo_data/right.png \
    --intrinsic_file Fast-FoundationStereo/demo_data/K.txt --out_dir $PWD/output/demo_ffs \
    --remove_invisible 0 --denoise_cloud 1 --scale 1 --get_pc 1 --valid_iters 8 --max_disp 192 --zfar 100
python view_demo.py --out_dir output/demo_ffs --intrinsic_file Fast-FoundationStereo/demo_data/K.txt
python cloud_to_html.py output/demo_ffs/cloud_denoise.ply --open
```

Pass `--out_dir` as an absolute path. The script runs `rm -rf` on it and then recreates it.

**Result:** a sharp disparity map, visually the same as FoundationStereo's (33–138 px), with 331k points after
denoising. The whole script took 22 s, against 55 s for FoundationStereo ViT-S.

**Speed on the GTX 1060** (forward pass only, fp16, 8 iterations, warm, random input; benchmarked separately):

| Checkpoint | 960×544 (demo) | 864×736 (sea frame, scale 0.35) | 1248×1056 (sea frame, scale 0.5) |
|---|---:|---:|---:|
| `23-36-37` | 1.7 s, 4.2 GB | 4.9 s, 5.1 GB (spills) | 40.5 s, 10.5 GB (spills) |
| `20-30-48` | 0.9 s, 1.6 GB | **1.0 s, 1.9 GB** | 5.1 s, 3.8 GB |

For comparison, FoundationStereo ViT-S takes 22.4 s at 860×720. So `20-30-48` is about 20× faster at the same
resolution, and it can run scale 0.5 within the free memory. Peak memory is far above upstream's figure (653 MB at
640×480). The likely cause is that, without `torch.compile`, the helper functions keep large intermediate tensors that
the compiled versions would fuse away, and `23-36-37` is hit hardest. Above about 4 GB the same spill problem as
FoundationStereo appears. On this GPU, use `20-30-48` (or `23-36-37` at smaller sizes).

## One frame vs WASS: compare_frame.py

```bash
cd 7_FoundationStereo && nix-shell
python compare_frame.py --wass ../5_WASS_Validation/data/compare/quickstart/wass --frame 0
```

It needs only a `run_wass.py` output folder (see [5_WASS_Validation](../5_WASS_Validation/README.md#one-command-run_wasspy)),
not the WASS tools. All output goes to `7_FoundationStereo/output/` (`--out` must be inside this folder).

### What it does

```
WASS run folder                                     compare_frame.py
---------------                                     ----------------
output/NNNNNN_wd/undistorted/0000000{0,1}.png  ──►  cv2.stereoRectify with WASS's K0, K1, ext_R, ext_T (alpha=0)
output/NNNNNN_wd/intrinsics_*.xml, ext_R/T.xml      │  scale by --scale -> FoundationStereo -> disparity
                                                    ▼
                                                    triangulate (Q) -> rectified cam0 -> cam0 -> WASS world
output/NNNNNN_wd/Cam0_poseR/T.txt               ──►  (x_cam0 = R0 X_world + T0)
output/planes.txt (mean plane)                  ──►  -> sea plane, z up, × baseline = metres  (same as wassgridsurface)
output/NNNNNN_wd/mesh_cam.xyzC (WASS cloud)     ──►  same transform -> compared
gridding/gridconfig.txt                         ──►  comparison area (the WASS grid area)
```

- **Rectification** is done here with OpenCV, not by reading WASS's rectified images, so any image pair with WASS-style
  calibration works. WASS's undistorted images have zero distortion. `|ext_T| = 1`, so triangulated points are in
  *baseline units*, like WASS's. The baseline (m) comes from the run's `timings.json` (`--baseline` overrides it).
- **Filtering** mirrors `wass_stereo`:
  1. the pixel must be inside both rectified images (`u − d` inside the right image);
  2. a RANSAC plane is fitted to FoundationStereo's own points;
  3. points within `PLANE_MAX_DISTANCE` = 1.5 baseline units of that plane are kept (WASS's default).

  This removes the sky, the camera housing and the platform. WASS's plane is used only to put both clouds in the same
  coordinates.
- **The three comparisons** treat WASS as the reference:
  1. **Image space.** WASS's 3D points are projected into the rectified left image as disparity (mean per pixel) and
     compared pixel by pixel with FoundationStereo's disparity. This is the most direct check of the matching itself.
  2. **Sea-plane space.** Both clouds are binned (mean z per cell, `--cell`, default 0.5 m) over the WASS grid area. Then
     RMSE, bias, correlation, Hs = 4·std(z) (on cells where both have data) and coverage.
  3. **Sea plane.** The same RANSAC + least-squares fit is run on both clouds, and the angle and offset between the
     planes are compared. WASS's own `plane.txt` is fitted differently (central third of the image only,
     `PLANE_USE_CENTRAL_THIRD_ONLY`): the same fit on WASS's points is about 1° away from it. So FoundationStereo is
     compared with *the same fit* on WASS's points, and the angle to WASS's own plane is listed only for reference.

### Options

| Option | Default | |
|---|---|---|
| `--wass DIR` | required | `run_wass.py` output folder |
| `--frame N` | 0 | WASS frame number (`output/NNNNNN_wd`) |
| `--method fs\|ffs` | `fs` | FoundationStereo or Fast-FoundationStereo |
| `--model M` | `vits` / `20-30-48` | fs: `vits` (`11-33-40`) or `vitl` (`23-51-11`); ffs: `20-30-48`, `23-36-37`, `20-26-39`, `15-44-51` |
| `--scale S` | 0.5 | image scale fed to the network. **The main lever for GPU memory and time** |
| `--valid_iters N` | 32 / 8 | refinement iterations (fs upstream suggests 16 for speed; ffs uses 4–8) |
| `--max_disp N` | 416 / auto | largest disparity the network can output, at its scale. For ffs (upstream 192) it is raised to 1.2× WASS's largest disparity in the frame: at scale 0.5 WASS reaches 196 px, and 192 clipped the near field (disparity RMSE 2.05 px → 0.99 px with 256). A warning is printed if it leaves no margin |
| `--hiera` | off | hierarchical inference (coarse pass at half resolution, then full) |
| `--fp32` | off | disable fp16 autocast (upstream uses fp16) |
| `--repeat N` | 2 | inference runs. The first includes CUDA warm-up; the summary reports the fastest |
| `--cell M` | 0.5 | binning cell for the elevation comparison, metres |
| `--reuse` | off | skip inference, reuse `disparity_fs.npy` in `--out` (to re-run only the analysis) |
| `--no-html` | off | skip `clouds.html` |
| `--out DIR` | `output/<run>_frameNNNNNN_<model>_s<scale>` | must be inside `7_FoundationStereo/` |

### Outputs (in `--out`)

| File | What |
|---|---|
| `comparison.png` | top: rectified left, FoundationStereo disparity (grey = removed by the plane filter), WASS as disparity, difference. Bottom: WASS elevation, FoundationStereo elevation, difference, per-cell scatter |
| `summary.md`, `summary.json` | the numbers (timings, GPU memory, point counts, disparity/elevation statistics, planes) |
| `clouds.html` | both clouds in the interactive viewer, sea-plane frame (z up), grid area + 25 %, coloured by height, 3× vertical exaggeration, starting from the camera |
| `rectified.jpg` | the rectified pair with horizontal lines, to check rectification by eye |
| `disparity_fs.npy` | FoundationStereo disparity at the scaled resolution (pixels at that scale) |
| `fs_cloud_sea.ply` | FoundationStereo cloud in sea-plane metres |

### Results: quick-start data, frame 0

Data: the `quickstart` WASS run in `5_WASS_Validation/data/compare/quickstart/wass`, frame `000000`, with
ViT-S (`11-33-40`) and 32 iterations. The images are 2456×2058 and the baseline is 3.0 m. To reproduce:

```bash
python compare_frame.py --wass ../5_WASS_Validation/data/compare/quickstart/wass --frame 0 --scale 0.35
```

| | scale 0.5, fp16 | scale 0.35, fp16 | scale 0.35, fp32 |
|---|---:|---:|---:|
| input to the network | 1228×1029 | 860×720 | 860×720 |
| inference time (warm run) | 72.5 s | **22.4 s** | 4659 s |
| peak GPU memory | 5756 MB | 2920 MB | 4246 MB |
| FoundationStereo points (after the plane filter) | 1.07 M | 0.53 M | 0.53 M |
| disparity vs WASS: median abs / RMSE / bias (full-res px) | 0.264 / 0.924 / +0.347 | 0.212 / 0.679 / +0.282 | 0.212 / 0.680 / +0.282 |
| elevation vs WASS: RMSE / bias | 13.6 / +3.5 cm | 14.6 / +3.9 cm | 14.6 / +3.9 cm |
| elevation correlation (0.5 m cells) | 0.936 | 0.927 | 0.926 |
| Hs = 4·std: FoundationStereo / WASS | 1.462 / 1.464 m | 1.456 / 1.474 m | 1.455 / 1.473 m |
| grid coverage: FoundationStereo / WASS | 82 % / 86 % | 72 % / 86 % | 72 % / 86 % |
| sea plane, same fit on both clouds: angle / offset | 0.25° / −4.2 cm | 0.37° / −4.4 cm | 0.35° / −3.8 cm |

For reference, WASS's SGBM dense stereo took **60.0 s** (CPU) on the same frame. WASS produced 3.2 M points.

What the comparison shows (see `comparison.png` in each output folder):

- **The same sea surface.** Both methods show the same wave pattern. Elevation correlates at about 0.93 per
  0.5 m cell, and Hs agrees to within about 1–2 cm out of 1.46 m. The two sea planes agree to within about
  0.3° when the same fit is applied to both clouds. (WASS's own `plane.txt` uses only the central third of the
  image, and the same fit applied to WASS's own points already differs from it by about 1°, so the angle is
  not compared against `plane.txt`.)
- **FoundationStereo is smoother.** It has less short-wave texture than WASS. Part of this is
  regularisation in the network, and part is the lower working resolution. Part of the 14 cm elevation RMSE is
  WASS's own pixel noise, so this is not FoundationStereo's error on its own; neither method is ground truth.
- **Small positive disparity bias** of about +0.3 px overall, rising to about +1 px (≈0.5 %) near the
  bottom of the image (near field). In elevation this appears as a +4 cm offset and a slight tilt. It is
  worth checking on more frames before reading anything into it.
- **Far-field stripes / lower coverage.** Beyond about y < −90 m one FoundationStereo pixel at scale 0.35
  is larger than a 0.5 m cell, so binning leaves empty stripes (coverage 72 % vs 82 % at scale 0.5). Proper
  gridding with interpolation, as `wassgridsurface` does, will fill these. They are not a stereo failure.
- **Non-sea content is handled.** Sky, the camera housing and the platform get confident disparities from
  the network, but the plane-distance filter (±1.5 baseline units) removes them. They are grey in the
  disparity panel.

#### Looking at the clouds

Open `output/<run>/clouds.html` in a Windows browser, e.g.
`\\wsl.localhost\NixOS\home\nixos\RowBot\7_FoundationStereo\output\quickstart_frame000000_vits_s0.35\clouds.html`,
or run `python -c "from cloud_to_html import open_in_windows; open_in_windows('output/.../clouds.html')"`.
It has two layers (FoundationStereo, WASS) that can be toggled. It is coloured by height with 3× vertical
exaggeration (a slider) and starts from camera 0.

### Performance on the GTX 1060

- **The run-time is set by memory, not compute.** About 2 GB of the 6 GB is held by Windows, so about 4 GB is
  free. Below that, inference is fast (scale 0.35: 2.9 GB, 22 s). Above it, the WSL driver silently spills
  into shared system memory and everything slows dramatically: scale 0.5 at 5.8 GB takes 72 s, and fp32 at
  4.2 GB takes over 75 minutes per frame.
- **Use fp16 autocast (the default).** fp32 gives identical accuracy here (0.680 vs 0.679 px RMSE) but
  needs about 45 % more memory. `--fp32` exists only for checks.
- **Recommended setting on this machine: `--scale 0.35`.** At about 22 s per frame it is about 3× faster than
  WASS's CPU SGBM, and it loses only about 1 cm of elevation RMSE against scale 0.5. On a GPU with ≥ 8 GB
  free, scale 0.5 (or full resolution on 24 GB) is the better choice.
- The first run in a process is slower (cuDNN autotuning, model load of about 5 s). `--repeat 2` reports the
  warm time.
- **Fast-FoundationStereo** (`--method ffs`, `20-30-48`) on the same frame: 1.6 s at scale 0.35 (1.9 GB) and 3.9 s
  at scale 0.5 (5.0 GB with `max_disp` 256). Its disparity agrees with WASS as well as FoundationStereo's does
  (median 0.25–0.29 px). Its elevation is smoother. See [All methods](#all-methods-on-one-frame-compare_methodspy).

## All methods on one frame: compare_methods.py

```bash
cd 7_FoundationStereo && nix-shell
# network runs first (each writes its own output folder)
python compare_frame.py --wass ../5_WASS_Validation/data/compare/quickstart/wass --frame 0 --scale 0.5
python compare_frame.py --wass ../5_WASS_Validation/data/compare/quickstart/wass --frame 0 --method ffs --scale 0.5
# then everything together
python compare_methods.py --wass ../5_WASS_Validation/data/compare/quickstart/wass \
    --wassfast ../5_WASS_Validation/data/compare/quickstart/wassfast --frame 0 \
    --runs output/quickstart_frame000000_vits_s0.5 output/quickstart_frame000000_ffs-20-30-48_s0.5
```

Everything is compared on the 256×256 grid (0.353 m) of WASS's `gridded.nc`, which WASSfast also uses:

| Surface | From |
|---|---|
| **WASS** | `gridded.nc`, this frame (WASS's own gridding) |
| **WASSfast** | `wassfast.nc`, this frame. WASSfast frame *i* = WASS pair *i* (as in `compare_runs.py`) |
| **each `--runs` folder** | its `fs_cloud_sea.ply`, gridded here: mean per grid cell, then holes up to `--fill` (1.5 m) from data filled by linear interpolation, inside WASS's `maskZ` |
| **WASS points** | WASS's own `mesh_cam.xyzC` gridded the same way as the runs. Not a method but a check. It matches `gridded.nc` to 4.4 cm RMSE, correlation 0.993, so the simpler gridding accounts for only a small part of the networks' differences |

All statistics use the cells where *every* surface has data (the common area, 87 % of the grid here). The outputs are:
- `methods.png`: maps, differences from WASS, wavenumber spectra in the largest square inside the common area, and
  an every-pair matrix.
- `methods.md` / `methods.json`: the same numbers.
- `clouds_all.html`: every point cloud overlaid in one interactive 3D viewer. It has WASS's cloud, each run's
  cloud, and WASSfast's grid as points (WASSfast has no point cloud). It uses sea-plane metres, 3× vertical
  exaggeration, and starts from camera 0. In "Layer colour" mode the layers use `methods.png`'s colours; tick them
  on and off, or switch to "Height". `--max_points` (default 250k) is per layer; `--no-html` skips it.

The spectrum is the most useful part when there is no ground truth. Real wind waves fall off roughly as k⁻³
(the saturation range), while matching noise flattens the spectrum, so each method's departure from that tail
shows what it actually resolves.

### Results: quick-start frame 0

| | time / frame | Hs | vs WASS: RMSE w/o offset | corr | energy 5–20 m vs WASS | energy 2–5 m vs WASS |
|---|---:|---:|---:|---:|---:|---:|
| **WASS** (SGBM, 16-core CPU) | 60 s | 1.44 m | – | – | 100 % | 100 % |
| WASSfast (CNN, CPU) | 0.79 s | 1.37 m | 12.7 cm | 0.936 | 84 % | 42 % |
| FoundationStereo ViT-S, scale 0.5 (GTX 1060) | 72.5 s | 1.48 m | 12.6 cm | 0.940 | 69 % | 1 % |
| FoundationStereo ViT-S, scale 0.35 | 22.4 s | 1.46 m | 13.6 cm | 0.929 | 42 % | 1 % |
| Fast-FoundationStereo 20-30-48, scale 0.5 | 3.9 s | 1.42 m | 13.7 cm | 0.927 | 41 % | 2 % |
| Fast-FoundationStereo 20-30-48, scale 0.35 | 1.6 s | 1.35 m | 16.5 cm | 0.890 | 14 % | 1 % |

The times measure different things on different hardware: WASS is SGBM only, WASSfast is the marginal time per pair
in batches of 16, and the networks are the forward pass only. Read them as orders of magnitude.

What it shows (one frame, so treat it as indicative):
- **All four agree on the sea state.** Hs is 1.35–1.48 m against WASS's 1.44 m, correlations are 0.89–0.94, and all
  share a +3 to +5 cm offset against WASS.
- **No method stands out against WASS.** WASSfast, FoundationStereo at 0.5 and Fast-FoundationStereo at 0.5 all
  sit at about 12.6–13.7 cm from WASS. In the pairwise table, FoundationStereo is *closer to WASSfast* (10.2 cm,
  0.962) than either is to WASS. That fits WASS carrying extra short-wave texture that the others don't have.
- **The networks are low-pass filters.** Their spectra follow WASS for waves longer than about 20 m, then fall
  below the k⁻³ tail that WASS and WASSfast follow. In the 5–20 m band FoundationStereo at 0.5 keeps 69 % of
  WASS's energy, Fast-FoundationStereo at 0.5 keeps 41 %, and at scale 0.35 they keep 42 % and 14 %. Below 5 m they have almost
  nothing (1–2 %). WASS has a flat plateau there, well above k⁻³, which is likely matching noise. So "less than
  WASS" below 5 m is not necessarily wrong, but the 5–20 m deficit is very likely real waves being smoothed
  away. Resolution (`--scale`) clearly matters, which points to tiling at full resolution as the next test.
- **Fast-FoundationStereo at scale 0.5 roughly matches FoundationStereo at scale 0.35** (RMSE 13.7 vs 13.6 cm, and 41 % vs 42 % in
  the 5–20 m band) at about 6× the speed (3.9 s vs 22.4 s).

## Next steps

1. **Resolution**: the networks smooth out short waves, and resolution matters. Run Fast-FoundationStereo at full resolution
   in overlapping tiles. It is fast enough (about 4 s per 1228×1029 tile), and the cost volume needs `max_disp` of
   about 400 at full resolution. Also try `--hiera` and ViT-L.
2. **Gridding**: write the network cloud as `mesh_cam.xyzC` into a copy of the WASS frame folder and run WASS's own
   `wassgridsurface`, so that every method shares the same interpolation and NetCDF output.
3. **Multi-frame runner** (`run_foundationstereo.py`): loop over all `*_wd` frames, keep the model loaded, and write a
   NetCDF file. Then `5_WASS_Validation/compare_runs.py` gives Hs time series and the frequency–wavenumber
   (dispersion relation) test, which is the strongest check available without ground truth.
4. **Photometric check**: warp the right image with each method's disparity and score the match against the left
   image. This needs no ground truth and directly tests the +0.3 px disparity offset.
5. **Speed**: ONNX/TensorRT export (both repos provide one; TensorRT needs Docker upstream).
