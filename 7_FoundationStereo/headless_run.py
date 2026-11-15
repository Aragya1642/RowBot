"""Run an upstream demo script without a display.

The upstream demos call cv2.imshow / cv2.waitKey and open an Open3D window, which fail on this headless WSL
setup (opencv is the headless build, and Open3D has no EGL). This stubs those calls out, then runs the
script with its own arguments, so the submodules stay unmodified:

    python headless_run.py Fast-FoundationStereo/scripts/run_demo.py --out_dir output/demo_ffs ...
"""
import os
import runpy
import sys

# Fast-FoundationStereo decorates a few functions with @torch.compile, whose Inductor backend needs Triton,
# and Triton needs compute capability >= 7.0. The GTX 1060 is 6.1, so run those functions eagerly (same maths).
# Set TORCHDYNAMO_DISABLE=0 to override on a newer GPU.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import cv2  # noqa: E402

cv2.imshow = lambda *a, **k: None
cv2.waitKey = lambda *a, **k: -1
try:
    import open3d as o3d

    class _NoWindow:
        def __getattr__(self, name):
            return lambda *a, **k: _NoWindow()

    o3d.visualization.Visualizer = lambda *a, **k: _NoWindow()
    o3d.visualization.draw_geometries = lambda *a, **k: None
except ImportError:
    pass

if len(sys.argv) < 2:
    sys.exit(__doc__)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
