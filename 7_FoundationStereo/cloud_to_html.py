#!/usr/bin/env python3
"""Export point clouds (.ply) to a single self-contained HTML file with an interactive 3D viewer.

    python cloud_to_html.py output/demo_vits/cloud_denoise.ply                 # -> output/demo_vits/cloud_denoise.html
    python cloud_to_html.py A.ply B.ply --out both.html --open                  # two layers, toggled in the viewer
    python cloud_to_html.py sea.ply --frame zup                                 # x, y horizontal, z up (sea-plane frame)

Open the HTML in any browser (from Windows: \\\\wsl.localhost\\<distro>\\home\\..., or use --open). No server is
needed; the points are embedded in the file and three.js is loaded from a CDN.

Controls: left-drag orbit, right-drag (or shift-drag) pan, wheel zoom, double-click a point to orbit around it.
The panel switches layers on/off, colouring (image RGB / distance / height), point size, and preset views.

Frames (--frame):
  opencv  camera frame, x right, y down, z forward (FoundationStereo run_demo.py output). Start view = the camera.
  zup     x, y horizontal, z up, e.g. WASS sea-plane coordinates. Start view = --eye if given, else oblique.

Also importable: write_viewer(out, layers, ...) with layers = [(name, Nx3 points, Nx3 colours in 0..1 or None)], each
optionally with a 4th element: the layer's colour for "Layer colour" mode ("#rrggbb" or RGB in 0..1).
"""

import argparse
import base64
import json
import subprocess
from pathlib import Path

import numpy as np

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg:#16181d; --panel:rgba(30,33,40,.9); --fg:#e6e8ec; --muted:#9aa1ad; --accent:#5aa9ff; }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--fg); font:13px/1.4 system-ui,sans-serif; overflow:hidden; }
  #panel { position:absolute; top:12px; left:12px; background:var(--panel); padding:10px 12px; border-radius:8px;
           width:290px; max-width:calc(100vw - 48px); box-shadow:0 2px 10px rgba(0,0,0,.4); }
  #panel h1 { font-size:14px; margin:0 0 4px; word-break:break-all; }
  #panel .muted { color:var(--muted); font-size:12px; }
  #panel label { display:flex; align-items:center; gap:8px; margin-top:6px; }
  #panel select, #panel input[type=range] { flex:1; min-width:0; }
  #panel button { margin-top:8px; margin-right:4px; background:#2b3140; color:var(--fg); border:1px solid #3c4456;
                  border-radius:5px; padding:4px 8px; cursor:pointer; }
  #panel button:hover { border-color:var(--accent); }
  #layers { margin-top:6px; border-top:1px solid #3c4456; padding-top:4px; }
  .swatch { width:10px; height:10px; border-radius:2px; display:inline-block; }
  #legend { margin-top:8px; display:none; }
  #legend .bar { height:10px; border-radius:3px; }
  #legend .ticks { display:flex; justify-content:space-between; color:var(--muted); font-size:11px; }
  #help { position:absolute; bottom:10px; left:12px; right:12px; color:var(--muted); font-size:12px; }
</style>
</head>
<body>
<div id="panel">
  <h1>__TITLE__</h1>
  <div class="muted">__SUBTITLE__</div>
  <div id="layers"></div>
  <label>Colour <select id="mode">
    <option value="rgb">Image</option>
    <option value="layer">Layer colour</option>
    <option value="height">Height (up axis)</option>
    <option value="depth">Distance from start view</option>
  </select></label>
  <label>Point size <input id="size" type="range" min="0.5" max="8" step="0.25" value="__PSIZE__"><span id="sizev"></span></label>
  <label>Vertical exaggeration <input id="vex" type="range" min="1" max="10" step="0.5" value="__VEX__"><span id="vexv"></span></label>
  <label><input id="axes" type="checkbox" checked> Axes / camera marker</label>
  <div id="legend"><div class="bar" id="bar"></div><div class="ticks"><span id="lo"></span><span id="hi"></span></div></div>
  <button id="reset">Start view</button><button id="top">Top</button><button id="side">Side</button>
</div>
<div id="help">Left-drag: orbit · Right/Shift-drag: pan · Wheel: zoom · Double-click: orbit around point</div>

<script type="importmap">
{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
} }
</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const DATA = __DATA__;
const b64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
const LAYER_COLOURS = [[0.35, 0.66, 1.0], [1.0, 0.55, 0.25], [0.45, 0.85, 0.45], [0.85, 0.45, 0.85],
                       [0.95, 0.85, 0.3], [0.6, 0.6, 0.6]];

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x16181d);

const box = new THREE.Box3();
const layers = DATA.layers.map((L, k) => {
  const pos = new Float32Array(b64(L.pos).buffer);   // viewer frame (y up)
  const rgb = b64(L.rgb);
  const n = pos.length / 3;
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geom.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  geom.computeBoundingBox();
  box.union(geom.boundingBox);
  const mat = new THREE.PointsMaterial({ size: 1, vertexColors: true, sizeAttenuation: false });
  const pts = new THREE.Points(geom, mat);
  scene.add(pts);
  return { name: L.name, total: L.total, pos, rgb, n, geom, mat, pts, colour: L.colour || LAYER_COLOURS[k % LAYER_COLOURS.length] };
});

const layersEl = document.getElementById("layers");
layers.forEach(L => {
  const lab = document.createElement("label");
  const cb = document.createElement("input");
  cb.type = "checkbox"; cb.checked = true;
  cb.onchange = () => { L.pts.visible = cb.checked; };
  const sw = document.createElement("span");
  sw.className = "swatch"; sw.style.background = `rgb(${L.colour.map(c => c * 255 | 0)})`;
  const txt = document.createElement("span");
  txt.textContent = `${L.name}: ${L.n.toLocaleString()} pts` + (L.total > L.n ? ` (of ${L.total.toLocaleString()})` : "");
  lab.append(cb, sw, txt);
  layersEl.append(lab);
});

const extent = box.getSize(new THREE.Vector3()).length();
const center = new THREE.Vector3(...DATA.center);
const eye = DATA.eye ? new THREE.Vector3(...DATA.eye) : null;

const helpers = new THREE.Group();
helpers.add(new THREE.AxesHelper(extent * 0.05));
if (DATA.frame === "opencv") {  // camera frustum at the origin, looking along -z
  const s = extent * 0.03, g = new THREE.BufferGeometry();
  const c = [[-1, .6, -1.5], [1, .6, -1.5], [1, -.6, -1.5], [-1, -.6, -1.5]].map(v => v.map(x => x * s));
  const v = [];
  for (const p of c) v.push(0, 0, 0, ...p);
  for (let i = 0; i < 4; i++) v.push(...c[i], ...c[(i + 1) % 4]);
  g.setAttribute("position", new THREE.Float32BufferAttribute(v, 3));
  helpers.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0xffcc33 })));
} else if (eye) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(extent * 0.006), new THREE.MeshBasicMaterial({ color: 0xffcc33 }));
  m.position.copy(eye);
  helpers.add(m);
}
scene.add(helpers);

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, extent * 1e-4, extent * 20);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.screenSpacePanning = true;

function startPos() {
  if (DATA.frame === "opencv") return new THREE.Vector3(0, 0, 0.001 * extent);
  if (eye) return eye.clone();
  return center.clone().add(new THREE.Vector3(0, extent * 0.35, extent * 0.6));
}
function view(kind) {
  const d = extent * 0.8;
  if (kind === "start") camera.position.copy(startPos());
  else if (kind === "top") camera.position.set(center.x, center.y + d, center.z + 1e-3);
  else camera.position.set(center.x + d, center.y + d * 0.1, center.z);
  controls.target.copy(center);
  camera.up.set(0, 1, 0);
  controls.update();
}

// Turbo colormap (polynomial approximation)
function turbo(t) {
  t = Math.min(1, Math.max(0, t));
  const r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  const g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  const b = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  return [r, g, b].map(x => Math.min(1, Math.max(0, x)));
}

function colour(mode) {
  const legend = document.getElementById("legend");
  const s = startPos();
  const val = (L, i) => mode === "height" ? L.pos[3 * i + 1]
    : Math.hypot(L.pos[3 * i] - s.x, L.pos[3 * i + 1] - s.y, L.pos[3 * i + 2] - s.z);
  let lo = 0, hi = 1;
  if (mode === "height" || mode === "depth") {  // shared 1-99 % range over all layers
    const sample = [];
    for (const L of layers) for (let i = 0; i < L.n; i += Math.max(1, L.n / 20000 | 0)) sample.push(val(L, i));
    sample.sort((a, b) => a - b);
    lo = sample[Math.floor(0.01 * (sample.length - 1))]; hi = sample[Math.floor(0.99 * (sample.length - 1))];
  }
  for (const L of layers) {
    const col = L.geom.attributes.color.array;
    for (let i = 0; i < L.n; i++) {
      if (mode === "rgb") { col[3 * i] = L.rgb[3 * i] / 255; col[3 * i + 1] = L.rgb[3 * i + 1] / 255; col[3 * i + 2] = L.rgb[3 * i + 2] / 255; }
      else if (mode === "layer") col.set(L.colour, 3 * i);
      else col.set(turbo((val(L, i) - lo) / (hi - lo)), 3 * i);
    }
    L.geom.attributes.color.needsUpdate = true;
  }
  const ranged = mode === "height" || mode === "depth";
  legend.style.display = ranged ? "block" : "none";
  if (ranged) {
    document.getElementById("bar").style.background =
      "linear-gradient(to right," + [0, .2, .4, .6, .8, 1].map(t => `rgb(${turbo(t).map(c => c * 255 | 0)})`).join(",") + ")";
    document.getElementById("lo").textContent = lo.toFixed(2) + " " + DATA.units;
    document.getElementById("hi").textContent = hi.toFixed(2) + " " + DATA.units;
  }
}

const sizeEl = document.getElementById("size");
const setSize = () => {
  for (const L of layers) L.mat.size = +sizeEl.value * renderer.getPixelRatio();
  document.getElementById("sizev").textContent = sizeEl.value;
};
sizeEl.oninput = setSize; setSize();
const vexEl = document.getElementById("vex");
const setVex = () => {  // stretch the up axis about the cloud centre (colours keep the true heights)
  const k = +vexEl.value;
  for (const L of layers) { L.pts.scale.set(1, k, 1); L.pts.position.set(0, center.y * (1 - k), 0); }
  document.getElementById("vexv").textContent = k + "x";
};
vexEl.oninput = setVex; setVex();
document.getElementById("mode").onchange = e => colour(e.target.value);
document.getElementById("axes").onchange = e => helpers.visible = e.target.checked;
document.getElementById("reset").onclick = () => view("start");
document.getElementById("top").onclick = () => view("top");
document.getElementById("side").onclick = () => view("side");

const ray = new THREE.Raycaster();
renderer.domElement.addEventListener("dblclick", e => {
  ray.params.Points.threshold = extent * 0.003;
  ray.setFromCamera(new THREE.Vector2(e.clientX / innerWidth * 2 - 1, -e.clientY / innerHeight * 2 + 1), camera);
  const hit = ray.intersectObjects(layers.filter(L => L.pts.visible).map(L => L.pts))[0];
  if (hit) { controls.target.copy(hit.point); controls.update(); }
});

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

document.getElementById("mode").value = DATA.mode;
colour(DATA.mode);
view("start");
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
</script>
</body>
</html>
"""


def to_viewer(pts, frame):
    """Input frame -> viewer frame (x right, y up, z towards the viewer)."""
    pts = np.asarray(pts, dtype=np.float64)
    if frame == "opencv":  # x right, y down, z forward
        return pts * np.array([1, -1, -1])
    if frame == "zup":  # x, y horizontal, z up
        return np.stack([pts[:, 0], pts[:, 2], -pts[:, 1]], axis=1)
    raise ValueError(frame)


def write_viewer(out, layers, frame="opencv", eye=None, title=None, subtitle="", units="m",
                 max_points=400_000, point_size=1.5, mode="rgb", vex=1.0, seed=0):
    """layers: list of (name, Nx3 points in `frame`, Nx3 colours in 0..1 or None). max_points is per layer."""
    out = Path(out)
    rng = np.random.default_rng(seed)
    data_layers, allpts = [], []
    for name, pts, cols, *lc in layers:
        pts = np.asarray(pts)
        total = len(pts)
        if cols is None:
            cols = np.full((total, 3), 0.8)
        if total > max_points:
            idx = rng.choice(total, max_points, replace=False)
            pts, cols = pts[idx], np.asarray(cols)[idx]
        v = to_viewer(pts, frame).astype(np.float32)
        allpts.append(v)
        data_layers.append({
            "name": name, "total": int(total),
            "pos": base64.b64encode(v.tobytes()).decode(),
            "rgb": base64.b64encode((np.clip(cols, 0, 1) * 255).astype(np.uint8).tobytes()).decode(),
        })
        if lc and lc[0] is not None:
            c = lc[0]
            if isinstance(c, str):
                c = [int(c.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
            data_layers[-1]["colour"] = [float(x) for x in c]
    data = {
        "layers": data_layers,
        "center": np.median(np.vstack(allpts), axis=0).tolist(),
        "eye": None if eye is None else to_viewer(np.atleast_2d(eye), frame)[0].tolist(),
        "frame": frame, "units": units, "mode": mode,
    }
    html = (TEMPLATE.replace("__TITLE__", title or out.stem)
            .replace("__SUBTITLE__", subtitle)
            .replace("__PSIZE__", str(point_size))
            .replace("__VEX__", str(vex))
            .replace("__DATA__", json.dumps(data)))
    out.write_text(html)
    return out


def open_in_windows(path):
    win = subprocess.run(["wslpath", "-w", str(Path(path).resolve())], capture_output=True, text=True).stdout.strip()
    subprocess.run(["cmd.exe", "/c", "start", "", win], cwd="/mnt/c", stderr=subprocess.DEVNULL)


def main():
    import open3d as o3d

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply", type=Path, nargs="+", help="one or more .ply files; each becomes a layer")
    ap.add_argument("--out", type=Path, help="default: first PLY with .html extension")
    ap.add_argument("--frame", choices=["opencv", "zup"], default="opencv")
    ap.add_argument("--eye", type=float, nargs=3, help="start-view position (zup frame), in the PLY's units")
    ap.add_argument("--max_points", type=int, default=400_000,
                    help="per layer: randomly subsample to this many points (file size ~20 bytes/point)")
    ap.add_argument("--units", default="m")
    ap.add_argument("--point_size", type=float, default=1.5)
    ap.add_argument("--open", action="store_true", help="open in the Windows default browser (WSL)")
    args = ap.parse_args()

    layers = []
    for p in args.ply:
        pcd = o3d.io.read_point_cloud(str(p))
        if len(pcd.points) == 0:
            raise SystemExit(f"{p} has no points")
        layers.append((p.stem, np.asarray(pcd.points), np.asarray(pcd.colors) if pcd.has_colors() else None))
    out = write_viewer(args.out or args.ply[0].with_suffix(".html"), layers, frame=args.frame, eye=args.eye,
                       title=" + ".join(p.name for p in args.ply), units=args.units,
                       max_points=args.max_points, point_size=args.point_size)
    n = sum(min(len(l[1]), args.max_points) for l in layers)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {n:,} points in {len(layers)} layer(s))")
    if args.open:
        open_in_windows(out)


if __name__ == "__main__":
    main()
