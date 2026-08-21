# Install

Everything the pipeline needs, from a bare Windows machine to a first
retarget. The Intel XPU path was validated on Windows with an Intel Arc
B580, `torch 2.13.0+xpu`, `torchvision 0.28.0+xpu`, Blender 5.1.2, and
Python 3.13. NVIDIA CUDA remains supported by the upstream GVHMR setup;
this fork's Windows requirements target Intel XPU.

## 0. Prerequisites

| Thing | Version | Notes |
|---|---|---|
| Intel Arc GPU + current driver | ~8 GB VRAM min | XPU PyTorch wheels are used |
| Blender | **5.1+** | the Blender MCP add-on requires it |
| Python | any (for `uv`) | the GVHMR venv is created as 3.10 |
| [uv](https://docs.astral.sh/uv/) | recent | manages the 3.10 venv painlessly |
| git, ffmpeg | recent | |

## 1. Mixamo character (Adobe — bring your own)

Any Mixamo character with the standard `mixamorig` skeleton works;
Y Bot is the reference the pipeline was developed on.

1. Log in at https://www.mixamo.com → Characters → pick a character.
2. Download: Format **FBX Binary**, Pose **T-pose** (with skin).
3. Save as `<repo>\ybot.fbx` (any name — pass it via `--fbx`).
4. Build + validate the working scene — this also writes
   `rig_profile.json`, the measured proportions every later stage uses:

```
"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" ^
    --background --python pipeline\setup_rig.py -- --fbx ybot.fbx --out ybot_rest.blend
```

Expected output ends with `rig_profile.json written — hip ... ground ...`
and `saved ... bones ... 30 fps`. If validation fails, the FBX is not a
plain Mixamo T-pose character download.

## 2. Blender MCP (official, Blender Lab)

The FK apply runs inside a live Blender through the official add-on's
bridge socket (`localhost:9876`).

1. In Blender: **Edit → Preferences → System → Network → Allow Online
   Access** (off by default — the server refuses to start without it).
2. Install the add-on from https://www.blender.org/lab/mcp-server/ —
   drag the install link into Blender **twice** (first adds the Blender
   Lab extension repository, second installs), or download the zip and
   use *Install from Disk*.
3. The server autostarts ~1 s after Blender launches (check
   Preferences → Add-ons → MCP: "Server is running").

For an **agent driving Blender**: either register the MCP server with
your client —

```
claude mcp add blender -s user -- uvx --from "git+https://projects.blender.org/lab/blender_mcp.git#subdirectory=mcp" --with "mcp[cli]<2" blender-mcp
```

(the `mcp[cli]<2` pin matters: the 2.x Python SDK removed
`mcp.server.fastmcp`, which the server imports) — or skip MCP entirely
and use `pipeline\blender_exec.py <script.py> [timeout]`, which speaks
the add-on's socket protocol directly.

## 3. GVHMR (the estimator)

```
git clone https://github.com/alvinsat/GVHMR tools/GVHMR
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r docs/requirements_gvhmr_windows.txt
.venv\Scripts\python.exe -m pip install --no-deps -e tools/GVHMR
.venv\Scripts\python.exe -m pip install yacs
```

`docs/requirements_gvhmr_windows.txt` is GVHMR's requirements with
Windows fixes baked in. It installs the tested native Intel XPU wheels;
do not replace them with CUDA wheels. `pytorch3d`, `chumpy`, and
`cython_bbox` are omitted because this adapter does not use those paths.

**Checkpoints** (~5.3 GB). Google Drive blocks scripted downloads;
the HuggingFace mirror works:

```
BASE=https://huggingface.co/camenduru/GVHMR/resolve/main
CK=tools/GVHMR/inputs/checkpoints
curl -L $BASE/gvhmr/gvhmr_siga24_release.ckpt   -o $CK/gvhmr/gvhmr_siga24_release.ckpt --create-dirs
curl -L $BASE/yolo/yolov8x.pt                   -o $CK/yolo/yolov8x.pt --create-dirs
curl -L $BASE/vitpose/vitpose-h-multi-coco.pth  -o $CK/vitpose/vitpose-h-multi-coco.pth --create-dirs
curl -L "$BASE/hmr2/epoch%3D10-step%3D25000.ckpt" -o "$CK/hmr2/epoch=10-step=25000.ckpt" --create-dirs
```

(dpvo.pth is only needed for moving cameras; plates are locked.)

**SMPL-X body model** (license-gated, free registration): register at
https://smpl-x.is.tue.mpg.de/ , download **SMPL-X v1.1 (NPZ+PKL)** and
copy `SMPLX_NEUTRAL.npz` to
`tools/GVHMR/inputs/checkpoints/body_models/smplx/`. GVHMR's network
asserts on this file at startup — nothing runs without it.

**Smoke test** (must print `xpu` and your Intel GPU without tracebacks):

```
.venv\Scripts\python.exe -c "import torch; assert torch.xpu.is_available(); print(torch.__version__, torch.xpu.get_device_name(0)); from hmr4d.utils.smplx_utils import make_smplx; from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL; from hmr4d.utils.preproc import Tracker; print('GVHMR XPU import chain OK')"
```

If a different GVHMR root is preferred, set the env var `GVHMR_ROOT`.

## 4. Verify end-to-end

With Blender open on `ybot_rest.blend`, run your first plate through
the five stages (README quickstart step 3 / docs/PIPELINE.md). Sanity
reference values from the author's accepted clips:

- estimator bind wrist span on a T-pose: ~1.25–1.44 m
- QA gate: 0 world-bound violations, max hip Z step well under
  0.12 m/frame (accepted clips: 0.026–0.061), single-support foot
  skate 0.000 m, end-frame rest hand error ~0.001 m, verdict PASS.
