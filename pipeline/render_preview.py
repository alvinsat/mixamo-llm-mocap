"""Render a clip's preview video — and the side-by-side showcase
against its source plate — straight from the live Blender rig.

Usage (run with the GVHMR venv python — it has cv2/PIL/PyAV; Blender
open on your rig scene with the clip's action applied):

  tools\\GVHMR\\.venv\\Scripts\\python.exe pipeline\\render_preview.py action_specs\\<motion>.json
  tools\\GVHMR\\.venv\\Scripts\\python.exe pipeline\\render_preview.py action_specs\\<motion>.json --showcase

Outputs into the spec's clip_dir:
  preview.mp4    the retarget alone (front camera, Workbench, 30 fps)
  showcase.mp4   (--showcase) source plate left, retarget right,
                 time-aligned, labeled — the proof-of-fidelity video

The source video for --showcase is auto-discovered as the first .mp4
next to the spec's landmarks file (override with --video). Frames are
rendered through a temporary camera that is removed afterwards; the
Blender window may be occluded or minimized (offscreen render).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_exec import execute_file  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

RENDER_SEQ = """import bpy, math
from mathutils import Vector
scene = bpy.context.scene

# Bind the SPEC's action before rendering — the live scene holds
# whatever action was applied last, which may be a different clip.
arm = bpy.data.objects["Armature"]
act = bpy.data.actions.get("{action}")
assert act is not None, "action '{action}' not found in the scene - run the apply first"
if arm.animation_data is None:
    arm.animation_data_create()
arm.animation_data.action = act
try:
    for slot in act.slots:
        arm.animation_data.action_slot = slot
        break
except Exception:
    pass
scene.frame_start = 1
scene.frame_end = int(act.frame_range[1])

cam_data = bpy.data.cameras.new("QA_Camera")
cam = bpy.data.objects.new("QA_Camera", cam_data)
scene.collection.objects.link(cam)
old = (scene.camera, scene.render.engine, scene.render.filepath,
       scene.render.resolution_x, scene.render.resolution_y)
try:
    scene.camera = cam
    cam.location = Vector((0.0, -4.6, 1.05))
    cam.rotation_euler = (math.radians(90), 0.0, 0.0)
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x, scene.render.resolution_y = 960, 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = r"{outdir}" + "\\\\f####"
    bpy.ops.render.render(animation=True)
finally:
    (scene.camera, scene.render.engine, scene.render.filepath,
     scene.render.resolution_x, scene.render.resolution_y) = old
    bpy.data.objects.remove(cam)
    bpy.data.cameras.remove(cam_data)
result = {{"frames": scene.frame_end}}
"""


def rpath(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO / p)


def encode(files: list[Path], out: Path, fps: int, width: int, height: int, make_frame) -> None:
    container = av.open(str(out), "w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height = width, height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "19"}
    for i, f in enumerate(files):
        frame = av.VideoFrame.from_ndarray(make_frame(i, f), format="rgb24")
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def label(img, text):
    cv2.putText(img, text, (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--showcase", action="store_true", help="also build the side-by-side against the source plate")
    ap.add_argument("--social", action="store_true",
                    help="with --showcase: also write showcase_social.mp4, 1920x1080 letterboxed "
                         "(social platforms reject >1920 width / >2.39:1 aspect)")
    ap.add_argument("--video", help="source plate video (default: first .mp4 next to the spec's landmarks)")
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args()

    spec = json.loads(rpath(args.spec).read_text(encoding="utf-8"))
    clip_dir = rpath(spec["clip_dir"])
    clip_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = clip_dir / "preview_frames"
    frames_dir.mkdir(exist_ok=True)

    # 1. PNG sequence from the live Blender.
    code = RENDER_SEQ.format(outdir=str(frames_dir), action=spec["action_name"])
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp = f.name
    try:
        res = execute_file(tmp, 600.0)
    finally:
        Path(tmp).unlink(missing_ok=True)
    if res.get("status") != "ok":
        raise SystemExit(res.get("message", "render failed"))

    files = sorted(frames_dir.glob("f*.png"))
    if not files:
        raise SystemExit(f"no frames rendered in {frames_dir}")

    # 2. preview.mp4
    dst_fps = int(spec.get("dst_fps", 30))
    encode(files, clip_dir / "preview.mp4", dst_fps, 960, 720,
           lambda i, f: np.asarray(Image.open(f).convert("RGB")))
    print("wrote", clip_dir / "preview.mp4", f"({len(files)} frames)")

    # 3. showcase.mp4
    if args.showcase:
        video = Path(args.video) if args.video else None
        if video is None:
            plate_dir = rpath(spec["landmarks"]).parent
            mp4s = sorted(plate_dir.glob("*.mp4"))
            if not mp4s:
                raise SystemExit(f"no source .mp4 found in {plate_dir}; pass --video")
            video = mp4s[0]
        cap = cv2.VideoCapture(str(video))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or float(spec.get("src_fps", 24))
        src = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            src.append(img)
        cap.release()
        sh, sw = src[0].shape[:2]
        scale = 720.0 / sh
        sw = int(sw * scale)
        W = sw + 4 + 960

        def make(i, f):
            si = min(len(src) - 1, int(round(i / dst_fps * src_fps)))
            left = label(cv2.resize(src[si], (sw, 720)), "SOURCE")
            right = label(cv2.imread(str(f)), "MIXAMO RETARGET")
            canvas = np.hstack([left, np.full((720, 4, 3), 40, np.uint8), right])
            return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        encode(files, clip_dir / "showcase.mp4", dst_fps, W, 720, make)
        print("wrote", clip_dir / "showcase.mp4", f"(source: {video.name})")

        if args.social:
            # 16:9 letterboxed variant within platform limits.
            sh2 = int(round(720 * 1920 / W))

            def make_social(i, f):
                content = cv2.resize(
                    cv2.cvtColor(make(i, f), cv2.COLOR_RGB2BGR), (1920, sh2),
                    interpolation=cv2.INTER_AREA)
                canvas = np.zeros((1080, 1920, 3), np.uint8)
                y0 = (1080 - sh2) // 2
                canvas[y0:y0 + sh2] = content
                return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

            encode(files, clip_dir / "showcase_social.mp4", dst_fps, 1920, 1080, make_social)
            print("wrote", clip_dir / "showcase_social.mp4", "(1920x1080 letterboxed)")

    if not args.keep_frames:
        for f in files:
            f.unlink()
        frames_dir.rmdir()


if __name__ == "__main__":
    main()
