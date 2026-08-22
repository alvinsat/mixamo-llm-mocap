"""Run MotionBERT 3D lifting on RTMPose 2D JSON output."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MOTIONBERT_ROOT = REPO_ROOT / "tools" / "MotionBERT"
SOURCE = MOTIONBERT_ROOT / "source"
CHECKPOINT = MOTIONBERT_ROOT / "checkpointes" / "pose3d" / "MB_ft_h36m" / "best_epoch.bin"
CONFIG = MOTIONBERT_ROOT / "checkpointes" / "pose3d" / "MB_ft_h36m" / "MB_ft_h36m.yaml"

if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from lib.utils.tools import get_config  # noqa: E402
from lib.utils.learning import load_backbone  # noqa: E402


# RTMPose/Open Model Zoo order: nose, neck, right body, left body, face.
# MotionBERT Human3.6M order: pelvis, right hip, right knee, right foot,
# left hip, left knee, left foot, spine, thorax, neck, head, left shoulder,
# left elbow, left wrist, right shoulder, right elbow, right wrist.
RTMPOSE_TO_H36M = [
    1, 9, 10, 11, 12, 13, 14, 1, 1, 1, 0, 6, 7, 8, 3, 4, 5
]


def load_rtmpose_json(path):
    """Load RTMPose mocap_data.json or AlphaPose-style detection JSON."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "frames" in payload:
        tracks = [frame["image"]["keypoints_2d"] for frame in payload["frames"]]
        tracks = np.asarray(tracks, dtype=np.float32)
        if payload["frames"] and payload["frames"][0]["image"].get("keypoint_space") != "image_pixels":
            valid_x = tracks[:, :, 0] >= 0
            valid_y = tracks[:, :, 1] >= 0
            heatmap_width = max(1.0, float(tracks[:, :, 0][valid_x].max() + 1.0))
            heatmap_height = max(1.0, float(tracks[:, :, 1][valid_y].max() + 1.0))
            width, height = payload.get("metadata", {}).get("resolution", [864, 1080])
            scale = min(448.0 / width, 256.0 / height)
            left = (448.0 - width * scale) * 0.5
            top = (256.0 - height * scale) * 0.5
            valid = valid_x & valid_y
            tracks[valid, 0] = (tracks[valid, 0] + 0.5) * 448.0 / heatmap_width
            tracks[valid, 1] = (tracks[valid, 1] + 0.5) * 256.0 / heatmap_height
            tracks[valid, 0] = (tracks[valid, 0] - left) / scale
            tracks[valid, 1] = (tracks[valid, 1] - top) / scale
        return tracks

    detections = []
    for item in payload:
        keypoints = np.asarray(item["keypoints"], dtype=np.float32).reshape(-1, 3)
        detections.append(keypoints)
    return np.asarray(detections, dtype=np.float32)


def to_h36m(keypoints):
    """Convert 19-joint RTMPose points into MotionBERT's 17-joint layout."""
    if keypoints.ndim != 3 or keypoints.shape[1] != 19 or keypoints.shape[2] != 3:
        raise ValueError("Expected RTMPose keypoints shaped as (frames, 19, 3)")

    output = keypoints[:, RTMPOSE_TO_H36M, :].copy()
    output[:, 0, :2] = (keypoints[:, 9, :2] + keypoints[:, 12, :2]) * 0.5
    output[:, 7, :2] = (keypoints[:, 1, :2] + output[:, 0, :2]) * 0.5
    output[:, 8, :2] = keypoints[:, 1, :2]
    output[:, 9, :2] = keypoints[:, 1, :2]
    output[:, 10, :2] = keypoints[:, 0, :2]
    return output


def normalize(keypoints, pixel_size):
    width, height = pixel_size
    scale = min(width, height) / 2.0
    normalized = keypoints.copy()
    normalized[:, :, :2] -= np.asarray([width, height], dtype=np.float32) / 2.0
    normalized[:, :, :2] /= scale
    return normalized


def run(input_json, output_npy, checkpoint=CHECKPOINT, config=CONFIG, pixel_size=None):
    device = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    keypoints = to_h36m(load_rtmpose_json(input_json))
    if pixel_size is None:
        raise ValueError("--width and --height are required for coordinate normalization")
    inputs = normalize(keypoints, pixel_size)

    args = get_config(str(config))
    model = load_backbone(args).to(device).eval()
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    state_dict = checkpoint_data.get("model_pos", checkpoint_data)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=True)

    maxlen = int(getattr(args, "maxlen", 243))
    sequence = torch.from_numpy(inputs)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(sequence), maxlen):
            chunk = sequence[start:start + maxlen].unsqueeze(0).to(device)
            prediction = model(chunk)
            if getattr(args, "rootrel", False):
                prediction[:, :, 0, :] = 0
            predictions.append(prediction.squeeze(0).cpu())
    result = torch.cat(predictions, dim=0).numpy()
    np.save(output_npy, result)
    return result


def preview_skeleton(result, interval_ms=33):
    """Show a lightweight animated 3D joint preview; no mesh is generated."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    connections = [
        (0, 7), (7, 8), (8, 10), (8, 9), (9, 11), (11, 12), (12, 13),
        (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 5), (5, 6),
    ]
    figure = plt.figure("MotionBERT 3D preview")
    axis = figure.add_subplot(111, projection="3d")
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)
    axis.set_zlim(-1, 1)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    points = axis.scatter([], [], [], c="crimson")
    lines = [axis.plot([], [], [], c="black")[0] for _ in connections]

    def update(frame_id):
        pose = result[frame_id]
        points._offsets3d = (pose[:, 0], pose[:, 1], pose[:, 2])
        for line, (start, end) in zip(lines, connections):
            line.set_data([pose[start, 0], pose[end, 0]], [pose[start, 1], pose[end, 1]])
            line.set_3d_properties([pose[start, 2], pose[end, 2]])
        axis.set_title(f"MotionBERT frame {frame_id + 1}/{len(result)}")
        return [points, *lines]

    animation = FuncAnimation(figure, update, frames=len(result), interval=interval_ms, blit=False)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="RTMPose mocap_data.json")
    parser.add_argument("--output", required=True, type=Path, help="Output MotionBERT .npy file")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--preview", action="store_true", help="Show an animated 3D skeleton after inference")
    options = parser.parse_args()
    result = run(
        options.input,
        options.output,
        options.checkpoint,
        options.config,
        (options.width, options.height),
    )
    print(f"saved {options.output} with shape {result.shape} on XPU/CPU")
    if options.preview:
        preview_skeleton(result)


if __name__ == "__main__":
    main()
