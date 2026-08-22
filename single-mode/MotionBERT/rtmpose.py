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


def _show_pose(pose, keypoints_2d, video_frame, frame_id, total_frames, window="MotionBERT camera-front preview"):
    import cv2

    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 0:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 720, 720)
    canvas = video_frame.copy() if video_frame is not None else np.zeros((720, 720, 3), dtype=np.uint8)
    connections = [
        (0, 7), (7, 8), (8, 10), (8, 9), (9, 11), (11, 12), (12, 13),
        (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 5), (5, 6),
    ]
    height, width = canvas.shape[:2]
    # MotionBERT uses X/Y for the front camera plane; Z is depth.
    points = pose[:, [0, 1]]
    reference = None
    if keypoints_2d is not None:
        reference = to_h36m(keypoints_2d[None])[0][:, :2]
    if reference is not None:
        valid = keypoints_2d[:, 2] > 0
        reference_scale = max(float(np.ptp(reference[valid[:17], 1])), 1.0)
        pose_scale = max(float(np.ptp(points[:, 1])), 1e-4)
        scale = reference_scale / pose_scale
        centre = points[0]
        anchor = reference[0]
    else:
        scale = min(width, height) * 0.42 / max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1e-4)
        centre = points[0]
        anchor = np.asarray([width * 0.5, height * 0.62])

    def pixel(point):
        x = int(round((point[0] - centre[0]) * scale + anchor[0]))
        y = int(round(anchor[1] + (point[1] - centre[1]) * scale))
        return x, y

    for start, end in connections:
        cv2.line(canvas, pixel(points[start]), pixel(points[end]), (255, 180, 40), 3, cv2.LINE_AA)
    for point in points:
        cv2.circle(canvas, pixel(point), 5, (40, 220, 255), -1, cv2.LINE_AA)

    if keypoints_2d is not None:
        input_points = reference
        for start, end in connections:
            if min(keypoints_2d[start, 2], keypoints_2d[end, 2]) > 0:
                cv2.line(canvas, tuple(np.rint(input_points[start]).astype(int)),
                         tuple(np.rint(input_points[end]).astype(int)), (70, 220, 70), 2, cv2.LINE_AA)
        for point, confidence in zip(input_points, keypoints_2d[:, 2]):
            if confidence > 0:
                cv2.circle(canvas, tuple(np.rint(point).astype(int)), 4, (70, 220, 70), -1, cv2.LINE_AA)

    cv2.putText(canvas, f"Camera front | BERT {frame_id + 1}/{total_frames}", (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 240, 245), 2, cv2.LINE_AA)
    cv2.putText(canvas, "GREEN: RTMPOSE  |  ORANGE: MOTIONBERT", (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 240, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, "SPACE: pause/play   R: restart   Q/ESC: close", (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 240, 245), 1, cv2.LINE_AA)
    cv2.imshow(window, canvas)
    return (cv2.waitKey(30) & 0xFF) not in (27, ord("q"))


def playback_preview(result, keypoints, video_path, fps=30.0):
    """Play the completed BERT result with simple media controls."""
    import cv2

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        print(f"Preview video could not be opened: {video_path}", flush=True)
        return
    window = "MotionBERT camera-front preview"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 720, 720)
    paused = False
    frame_id = 0
    delay = max(1, int(1000.0 / max(float(fps), 1.0)))
    print("Preview controls: Space pause/play, R restart, Q or Esc close.", flush=True)
    while True:
        if not paused:
            ok, video_frame = video.read()
            if not ok or frame_id >= len(result):
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_id = 0
                continue
            _show_pose(result[frame_id], keypoints[frame_id], video_frame, frame_id, len(result))
            frame_id += 1
        key = cv2.waitKey(delay if not paused else 100) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("r"):
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_id = 0
            paused = False
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
    video.release()
    cv2.destroyAllWindows()


def run(input_json, output_npy, checkpoint=CHECKPOINT, config=CONFIG, pixel_size=None, preview=False, video_path=None):
    device = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    source_keypoints = load_rtmpose_json(input_json)
    keypoints = to_h36m(source_keypoints)
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
    preview_open = False
    video = None
    video_index = -1
    if preview and video_path:
        import cv2
        video = cv2.VideoCapture(str(video_path))
        cv2.namedWindow("MotionBERT camera-front preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("MotionBERT camera-front preview", 720, 720)
        ok, first_frame = video.read()
        if ok:
            blank_pose = np.zeros((17, 3), dtype=np.float32)
            _show_pose(blank_pose, None, first_frame, 0, len(sequence))
            video_index = 0
        print("MotionBERT preview window opened; processing is in progress.", flush=True)
    with torch.no_grad():
        for start in range(0, len(sequence), maxlen):
            chunk = sequence[start:start + maxlen].unsqueeze(0).to(device)
            prediction = model(chunk)
            if getattr(args, "rootrel", False):
                prediction[:, :, 0, :] = 0
            if preview:
                preview_open = True
                for offset, pose in enumerate(prediction.squeeze(0).cpu().numpy()):
                    if (start + offset) % 5 == 0 or start + offset == len(sequence) - 1:
                        video_frame = None
                        if video is not None:
                            target_frame = start + offset
                            while video_index < target_frame:
                                ok, video_frame = video.read()
                                video_index += 1
                                if not ok:
                                    video_frame = None
                                    break
                        if not _show_pose(pose, keypoints[start + offset], video_frame, start + offset, len(sequence)):
                            preview = False
                            break
            predictions.append(prediction.squeeze(0).cpu())
    if preview_open:
        import cv2
        if video is not None:
            video.release()
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    result = torch.cat(predictions, dim=0).numpy()
    np.save(output_npy, result)
    if preview and video_path:
        playback_preview(result, source_keypoints, video_path, fps=30.0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="RTMPose mocap_data.json")
    parser.add_argument("--output", required=True, type=Path, help="Output MotionBERT .npy file")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--preview", action="store_true", help="Show a fast OpenCV skeleton during inference")
    parser.add_argument("--video", type=Path, help="Source video to show behind the camera-front preview")
    options = parser.parse_args()
    result = run(
        options.input,
        options.output,
        options.checkpoint,
        options.config,
        (options.width, options.height),
        options.preview,
        options.video,
    )
    print(f"saved {options.output} with shape {result.shape} on XPU/CPU")
    if options.preview:
        print("Live OpenCV preview finished")


if __name__ == "__main__":
    main()
