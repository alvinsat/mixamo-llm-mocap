import os
import json
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import openvino as ov
from scipy.interpolate import CubicSpline
import logging
import tkinter as tk
from tkinter import filedialog
import argparse

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("OpenVINO_3DPose")

# --- CONFIGURATION ---
MODEL_PATH = r"C:\Users\User\Documents\GitHub\openvino-anim-2-pose\local-anim-pre\models\human-pose-estimation-3d-0001.xml"
OUTPUT_JSON = r"mocap_data.json"
SHOW_REALTIME_3D = True      # Set to False for maximum batch export speed
RENDER_EVERY_N_FRAMES = 1   # Increase (e.g., 2 or 3) to skip UI frames for higher FPS

# VIEW SELECTION CONFIGURATION
# Available options: "Front", "Left", "Right", "Top", "Bottom", "3D_Free"
# Use ["Front"] for maximum speed (~25+ FPS). Use ["ALL"] for the full 6-grid (~3 FPS).
SELECTED_VIEWS = ["Front"]

# Panoptic 19-Joint Kinematic Map
PANOPTIC_LIMBS = [
    [18, 17, 1],   # Head/Face
    [16, 15, 1],   # Eyes
    [5, 4, 3],     # Left Arm: Wrist -> Elbow -> Shoulder
    [8, 7, 6],     # Left Leg: Ankle -> Knee -> Hip
    [11, 10, 9],   # Right Arm: Wrist -> Elbow -> Shoulder
    [14, 13, 12]   # Right Leg: Ankle -> Knee -> Hip
]

BONE_CONNECTIONS = {
    "mixamorig:LeftArm": (6, 7),
    "mixamorig:LeftForeArm": (7, 8),
    "mixamorig:RightArm": (3, 4),
    "mixamorig:RightForeArm": (4, 5),
    "mixamorig:LeftUpLeg": (12, 13),
    "mixamorig:LeftLeg": (13, 14),
    "mixamorig:RightUpLeg": (9, 10),
    "mixamorig:RightLeg": (10, 11)
}

OV_JOINTS = {
    "nose": 0, "neck": 1,
    "left_shoulder": 6, "left_elbow": 7, "left_wrist": 8,
    "left_hip": 12, "left_knee": 13, "left_ankle": 14,
    "right_shoulder": 3, "right_elbow": 4, "right_wrist": 5,
    "right_hip": 9, "right_knee": 10, "right_ankle": 11,
    "right_eye": 15, "left_eye": 16,
    "right_ear": 17, "left_ear": 18,
}

OV_TO_PANOPTIC = [1, 0, 9, 10, 11, 3, 4, 5, 12, 13, 14, 6, 7, 8, 15, 16, 17, 18]


def ov_to_world(point_mm, hip_mm):
    """Convert Open Model Zoo centimetres into centred pipeline metres."""
    point = (np.asarray(point_mm, dtype=np.float64) - hip_mm) / 100.0
    return point


def pipeline_landmarks(joints_mm, hip_mm):
    """Build the named landmark contract consumed by lift_to_mixamo.py."""
    points = {name: ov_to_world(joints_mm[index], hip_mm)
              for name, index in OV_JOINTS.items()}
    points["pelvis"] = (points["left_hip"] + points["right_hip"]) * 0.5

    for side in ("left", "right"):
        points[f"{side}_index"] = points[f"{side}_wrist"].copy()
        points[f"{side}_heel"] = points[f"{side}_ankle"].copy()
        points[f"{side}_foot_index"] = points[f"{side}_ankle"].copy()

    return {
        name: {"x": float(value[0]), "y": float(value[1]), "z": float(value[2]),
               "visibility": 1.0, "presence": 1.0}
        for name, value in points.items()
    }


def filter_2d_tracks(tracks, confidence_threshold=0.1, max_gap=12):
    """Fill short low-confidence joint gaps using neighboring detections."""
    filtered = np.asarray(tracks, dtype=np.float64).copy()
    if filtered.ndim != 3 or filtered.shape[2] != 3:
        raise ValueError("Expected 2D tracks shaped as (frames, joints, 3)")

    for joint_id in range(filtered.shape[1]):
        confidence = filtered[:, joint_id, 2]
        valid = np.isfinite(confidence) & (confidence >= confidence_threshold)
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size < 2:
            continue

        for gap_start, gap_end in zip(valid_indices[:-1], valid_indices[1:]):
            gap_length = gap_end - gap_start - 1
            if gap_length <= 0 or gap_length > max_gap:
                continue

            gap_indices = np.arange(gap_start + 1, gap_end)
            support_start = max(0, gap_start - 2)
            support_end = min(filtered.shape[0], gap_end + 3)
            support_indices = np.flatnonzero(valid[support_start:support_end]) + support_start
            if support_indices.size >= 3:
                for axis in (0, 1):
                    spline = CubicSpline(
                        support_indices,
                        filtered[support_indices, joint_id, axis],
                        bc_type="natural",
                    )
                    filtered[gap_indices, joint_id, axis] = spline(gap_indices)
            else:
                for axis in (0, 1):
                    filtered[gap_indices, joint_id, axis] = np.interp(
                        gap_indices,
                        [gap_start, gap_end],
                        filtered[[gap_start, gap_end], joint_id, axis],
                    )
            filtered[gap_indices, joint_id, 2] = np.minimum(
                filtered[[gap_start, gap_end], joint_id, 2].min(),
                confidence_threshold,
            )

    return filtered


def heatmap_to_image_keypoints(keypoints_2d, heatmap_shape, frame_size, target_shape=(256, 448)):
    """Convert heatmap coordinates into original video pixel coordinates."""
    heatmap_height, heatmap_width = heatmap_shape
    frame_width, frame_height = frame_size
    target_height, target_width = target_shape
    scale = min(target_width / frame_width, target_height / frame_height)
    resized_width = frame_width * scale
    resized_height = frame_height * scale
    left = (target_width - resized_width) * 0.5
    top = (target_height - resized_height) * 0.5

    pixels = np.asarray(keypoints_2d, dtype=np.float32).copy()
    valid = pixels[:, 2] > 0.0
    pixels[valid, 0] = (pixels[valid, 0] + 0.5) * target_width / heatmap_width
    pixels[valid, 1] = (pixels[valid, 1] + 0.5) * target_height / heatmap_height
    pixels[valid, 0] = (pixels[valid, 0] - left) / scale
    pixels[valid, 1] = (pixels[valid, 1] - top) / scale
    return pixels

SKELETON_DRAW_LINES = [
    (0, 1), (0, 2), (0, 3), (3, 4), (4, 5),
    (0, 9), (9, 10), (10, 11), (2, 6), (6, 7),
    (7, 8), (2, 12), (12, 13), (13, 14)
]

ALL_VIEW_CONFIGS = {
    "Front": {"elev": 0, "azim": -90},
    "Left": {"elev": 0, "azim": 180},
    "Right": {"elev": 0, "azim": 0},
    "Top": {"elev": 90, "azim": -90},
    "Bottom": {"elev": -90, "azim": -90},
    "3D_Free": {"elev": 15, "azim": -60}
}


def select_video_file_ui() -> str:
    """Opens a UI dialog for selecting an input video file."""
    logger.info("Opening UI dialog to select input video file...")
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    
    file_path = filedialog.askopenfilename(
        title="Select Input Video File",
        filetypes=[
            ("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
            ("All Files", "*.*")
        ]
    )
    root.destroy()
    return file_path


def resize_with_pad(frame, target_shape=(256, 448)):
    """Resizes frame maintaining aspect ratio and adds black padding (letterboxing)."""
    target_h, target_w = target_shape
    h, w, _ = frame.shape
    
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    
    padded[top:top + new_h, left:left + new_w] = resized
    return padded


def vector_to_quaternion(v_src, v_dst):
    """Calculates quaternion rotation from v_src vector to v_dst vector."""
    v_src = v_src / (np.linalg.norm(v_src) + 1e-8)
    v_dst = v_dst / (np.linalg.norm(v_dst) + 1e-8)
    cross = np.cross(v_src, v_dst)
    dot = np.dot(v_src, v_dst)

    if dot < -0.999999:
        return [0.0, 1.0, 0.0, 0.0]

    s = np.sqrt((1 + dot) * 2)
    qw = s / 2.0
    qx = cross[0] / s
    qy = cross[1] / s
    qz = cross[2] / s
    return [float(qw), float(qx), float(qy), float(qz)]


def extract_2d_keypoints(heatmaps, confidence_threshold=0.1):
    """Decode Open Model Zoo heatmap peaks and their confidence scores."""
    keypoints_2d = [(-1, -1, 0.0)] * 19
    for keypoint_id in range(18):
        peak = np.unravel_index(np.argmax(heatmaps[keypoint_id]), heatmaps[keypoint_id].shape)
        cy, cx = peak
        confidence = float(heatmaps[keypoint_id, cy, cx])
        panoptic_id = OV_TO_PANOPTIC[keypoint_id]
        keypoints_2d[panoptic_id] = (
            (cx, cy, confidence)
            if confidence > confidence_threshold
            else (-1, -1, 0.0)
        )
    return np.asarray(keypoints_2d, dtype=np.float32)


def process_pose_3d(heatmaps, features, img_w, img_h, avg_height=180.0, keypoints_2d=None):
    """Decode one pose using the Open Model Zoo postprocessing contract."""
    if keypoints_2d is None:
        keypoints_2d = extract_2d_keypoints(heatmaps)

    neck_x, neck_y, neck_conf = keypoints_2d[0]
    if neck_conf <= 0.1:
        return np.zeros((19, 3), dtype=np.float32), keypoints_2d
    neck_x = int(np.clip(np.rint(neck_x), 0, features.shape[2] - 1))
    neck_y = int(np.clip(np.rint(neck_y), 0, features.shape[1] - 1))

    joints_3d = np.zeros((19, 3), dtype=np.float32)
    for keypoint_id in range(19):
        channel = features[keypoint_id * 3:keypoint_id * 3 + 3]
        joints_3d[keypoint_id] = channel[:, neck_y, neck_x]

    for limb in PANOPTIC_LIMBS:
        for source_id in limb:
            x, y, confidence = keypoints_2d[source_id]
            if confidence <= 0.1:
                continue
            x = int(np.clip(np.rint(x), 0, features.shape[2] - 1))
            y = int(np.clip(np.rint(y), 0, features.shape[1] - 1))
            for target_id in limb:
                channel = features[target_id * 3:target_id * 3 + 3]
                joints_3d[target_id] = channel[:, y, x]
            break

    joints_3d *= avg_height
    return joints_3d, keypoints_2d


def main():
    logger.info("=== Starting OpenVINO 3D Pose Extractor ===")

    # 1. UI Video Selection
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Input video path; opens a picker when omitted")
    args = parser.parse_args()
    video_path = args.video or select_video_file_ui()
    if not video_path:
        logger.warning("No video file selected. Exiting pipeline.")
        return
    logger.info(f"Selected Input Video: {os.path.abspath(video_path)}")

    # 2. Validate Model Path
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file missing at: {MODEL_PATH}")
        raise FileNotFoundError(f"Model file missing at: {MODEL_PATH}")

    # 3. Initialize OpenVINO Runtime Engine
    core = ov.Core()
    device = "GPU" if "GPU" in core.available_devices else "CPU"
    logger.info(f"Selected Accelerator Device: {device}")

    model = core.read_model(model=MODEL_PATH)
    compiled_model = core.compile_model(model=model, device_name=device)

    # 4. Open Video Stream & Extract Metadata
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mocap_export_data = {
        "metadata": {
            "source_video": os.path.basename(video_path),
            "resolution": [width, height],
            "fps": fps,
            "total_frames": total_frames,
            "model": "human-pose-estimation-3d-0001"
        },
        "frames": []
    }

    # Setup Dynamic Matplotlib Viewport
    if SHOW_REALTIME_3D:
        plt.ion()

        # Parse View Selection
        if "ALL" in SELECTED_VIEWS:
            active_views = ["Front", "Left", "Right", "Top", "Bottom"]
        else:
            active_views = [v for v in SELECTED_VIEWS if v in ALL_VIEW_CONFIGS]
            if not active_views:
                active_views = ["Front"]

        num_plots = 1 + len(active_views)
        cols = min(num_plots, 3)
        rows = (num_plots + cols - 1) // cols

        fig = plt.figure(figsize=(5 * cols, 4 * rows))

        # 2D Video Input Subplot
        ax_2d = fig.add_subplot(rows, cols, 1)
        dummy_img = np.zeros((256, 448, 3), dtype=np.uint8)
        img_artist = ax_2d.imshow(dummy_img)
        ax_2d.axis("off")
        title_artist_2d = ax_2d.set_title("2D Input (Padded)")

        plot_artists = []

        # Configured 3D View Subplots
        for idx, view_name in enumerate(active_views):
            cfg = ALL_VIEW_CONFIGS[view_name]
            ax = fig.add_subplot(rows, cols, idx + 2, projection="3d")
            ax.set_title(f"3D View: {view_name}", fontsize=10)
            ax.set_xlim([-300, 300])
            ax.set_ylim([0, 800])
            ax.set_zlim([-200, 200])
            ax.view_init(elev=cfg["elev"], azim=cfg["azim"])

            scat = ax.scatter([], [], [], c="red", s=12)
            lines = [ax.plot([], [], [], c="blue")[0] for _ in SKELETON_DRAW_LINES]
            plot_artists.append({"scatter": scat, "lines": lines})

        fig.tight_layout()

    frame_idx = 0
    t_pipeline_start = time.time()
    pipeline_frames = []
    tracks_2d = []

    while cap.isOpened():
        t_frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        img_h, img_w, _ = frame.shape

        # Aspect-Ratio Preserving Preprocessing (Padded to 448x256)
        padded_frame = resize_with_pad(frame, target_shape=(256, 448))
        normalized = (padded_frame.astype(np.float32) - 128.0) / 255.0
        input_tensor = np.expand_dims(normalized.transpose(2, 0, 1), axis=0)

        # Inference Execution
        t_infer_start = time.time()
        results = compiled_model([input_tensor])
        infer_latency_ms = (time.time() - t_infer_start) * 1000.0

        heatmaps, features = None, None
        for out_tensor in results.values():
            if len(out_tensor.shape) == 4:
                if out_tensor.shape[1] == 19:
                    heatmaps = out_tensor[0]
                elif out_tensor.shape[1] == 57:
                    features = out_tensor[0]

        if heatmaps is None or features is None:
            continue

        raw_keypoints_2d = extract_2d_keypoints(heatmaps)
        tracks_2d.append(raw_keypoints_2d)
        filtered_keypoints_2d = filter_2d_tracks(np.asarray(tracks_2d))[-1]
        joints_3d, keypoints_2d = process_pose_3d(
            heatmaps,
            features,
            img_w,
            img_h,
            keypoints_2d=filtered_keypoints_2d,
        )
        hip_center = (joints_3d[6] + joints_3d[12]) / 2.0
        landmarks = pipeline_landmarks(joints_3d, hip_center)

        # Mocap Payload Build
        frame_payload = {
            "frame": frame_idx,
            "t": float((frame_idx - 1) / fps),
            "ok": True,
            "pelvis_height": 0.0,
            "root": [0.0, 0.0],
            "incam_root": [0.0, 0.0, 0.0],
            "hips_position": [float(hip_center[0]), float(hip_center[2]), float(-hip_center[1])],
            "joints_3d": {str(i): joints_3d[i].tolist() for i in range(19)},
            "bones": {},
            "world": landmarks,
            "incam": {},
            "image": {
                "keypoints_2d": heatmap_to_image_keypoints(
                    keypoints_2d,
                    heatmaps.shape[1:3],
                    (img_w, img_h),
                ).tolist(),
                "keypoint_space": "image_pixels"
            }
        }

        def landmark_array(name):
            point = landmarks[name]
            return np.array([point["x"], point["y"], point["z"]], dtype=np.float64)

        ear_mid = (landmark_array("left_ear") + landmark_array("right_ear")) * 0.5
        gaze = landmark_array("nose") - ear_mid
        gaze /= np.linalg.norm(gaze) + 1e-8
        frame_payload["gaze"] = gaze.tolist()

        default_bone_vec = np.array([0.0, 1.0, 0.0])
        for bone_name, (idx_start, idx_end) in BONE_CONNECTIONS.items():
            vec = joints_3d[idx_end] - joints_3d[idx_start]
            quat = vector_to_quaternion(default_bone_vec, vec)
            frame_payload["bones"][bone_name] = {"rotation_quat_wxyz": quat}

        mocap_export_data["frames"].append(frame_payload)
        pipeline_frames.append(frame_payload)

        # Matplotlib Rendering
        if SHOW_REALTIME_3D and (frame_idx % RENDER_EVERY_N_FRAMES == 0):
            img_artist.set_data(cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB))
            title_artist_2d.set_text(f"2D Input ({frame_idx}/{total_frames})")

            xs = joints_3d[:, 0]
            ys = joints_3d[:, 2]
            zs = -joints_3d[:, 1]

            for artist_dict in plot_artists:
                artist_dict["scatter"]._offsets3d = (xs, ys, zs)

                for line, (p1, p2) in zip(artist_dict["lines"], SKELETON_DRAW_LINES):
                    line.set_data(
                        [joints_3d[p1, 0], joints_3d[p2, 0]],
                        [joints_3d[p1, 2], joints_3d[p2, 2]]
                    )
                    line.set_3d_properties([-joints_3d[p1, 1], -joints_3d[p2, 1]])

            fig.canvas.flush_events()

        # Telemetry
        total_frame_latency_ms = (time.time() - t_frame_start) * 1000.0
        elapsed_time = time.time() - t_pipeline_start
        avg_fps = frame_idx / elapsed_time
        remaining_frames = total_frames - frame_idx
        eta_sec = remaining_frames / avg_fps if avg_fps > 0 else 0.0

        if frame_idx % 10 == 0 or frame_idx == total_frames:
            logger.info(
                f"Frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%) | "
                f"Infer: {infer_latency_ms:.1f}ms | Loop: {total_frame_latency_ms:.1f}ms | "
                f"Speed: {avg_fps:.1f} FPS | ETA: {eta_sec:.1f}s"
            )

    cap.release()
    if SHOW_REALTIME_3D:
        plt.close()

    if pipeline_frames:
        raw_tracks = np.asarray(
            [frame["image"]["keypoints_2d"] for frame in pipeline_frames],
            dtype=np.float64,
        )
        filtered_tracks = filter_2d_tracks(raw_tracks)
        for frame, keypoints_2d in zip(pipeline_frames, filtered_tracks):
            frame["image"]["keypoints_2d"] = keypoints_2d.tolist()

        ankle_heights = [
            (frame["world"]["left_ankle"]["y"] + frame["world"]["right_ankle"]["y"]) * 0.5
            for frame in pipeline_frames[:min(10, len(pipeline_frames))]
        ]
        ground_ref = float(np.median(ankle_heights))
        for frame in pipeline_frames:
            frame["pelvis_height"] = ground_ref

    # Save Output
    with open(OUTPUT_JSON, "w") as f:
        json.dump(mocap_export_data, f, indent=2)

    logger.info(f"Mocap export complete. Saved to: {os.path.abspath(OUTPUT_JSON)}")


if __name__ == "__main__":
    main()