"""Generic lift: estimator landmarks -> Mixamo-world joints, driven by an
action_spec JSON (see docs/PIPELINE.md for the spec schema).

Usage:
  python pipeline\\lift_to_mixamo.py --spec action_specs\\<motion>.json

This is a direction-preserving retarget: keep the estimator's segment
DIRECTIONS, rebuild positions from the Mixamo sockets with Mixamo bone
lengths. With mesh-quality input (GVHMR) no inverse kinematics, no
authored limb targets and no action envelopes are needed. The spec
contributes only what the estimator cannot know:

  - where the clip blends into exact Y Bot rest (start and/or end)
  - when fists close
  - which foot is the support in each phase (left/right/both/none)
  - authored arm overrides for beats where the owner's read of the
    video beats the estimator (e.g. an occluded arm)
  - the QA frames

Two structural jobs happen here rather than in the FK apply:

  - support-ankle pinning: the estimator world is per-frame hip-centered,
    so a planted foot drifts while the body leans. During single-support
    windows the WHOLE pose is translated so the support ankle stays where
    it touched down (the pelvis then sways over the planted foot, which
    is the correct physics). The correction decays over 10 frames after
    the window.
  - hips world Z stays at rest height; the FK apply searches the real
    hip height per frame so the support foot plants at ground level —
    that is what makes crouches and wide stances drop the pelvis. For
    airborne windows the per-frame `pelvis_height` is passed through and
    the apply integrates the arc instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

REPO = Path(__file__).resolve().parents[1]

# Y Bot rest joints, world meters (measured from the live rig; see docs/RIG.md).
REST = {
    "hips": np.array([0.0, 0.0, 0.99792]),
    "spine": np.array([0.0, 0.01227, 1.09715]),
    "spine1": np.array([0.0, 0.0265, 1.21361]),
    "spine2": np.array([0.0, 0.04276, 1.34721]),
    "neck": np.array([0.0, 0.03483, 1.49754]),
    "head": np.array([0.0, 0.00341, 1.60075]),
    "l_shoulder": np.array([0.06103, 0.03571, 1.43832]),
    "l_arm": np.array([0.18758, 0.06171, 1.43567]),
    "l_elbow": np.array([0.46163, 0.06171, 1.4357]),
    "l_wrist": np.array([0.73777, 0.06171, 1.43572]),
    "l_hand": np.array([0.84757, 0.06171, 1.43573]),
    "r_shoulder": np.array([-0.06109, 0.03571, 1.43831]),
    "r_arm": np.array([-0.18764, 0.06171, 1.43564]),
    "r_elbow": np.array([-0.46168, 0.06171, 1.43562]),
    "r_wrist": np.array([-0.73783, 0.06171, 1.43559]),
    "r_hand": np.array([-0.84763, 0.06171, 1.43558]),
    "l_upleg": np.array([0.09124, 0.00055, 0.93136]),
    "l_knee": np.array([0.09369, 0.00571, 0.5254]),
    "l_ankle": np.array([0.09124, 0.0263, 0.10492]),
    "l_foot": np.array([0.09498, -0.11336, 0.03284]),
    "l_toe": np.array([0.09356, -0.21334, 0.03108]),
    "r_upleg": np.array([-0.09124, 0.00055, 0.93136]),
    "r_knee": np.array([-0.09369, 0.0057, 0.5254]),
    "r_ankle": np.array([-0.09124, 0.0263, 0.10492]),
    "r_foot": np.array([-0.09498, -0.11336, 0.03284]),
    "r_toe": np.array([-0.09356, -0.21334, 0.03109]),
}

# Mixamo bone lengths, meters.
LEN = {
    "spine1": 0.13459,
    "spine2": 0.12344,
    "neck": 0.10790,
    "head": 0.19630,
    "l_shoulder": 0.12922,
    "l_arm": 0.27405,
    "l_fore": 0.27614,
    "l_hand": 0.10980,
    "r_shoulder": 0.12922,
    "r_arm": 0.27405,
    "r_fore": 0.27614,
    "r_hand": 0.10980,
    "l_upleg": 0.40599,
    "l_leg": 0.42099,
    "l_foot": 0.15722,
    "r_upleg": 0.40599,
    "r_leg": 0.42099,
    "r_foot": 0.15722,
}

# A character-specific rig_profile.json (written by pipeline/setup_rig.py
# when you build the scene) overrides the built-in Y Bot measurements —
# this is what makes the lift work with any Mixamo character's
# proportions.
_PROFILE = REPO / "rig_profile.json"
if _PROFILE.exists():
    _p = json.loads(_PROFILE.read_text(encoding="utf-8"))
    REST = {k: np.array(v, dtype=float) for k, v in _p["rest"].items()}
    LEN = {k: float(v) for k, v in _p["lengths"].items()}

MP_USED = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_index", "right_index",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

HIP_Z = float(REST["hips"][2])


def rpath(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO / p)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros(3)


def smoother(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def lerp(a, b, t):
    return a * (1.0 - t) + b * t


def mp_to_mix(p):
    """Estimator world (y down, z away-from-camera) -> Mixamo world
    (z up, character faces -Y, left +X)."""
    return np.array([p[0], p[2], -p[1] + HIP_Z], dtype=np.float64)


def smooth_series(arr, window=7, poly=2):
    n = arr.shape[0]
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 5:
        return arr
    out = np.empty_like(arr)
    for c in range(arr.shape[1]):
        out[:, c] = savgol_filter(arr[:, c], window_length=w, polyorder=poly, mode="interp")
    return out


def load_mp(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = [f for f in data["frames"] if f.get("ok")]
    world = {n: np.zeros((len(frames), 3)) for n in MP_USED}
    times = np.zeros(len(frames))
    pelvis_h = np.zeros(len(frames))
    for i, f in enumerate(frames):
        times[i] = f["t"]
        pelvis_h[i] = float(f.get("pelvis_height", 0.0))
        for n in MP_USED:
            w = f["world"][n]
            world[n][i] = (w["x"], w["y"], w["z"])
    return data["fps"], times, world, pelvis_h


def resample(times, series, dst_times):
    out = {}
    for k, arr in series.items():
        sm = smooth_series(arr)
        cols = [np.interp(dst_times, times, sm[:, c]) for c in range(arr.shape[1])]
        out[k] = np.stack(cols, axis=1)
    return out


def place_chain(up, left):
    z = unit(up)
    x = unit(left - z * np.dot(left, z))
    y = unit(np.cross(z, x))
    x = unit(np.cross(y, z))
    return x, y, z


def rest_pose():
    out = {k: v.copy() for k, v in REST.items()}
    out["basis_x"] = np.array([1.0, 0.0, 0.0])
    out["basis_y"] = np.array([0.0, 1.0, 0.0])
    out["basis_z"] = np.array([0.0, 0.0, 1.0])
    return out


def blend_pose(a, b, t):
    t = float(np.clip(t, 0.0, 1.0))
    out = {}
    for k in set(a) | set(b):
        if k not in a:
            out[k] = b[k]
        elif k not in b:
            out[k] = a[k]
        else:
            out[k] = lerp(np.asarray(a[k], float), np.asarray(b[k], float), t)
    return out


def seg(mp, i, a, b):
    """Unit direction of estimator segment a->b at frame i."""
    return unit(mp[b][i] - mp[a][i])


def reconstruct(mp, i):
    """Direction-preserving retarget of one frame onto Mixamo lengths."""
    ls, rs = mp["left_shoulder"][i], mp["right_shoulder"][i]
    lh, rh = mp["left_hip"][i], mp["right_hip"][i]
    nose = mp["nose"][i]

    hip_mid = 0.5 * (lh + rh)
    sh_mid = 0.5 * (ls + rs)
    up = unit(sh_mid - hip_mid)
    if np.linalg.norm(up) < 0.2:
        up = np.array([0.0, 0.0, 1.0])
    left = unit((ls - rs) + (lh - rh) * 0.35)
    x, y, z = place_chain(up, left)

    hips = np.array([hip_mid[0], hip_mid[1], HIP_Z])

    def in_basis(local_xyz):
        return hips + x * local_xyz[0] + y * local_xyz[1] + z * local_xyz[2]

    # Torso: the SMPL-derived landmarks carry no spine names, so the
    # chest chain is built from the hip basis + shoulder line.
    chest_dir = unit(0.65 * z + 0.35 * up)
    spine = hips + chest_dir * np.linalg.norm(REST["spine"] - REST["hips"])
    spine1 = spine + chest_dir * LEN["spine1"]
    spine2 = spine1 + chest_dir * LEN["spine2"]
    neck = spine2 + z * LEN["neck"]
    head = neck + z * LEN["head"] * 0.85 + unit(np.array([nose[0], nose[1], 0.0])) * 0.02

    # Sockets from the posed hip basis.
    l_shoulder = in_basis(REST["l_shoulder"] - REST["hips"])
    r_shoulder = in_basis(REST["r_shoulder"] - REST["hips"])
    l_arm = l_shoulder + unit((ls - sh_mid) + x * 0.35) * LEN["l_shoulder"]
    r_arm = r_shoulder + unit((rs - sh_mid) - x * 0.35) * LEN["r_shoulder"]
    l_upleg = in_basis(REST["l_upleg"] - REST["hips"])
    r_upleg = in_basis(REST["r_upleg"] - REST["hips"])

    # Limbs: estimator directions, Mixamo lengths.
    l_elbow = l_arm + seg(mp, i, "left_shoulder", "left_elbow") * LEN["l_arm"]
    l_wrist = l_elbow + seg(mp, i, "left_elbow", "left_wrist") * LEN["l_fore"]
    l_hand = l_wrist + seg(mp, i, "left_wrist", "left_index") * LEN["l_hand"]
    r_elbow = r_arm + seg(mp, i, "right_shoulder", "right_elbow") * LEN["r_arm"]
    r_wrist = r_elbow + seg(mp, i, "right_elbow", "right_wrist") * LEN["r_fore"]
    r_hand = r_wrist + seg(mp, i, "right_wrist", "right_index") * LEN["r_hand"]

    l_knee = l_upleg + seg(mp, i, "left_hip", "left_knee") * LEN["l_upleg"]
    l_ankle = l_knee + seg(mp, i, "left_knee", "left_ankle") * LEN["l_leg"]
    r_knee = r_upleg + seg(mp, i, "right_hip", "right_knee") * LEN["r_upleg"]
    r_ankle = r_knee + seg(mp, i, "right_knee", "right_ankle") * LEN["r_leg"]

    l_foot = l_ankle + seg(mp, i, "left_ankle", "left_foot_index") * LEN["l_foot"]
    r_foot = r_ankle + seg(mp, i, "right_ankle", "right_foot_index") * LEN["r_foot"]
    l_fwd = unit(np.array([*(mp["left_foot_index"][i] - mp["left_heel"][i])[:2], 0.0]))
    r_fwd = unit(np.array([*(mp["right_foot_index"][i] - mp["right_heel"][i])[:2], 0.0]))
    if np.linalg.norm(l_fwd) < 0.1:
        l_fwd = -y
    if np.linalg.norm(r_fwd) < 0.1:
        r_fwd = -y
    l_toe = l_foot + l_fwd * 0.10
    r_toe = r_foot + r_fwd * 0.10

    return {
        "hips": hips, "spine": spine, "spine1": spine1, "spine2": spine2,
        "neck": neck, "head": head,
        "l_shoulder": l_shoulder, "l_arm": l_arm, "l_elbow": l_elbow,
        "l_wrist": l_wrist, "l_hand": l_hand,
        "r_shoulder": r_shoulder, "r_arm": r_arm, "r_elbow": r_elbow,
        "r_wrist": r_wrist, "r_hand": r_hand,
        "l_upleg": l_upleg, "l_knee": l_knee, "l_ankle": l_ankle,
        "l_foot": l_foot, "l_toe": l_toe,
        "r_upleg": r_upleg, "r_knee": r_knee, "r_ankle": r_ankle,
        "r_foot": r_foot, "r_toe": r_toe,
        "basis_x": x, "basis_y": y, "basis_z": z,
    }


def window_amount(dest_f, rise, fall, src2dest):
    """1 inside [rise..fall] windows given in src frames, smooth edges."""
    r0, r1 = src2dest(rise[0]), src2dest(rise[1])
    f0, f1 = src2dest(fall[0]), src2dest(fall[1])
    if dest_f <= r0 or dest_f >= f1:
        return 0.0
    if dest_f < r1:
        return smoother((dest_f - r0) / max(1.0, r1 - r0))
    if dest_f <= f0:
        return 1.0
    return 1.0 - smoother((dest_f - f0) / max(1.0, f1 - f0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, type=Path)
    args = ap.parse_args()
    spec = json.loads(rpath(args.spec).read_text(encoding="utf-8"))

    fps, times, world, pelvis_h = load_mp(rpath(spec["landmarks"]))
    dst_fps = float(spec.get("dst_fps", 30))
    duration = float(times[-1])
    n_dst = int(round(duration * dst_fps)) + 1
    dst_times = np.clip(np.arange(n_dst) / dst_fps, times[0], times[-1])

    mp = {n: np.stack([mp_to_mix(p) for p in world[n]]) for n in world}
    mp = resample(times, mp, dst_times)
    if len(pelvis_h) >= 7:
        pelvis_h = savgol_filter(pelvis_h, 7, 2, mode="interp")
    pelvis_h = np.interp(dst_times, times, pelvis_h)

    def src2dest(sf):
        return (sf - 1) * dst_fps / fps + 1

    rb = spec["rest_blend_end"]
    rest = rest_pose()

    plant_windows = [(w["src"][0], w["src"][1], w["support"]) for w in spec["plant"]]

    def plant_at(sf):
        for a, b, sup in plant_windows:
            if a <= sf <= b:
                return sup
        return "both"

    recs, extras = [], []
    for i in range(n_dst):
        f = i + 1
        sf = 1 + (f - 1) * fps / dst_fps  # dest -> src frame (float)
        rec = reconstruct(mp, i)

        # Authored arm overrides (spec): pull a whole arm chain to
        # hip-local targets — for beats where the owner's read of the
        # video beats the estimator (e.g. occluded arm chambered at the
        # chest). Targets are meters in the hip basis, ramped in/out.
        for ov in spec.get("arm_overrides", []):
            amt = window_amount(
                f,
                [ov["src"][0], ov["src"][0] + ov.get("ramp_src", 6)],
                [ov["src"][1] - ov.get("ramp_src", 6), ov["src"][1]],
                src2dest,
            )
            if amt <= 1e-4:
                continue
            s = "l" if ov["side"] == "left" else "r"
            bx, by, bz = rec["basis_x"], rec["basis_y"], rec["basis_z"]
            hips_p = np.asarray(rec["hips"], float)

            def hip_local(v3):
                return hips_p + bx * v3[0] + by * v3[1] + bz * v3[2]

            sock = np.asarray(rec[f"{s}_arm"], float)
            e_t = sock + unit(hip_local(ov["elbow_local"]) - sock) * LEN[f"{s}_arm"]
            w_t = e_t + unit(hip_local(ov["wrist_local"]) - e_t) * LEN[f"{s}_fore"]
            e = lerp(np.asarray(rec[f"{s}_elbow"], float), e_t, amt)
            w = lerp(np.asarray(rec[f"{s}_wrist"], float), w_t, amt)
            e = sock + unit(e - sock) * LEN[f"{s}_arm"]
            w = e + unit(w - e) * LEN[f"{s}_fore"]
            rec[f"{s}_elbow"], rec[f"{s}_wrist"] = e, w
            rec[f"{s}_hand"] = w + unit(w - e) * LEN[f"{s}_hand"]

        rest_amt = smoother((sf - rb["start_src"]) / max(1.0, rb["full_src"] - rb["start_src"]))
        rbs = spec.get("rest_blend_start")
        if rbs:
            start_amt = 1.0 - smoother(
                (sf - rbs["full_src"]) / max(1.0, rbs["release_src"] - rbs["full_src"])
            )
            rest_amt = max(rest_amt, start_amt)
        if rest_amt > 0.0:
            rec = blend_pose(rec, rest, rest_amt)
            for k in ("basis_x", "basis_y", "basis_z"):
                rec[k] = unit(np.asarray(rec[k], float))

        fist = window_amount(f, spec["fists"]["rise_src"], spec["fists"]["fall_src"], src2dest)
        recs.append(rec)
        extras.append({"frame": f, "t": float(dst_times[i]), "rest": rest_amt, "fist": fist,
                       "plant": plant_at(sf) if rest_amt < 0.65 else "both",
                       "pelvis_height": float(pelvis_h[i])})

    # Optional extra temporal smoothing on selected joints in selected
    # windows (spec "smooth") — for fast flurries the raw estimator
    # tracks can look staccato; a windowed Savitzky-Golay pass with
    # smooth edge blending calms them without killing the snap.
    for sm in spec.get("smooth", []):
        a = max(1, int(round(src2dest(sm["src"][0]))))
        b = min(n_dst, int(round(src2dest(sm["src"][1]))))
        w = int(sm.get("window", 9))
        w = min(w if w % 2 == 1 else w - 1, (b - a + 1) | 1)
        if w < 5 or b - a < 4:
            continue
        keys = sm.get("joints", ["l_elbow", "l_wrist", "l_hand", "r_elbow", "r_wrist", "r_hand"])
        ramp = max(2, w // 2)
        for k in keys:
            series = np.array([np.asarray(recs[f - 1][k], float) for f in range(a, b + 1)])
            smoothed = np.stack(
                [savgol_filter(series[:, c], window_length=w, polyorder=2, mode="interp")
                 for c in range(3)], axis=1)
            for j, f in enumerate(range(a, b + 1)):
                t = min(1.0, min(j, (b - a) - j) / float(ramp))
                blend = smoother(t)
                recs[f - 1][k] = series[j] * (1.0 - blend) + smoothed[j] * blend

    # Support-ankle pinning (see module docstring).
    RELEASE = 10
    pos_keys = [k for k in recs[0] if not k.startswith("basis_")]
    for a_src, b_src, sup in plant_windows:
        if sup not in ("left", "right"):
            continue
        a = max(1, int(round(src2dest(a_src))))
        b = min(n_dst, int(round(src2dest(b_src))))
        ankle = "l_ankle" if sup == "left" else "r_ankle"
        anchor = np.asarray(recs[a - 1][ankle], float)[:2].copy()
        last_delta = np.zeros(2)
        for f in range(a, b + 1):
            delta = anchor - np.asarray(recs[f - 1][ankle], float)[:2]
            for k in pos_keys:
                recs[f - 1][k] = np.asarray(recs[f - 1][k], float)
                recs[f - 1][k][:2] += delta
            last_delta = delta
        for j, f in enumerate(range(b + 1, min(n_dst, b + RELEASE) + 1)):
            fade = last_delta * (1.0 - smoother((j + 1) / float(RELEASE)))
            for k in pos_keys:
                recs[f - 1][k] = np.asarray(recs[f - 1][k], float)
                recs[f - 1][k][:2] += fade

    joints = []
    for rec, ex in zip(recs, extras):
        out = {k: [round(float(v), 5) for v in np.asarray(val)] for k, val in rec.items()}
        out["frame"] = ex["frame"]
        out["t"] = ex["t"]
        out["rest_amount"] = round(float(ex["rest"]), 4)
        out["fist_amount"] = round(float(ex["fist"]), 4)
        out["plant"] = ex["plant"]
        out["pelvis_height"] = round(float(ex["pelvis_height"]), 5)
        joints.append(out)

    payload = {
        "src_fps": fps,
        "dst_fps": dst_fps,
        "duration": duration,
        "frame_count": n_dst,
        "estimator": spec.get("estimator", "unknown"),
        "action_spec": spec["name"],
        "frames": joints,
    }
    out_path = rpath(spec["joints_out"])
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    print("wrote", out_path, "frames", n_dst)

    for sfq in spec.get("qa_src_frames", []):
        f = int(round(src2dest(sfq)))
        f = max(1, min(n_dst, f))
        j = joints[f - 1]
        bx = np.asarray(j["basis_x"], float)
        yaw = np.degrees(np.arctan2(-bx[1], bx[0]))
        print(
            f"src{sfq:3d}/dest{f:3d} rest={j['rest_amount']:.2f} fist={j['fist_amount']:.2f} "
            f"plant={j['plant']:5s} yaw={yaw:+4.0f} "
            f"lw={j['l_wrist']} rw={j['r_wrist']} ra={j['r_ankle']}"
        )


if __name__ == "__main__":
    main()
