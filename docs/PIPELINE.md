# The pipeline, operationally

How a video becomes a Y Bot clip, stage by stage, with the decision
rules an operator (human or AI) needs. Everything here was validated on
the two shipped clips.

## 0. The plate (source video)

Quality is won or lost here. Rules (see docs/PROMPTING.md for
generating plates with AI video):

- Locked camera, full body in frame the whole time, even light, 720p+.
- **T-pose held ~1 s at the start AND the end**, facing the camera
  square. The T-poses are the rest anchors; the pipeline blends into
  exact Y Bot rest there.
- One continuous take, on the spot. Every moving limb must stay
  visible — an occluded limb cannot be tracked (it can be fixed with an
  `arm_override`, but only because a human saw the video).
- Facing changes are fine (the kungfu clip turns ±100°); occlusion is
  the enemy, not rotation.

Save as `plates/<name>/<clip>.mp4` with a short `SOURCE.md`.

## 1. Estimate

```
tools\GVHMR\.venv\Scripts\python.exe pipeline\estimate_pose_gvhmr.py ^
    --video plates\<name>\<clip>.mp4 --out plates\<name>\landmarks.json
```

Output: 33 landmarks/frame (MediaPipe-compatible convention: per-frame
mid-hip origin, y down, z **negative toward camera**, meters),
`pelvis_height` (absolute, above the estimator's ground plane) and
`gaze` (unit vector, the performer's real face direction taken from
mesh vertices — nose vs ear midpoint). The face landmarks are real mesh
points, not torso-derived: head orientation cannot be recovered from
joint positions, and a retarget without it can only lock the character's
gaze to its chest.
GVHMR caches its preprocessing under `tools/GVHMR/outputs/demo/<stem>/`
— re-runs are fast.

## 2. Numeric beat sheet — BEFORE any spec

```
tools\GVHMR\.venv\Scripts\python.exe pipeline\analyze_landmarks.py --landmarks plates\<name>\landmarks.json
```

Write `plates/<name>/BEATSHEET.md`: a table of beats with src frame
windows, the acting limb, the support foot, and the *numeric evidence*
(this tool's output). Hard rules:

- **Take every limb decision from the numbers, never from stills.**
  Facing the camera, viewer-left is the character's RIGHT; every
  screen-read limb guess in this project's history was wrong at least
  once. Gen-video also deviates from its prompt (a prompted left-knee
  jump arrived as right-knee).
- Punches: wrist z dips below −0.32 (toward camera). Kicks/knees: one
  ankle height rises with the other grounded. Flight: both ankles
  > 0.25 m. Ducks/jumps: `pelvis_height` dips/peaks.
- Shoulder-line yaw swings ±40–60° during punches without the hips
  turning — don't read those as facing changes.

## 3. The action_spec (the motion as data)

`action_specs/<motion>.json` — all frame numbers are **source** frames
(the lift converts to 30 fps dest internally):

```jsonc
{
  "name": "combo",                          // slug
  "action_name": "YBot_ComboFromVideo",     // Blender action name
  "clip_dir": "clips/ybot_combo_from_video",// outputs (repo-relative)
  "landmarks": "plates/combo/landmarks.json",
  "joints_out": "plates/combo/joints_mixamo.json",
  "estimator": "gvhmr_siga24",
  "src_fps": 24, "dst_fps": 30,

  // Blend into exact Y Bot rest. End is required; start is optional
  // (use it when the plate opens on a clean T-pose).
  "rest_blend_start": { "full_src": 1, "release_src": 16 },
  "rest_blend_end":   { "start_src": 214, "full_src": 232 },

  // Fist amount 0->1->0, smooth edges.
  "fists": { "rise_src": [20, 28], "fall_src": [200, 212] },

  // Optional authored overrides for beats where the human's read of
  // the video beats the estimator (occluded arm etc.). Targets are
  // METERS in the hip basis (x=character left, y=back, z=up), ramped.
  "arm_overrides": [
    { "side": "right", "src": [76, 163], "ramp_src": 6,
      "wrist_local": [-0.17, -0.12, 0.33],
      "elbow_local": [-0.25, 0.05, 0.12] }
  ],

  // The support schedule — the heart of the spec. Windows must cover
  // the clip. Supports: "both", "left", "right", "none" (airborne).
  "plant": [
    { "src": [1, 129],   "support": "both"  },
    { "src": [130, 154], "support": "left"  },   // right leg acts
    { "src": [155, 158], "support": "right" },
    { "src": [159, 168], "support": "none"  },   // flight
    { "src": [169, 177], "support": "left"  },   // landing
    { "src": [178, 241], "support": "both"  }
  ],

  "qa_src_frames": [1, 30, 60, 109, 139, 163, 241]  // beat frames for QA/stills
}
```

What each support does in the FK apply:
- `left`/`right`: hip height searched so that foot lands at 0.105 m;
  the foot is flattened (Z-only, keeps estimator XZ and its own
  heading); the lift also **pins the support ankle's XZ** for the whole
  window (the pose translates so the planted foot never skates; the
  correction decays over 10 frames after the window).
- `both`: plant the lower foot, flatten both if near ground. Feet may
  genuinely step inside a `both` window — that is fine.
- `none`: no plant, no snapping; hip height integrates the estimator's
  `pelvis_height` arc from the last planted frame (continuous takeoff,
  ballistic flight). Landing pops are caught by QA's hip-step check.
  Two tuning notes from real passes: (a) airborne detection fires only
  once feet clear ~0.25 m, which EATS the launch rise if the window
  starts there — start the `none` window at the true liftoff (watch
  `pelvis_height` start climbing steeply); (b) the estimator tends to
  understate jump amplitude — an optional `"boost": 1.3` on the window
  scales the integrated arc (landing stays continuous because the arc's
  endpoint is scaled too).

An optional top-level `"smooth"` list applies extra windowed
Savitzky-Golay smoothing to selected joints (default: both arm chains)
where fast flurries read staccato:

```jsonc
"smooth": [ { "src": [172, 204], "window": 9 } ]
```

Windowed correctors for systematic estimator bias inside a phase (all
ramped; they leave the rest of the clip untouched):

```jsonc
// Rotate a whole arm rigidly about its shoulder socket. Targets are in
// METRES and the angle is solved per frame, because a fixed angle over-
// or under-shoots as the arm swings. Rigid rotation preserves elbow
// bend, wrist alignment and the distance between the hands — offsetting
// the joints instead (and re-normalizing bone lengths) collapses the
// hands together and folds the wrists at anything beyond ~2 cm.
"arm_pose": [
  { "src": [138, 206], "ramp_src": 6, "side": "both",
    "drop_m": 0.145,      // + lowers the hands along an arc
    "widen_m": 0.085 }    // + opens the hands apart
  // also accepts raw "pitch_deg" / "yaw_deg" if you prefer angles
],

// Restore strike extension. Scales the hand's distance from its
// shoulder socket along its own direction, then re-places the chain
// with exact bone lengths (current elbow as pole). Only arms already
// past `min_extension` of full length are affected, so guards stay put.
"reach": [
  { "src": [172, 204], "ramp_src": 4, "side": "both",
    "factor": 1.22, "max_fraction": 0.97, "min_extension": 0.30 }
],

// Authored targets for a beat the estimator got wrong (occluded arm).
// Positions in metres in the hip basis (x=left, y=back, z=up).
"arm_overrides": [
  { "side": "right", "src": [76, 163], "ramp_src": 6,
    "wrist_local": [-0.17, -0.12, 0.33], "elbow_local": [-0.25, 0.05, 0.12] }
]
```

**Sizing a correction:** measure, don't guess. Compare the retarget
against the reference on quantities the eye actually reads — hand height
relative to the HEAD, distance between the hands, hand distance from the
chest — and remember to subtract the part that is proportion rather than
pose (a character whose head sits lower on its shoulders will always
read "hands high" at identical joint angles).

### Gaze

The head follows the estimator's real gaze by default — no spec entry
needed. Two top-level knobs tune it:

```jsonc
"gaze_follow": 1.0,          // 0..1, how strongly the head tracks the real gaze
"gaze_max_offset_deg": 70    // cap on head-vs-chest yaw, so a 360 spin
                             // does not twist the neck off
```

Per-window overrides go in `head_look` entries as `"follow_gaze"`, and
the manual correctors remain for the rare case where the estimator's
head is wrong: `"amount"` (blend heading toward character-forward),
`"level_face"` + `"level_target_deg"` (put the face at a fixed elevation
by closed loop on the rig).

`qa_clip.py` audits the result: it flags a head locked to the chest
(gaze-vs-chest std < 2°, the signature of an estimator that isn't
emitting real head orientation) and a head staring upward (mean
elevation > 12°).

**Smoothing eats strikes.** A `smooth` window wide enough to calm a
fast flurry (9 frames) also averages away its extension peaks — the
punches get visibly shorter. Keep flurry windows narrow (5) and
restore amplitude with `reach` rather than widening the smoothing.

## 4. Lift

```
tools\GVHMR\.venv\Scripts\python.exe pipeline\lift_to_mixamo.py --spec action_specs\<motion>.json
```

Direction-preserving retarget: estimator segment directions × Mixamo
bone lengths from the Mixamo sockets. No IK. Read the QA lines it
prints (one per `qa_src_frames`): rest/fist amounts, plant, yaw, key
positions. Sanity: rest frames print the exact Y Bot rest
(`lw=[0.73777, 0.06171, 1.43572]`).

## 5. Apply (inside live Blender)

Blender must be open on `ybot_rest.blend` with the MCP add-on server
running. One command drives all three Blender-side stages:

```
python pipeline\run_in_blender.py all action_specs\<motion>.json
```

(`apply` keys the action — ~2–4 min per 300 frames, Blender's UI
freezes, that's normal; `curves` dumps every bone every frame;
`stills` renders front+side PNGs at the spec's QA frames. Run stages
individually with `apply|curves|stills`.) Agents with MCP can instead
call `apply_mixamo_fk.run/dump_curves/run_stills_render` directly via
`execute_blender_code`.

Stills and previews use a temporary camera + Workbench render
(window-independent); never use viewport screenshots — they capture
black when the Blender window is occluded.

To produce the review videos (`preview.mp4` + the side-by-side
`showcase.mp4` against the source plate):

```
tools\GVHMR\.venv\Scripts\python.exe pipeline\render_preview.py action_specs\<motion>.json --showcase
```

## 6. QA

```
tools\GVHMR\.venv\Scripts\python.exe pipeline\qa_clip.py --spec action_specs\<motion>.json
```

| Check | Limit | Meaning |
|---|---|---|
| world bounds | \|x\|,\|y\| < 2, z ∈ (−0.5, 3) | exploded bones |
| hip Z step | < 0.12 m/frame | pops (takeoff/landing included) |
| in-place travel | end-to-end ≈ 0 | root discipline |
| single-support | z-err ≈ 0, XZ wander ≈ 0 | plant + no skate |
| airborne | lower-foot clearance > 0.12 m | the jump actually flies |
| end frame | hand error < 0.03 vs rest | clean T-pose out |
| frame-jumps | < 0.30 m per bone | teleporting limbs |

`WARN` on a `both` window usually means a genuine step (weight shift,
stance widening) — check the video before "fixing" it. **Numbers can
pass while the motion is wrong: always look at the stills and play the
clip. The human eye is the last word.**

## 7. Iteration protocol (how 16 passes became 1–2)

- Beat sheet before any spec. A wrong limb in the spec is a wasted pass.
- Human notes come as frame ranges + a region ("frames 18–50, punching
  side"). Map the range to the beat and to the spec field that owns it
  BEFORE editing anything.
- One constraint per pass. Never touch a region the human has signed
  off to polish a different one.
- Re-render stills after every apply; never argue from a stale image.
- When the estimator and the human's read of the video disagree, the
  human is right (occlusion, foreshortening) — encode the fix as an
  `arm_override`, don't fight the estimator globally.
