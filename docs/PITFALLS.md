# Pitfalls

Every one of these was paid for in real passes across three model
generations working on this rig. If you are an AI operating this
pipeline: read this file before your first pass and again before
touching a clip a human has partially signed off.

## Space and measurement

1. **Blender IK explodes this rig.** `POSE_IK` on the legs sent feet to
   −81 m. The skeleton is FK-only; plant feet via hip-height search +
   Z-only flattening. If FK plant looks wrong, fix the spec or the
   lift — never "just add IK for the landing".
2. **World-space aim is 100× off.** Composing a pose matrix through
   `arm.matrix_world` (scale 0.01) and assigning it sends hands to
   z ≈ 10 m. Aim in armature centimeters (`aim_bone` does).
3. **`pb.head` lies on a posed rig.** Only
   `(arm.matrix_world @ pb.matrix).to_translation()` after a depsgraph
   update is truth.
4. **Stale location keys survive rewrites.** A failed pass that keyed
   locations on non-hip bones will haunt a rotations-only rewrite.
   `apply_mixamo_fk` deletes and rebuilds the action and re-keys zero
   locations everywhere for this reason.

## Reading the video

5. **Never decide a limb from a still.** Facing the camera,
   viewer-left = character-RIGHT. Three separate incidents: a kick
   built on the wrong leg (full wasted pass), a "left knee" jump that
   was numerically right-knee, a "chambered fist" that was a
   foreshortened punch toward the camera. `analyze_landmarks.py` output
   is the only admissible evidence for the beat sheet.
6. **Gen-video does not follow its prompt.** Prompted left-knee jump →
   right knee; prompted extended front kick → high-knee snap; prompted
   A-pose → the model's own idea of one. Retarget what was filmed;
   never author toward the prompt.
7. **The estimator loses occluded limbs.** An arm behind the torso in
   3/4 view came back as garbage (lateral-back instead of chambered at
   the chest). When a human's read of the video contradicts the
   estimator, the human is right — encode it as a spec `arm_override`
   with ramps, and leave the estimator's other limbs alone.

## Feet and ground

8. **Per-frame hip-centered landmarks make planted feet skate.** The
   support foot wandered 0.30 m during a kick. Fix is structural (the
   lift pins the support ankle's XZ and translates the whole pose;
   pelvis sways over the foot, which is correct physics) — do not
   patch it per-frame in the apply.
9. **A degenerate aim target bends toes randomly.** Aiming ToeBase at
   its own head position is a near-zero-length target. Toe aims need a
   point several cm FORWARD along the foot's own heading.
10. **Foot heading follows the foot/hips, never a world axis.**
    "Straightening" toes toward world −Y on a yawed stance turns them
    outward on screen. Keep estimator XZ; constrain heights only.
11. **Never snap the acting limb to the floor.** A floor snap that kept
    running while a kick lifted pinned the foot for one frame and made
    the next frame pop. The plant schedule exists so ground logic
    knows whose foot is acting.

## Blender / tooling

12. **Viewport screenshots capture black** when the Blender window is
    occluded or minimized. Stills go through a temporary camera +
    Workbench render (`run_stills_render`), always.
13. **The long apply blocks Blender's UI.** 300 frames ≈ 2–4 minutes of
    frozen window while the socket request executes. Normal; don't
    kill it. Results are also written to disk (`apply_result.json`) so
    a dropped socket loses nothing.
14. **mcp SDK 2.x breaks the official Blender MCP server** (it imports
    `mcp.server.fastmcp`, removed in 2.0). Pin `mcp[cli]<2` when
    registering the server.
15. **Stale stills burn passes.** After every re-key, delete and
    re-render the stills before judging anything. An old PNG has
    cost this project at least two decisions.
16. **The live scene holds ONE action — the last one applied.** A
    preview rendered after working on another clip silently shows the
    wrong animation (a source-vs-retarget showcase then looks
    "desynchronized" when it is in fact a different clip).
    `render_preview.py` binds the spec's action before rendering for
    exactly this reason; anything else that renders the scene must do
    the same.

17. **Head orientation must come from the estimator, or the gaze is a
    lie.** Joint positions cannot describe where a head looks (the head
    joint sits inside the skull), so face landmarks must be read from
    real mesh vertices — synthesizing nose/ears from the torso basis
    makes the ear line *identical* to the shoulder line and locks the
    character's gaze to its chest for the whole clip. Symptoms: the head
    swings wide whenever the body twists, and "head vs chest yaw"
    measures exactly 0.000 at every frame — a number too clean to be
    real. Compounding it: the FK apply aims only the skull axis (+Y), so
    the FACE axis (head local +Z) inherits torso lean and yaw. Fix at
    the source (estimator emits `gaze`), orient the head to it in the
    apply, and let `qa_clip.py` flag a chest-locked head (gaze-vs-chest
    std < 2°) and sky-staring (mean elevation > 12°). Three passes were
    burned "fixing" this downstream before the estimator was suspected.
18. **Smoothing eats strikes.** A `smooth` window wide enough to calm a
    fast flurry (9 frames) also averages away its extension peaks —
    punches visibly shrink. Keep flurry windows narrow (5) and restore
    amplitude with `reach`.
19. **Judge limb height against the HEAD, not the shoulders.** A
    hand height that looks correct relative to the shoulder line can sit
    at face level on a character whose head/shoulder proportions differ
    from the performer's — the eye reads hands against the face. A
    guard measured "3 cm off" shoulder-relative was 10-19 cm off
    head-relative, which is what the owner saw.
20. **Abutting correction windows cancel at the handover.** Two
    `nudges` windows that meet (138-171, 168-204) each ramp through the
    junction, so the total correction dips exactly where they meet — a
    visible defect at that instant. Use one continuous window plus a
    smaller overlapping one, and make a correction reach full strength
    BEFORE the beat it is meant to fix.
21. **A "Hand" bone's world position is the WRIST.** Blender reports a
    bone's head; `mixamorig:*Hand` sits at the wrist, so dividing its
    distance-from-shoulder by (arm+fore+hand) understates extension by
    ~15%. Measure the wrist against arm+fore, or the fingertip bone.

## Process

22. **Beat sheet before any spec.** The 16-pass punch and the 4-pass
    kick differ by exactly this discipline; the two shipped clips took
    1–2 passes each.
23. **One constraint per pass; signed-off regions are frozen.** The
    single historical regression came from "improving" feet while
    polishing something else after the upper body was approved.
24. **Map human notes to the owning system before editing.** "Frames
    18–50, punching side" once meant the *guard*, not the punch — a
    pass was wasted editing the wrong function. Convert the frame range
    to src frames, find the beat, find the spec field; only then edit.
25. **QA passing ≠ done.** Numbers catch explosions, pops, skate and
    drift; they cannot see a wrong silhouette. Play the clip, read the
    stills, ask the human.
