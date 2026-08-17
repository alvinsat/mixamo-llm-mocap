# Prompting gen-video plates

The pipeline retargets *any* locked-camera footage, but its cheapest
source is AI video (the shipped plates came from Seedance and an
earlier image-to-video tool). This is how to prompt so the result
survives retargeting.

## The contract a plate must honor

1. **Camera locked**, full body in frame every frame, even light,
   1080p (720p minimum), 24–30 fps, ~10 s.
2. **T-pose held ~1 s at the START and the END**, square to camera.
   These bookends are the rest anchors (`rest_blend_start/_end`) and
   the proof the identity didn't drift. If the start pose is dirty the
   end T-pose alone is enough — but demand both anyway.
3. **Chest and hips face the camera** unless a turn IS the motion.
   Turns are retargetable; occlusion is not.
4. **Every moving limb stays visible.** Kicks travel just to the
   performer's side of the centerline so the leg never hides behind
   the torso. No props, tight kit, bare feet.
5. One continuous take, on the spot: no walking, no spinning, no
   extra strikes. Name the exact moves and forbid the rest.
6. Reuse the same first-frame still (image-to-video) across plates —
   same performer, same studio, same grid. Identity drift between
   plates breaks nothing technically but muddies comparison.

## Template

> Same man, same grey studio, same floor grid, camera locked, full body
> in frame the whole time. He starts holding this exact T-pose,
> perfectly still and facing the camera straight on, for a full second.
> Then he [MOVE 1 in one clause]. He then [MOVE 2]. Next he [MOVE 3,
> with visibility hints like "the kicking foot staying just to HIS
> right of the centerline so the whole leg stays visible"]. Finally he
> lowers his arms and spreads them straight out horizontally into a
> T-pose, feet returning under his shoulders, and holds that T-pose
> perfectly still, facing the camera, for the last full second. He
> stays on the same spot the whole time. His chest and hips face the
> camera the entire clip. No walking, no turning, no spinning, no
> extra strikes. Even lighting, bare feet, tight athletic kit, no
> props.

Write a must-see beat checklist next to the prompt (time window → what
must be visible) and regenerate the video if any beat fails — a hidden
foot cannot be retargeted, and regeneration is cheaper than a wasted
pass.

## Expect the model to disobey

Every plate so far deviated from its prompt somewhere: a left-knee jump
delivered as right-knee, an extended kick delivered as a knee snap, an
opening T-pose delivered as an A-pose. None of that matters IF you take
the beats and limbs from `analyze_landmarks.py` instead of from the
prompt (docs/PITFALLS.md #5–6). Prompt for what you want; retarget what
you got.
