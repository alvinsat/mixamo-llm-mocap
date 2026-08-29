# make sure inside root folder
hf download camenduru/GVHMR hmr2/epoch=10-step=25000.ckpt --local-dir tools/GVHMR/inputs/checkpoints/hmr2
hf download camenduru/GVHMR vitpose/vitpose-h-multi-coco.pth --local-dir tools/GVHMR/inputs/checkpoints/vitpose