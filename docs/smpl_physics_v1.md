# SMPL 151D physics-v1 fine-tuning

This experiment is isolated in
`exp=gem_smpl_music_only_4set_physics_v1`. It keeps the 151D network and all
baseline losses unchanged, then adds low-weight GT derivative and sole-ground
losses. A baseline checkpoint is therefore loaded directly as model weights;
the optimizer and scheduler start from step zero.

## 1. Build the ground sidecars

Run these once against complete, canonical 30 FPS motions. Generation never
creates contact labels and never substitutes Y=0 for an invalid estimate.

```bash
python scripts/build_ground_sidecars.py --kind aist \
  --root inputs/AIST++ --device cuda

python scripts/build_ground_sidecars.py --kind manifest \
  --root /data0/user/liwei/datasets/music_dance_genmo/AIOZ-GDANCE --device cuda
python scripts/build_ground_sidecars.py --kind manifest \
  --root /data0/user/liwei/datasets/music_dance_genmo/FineDance --device cuda
python scripts/build_ground_sidecars.py --kind manifest \
  --root /data0/user/liwei/datasets/music_dance_genmo/CoMPAS3D --device cuda
```

The default output is `<dataset>/physics/ground_v1.jsonl`, matching the
experiment config. Rebuilding an existing sidecar requires `--overwrite`.

## 2. Mandatory stale-data preflight

The verification path checks contract versions, IDs, frame counts and the
SHA256 of every source motion (the AIST++ annotation artifact is hashed once).

```bash
python scripts/build_ground_sidecars.py --kind aist \
  --root inputs/AIST++ --verify-only
python scripts/build_ground_sidecars.py --kind manifest \
  --root /data0/user/liwei/datasets/music_dance_genmo/AIOZ-GDANCE --verify-only
python scripts/build_ground_sidecars.py --kind manifest \
  --root /data0/user/liwei/datasets/music_dance_genmo/FineDance --verify-only
python scripts/build_ground_sidecars.py --kind manifest \
  --root /data0/user/liwei/datasets/music_dance_genmo/CoMPAS3D --verify-only
```

## 3. One-step and 100-step checks

Use the exact checkpoint selected for fine-tuning. These commands intentionally
load weights through `pretrain_ckpt`; they do not resume the old optimizer.

```bash
python scripts/train.py exp=gem_smpl_music_only_4set_physics_v1 \
  pretrain_ckpt=/path/to/baseline.ckpt pl_trainer.devices=1 \
  pl_trainer.max_steps=1 use_wandb=false \
  data.loader_opts.train.num_workers=0

python scripts/train.py exp=gem_smpl_music_only_4set_physics_v1 \
  pretrain_ckpt=/path/to/baseline.ckpt pl_trainer.devices=1 \
  pl_trainer.max_steps=100 use_wandb=false
```

Inspect peak GPU memory during both runs. At step zero every new weighted term
is zero; it ramps linearly to the configured target at step 10000. The sum of
`physics_*_weighted_loss` should initially stay below about 2% of the original
total loss. Per-dataset discontinuity logs contain both a displayed masked
rate and exact `masked_count`/`candidate_count` totals.

## 4. Full fine-tuning

```bash
python scripts/train.py exp=gem_smpl_music_only_4set_physics_v1 \
  pretrain_ckpt=/path/to/baseline.ckpt
```

Defaults are 8 GPUs, per-GPU batch 128, 52224 global samples per epoch (51
steps), learning rate `2e-5`, 50k steps, LR halving at 30k/45k and checkpoint
saving every 5k steps. Use `resume_mode=last` only to resume a physics-v1 run.

Inference uses the existing music-only demo/server interface because no model
parameters or 151D fields changed; point it at a physics-v1 checkpoint in the
same way as the baseline checkpoint.
