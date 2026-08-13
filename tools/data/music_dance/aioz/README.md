# AIOZ-GDANCE → GENMO canonical data

This directory implements the data-only first stage. It does not change the GENMO network,
existing AIST++ artifacts, or training configuration.

The downloaded AIOZ archive contains one group-level motion pickle and one same-stem WAV per
sequence. Each pickle is validated as:

- `smpl_poses [P,T,72]`: standard SMPL axis-angle order;
- `root_trans [P,T,3]`: preserved without scale/axis transforms;
- `smpl_betas [P,T,10]`;
- `meta`: includes the original clip interval and person count.

Each dancer becomes one output motion. The exact pose slice is `global_orient = pose[:3]` and
`body_pose = pose[3:66]`; source dimensions `66:72` are discarded. Every dancer in a group
references the one shared `musicfeat_v2/<group>_musicfeat_fps30.pt` file.

## Inspect

Inspect ten random group sequences (with at least one from each official split):

```bash
python tools/data/music_dance/aioz/inspect_aioz.py \
  --root /data0/user/liwei/datasets/music_dance_raw/AIOZ-GDANCE \
  --num-groups 10 \
  --source-fps 30 \
  --strict \
  --output outputs/aioz_inspection_10.json
```

Audit every group:

```bash
python tools/data/music_dance/aioz/inspect_aioz.py \
  --root /data0/user/liwei/datasets/music_dance_raw/AIOZ-GDANCE \
  --all \
  --source-fps 30 \
  --strict \
  --output outputs/aioz_inspection_full.json
```

The downloaded PKL has no explicit FPS key. Inspection verifies the configured source FPS from
the motion tensor length, `meta.orig_end - meta.orig_start`, filename span, and WAV duration.

## Ten-group conversion and validation

```bash
python tools/data/music_dance/aioz/convert_aioz_to_genmo.py \
  --root /data0/user/liwei/datasets/music_dance_raw/AIOZ-GDANCE \
  --output-root /data0/user/liwei/datasets/music_dance_genmo/AIOZ-GDANCE-smoke10 \
  --source-fps 30 \
  --target-fps 30 \
  --sample-groups 10 \
  --seed 20260813 \
  --workers 4 \
  --strict

EGL_PLATFORM=surfaceless python tools/data/music_dance/aioz/validate_aioz_genmo.py \
  --root /data0/user/liwei/datasets/music_dance_genmo/AIOZ-GDANCE-smoke10 \
  --strict \
  --render-samples 3 \
  --seed 20260813
```

## Full conversion

Use a fresh output directory. The converter calls
`gem.utils.music_features.extract_edge_baseline35` once per group and explicitly aligns only a
small (default at most two-frame) STFT boundary difference. Each worker uses an isolated Numba
cache below `/tmp/genmo_aioz_numba_cache`; this avoids stale shared-cache crashes without changing
the librosa/EDGE feature computation.

```bash
python tools/data/music_dance/aioz/convert_aioz_to_genmo.py \
  --root /data0/user/liwei/datasets/music_dance_raw/AIOZ-GDANCE \
  --output-root /data0/user/liwei/datasets/music_dance_genmo/AIOZ-GDANCE \
  --source-fps 30 \
  --target-fps 30 \
  --workers 8 \
  --strict

python tools/data/music_dance/aioz/validate_aioz_genmo.py \
  --root /data0/user/liwei/datasets/music_dance_genmo/AIOZ-GDANCE \
  --strict
```

The output manifests retain the official group-level split. The validator rejects split leakage,
duplicate samples, non-finite motion/music, wrong dimensions, unshared group music references,
or any final `motion_T != music_T`.
