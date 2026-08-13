# FineDance → GENMO canonical data

The downloaded release stores each motion as `[T,315]`, not raw axis-angle:

```text
translation [T,3] + 52 SMPL-H joints × continuous rotation-6D [T,312]
```

The converter uses GENMO's rotation conversion functions to recover axis-angle, keeps the
first 22 SMPL body joints as `pose [T,66]`, preserves `transl [T,3]`, and records neutral
`betas [T,10]` because FineDance supplies no subject shape. It drops 30 hand joints only after
the complete 52-joint rotation-6D tensor has been decoded. No coordinate, joint-order,
translation-scale, or second FPS conversion is applied to this 30 FPS release.

The bundled `music_npy` is audited but never used. Every WAV is processed once by
`gem.utils.music_features.extract_edge_baseline35`, yielding `[T,35] @ 30 Hz`.
The inspection report also preserves the release anomaly in `187.json`, whose song `name` is
the numeric value `711` rather than a string; this does not invalidate its motion/WAV pair.

## Inspect all 203 complete motion/WAV pairs

```bash
python tools/data/music_dance/finedance/inspect_finedance.py \
  --root /data0/user/liwei/datasets/music_dance_raw/FineDance/raw/finedance \
  --sample-count 10 \
  --source-fps 30 \
  --output /data0/user/liwei/datasets/music_dance_genmo/FineDance-inspection.json \
  --strict
```

## Ten-sample smoke conversion

The IDs below cover train/val/test and have an independently audited WAV/motion disagreement
of at most two 30 Hz frames.

```bash
python tools/data/music_dance/finedance/convert_finedance_to_genmo.py \
  --root /data0/user/liwei/datasets/music_dance_raw/FineDance/raw/finedance \
  --output-root /data0/user/liwei/datasets/music_dance_genmo/FineDance-smoke10 \
  --sample-id 001 --sample-id 074 --sample-id 193 --sample-id 202 \
  --sample-id 037 --sample-id 063 --sample-id 098 --sample-id 144 \
  --sample-id 161 --sample-id 211 \
  --source-fps 30 --target-fps 30 \
  --strict

python tools/data/music_dance/finedance/validate_finedance_genmo.py \
  --root /data0/user/liwei/datasets/music_dance_genmo/FineDance-smoke10 \
  --smpl-forward-samples 3 \
  --strict
```

To render three samples on a machine with a working EGL/OpenGL setup, append:

```bash
  --render-samples 3 --render-dir outputs/finedance-smoke10-renders
```

## Full conversion

The default strict time-alignment contract rejects rather than silently trims any sample whose
WAV/motion duration differs by more than two frames. The downloaded public subset contains
20 such samples, so a strict all-ID run is intentionally expected to return non-zero while
still exporting the 183 safe pairs and a complete failure report:

```bash
python tools/data/music_dance/finedance/convert_finedance_to_genmo.py \
  --root /data0/user/liwei/datasets/music_dance_raw/FineDance/raw/finedance \
  --output-root /data0/user/liwei/datasets/music_dance_genmo/FineDance \
  --source-fps 30 --target-fps 30 \
  --strict

python tools/data/music_dance/finedance/validate_finedance_genmo.py \
  --root /data0/user/liwei/datasets/music_dance_genmo/FineDance \
  --smpl-forward-samples 10 \
  --strict
```

Do not increase `--max-audio-motion-frame-mismatch` merely to force the 20 bad pairs through.
Their discrepancies reach hundreds of frames and require dataset-specific synchronization
metadata that the public release does not bundle.
