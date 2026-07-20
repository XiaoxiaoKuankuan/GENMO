# Music-only SMPL motion generation

`scripts/demo/demo_music.py` generates SMPL human motion directly from a WAV, MP3, or FLAC file:

```text
music file
  -> EDGE baseline35 features at 30 FPS
  -> full PyTorch gem_smpl.ckpt
  -> DDIM + classifier-free guidance
  -> 151-D GEM motion representation
  -> EnDecoder
  -> global and in-camera SMPL parameters
```

This path uses the complete `inputs/pretrained/gem_smpl.ckpt` model composed with `exp=gem_smpl`. The current real-time `gem_smpl_denoiser.onnx` export is regression-only and does not contain the music-conditioned diffusion sampler, so it cannot be used for music generation. A `gem_smpl_regression` checkpoint is also unsupported.

## Install the audio feature dependency

EDGE baseline features require librosa. It is declared as a GENMO runtime dependency; for an existing environment, install it with:

```bash
python -m pip install "librosa>=0.10,<0.11"
```

No T5 model is loaded for music-only generation.

## EDGE baseline35 definition

The extractor follows [`Stanford-TML/EDGE/data/audio_extraction/baseline_features.py`](https://github.com/Stanford-TML/EDGE/blob/main/data/audio_extraction/baseline_features.py):

| Channels | Feature |
|---|---|
| `0` | onset strength |
| `1:21` | 20 MFCC channels |
| `21:33` | 12 chroma CENS channels |
| `33` | binary onset peak |
| `34` | binary beat peak |

It uses 30 FPS, `hop_length=512`, and `sample_rate=15360`. EDGE's original preparation first sliced audio into five-second windows and then retained 150 frames. This implementation removes that fixed five-second crop because it accepts complete songs and user-selected ranges. AIST music IDs use EDGE's BPM priors; arbitrary filenames use librosa tempo estimation.

Feature dimension mismatches are fatal. The code does not truncate columns, append zero columns, apply PCA, or use random projections.

## Arbitrary music inference

```bash
python scripts/demo/demo_music.py \
  --audio /path/to/song.wav \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt \
  --duration_sec 10 \
  --output_root outputs/music_demo \
  --save_features
```

Add `--no_render` to save parameters without Open3D rendering. Add `--mux_audio` to create a rendered video with the selected original audio segment when ffmpeg is available:

```bash
python scripts/demo/demo_music.py \
  --audio /path/to/song.flac \
  --duration_sec 10 \
  --mux_audio
```

The first version intentionally does not segment and stitch long diffusion generations. If the selected audio produces more than `--max_frames` (default 600), inference stops with an error. Use `--start_sec` and `--duration_sec` to choose a shorter range; this also reduces GPU memory usage.

For multiple stochastic samples, use `--num_samples N`. Sample `i` uses `seed + i`.

## Output layout

With `--save_features --mux_audio`, output is:

```text
outputs/music_demo/<audio_stem>/
  music_features.pt
  music_features_meta.json
  run_config.json
  sample_000/
    smpl_params.pt
    motion_151d.pt
    motion_global.mp4
    motion_with_audio.mp4
  sample_001/
    ...
```

`smpl_params.pt` contains global and in-camera `body_pose`, `global_orient`, `transl`, and `betas`, plus camera intrinsics, source audio selection, FPS, and seed. `motion_151d.pt` is the actual final `pred_x` returned by GEM's diffusion pipeline. Parameters are always saved even when rendering or ffmpeg is unavailable.

The result is SMPL human motion, not robot motor commands. A physical robot still requires morphology-aware retargeting, safety constraints, a controller, and hardware validation.

## AIST++ feature preparation

The official annotations directory is not required for arbitrary `--audio` inference and normally contains annotations only. Batch preparation additionally requires already aligned, same-stem per-sequence WAV files; a shared music-ID file such as `mBR0.wav` cannot establish the sequence offset.

Audit source data without modifying it:

```bash
python tools/data/aistpp/audit_aistpp.py \
  --annotations-root /home/weili/datasets/AISTPP_official/annotations
```

Extract one file:

```bash
python tools/data/aistpp/extract_musicfeat_v2.py \
  --audio /path/song.wav \
  --output /path/song_musicfeat_fps30.pt
```

Extract aligned AIST++ sequences:

```bash
python tools/data/aistpp/extract_musicfeat_v2.py \
  --annotations-root /home/weili/datasets/AISTPP_official/annotations \
  --aligned-wav-dir /path/to/per_sequence_wavs \
  --output-dir inputs/AIST++/musicfeat_v2
```

Missing same-stem WAVs are written to `missing_wavs.json`; the command returns nonzero unless `--allow-missing` is explicitly selected. The annotations root is read-only to these tools.
