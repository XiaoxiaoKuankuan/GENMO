# Music-to-motion and robot streaming

`scripts/demo/demo_music.py` turns an existing WAV, MP3, or FLAC file into
SMPL-X human motion. It does **not** generate music audio:

```text
music file
  -> EDGE baseline35 at 30 FPS
  -> full PyTorch gem_smpl.ckpt
  -> DDIM + classifier-free guidance
  -> raw 151-D GEM motion
  -> EnDecoder
  -> zero-shape global/in-camera SMPL-X parameters
  -> atomic READY generation directory
```

No video is read and YOLO, ByteTrack, ViTPose, HMR2, and T5 are not loaded.
Music generation requires the complete `inputs/pretrained/gem_smpl.ckpt`
composed with `exp=gem_smpl`. The real-time ONNX exports are regression-only;
they do not contain music-conditioned DDIM sampling and are never used here.

## EDGE baseline35 contract

The extractor follows EDGE's `data/audio_extraction/baseline_features.py`:

| Channels | Feature |
|---|---|
| `0` | onset strength |
| `1:21` | 20 MFCC channels |
| `21:33` | 12 chroma CENS channels |
| `33` | binary onset peak |
| `34` | binary beat peak |

Timing is fixed to 30 FPS, `sample_rate=15360`, and `hop_length=512`. The
checkpoint, model's first `music_embedder` linear layer, and feature tensor
must all have dimension 35. A mismatch is fatal: columns are never truncated,
padded, projected, or replaced by another audio representation.

EDGE's dataset script operated on already sliced five-second clips and then
kept 150 frames. This demo removes that fixed crop because it supports an
explicit `--start_sec`/`--duration_sec` range. Librosa boundary framing can
produce one boundary frame more than `duration × 30`; the generated duration is
therefore recorded as `feature_frames / 30`.

Install the existing audio dependency when needed:

```bash
python -m pip install "librosa>=0.10,<0.11"
```

## CPU dry-run

Dry-run reads the real audio and extracts real baseline35 features, but does
not load a checkpoint, GEM, CUDA, SMPL-X, Open3D, or create output/READY:

```bash
python scripts/demo/demo_music.py \
  --audio /path/to/song.wav \
  --duration_sec 10 \
  --dry_run
```

It prints feature shape/finiteness, BPM and peak counts, synthetic camera
shapes, and confirms that only `has_music_mask` is enabled.

## Generate a READY motion

```bash
python scripts/demo/demo_music.py \
  --audio /path/to/song.wav \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt \
  --start_sec 0 \
  --duration_sec 10 \
  --output_root outputs/music_motion \
  --seed 42 \
  --shape_mode zero \
  --guidance_scale 2.5 \
  --ddim_steps 50 \
  --no_render
```

`--num_samples N` publishes N independent directories using seeds `seed+i`.
`--save_features` remains accepted for compatibility; the READY protocol now
always includes `music_features.pt`. Without `--no_render`, the demo attempts
`motion_global.mp4`. `--mux_audio` additionally attempts
`motion_with_audio.mp4` with the selected source range. Open3D or ffmpeg
failure is a warning: valid motion parameters are still published.

The first implementation deliberately rejects selections above `--max_frames`
(default 600, about 20 seconds). Use `--start_sec` and `--duration_sec` for a
shorter segment; five to ten seconds is recommended for initial robot tests.
Long-song generation needs overlap handling, root and foot-contact continuity,
and beat-continuous stitching. Unsafe concatenation is not implemented.

## Atomic output protocol

Every sample is a direct child of `output_root`, so `MotionWatcher` can see it:

```text
outputs/music_motion/
  song_start0p000_seed42_<UTC>_<uuid>/
    smpl_params.pt
    motion.npz
    raw_motion_151d.pt
    music_features.pt
    metadata.json
    source_audio.txt
    motion_global.mp4          # optional
    motion_with_audio.mp4      # optional
    READY                      # always created last
```

Files are first written and flushed in `outputs/music_motion/.tmp_<uuid>/`.
The directory is atomically renamed with `os.replace`; only then is `READY`
written and flushed. A failed generation cleans its own temporary directory
and cannot expose READY. Existing generations are never replaced.

`smpl_params.pt` contains global and in-camera `body_pose [L,63]`,
`global_orient [L,3]`, `transl [L,3]`, and `betas [L,10]`, plus camera tensors,
FPS, audio selection, feature/checkpoint/seed settings, and metadata. Both beta
tensors are exact zeros. `motion.npz` also stores zero betas.
`raw_motion_151d.pt` preserves GEM's raw diffusion diagnostic output; its shape
components are not rewritten, and GMR never reads this file.

## Music-to-motion robot streaming

The four programs remain decoupled:

```text
demo_music.py                       music -> complete SMPL-X + READY
stream_smpl_params_to_gmr.py        watch/cache/interpolate -> SMP1 UDP
run_smplx_bumi3.sh                  SMPL-X targets -> BUMI3 retargeting
GMT                                 reference -> tracking policy -> robot
```

Terminal 1 — GMR-CPP/MuJoCo:

```bash
cd /home/weili/GMR-CPP_e1jump_lowdpi
./run_smplx_bumi3.sh \
  --always \
  --vis \
  --vis-smplx-targets \
  --vis-smplx-frames
```

Terminal 2 — persistent simulation streamer:

```bash
cd /home/weili/GENMO
source .venv/bin/activate
python scripts/demo/stream_smpl_params_to_gmr.py \
  --watch_dir outputs/music_motion \
  --source_filter music_only \
  --gmr_host 127.0.0.1 \
  --gmr_port 7006 \
  --publish_fps 30 \
  --shape_mode zero \
  --mode sim \
  --new_motion_policy queue
```

Add `--audio_playback ffplay` for best-effort local sound. The child process is
started only when BLENDING enters PLAYING and is terminated on RETURNING,
ERROR, ESTOP, interrupt, or streamer exit. `--audio_offset_sec` is added to the
source seek time to compensate local startup latency. This is not a hard-real-
time audio clock, and GMR transmission never depends on ffplay succeeding.

Terminal 3 — generate new music motions as needed using the command above.
Once READY exists, the running watcher adds the directory to its queue. Use
`--replay_existing` if the directory predates the watcher, or play it directly:

```bash
python scripts/demo/stream_smpl_params_to_gmr.py \
  --motion outputs/music_motion/<generation>/smpl_params.pt \
  --gmr_host 127.0.0.1 \
  --gmr_port 7006 \
  --publish_fps 30 \
  --shape_mode zero \
  --mode sim \
  --once
```

For robot mode, a stand pose already verified in simulation and on the target
platform is mandatory:

```bash
python scripts/demo/stream_smpl_params_to_gmr.py \
  --watch_dir outputs/music_motion \
  --source_filter music_only \
  --gmr_host 127.0.0.1 \
  --gmr_port 7006 \
  --publish_fps 30 \
  --shape_mode zero \
  --mode robot \
  --idle_motion inputs/motions/smplx_idle_stand.pt \
  --new_motion_policy queue \
  --estop_file /tmp/genmo_estop
```

The streamer uses the same validated state machine and SMP1 path as text
motions. With no action it keeps sending idle at `publish_fps`. New motions
blend from the current pose after root translation/yaw alignment. Completed
motions enter RETURNING and blend to idle rather than holding a potentially
unsafe last dance frame. Source betas are ignored and every `EnDecoder.fk_v2`
call receives exact `zeros(1,1,10)`.

`--source_filter music_only` reads `metadata.json`; text outputs and invalid or
incomplete directories are ignored and are not marked consumed. The existing
SMP1 412-byte packet, 14-target order, coordinate conversion, GMR-CPP, BUMI3,
Redis, and GMT interfaces are unchanged.

## Safety

The output is human SMPL-X motion, not motor torque. `mode=sim` is the default.
Robot mode requires a verified idle motion, defaults to queueing, and rejects
interrupt unless explicitly overridden. The software ESTOP returns toward idle
but cannot replace the physical emergency stop. Before hardware use, inspect
feet, root motion, speed, and pose limits in MuJoCo, then use reduced speed,
suspended support, and the robot's normal hardware protections.

## AIST++ preparation

The official annotations root is not a dependency for arbitrary `--audio`
inference. Batch AIST++ feature preparation additionally requires aligned,
same-stem per-sequence WAV files; a shared music-ID WAV cannot determine the
sequence offset. The preparation/audit tools never modify the annotation root:

```bash
python tools/data/aistpp/audit_aistpp.py \
  --annotations-root /home/weili/datasets/AISTPP_official/annotations

python tools/data/aistpp/extract_musicfeat_v2.py \
  --annotations-root /home/weili/datasets/AISTPP_official/annotations \
  --aligned-wav-dir /path/to/per_sequence_wavs \
  --output-dir inputs/AIST++/musicfeat_v2
```
