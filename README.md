<p align="center">
  <h1 align="center">GEM: A Generalist Model for Human Motion</h1>
  <p align="center">
    <a href="https://jeffli.site/"><strong>Jiefeng Li</strong></a>
    ·
    <a href="https://www.jinkuncao.com/"><strong>Jinkun Cao</strong></a>
    ·
    <a href="https://cs.stanford.edu/~haotianz/"><strong>Haotian Zhang</strong></a>
    ·
    <a href="https://davrempe.github.io/"><strong>Davis Rempe</strong></a>
    ·
    <a href="https://jankautz.com/"><strong>Jan Kautz</strong></a>
    ·
    <a href="https://www.umariqbal.info/"><strong>Umar Iqbal</strong></a>
    ·
    <a href="https://ye-yuan.com/"><strong>Ye Yuan</strong></a>
  </p>
  <h2 align="center">ICCV 2025 (Highlight)</h2>
  <div align="center">
    <img src="./assets/teaser.png" alt="Logo" width="100%">
  </div>
</p>
<p align="center">
  <a href="https://research.nvidia.com/labs/dair/gem/"><img src="https://img.shields.io/badge/Project-Page-0099cc"></a>
  <a href="https://arxiv.org/abs/2505.01425"><img src="https://img.shields.io/badge/arXiv-2505.01425-b31b1b.svg"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-NVIDIA_OneWay_Noncommercial-green"></a>
</p>

GEM is a unified generative framework for human motion estimation and generation. GEM accepts multiple conditioning modalities — video, 2D keypoints, text, and audio — and handles multiple tasks without task-specific heads.

> For full-body motion estimation (hands + face), see [GEM-X](https://github.com/NVlabs/GEM-X).

---

## 📰 News
- **[March 2026]** 📢 **GEM-SMPL** is released with a multi-modal demo script.
- **[December 2025]** 📢 GENMO has been renamed to **GEM**.
- **[October 2025]** 📢 The **GEM** codebase is **released!**.

---

## 🚀 Quick Start

```bash
pip install uv && uv venv .venv --python 3.10 && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
bash scripts/install_env.sh
python scripts/demo/demo_smpl.py --input_list path/to/video.mp4 "text:a person walks forward" --ckpt_path inputs/pretrained/gem_smpl.ckpt
```

For full installation instructions (body model, checkpoints), see [docs/INSTALL.md](docs/INSTALL.md).

---

## 📦 Pretrained Models

| Model | Body Model | Description | Download |
|-------|-----------|-------------|----------|
| GEM-SMPL | SMPL | Regression + generation (text/audio/music/video) | [HuggingFace](https://huggingface.co/nvidia/GEM-X) |

Place checkpoints under `inputs/pretrained/` or pass the path directly via `--ckpt_path`. The demo scripts will automatically download the checkpoint from HuggingFace if `--ckpt` is not provided.

---

## 🎬 Demo

### Multi-modal demo (video + text)

The main demo supports mixed video and text conditioning — the core contribution of GEM.

**Video + inline text:**
```bash
python scripts/demo/demo_smpl.py \
  --input_list video1.mp4 "text:a person acting like a monkey" video2.mp4 \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt
```

**Video + text file:**
```bash
python scripts/demo/demo_smpl.py \
  --input_list video1.mp4 prompt.txt video2.mp4 \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt
```

**Multiple videos + multiple text prompts:**
```bash
python scripts/demo/demo_smpl.py \
  --input_list video1.mp4 "text:a person acting like a monkey" video2.mp4 "text:a person dances" \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--input_list` | — | Input list (required): `.mp4`/`.avi`/`.mov` files, `.txt` files, or `text:prompt` strings |
| `--ckpt_path` | `null` | Pretrained checkpoint path |
| `--text_length` | `300` | Number of frames for each text segment (300 = 10s at 30fps) |
| `--hmr2_ckpt` | `inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt` | HMR2 checkpoint for image features |
| `-s` / `--static_cam` | off | Assume static camera |
| `--output_root` | `outputs` | Output directory |
| `--no_render` | off | Skip visualization, only save SMPL parameters |

### Outputs

Results are saved to `outputs/<first_video_name>_mix/`:

| File | Description |
|---|---|
| `1_incam.mp4` | In-camera mesh overlay |
| `2_global.mp4` | Global-coordinate render |
| `3_incam_global_horiz.mp4` | Side-by-side comparison |
| `smpl_params.pt` | SMPL parameters (`body_params_global`, `body_params_incam`, `K_fullimg`, `segment_info`) |

### Text-only motion generation

Generate a complete SMPL motion directly from one text prompt, without an input video:

```bash
python scripts/demo/demo_smpl_text.py \
  --prompt "a person walks forward and waves" \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt \
  --num_frames 300
```

This path does not read video and does not run YOLO, ByteTrack, ViTPose, or HMR2. It uses T5-3B text embeddings and the full GEM DDIM/CFG diffusion sampler, so it requires the complete `gem_smpl` checkpoint. A checkpoint trained with `exp=gem_smpl_regression` cannot generate motion from text, and the real-time ONNX denoiser does not contain this text diffusion sampling path. The first run may need to download T5-3B; use `--t5_model /path/to/t5-3b --local_files_only` for an existing local model.

T5 loading is cache-first: a complete local Hugging Face cache is used without
making a network metadata request, so repeat runs are not affected by proxy
availability. If a download is required, do not use the unsupported
`socks://` proxy scheme with the default HTTPX installation. For a local mixed
HTTP proxy, use `HTTP_PROXY=http://127.0.0.1:7897` and
`HTTPS_PROXY=http://127.0.0.1:7897`, then unset `ALL_PROXY`/`all_proxy`. A real
SOCKS proxy requires HTTPX SOCKS support and a `socks5://` URL.

Results are written under `outputs/text_motion/` as SMPL parameter files, metadata, and (unless `--no_render` is set) a global-coordinate `global.mp4`. Use `--dry_run` to validate the synthetic camera/input tensor contract without loading T5, GEM, a checkpoint, or CUDA.

Each completed generation is published atomically in a unique directory. The
`READY` marker is created last, after `smpl_params.pt`, `motion.npz`,
`metadata.json`, `prompt.txt`, and any successfully generated render have been
closed and flushed. Runtime consumers must ignore directories without `READY`.

### Text-to-motion robot streaming

The text generator and robot player are separate processes. GEM may take longer
than real time to generate a complete action, while the persistent streamer
continues sending a cached motion, a smooth safety transition, or an idle pose
to GMR-CPP at a fixed rate:

```text
demo_smpl_text.py                  text -> complete SMPL-X action + READY
stream_smpl_params_to_gmr.py       watch/cache/interpolate -> SMP1 UDP
run_smplx_bumi3.sh                 SMPL-X targets -> BUMI3 retargeting
GMT                                reference trajectory -> tracking policy
```

Terminal 1 — start GMR-CPP and its MuJoCo visualization:

```bash
cd /home/weili/GMR-CPP_e1jump_lowdpi
./run_smplx_bumi3.sh \
  --always \
  --vis \
  --vis-smplx-targets \
  --vis-smplx-frames
```

Terminal 2 — keep the simulation streamer running:

```bash
cd /home/weili/GENMO
source .venv/bin/activate
python scripts/demo/stream_smpl_params_to_gmr.py \
  --watch_dir outputs/text_motion \
  --gmr_host 127.0.0.1 \
  --gmr_port 7006 \
  --publish_fps 30 \
  --shape_mode zero \
  --mode sim \
  --new_motion_policy queue
```

Terminal 3 — generate actions as needed:

```bash
python scripts/demo/demo_smpl_text.py \
  --prompt "A person walks forward, turns left, raises both arms and returns to a standing pose." \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt \
  --num_frames 300 \
  --fps 30 \
  --seed 42 \
  --shape_mode zero \
  --no_render
```

For robot mode, first extract a frame from a stand motion that has already been
validated in simulation and on the target platform:

```bash
python scripts/tools/extract_smpl_idle_pose.py \
  --motion outputs/verified_stand/smpl_params.pt \
  --frame 0 \
  --output inputs/motions/smplx_idle_stand.pt \
  --shape_mode zero

python scripts/demo/stream_smpl_params_to_gmr.py \
  --watch_dir outputs/text_motion \
  --gmr_host 127.0.0.1 \
  --gmr_port 7006 \
  --publish_fps 30 \
  --shape_mode zero \
  --mode robot \
  --idle_motion inputs/motions/smplx_idle_stand.pt \
  --new_motion_policy queue \
  --estop_file /tmp/genmo_estop
```

The player uses monotonic-time resampling, quaternion shortest-path
interpolation, root-position/yaw alignment, and explicit BLENDING, PLAYING,
RETURNING, HOLDING, ERROR, and ESTOP states. With no action it keeps publishing
idle targets; the default simulation idle is an arms-down standing pose rather
than the SMPL-X horizontal-arm T-pose. After every action it returns smoothly
to the aligned idle pose instead of holding the last frame. `queue` is the
recommended robot policy. `interrupt` is rejected in robot mode unless
explicitly enabled.

The synthetic arms-down pose is for simulation only. Robot mode continues to
require `--idle_motion`; that verified file determines the real robot idle body
pose and should itself contain a tested, arms-down safe stance.

`--shape_mode zero` is the only streamer shape policy: source betas are ignored
and every SMPL-X FK call receives `zeros(1, 1, 10)`. The software ESTOP file
causes a finite, short return to idle and latches until the file is removed and
a new action arrives. It does not replace the robot's physical emergency stop.
Before real-hardware testing, validate the idle/action in MuJoCo, use reduced
speed and suspended support, and retain the robot's normal safety systems. The
streamer sends pose references only; it does not send motor torques and does not
change the existing SMP1, GMR-CPP, BUMI3, Redis, or GMT protocols.

Validate an action without opening a UDP socket:

```bash
python scripts/demo/stream_smpl_params_to_gmr.py \
  --motion outputs/text_motion/example/smpl_params.pt \
  --shape_mode zero \
  --publish_fps 30 \
  --mode sim \
  --dry_run
```

### Music-to-motion robot streaming

Generate human SMPL-X motion from WAV, MP3, or FLAC without video, YOLO,
ViTPose, HMR2, or T5. This uses EDGE baseline35 and the complete PyTorch
`gem_smpl.ckpt` DDIM/CFG path; the regression-only ONNX exports cannot perform
music-conditioned generation.

Terminal 1 — start GMR-CPP/MuJoCo:

```bash
cd /home/weili/GMR-CPP_e1jump_lowdpi
./run_smplx_bumi3.sh \
  --always \
  --vis \
  --vis-smplx-targets \
  --vis-smplx-frames
```

Terminal 2 — keep the music streamer running:

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

Terminal 3 — generate a complete action and atomically publish READY:

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

Every sample is a direct `outputs/music_motion/<generation>/` child containing
SMPL parameters, raw diagnostics, music features, metadata, and a READY marker
created last. Both saved global/in-camera betas and every streamer FK beta are
zero. With no action, the streamer keeps sending idle at a fixed rate; after an
action it blends back to idle instead of holding the last frame.

Add `--audio_playback ffplay` to terminal 2 for best-effort local audio. Audio
failure never stops the control stream and is not a hard-real-time clock. Robot
mode requires `--idle_motion inputs/motions/smplx_idle_stand.pt`; first validate
in MuJoCo, then use reduced speed, suspended support, physical emergency stop,
and normal hardware protection. Software ESTOP cannot replace physical ESTOP.

See [Music-to-motion and robot streaming](docs/MUSIC_DEMO.md) for the exact
35-channel contract, dry-run, atomic output protocol, long-audio limit, direct
playback, robot command, and safety details.

### Video-only demo

For simple pose estimation without text conditioning, use `demo_smpl_hpe.py`:

```bash
python scripts/demo/demo_smpl_hpe.py \
  --video path/to/video.mp4 \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt
```

### Real-time webcam demo

`demo_webcam.py` runs the whole pipeline (YOLOX → ViTPose-H → HMR2 → GEM denoiser) frame-by-frame via ONNX Runtime, with a sliding window and streaming global rollout. See [docs/INSTALL.md](docs/INSTALL.md) Steps 8–10 for ONNX Runtime setup and ONNX export commands (one-time).

```bash
# Video file, OpenCV in-camera mesh overlay, no image features (fastest)
python scripts/demo/demo_webcam.py \
  --video path/to/video.mp4 --no_imgfeat \
  --render --render_mode opencv

# Webcam with Viser-based 3D world viewer (open http://localhost:8012)
python scripts/demo/demo_webcam.py \
  --camera_id 0 --no_imgfeat \
  --render --render_mode viser

# Webcam with neutral SMPL-X shape and GMR-CPP SMP1 streaming
python scripts/demo/demo_webcam.py \
  --camera_id 2 --no_imgfeat --display \
  --gmr_host 127.0.0.1 --gmr_port 7006 \
  --shape_mode zero
```

| Flag | Default | Purpose |
|---|---|---|
| `--video` / `--camera_id` | camera 0 | input source |
| `--context_frames` | 120 | sliding-window length (must match exported denoiser `--seq_len`) |
| `--no_imgfeat` | off | use the no-imgfeat denoiser variant; skips HMR2 entirely |
| `--render_mode {opencv,viser}` | `viser` | mesh-overlay window vs web 3D viewer |
| `--shape_mode {zero,first,mean,ema,per_frame}` | `zero` | control SMPL-X body shape consistently for rendering and GMR FK; `zero` uses the neutral mean body shape and avoids frame-to-frame or run-to-run body-proportion changes |
| `--no_async_pipeline` | off | force synchronous mode (lower throughput, zero pipeline lag) |

`--shape_mode` controls the SMPL-X body shape consistently for rendering and GMR FK. The default `zero` mode uses the neutral mean SMPL-X body shape and avoids frame-to-frame or run-to-run body-proportion changes.

For per-module latency profiling: `python tools/benchmark/benchmark_modules.py`.

---

## 🏋️ Training

See [Dataset Preparation](docs/DATA.md) for download links and directory structure.

**Regression model** (video → SMPL):

```bash
python scripts/train.py exp=gem_smpl_regression
```

**Full model** (regression + text/audio generation):

```bash
python scripts/train.py exp=gem_smpl
```

**Multi-GPU (DDP)**:

```bash
python scripts/train.py exp=gem_smpl_regression pl_trainer.devices=4
```

**SLURM**:

```bash
python scripts/train_slurm.py exp=gem_smpl_regression
```

### Key settings

From `configs/exp/gem_smpl_regression.yaml`:

- Body model: SMPLx
- Optimizer: AdamW (lr=2e-4)
- Precision: 16-mixed
- Max steps: 500K
- Gradient clipping: 0.5
- Validation every 3000 steps

Logging uses W&B by default. To disable:

```bash
python scripts/train.py exp=gem_smpl_regression use_wandb=false
```

---

See [FAQ](docs/FAQ.md) for common issues.

---


## 🤝 Related Humanoid Work at NVIDIA
GEM is part of a larger effort to enable humanoid motion data for robotics, physical AI, and other applications.

Check out these related works:
* [GEM-X](https://github.com/NVlabs/GEM-X)
* [SOMA Body Model](https://github.com/NVlabs/SOMA-X)
* [BONES-SEED Dataset](ttps://huggingface.co/datasets/bones-studio/seed)
* [ProtoMotions](https://github.com/NVlabs/ProtoMotions)
* [SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter)
* [SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl)
* [Kimodo](https://github.com/nv-tlabs/kimodo)


## 📖 Citation

```bibtex
@inproceedings{genmo2025,
  title     = {GENMO: A GENeralist Model for Human MOtion},
  author    = {Li, Jiefeng and Cao, Jinkun and Zhang, Haotian and Rempe, Davis and Kautz, Jan and Iqbal, Umar and Yuan, Ye},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```

---

## 📄 License

This project is released under the NVIDIA OneWay Noncommercial License — see [LICENSE](LICENSE) for details. Third-party components are subject to their own licenses; see [ATTRIBUTIONS.md](ATTRIBUTIONS.md) for specifics.
