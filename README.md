<p align="center">
  <h1 align="center">GEM：通用人体动作模型</h1>
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
  <h2 align="center">ICCV 2025（Highlight）</h2>
  <div align="center">
    <img src="./assets/teaser.png" alt="GEM 项目预览" width="100%">
  </div>
</p>
<p align="center">
  <a href="https://research.nvidia.com/labs/dair/gem/"><img src="https://img.shields.io/badge/Project-Page-0099cc"></a>
  <a href="https://arxiv.org/abs/2505.01425"><img src="https://img.shields.io/badge/arXiv-2505.01425-b31b1b.svg"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-NVIDIA_OneWay_Noncommercial-green"></a>
</p>

GEM 是一个用于人体动作估计与生成的统一生成式框架。GEM 可接收视频、2D 关键点、文本和音频等多种条件，并通过统一模型完成多类任务，无需为每项任务设计独立的专用头。

> 如需包含手部和面部的全身动作估计，请参阅 [GEM-X](https://github.com/NVlabs/GEM-X)。

---

## 📰 最新动态

- **[2026 年 3 月]** 📢 发布带有多模态 Demo 脚本的 **GEM-SMPL**。
- **[2025 年 12 月]** 📢 GENMO 正式更名为 **GEM**。
- **[2025 年 10 月]** 📢 **GEM** 代码库正式发布。

---

## 🚀 快速开始

```bash
pip install uv && uv venv .venv --python 3.10 && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
bash scripts/install_env.sh
python scripts/demo/demo_smpl.py --input_list path/to/video.mp4 "text:a person walks forward" --ckpt_path inputs/pretrained/gem_smpl.ckpt
```

完整安装说明（包括身体模型和 checkpoint）请参阅 [docs/INSTALL.md](docs/INSTALL.md)。

---

## 📦 预训练模型

| 模型 | 身体模型 | 说明 | 下载地址 |
|---|---|---|---|
| GEM-SMPL | SMPL | 回归与生成（文本、语音、音乐、视频） | [HuggingFace](https://huggingface.co/nvidia/GEM-X) |

请将 checkpoint 放在 `inputs/pretrained/` 下，也可以通过 `--ckpt_path` 直接指定路径。如果没有显式提供 checkpoint，Demo 脚本会按其默认逻辑从 Hugging Face 下载。

---

## 🎬 使用演示

### 多模态演示（视频 + 文本）

主演示支持视频与文本混合条件，这是 GEM 的核心能力之一。

**视频 + 行内文本：**

```bash
python scripts/demo/demo_smpl.py \
  --input_list video1.mp4 "text:a person acting like a monkey" video2.mp4 \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt
```

**视频 + 文本文件：**

```bash
python scripts/demo/demo_smpl.py \
  --input_list video1.mp4 prompt.txt video2.mp4 \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt
```

**多个视频 + 多个文本提示：**

```bash
python scripts/demo/demo_smpl.py \
  --input_list video1.mp4 "text:a person acting like a monkey" video2.mp4 "text:a person dances" \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt
```

### 主要参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input_list` | — | 必需输入列表：`.mp4`、`.avi`、`.mov`、`.txt` 文件或 `text:提示词` 字符串 |
| `--ckpt_path` | `null` | 预训练 checkpoint 路径 |
| `--text_length` | `300` | 每个文本片段的帧数；30 FPS 下 300 帧为 10 秒 |
| `--hmr2_ckpt` | `inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt` | 图像特征使用的 HMR2 checkpoint |
| `-s` / `--static_cam` | 关闭 | 假设相机静止 |
| `--output_root` | `outputs` | 输出目录 |
| `--no_render` | 关闭 | 跳过可视化，只保存 SMPL 参数 |

### 输出文件

结果保存在 `outputs/<first_video_name>_mix/`：

| 文件 | 说明 |
|---|---|
| `1_incam.mp4` | 相机坐标中的人体网格叠加视频 |
| `2_global.mp4` | 全局坐标动作渲染视频 |
| `3_incam_global_horiz.mp4` | 相机视图与全局视图的横向对比视频 |
| `smpl_params.pt` | SMPL 参数：`body_params_global`、`body_params_incam`、`K_fullimg`、`segment_info` |

### 纯文本动作生成

无需输入视频，直接通过一条文本提示生成完整 SMPL 动作：

```bash
python scripts/demo/demo_smpl_text.py \
  --prompt "a person walks forward and waves" \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt \
  --num_frames 300
```

这条路径不读取视频，也不运行 YOLO、ByteTrack、ViTPose 或 HMR2。它使用 T5-3B 文本特征和完整 GEM DDIM/CFG 扩散采样器，因此必须使用完整 `gem_smpl` checkpoint。通过 `exp=gem_smpl_regression` 训练的 checkpoint 无法从文本生成动作，实时 ONNX denoiser 也不包含文本扩散采样路径。

第一次运行可能需要下载 T5-3B。若本地已有模型，请使用：

```bash
--t5_model /path/to/t5-3b --local_files_only
```

T5 采用“本地缓存优先”加载：如果 Hugging Face 本地缓存完整，则不会发起网络元数据请求，因此重复运行不受代理可用性影响。需要下载时，默认 HTTPX 环境不支持 `socks://` 代理协议。对于本地 HTTP 混合代理，可设置：

```bash
HTTP_PROXY=http://127.0.0.1:7897
HTTPS_PROXY=http://127.0.0.1:7897
```

同时取消 `ALL_PROXY` 和 `all_proxy`。真正的 SOCKS 代理需要安装 HTTPX SOCKS 支持，并使用 `socks5://` 地址。

结果写入 `outputs/text_motion/`，其中包含 SMPL 参数、元数据；未指定 `--no_render` 时还会生成全局坐标视频 `global.mp4`。使用 `--dry_run` 可以在不加载 T5、GEM、checkpoint 或 CUDA 的情况下验证合成相机与输入张量契约。

每次成功生成都会原子发布到一个唯一目录。`smpl_params.pt`、`motion.npz`、`metadata.json`、`prompt.txt` 和成功生成的渲染文件全部关闭并刷新后，才会最后创建 `READY`。运行时消费者必须忽略没有 `READY` 的目录。

### 常驻文本动作生成服务

`demo_smpl_text.py` 的单次模式会依次加载 T5-3B、编码文本、释放 T5、加载 GEM、初始化 DDIM、生成一次后退出，适合离线使用，但每条命令都要重复约 20 秒的模型启动开销。`demo_smpl_text_server.py` 在启动时把 FP16 T5-3B 和完整 GEM-SMPL 同时常驻同一张 GPU，并且只初始化一次固定的 DDIM/CFG；后续请求只执行文本编码、扩散生成和原子保存，实测通常约 1 秒以内。

服务正常请求路径不会卸载模型、不会 CPU offload、不会量化，也不会调用 `torch.cuda.empty_cache()`。相同 prompt 的 `[50,1024]` T5 embedding 会进入 CPU LRU 缓存，不会随 prompt 数量占用更多 GPU 显存。只有 CUDA OOM 恢复或服务最终关闭时才允许清理 CUDA cache。

stdin 常驻模式：

```bash
cd /home/weili/GENMO
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 \
python scripts/demo/demo_smpl_text_server.py \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt \
  --t5_model t5-3b \
  --local_files_only \
  --device cuda:0 \
  --text_encoder_dtype float16 \
  --output_root outputs/text_motion \
  --transport stdin \
  --num_frames 120 \
  --fps 30 \
  --seed 42 \
  --ddim_steps 20 \
  --guidance_scale 2.5 \
  --shape_mode zero
```

看到 `SERVICE READY` 和 `text-motion>` 后，可直接输入动作文本，也可输入一行 JSON：

```text
A person performs exactly one squat and returns to standing.
{"request_id":"walk-001","prompt":"A person continuously walks straight forward.","num_frames":300,"fps":30,"seed":44}
```

管理命令为 `/status`、`/help`、`/clear-cache` 和 `/quit`。`/quit`、Ctrl+C 或 SIGTERM 会停止接收新请求，释放 T5/GEM，最后清理 CUDA cache。

ZMQ 常驻模式：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/demo/demo_smpl_text_server.py \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt \
  --t5_model t5-3b \
  --local_files_only \
  --device cuda:0 \
  --text_encoder_dtype float16 \
  --output_root outputs/text_motion \
  --transport zmq \
  --bind tcp://127.0.0.1:7010 \
  --num_frames 120 \
  --fps 30 \
  --seed 42 \
  --ddim_steps 20 \
  --guidance_scale 2.5 \
  --shape_mode zero
```

另一个终端发送请求：

```bash
python scripts/demo/text_motion_client.py \
  --endpoint tcp://127.0.0.1:7010 \
  --request_id squat-001 \
  --prompt "A person performs exactly one squat and returns to standing." \
  --num_frames 120 \
  --fps 30 \
  --seed 42 \
  --timeout_seconds 30
```

每次成功请求仍按现有协议生成直接位于 `outputs/text_motion/` 下的唯一 READY 目录，其中包含 `smpl_params.pt`、`motion.npz`、`prompt.txt` 和 `metadata.json`；同时原子更新 `outputs/text_motion/latest_ready.json`。现有 GMR streamer 已直接监视 READY 目录，不需要读取 latest 文件，也不需要改变 SMP1：

```bash
python scripts/demo/stream_smpl_params_to_gmr.py \
  --watch_dir outputs/text_motion \
  --source_filter text_only \
  --gmr_host 127.0.0.1 \
  --gmr_port 7006 \
  --publish_fps 30 \
  --shape_mode zero \
  --mode sim \
  --new_motion_policy queue
```

RTX 4090 实测：T5 加载后 allocated 约 3.061 GiB，T5+GEM 加载后约 5.041 GiB，预热后约 5.048 GiB；120 帧请求约 0.51～0.55 秒，300 帧请求约 0.79 秒。连续 50 次混合请求后 PyTorch allocated 增长为 0 MiB。该性能取决于 GPU、CUDA、checkpoint、磁盘与是否命中文本缓存。

### 文本动作到机器人实时播放

文本生成器与机器人播放器是两个独立进程。即使 GEM 生成一整段动作的速度慢于实时，常驻 streamer 仍会以固定频率向 GMR-CPP 发送缓存动作、安全过渡姿态或 idle 姿态：

```text
demo_smpl_text.py                  文本 -> 完整 SMPL-X 动作 + READY
stream_smpl_params_to_gmr.py       监视/缓存/插值 -> SMP1 UDP
run_smplx_bumi3.sh                 SMPL-X 目标 -> BUMI3 重定向
GMT                                参考轨迹 -> tracking policy
```

终端 1：启动 GMR-CPP 与 MuJoCo 可视化。

```bash
cd /home/weili/GMR-CPP_e1jump_lowdpi
./run_smplx_bumi3.sh \
  --always \
  --vis \
  --vis-smplx-targets \
  --vis-smplx-frames
```

终端 2：保持仿真 streamer 常驻运行。

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

终端 3：按需生成新动作。

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

实物机器人模式下，首先从已在仿真和目标平台上验证过的站立动作中提取一帧：

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

播放器采用基于单调时钟的重采样、四元数最短路径插值、根节点位置/yaw 对齐，并使用明确的 `BLENDING`、`PLAYING`、`RETURNING`、`HOLDING`、`ERROR` 和 `ESTOP` 状态。没有动作时仍持续发送 idle 目标；默认仿真 idle 是双臂下垂的站立姿态，而不是 SMPL-X 双臂水平展开的 T-pose。每段动作结束后会平滑返回对齐后的 idle，不会停留在最后一帧。

`queue` 是实物机器人推荐策略。除非显式允许，否则 robot 模式拒绝 `interrupt`。仿真用的合成双臂下垂姿态不能直接用于实物；robot 模式仍要求传入经过验证的 `--idle_motion`。

`--shape_mode zero` 是 streamer 唯一支持的体型策略：忽略源动作中的 betas，每次 SMPL-X FK 都接收 `zeros(1, 1, 10)`。软件 ESTOP 文件触发后，播放器会在有限时间内返回 idle 并保持锁存，直到文件删除且有新动作到达。软件 ESTOP 不能替代机器人硬件急停。

streamer 只发送姿态参考，不发送电机力矩，也不会修改 SMP1、GMR-CPP、BUMI3、Redis 或 GMT 协议。实机测试前必须先在 MuJoCo 中验证 idle 和动作，并使用低速、吊装保护及机器人原有安全系统。

不创建 UDP socket 的动作验证命令：

```bash
python scripts/demo/stream_smpl_params_to_gmr.py \
  --motion outputs/text_motion/example/smpl_params.pt \
  --shape_mode zero \
  --publish_fps 30 \
  --mode sim \
  --dry_run
```

### 音乐动作到机器人实时播放

无需视频、YOLO、ViTPose、HMR2 或 T5，直接从 WAV、MP3 或 FLAC 生成 SMPL-X 人体动作。该路径使用 EDGE baseline35 和完整 PyTorch `gem_smpl.ckpt` 的 DDIM/CFG；回归型 ONNX 导出不能进行音乐条件动作生成。

终端 1：启动 GMR-CPP/MuJoCo。

```bash
cd /home/weili/GMR-CPP_e1jump_lowdpi
./run_smplx_bumi3.sh \
  --always \
  --vis \
  --vis-smplx-targets \
  --vis-smplx-frames
```

终端 2：保持音乐动作 streamer 常驻运行。

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

终端 3：生成完整动作并原子发布 READY。

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

#### 常驻音乐动作生成服务

上面的 `demo_music.py` 单次模式保持不变。需要连续提交多首音乐或多个片段时，可让完整 GEM-SMPL 和固定 DDIM/CFG 常驻同一张 GPU，避免每次请求重新加载 checkpoint 和初始化扩散采样器。音乐服务不加载 T5，也不修改文本常驻服务协议。

stdin 模式：

```bash
cd /home/weili/GENMO
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 \
python scripts/demo/demo_music_server.py \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt \
  --device cuda:0 \
  --output_root outputs/music_motion \
  --transport stdin \
  --duration_sec 10 \
  --ddim_steps 20 \
  --guidance_scale 2.5 \
  --shape_mode zero
```

看到 `[ResidentMusic] SERVICE READY` 和 `music-motion>` 后，可以直接粘贴一行 WAV、MP3 或 FLAC 路径。路径中可以包含空格，也可以使用一对单引号或双引号：

```text
/home/weili/music/song.wav
"/home/weili/music/My Song.mp3"
{"request_id":"music-001","audio_path":"/home/weili/music/song.flac","start_sec":15,"duration_sec":10,"seed":7}
```

直接输入路径时默认从第 0 秒生成 10 秒动作。JSON 中可将 `duration_sec` 设为 `null`，表示从 `start_sec` 生成到文件末尾，但仍受 `--max_frames` 限制；默认 600 帧约为 20 秒，不会自动截断或拼接长音乐。

管理命令为 `/status`、`/help`、`/clear-cache` 和 `/quit`。`/clear-cache` 只清除 CPU 中的 EDGE baseline35 特征 LRU 缓存，不卸载常驻 GEM，也不会重新初始化 DDIM。缓存 key 包含解析后的绝对路径、文件 inode/大小/mtime、音频范围和特征版本；源文件被替换或修改后会自动 cache miss。

可通过 `--allowed_audio_root` 重复配置允许访问的服务端目录：

```bash
python scripts/demo/demo_music_server.py \
  --allowed_audio_root /home/weili/music \
  --allowed_audio_root /mnt/shared/audio
```

路径会先解析符号链接再检查，不能使用 `../` 或 symlink 越过白名单。stdin 和 ZMQ 请求中的 `audio_path` 都是**服务端机器上的文件路径**，协议不会上传音频字节；跨机器调用需要共享挂载。

ZMQ 服务默认只监听本机 `127.0.0.1:7011`：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/demo/demo_music_server.py \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt \
  --device cuda:0 \
  --output_root outputs/music_motion \
  --transport zmq \
  --bind tcp://127.0.0.1:7011 \
  --duration_sec 10 \
  --ddim_steps 20 \
  --guidance_scale 2.5
```

另一个终端发送路径请求：

```bash
python scripts/demo/music_motion_client.py \
  --endpoint tcp://127.0.0.1:7011 \
  --audio "/home/weili/music/My Song.wav" \
  --start_sec 0 \
  --duration_sec 10 \
  --seed 42 \
  --timeout_seconds 60
```

服务从选定音频范围提取固定 30 FPS 的 35 维特征，生成帧数由实际特征长度决定。相同文件和范围的第二次请求会命中特征缓存。每次成功请求仍在 `outputs/music_motion/` 下发布现有 `music_only` READY 直接子目录，现有 `MotionWatcher(source_filter=music_only)` 无需修改即可消费。

常驻音乐服务 v1 专注低延迟参数生成，不执行 Open3D 渲染和 ffmpeg mux。需要离线视频时继续使用单次 `demo_music.py`；需要播放原音乐时继续由 streamer 的 `--audio_playback ffplay` 控制。音频播放仍是 best-effort，不属于机器人硬实时控制时钟。

每个 sample 都是 `outputs/music_motion/<generation>/` 的直接子目录，包含 SMPL 参数、原始诊断数据、音乐特征、元数据，以及最后创建的 READY。保存的 global/incam betas 和 streamer 每次 FK 使用的 betas 都是全零。没有动作时 streamer 继续定频发送 idle；动作结束后平滑返回 idle，不保持最后一帧。

终端 2 增加 `--audio_playback ffplay` 可同时进行尽力而为的本地音乐播放。音频失败不会中断控制流，这也不是硬实时音频时钟。robot 模式必须提供 `--idle_motion inputs/motions/smplx_idle_stand.pt`。先在 MuJoCo 中验证，再使用低速、吊装保护、物理急停和机器人正常硬件保护；软件 ESTOP 不能替代物理急停。

完整的 35 通道契约、dry-run、原子输出协议、长音频限制、直接播放命令和安全说明请参阅 [音乐动作生成与机器人实时播放](docs/MUSIC_DEMO.md)。

### 统一多模态常驻服务

`demo_multimodal_server.py` 将实时视频跟随、文本动作生成、音乐动作生成和文本+音乐联合生成放在同一个常驻进程中，并统一交给一个 30 Hz GMR 输出端。它不会创建 `ResidentTextMotionEngine` 和 `ResidentMusicMotionEngine` 两套模型，而是由 `ResidentMultimodalMotionEngine` 只持有一个 T5 tokenizer、一个 FP16 T5-3B、一个完整 GEM-SMPL 和一个初始化后的 DDIM。文本 embedding 与 EDGE baseline35 音乐特征仅缓存在 CPU。

支持的生成模式：

| 模式 | GEM 条件 | 输出 `source` |
|---|---|---|
| `text` | `[50,1024]` T5 文本特征 | `text_only` |
| `music` | `[L,35]` EDGE baseline35 | `music_only` |
| `text_music` | 同一个 GEM batch 中同时包含文本和音乐条件，只调用一次 `GEM.predict()` | `text_music` |

`text_music` 不是分别生成两条动作后混合。它在同一个完整 PyTorch GEM DDIM/CFG 请求中同时设置 `text_embed`、`has_text=true`、`music_embed` 和全 True 的 `has_music_mask`，元数据会记录 `fusion_mode=joint_gem_condition` 与 `fusion_training_status=zero_shot_cross_dataset`。所有图像、2D、相机和 speech/audio 条件 mask 都保持关闭。

第一版明确不支持 `video_text`、`video_music` 和 `video_text_music`。实时视频走 ONNX regression denoiser，其输入只有 `obs`、`bbx_xys`、`K_fullimg`、`f_imgseq` 和 `f_cam_angvel`，没有文本或音乐输入。收到这些模式时服务返回 `UnsupportedModeError`，不会静默退化成视频、文本或音乐模式。真正的视频多条件融合需要完整 PyTorch diffusion 的固定窗口方案和额外验证。

先启动 GMR-CPP/MuJoCo：

```bash
cd /home/weili/GMR-CPP_e1jump_lowdpi

./run_smplx_bumi3.sh \
  --always \
  --vis \
  --vis-smplx-targets \
  --vis-smplx-frames
```

再启动统一服务。下面的 eager 模式会在启动阶段依次加载 T5、完整 GEM 和视频模型栈，后续切换相机或视频文件不会重新加载模型：

```bash
cd /home/weili/GENMO
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 \
python scripts/demo/demo_multimodal_server.py \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt \
  --t5_model t5-3b \
  --local_files_only \
  --device cuda:0 \
  --video_init eager \
  --no_imgfeat \
  --clip_frames 120 \
  --clip_fps 30 \
  --ddim_steps 20 \
  --guidance_scale 2.5 \
  --output_root outputs/multimodal_motion \
  --bind tcp://127.0.0.1:7020 \
  --gmr_host 127.0.0.1 \
  --gmr_port 7006 \
  --publish_fps 30 \
  --shape_mode zero \
  --mode sim
```

实物机器人模式必须改用经过 MuJoCo 和实机低速验证的站立动作，并保持安全的 `queue` 策略：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/demo/demo_multimodal_server.py \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt \
  --t5_model t5-3b \
  --local_files_only \
  --device cuda:0 \
  --video_init eager \
  --no_imgfeat \
  --clip_frames 120 \
  --clip_fps 30 \
  --ddim_steps 20 \
  --output_root outputs/multimodal_motion \
  --bind tcp://127.0.0.1:7020 \
  --gmr_host 127.0.0.1 \
  --gmr_port 7006 \
  --publish_fps 30 \
  --shape_mode zero \
  --mode robot \
  --idle_motion inputs/motions/smplx_idle_stand.pt \
  --new_motion_policy queue
```

`clip_fps` 固定为 30，`clip_frames` 是服务级固定长度，请求不能覆盖。音乐和文本+音乐会从 `start_sec` 提取恰好 `clip_frames / 30` 秒音频；librosa 边界造成的 ±2 帧差异可用显式 `trim_or_pad_last` 对齐，更大的差异会拒绝请求。

另一个终端使用统一客户端：

```bash
# 启动实时摄像头跟随
python scripts/demo/multimodal_motion_client.py \
  --endpoint tcp://127.0.0.1:7020 \
  --video_start \
  --camera_id 0

# 启动服务端路径中的视频文件
python scripts/demo/multimodal_motion_client.py \
  --endpoint tcp://127.0.0.1:7020 \
  --video_start \
  --video_path /server/path/demo.mp4

# 文本动作
python scripts/demo/multimodal_motion_client.py \
  --endpoint tcp://127.0.0.1:7020 \
  --mode text \
  --prompt "A person walks forward and raises both arms." \
  --seed 42

# 音乐动作
python scripts/demo/multimodal_motion_client.py \
  --endpoint tcp://127.0.0.1:7020 \
  --mode music \
  --audio /server/path/song.wav \
  --start_sec 0 \
  --seed 42

# 文本+音乐联合条件
python scripts/demo/multimodal_motion_client.py \
  --endpoint tcp://127.0.0.1:7020 \
  --mode text_music \
  --prompt "A person dances energetically and turns once." \
  --audio /server/path/song.wav \
  --start_sec 10 \
  --seed 42

# 状态、停止视频、静止、软件急停、清除急停和关闭
python scripts/demo/multimodal_motion_client.py --status
python scripts/demo/multimodal_motion_client.py --video_stop
python scripts/demo/multimodal_motion_client.py --idle
python scripts/demo/multimodal_motion_client.py --estop
python scripts/demo/multimodal_motion_client.py --clear_estop
python scripts/demo/multimodal_motion_client.py --shutdown
```

客户端中的音频和视频都是服务端本地路径，不会上传文件。可重复指定 `--allowed_audio_root` 和 `--allowed_video_root` 限制服务可读取的目录；服务会执行 `expanduser()`、`resolve(strict=True)`，拒绝 `..`、软链接越界、非普通文件和不支持的扩展名。摄像头 ID 不经过路径检查。

统一服务也支持本机 JSON 行诊断：

```bash
python scripts/demo/demo_multimodal_server.py \
  --transport stdin \
  --video_init lazy \
  --no_imgfeat \
  --mode sim
```

每行输入一个 JSON 对象，例如：

```json
{"op":"generate","mode":"text","request_id":"text-001","prompt":"A person waves.","seed":42}
{"op":"generate","mode":"music","request_id":"music-001","audio_path":"/server/path/song.wav","start_sec":0,"seed":42}
{"op":"generate","mode":"text_music","request_id":"mix-001","prompt":"A person dances.","audio_path":"/server/path/song.wav","start_sec":0,"seed":42}
{"op":"video_start","camera_id":0}
{"op":"video_stop"}
{"op":"idle"}
{"op":"estop"}
{"op":"clear_estop"}
{"op":"status"}
{"op":"clear_cache","target":"all"}
{"op":"shutdown"}
```

所有生成结果都是 `outputs/multimodal_motion/` 的直接子目录。文本结果包含 `smpl_params.pt`、`motion.npz`、`metadata.json`、`prompt.txt`；音乐结果增加 `music_features.pt` 和 `source_audio.txt`；联合结果同时包含文本与音乐文件。普通文件全部写完、关闭并 fsync 后才原子重命名目录并最后创建 `READY`。global/incam betas 都被强制为全零。

`MotionSourceMux` 是统一服务中唯一持有 `GMRUDPBridge`、`SMPLXGMRReference` 和 GMR FK 发送循环的组件。Webcam 只通过 `frame_sink(SMPLFrame)` 提交最新帧，不创建第二个 UDP socket。生成期间视频 GPU 推理暂停，但 Mux 的 30 Hz 线程继续发送最后一个新鲜视频帧或安全 idle；READY 后平滑切入生成 clip，动作结束后平滑返回视频快照或 idle，恢复视频前会请求清空 tracker、滑窗、rollout state 和 frame index。视频帧超过 `--video_stale_sec` 未更新时自动回到 idle，ESTOP 始终具有最高优先级。

每次新执行 `video_start` 时，视频 rollout 第一帧会对齐当前正在发送的根节点水平位置和 yaw，后续帧复用同一个刚体变换，并在 `blend_seconds` 内平滑进入视频姿态。因此先前文本或音乐动作已经移动到其他位置时，启动视频不会把人物拉回视频自身的局部原点；视频暂时过期并回到安全 idle 时也会保留当前脚下的水平位置。

`mode=robot` 必须提供经过 MuJoCo 和实机验证的 `--idle_motion`，默认禁止 interrupt；`shape_mode` 只允许 `zero`，保存的 global/incam betas 以及每次 EnDecoder FK 的 betas 都是全零。软件 ESTOP 不能替代物理急停。实机前必须先验证根节点、足部接触、速度和姿态范围，并使用低速、吊装保护和机器人原有硬件安全系统。

原有 `demo_smpl_text_server.py`、`demo_music_server.py`、`demo_webcam.py` 和 `stream_smpl_params_to_gmr.py` 仍可独立运行，接口没有被统一服务替换。SMP1 magic/version、14 个目标名称和顺序、GMR-CPP、BUMI3、Redis 与 GMT 协议均未修改。

### 纯视频演示

如需进行不含文本条件的简单姿态估计，请使用 `demo_smpl_hpe.py`：

```bash
python scripts/demo/demo_smpl_hpe.py \
  --video path/to/video.mp4 \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt
```

### 实时摄像头演示

`demo_webcam.py` 通过 ONNX Runtime 逐帧运行 YOLOX → ViTPose-H → HMR2 → GEM denoiser，采用滑动窗口和流式全局 rollout。ONNX Runtime 安装和一次性 ONNX 导出命令请参阅 [docs/INSTALL.md](docs/INSTALL.md) 的步骤 8～10。

```bash
# 视频文件：OpenCV 相机内人体网格叠加，不使用图像特征，速度最快
python scripts/demo/demo_webcam.py \
  --video path/to/video.mp4 --no_imgfeat \
  --render --render_mode opencv

# Webcam：Viser 三维世界查看器，浏览器打开 http://localhost:8012
python scripts/demo/demo_webcam.py \
  --camera_id 0 --no_imgfeat \
  --render --render_mode viser

# Webcam：中性 SMPL-X 体型，并通过 SMP1 向 GMR-CPP 实时发送
python scripts/demo/demo_webcam.py \
  --camera_id 2 --no_imgfeat --display \
  --gmr_host 127.0.0.1 --gmr_port 7006 \
  --shape_mode zero
```

| 参数 | 默认值 | 用途 |
|---|---|---|
| `--video` / `--camera_id` | camera 0 | 输入源 |
| `--context_frames` | `120` | 滑动窗口长度，必须与导出 denoiser 时的 `--seq_len` 一致 |
| `--no_imgfeat` | 关闭 | 使用无图像特征版本的 denoiser，完全跳过 HMR2 |
| `--render_mode {opencv,viser}` | `viser` | 人体网格叠加窗口或 Web 三维查看器 |
| `--shape_mode {zero,first,mean,ema,per_frame}` | `zero` | 统一控制渲染和 GMR FK 的 SMPL-X 体型；`zero` 使用中性平均体型，避免不同帧或不同运行之间人体比例变化 |
| `--no_async_pipeline` | 关闭 | 强制同步运行，降低吞吐量但消除流水线延迟 |

`--shape_mode` 会统一控制渲染与 GMR FK 使用的 SMPL-X 体型。默认 `zero` 使用中性平均 SMPL-X 体型，避免身体比例逐帧变化或不同运行之间发生变化。

各模块延迟分析命令：

```bash
python tools/benchmark/benchmark_modules.py
```

---

## 🏋️ 训练

数据集下载链接和目录结构请参阅 [数据集准备文档](docs/DATA.md)。

### 构建 HumanML3D SMPL-X 动作元数据

已有 HumanML3D 官方仓库、精确 AMASS 映射报告和 GENMO 预处理 AMASS 文件时，可生成 `Humanml3dDataset` 直接读取的动作与文本元数据：

```bash
python tools/data/humanml3d/build_humanml3d_smpl.py
```

默认输出为：

```text
inputs/HumanML3D_SMPL/hmr4d_support/humanml3d_smplhpose_train.pth
```

工具只使用 `exact_family_path` 记录，不对 unmatched 动作做模糊匹配，也不处理 HumanAct12。它将 HumanML3D 的 20 FPS 时间范围换算到预处理 AMASS 的 30 FPS 时间轴，复用官方五类前缀裁剪，读取官方原动作/镜像文本，并按确定性 key 保存带独立描述的子片段。输出坐标仍保持 AMASS AZ；训练数据集加载时会执行现有的 AZ → AY 转换。

建议先进行不保存主 PTH 的全量审计：

```bash
python tools/data/humanml3d/build_humanml3d_smpl.py \
  --dry-run \
  --report-dir outputs/humanml3d_full_dryrun
```

正式构建会先写入 `.tmp` 文件，重新加载并验证全部记录，再原子替换正式输出。详细审计报告写入 `outputs/humanml3d_build_report/`。该步骤只构建 SMPL-X 动作和原始 caption，不在内存中加载 T5-3B。

### 提取 HumanML3D T5-3B 文本特征

动作 PTH 构建完成后，先估算 FP16 特征体积和“分片 + 最终临时文件”的峰值磁盘需求；该命令不加载 T5：

```bash
python tools/data/humanml3d/extract_t5_embeddings.py \
  --estimate-only
```

使用本地缓存中的 T5-3B 执行全量提取：

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/data/humanml3d/extract_t5_embeddings.py \
  --model-name-or-path t5-3b \
  --local-files-only \
  --device cuda:0 \
  --batch-size 16 \
  --motions-per-shard 256 \
  --resume \
  --cleanup-shards
```

默认输出为：

```text
inputs/HumanML3D_SMPL/t5_embeddings_v1_half/all_text_embed.pth
```

最终文件是纯 `motion_id -> Tensor[num_captions, 50, 1024]` 字典：T5 推理默认使用 float32，保存为 CPU float16；第 0 维与动作记录中 `text_data` 的原始顺序严格一致，padding token 对应位置为零。工具使用原始 caption 字符串，不使用 processed tokens，不去重，也不进行 mean pooling、PCA 或维度裁剪。

提取过程先复制轻量 caption 元数据并释放约 1.4 GB 的动作字典，再加载 T5-3B。每个分片都采用“写临时文件 → 回读完整验证 → 原子替换”，`--resume` 只复用 fingerprint 一致且实际回读通过的分片；最终文件也会回读并核对全部 key、caption 数、shape、dtype、连续内存和有限值。`--cleanup-shards` 只在最终文件验证成功后删除分片 PTH，保留 manifest 和 `outputs/humanml3d_t5_report/` 审计报告。

### 构建官方 AIST++ crossmodal 音乐训练集

完整 AIST++ motion、keypoints2d、camera 标注和对齐后的 EDGE baseline35 特征齐备时，使用官方 crossmodal 文件原始顺序构建标准训练产物：

```bash
python tools/data/aistpp/build_annot_aist_official_30fps.py \
  --annotations-root /home/weili/datasets/AISTPP_fullset/aist_plusplus_final \
  --musicfeat-dir /home/weili/GENMO/inputs/AIST++/musicfeat_v2 \
  --output-root /home/weili/GENMO/inputs/AIST++ \
  --view c01 \
  --dry-run \
  --allow-ignored-official \
  --report-dir outputs/aistpp_official_dryrun
```

dry-run 完整通过后再去掉 `--dry-run` 并增加 `--overwrite`。正式输出固定为：

```text
inputs/AIST++/annot_aist_30fps.pt   # train/val/test 官方并集 1020 条
inputs/AIST++/train.pt              # 980 条
inputs/AIST++/val.pt                # 20 条
inputs/AIST++/test.pt               # 20 条
inputs/AIST++/minitrain.pt          # 按官方 train 顺序选取 16 条
```

该工具复用 partial builder 已验证的 60→30 FPS、关键点、tight bbox、相机和 `get_c_rootparam()` 逻辑，不重新提取、截断或填充音乐特征。所有官方 ID 都必须同时具备 motion、keypoints2d、`[L,35]` 音乐特征和相机标定；官方 split 与 `ignore_list.txt` 有交集时也会拒绝发布，避免静默删除后伪装成完整 980/20/20。额外 motion、keypoints 或音乐特征只写入报告，不会自动加入 train。

如果已经人工审计并明确确认这些交集序列可以使用，可显式增加 `--allow-ignored-official`。该参数只授权保留官方 split 中的交集 ID，不修改原始 `ignore_list.txt`；省略时仍按默认安全策略失败。当前本机 28 条交集序列已经确认可用，因此正式构建命令需要带上此参数。

五个文件先全部写入 `.tmp` 并重新加载验证，确认 annot 契约、音乐长度、split 顺序/互斥和 minitrain 后才发布；任一步失败都会清理临时文件并保留审计报告。`--limit` 仅用于 dry-run 或独立测试输出目录，不能覆盖标准正式文件。默认训练配置 `configs/train_datasets/aistpp_train.yaml` 已使用 `feat_version: v2` 和上述标准文件名。

### 构建本地 AIST++ partial 音乐训练集

当本地只有部分官方 AIST++ 动作、对应关键点/相机标注和已经对齐的 EDGE baseline35 音乐特征时，可以构建用于验证音乐条件训练链路的 partial 数据集：

```bash
python tools/data/aistpp/build_annot_aist_30fps.py \
  --annotations-root /home/weili/datasets/AISTPP_official/annotations \
  --musicfeat-dir inputs/AIST++/musicfeat_v2 \
  --output-root inputs/AIST++ \
  --view c01 \
  --overwrite
```

该工具将 60 FPS 动作同步按 `[::2]` 转为 30 FPS，使用具名相机参数和现有 `get_c_rootparam()` 构造相机坐标 SMPL，通过 COCO17 关键点生成紧 bbox，并跳过所有缺少或不匹配音乐特征的序列。它生成：

```text
inputs/AIST++/annot_aist_30fps_partial.pt
inputs/AIST++/train_partial.pt
inputs/AIST++/val_partial.pt
inputs/AIST++/test_partial.pt
inputs/AIST++/minitrain_partial.pt
```

partial 训练配置位于 `configs/train_datasets/aistpp_partial_train.yaml`。`train_partial` 使用本地所有成功构建序列减去当前可用的官方 crossmodal val/test 交集，绝不是官方 980 条 crossmodal training split，不能用于声明官方 AIST++ benchmark 或论文指标复现。构建器不下载数据、不重新提取音乐特征，也不会用零值或其他歌曲特征填补缺失序列。

### 构建 BEAT2 四语言 split 索引

当 BEAT2 四个语言子集已经下载到 `/home/weili/datasets/BEAT2_official` 后，可生成 `BEAT2SmplDataset` 直接读取的 `all_splits.pth`。工具只构建索引，不复制或修改 NPZ/WAV，也不预提取音频特征：

```bash
python tools/data/beat2/build_all_splits.py \
  --root /home/weili/datasets/BEAT2_official \
  --dry-run \
  --report-dir outputs/beat2_build_dryrun
```

默认策略会在正式 train/val/test 条目缺少 NPZ 或 WAV 时拒绝发布。当前本机数据有 7 条中文 train 记录缺少 NPZ；重新下载缺失文件是首选方案。如果确认要跳过并在报告中保留缺失清单，则必须显式运行：

```bash
python tools/data/beat2/build_all_splits.py \
  --root /home/weili/datasets/BEAT2_official \
  --output /home/weili/datasets/BEAT2_official/all_splits.pth \
  --report-dir outputs/beat2_build_report \
  --allow-missing-pairs \
  --overwrite
```

默认不会把 `additional` 合入 train；如确实需要该策略，必须显式增加 `--include-additional-as-train`。输出顶层仅包含 `train`、`val`、`test`、`minitrain` 和 `additional`，每项只保存 `video_id`、相对语言 `subset` 与 NPZ 的真实帧数。所有引用在临时 PTH 回读验证后才通过 `os.replace` 原子发布。

确保 GENMO 输入路径指向数据根目录：

```bash
ln -s /home/weili/datasets/BEAT2_official inputs/BEAT2
```

如果 `inputs/BEAT2` 已存在，应先用 `readlink -f inputs/BEAT2` 核对目标，不要覆盖用户已有路径。详细审计报告位于 `outputs/beat2_build_report/`，其中会列出缺失配对、CSV 外孤立文件、非法 NPZ、Git LFS pointer、短音频和过短动作。

### 服务器完整训练预检

`configs/exp/gem_smpl_server.yaml` 保留 AMASS、BEDLAM、H36M、3DPW、AIST++、BEAT2 和 HumanML3D 七个训练集，以及 EMDB1、EMDB2、3DPW、RICH 四个验证集。它仍使用完整 regression + DDIM diffusion、多模态条件、151 维 EnDecoder、T5-3B 预计算特征、AdamW 和 16-bit mixed precision；仅移除了本地缺少 `imgfeats/3dpw_occ_train` 的 3DPW-OCC 额外遮挡增强源及其 metric。默认单卡起始设置为 batch size 4、4 个 worker，可通过 Hydra 命令行覆盖。

开始服务器训练前运行统一检查：

```bash
python tools/train/preflight_gem_smpl.py \
  --exp gem_smpl_server \
  --samples-per-dataset 1 \
  --batch-size 2 \
  --num-workers 0 \
  --instantiate-model \
  --check-pretrained \
  --strict
```

该工具通过 Hydra compose 读取真实配置，检查所有正式数据产物、HumanML3D motion/T5 key 与有限值、AIST++ 官方 split/35 维音乐特征、BEAT2 索引配对、回归/验证数据和身体模型；随后逐个实例化 Dataset、读取样本、构造真实混合 batch，并可实例化完整 GEM 模型。结果写入 `outputs/preflight_gem_smpl/report.json`。RICH 验证依赖的小型 `cam2params.pt` 使用 [GVHMR 官方资源](https://github.com/zju3dv/GVHMR/blob/main/hmr4d/dataset/rich/resource/cam2params.pt)，loader 已兼容 PyTorch 2.6 的可信本地 artifact 加载。

`MetricRICH` 只创建实际用于 FK 和指标计算的 male/female/neutral 三个 SMPL-X 模型，并加载 `smpl_neutral_J_regressor.pt` 与 `smplx2smpl_sparse.pt`。已移除从未参与相机坐标指标、全局指标或日志的 SMPL mesh faces 初始化，因此 RICH 回调不再非必要地依赖 `SMPL_NEUTRAL.pkl`；所有 RICH 指标公式、metric key 和日志名称保持不变。

预检通过后可启动完整服务器训练：

```bash
python scripts/train.py exp=gem_smpl_server
```

例如调整实际 batch 与 worker：

```bash
python scripts/train.py exp=gem_smpl_server \
  data.loader_opts.train.batch_size=8 \
  data.loader_opts.train.num_workers=8
```

**回归模型（视频 → SMPL）：**

```bash
python scripts/train.py exp=gem_smpl_regression
```

**完整模型（回归 + 文本/音频生成）：**

```bash
python scripts/train.py exp=gem_smpl
```

**多 GPU（DDP）：**

```bash
python scripts/train.py exp=gem_smpl_regression pl_trainer.devices=4
```

**SLURM：**

```bash
python scripts/train_slurm.py exp=gem_smpl_regression
```

### 主要训练配置

来自 `configs/exp/gem_smpl_regression.yaml`：

- 身体模型：SMPL-X
- 优化器：AdamW，学习率 `2e-4`
- 精度：`16-mixed`
- 最大步数：500K
- 梯度裁剪：0.5
- 每 3000 步验证一次

默认使用 W&B 记录日志。关闭方法：

```bash
python scripts/train.py exp=gem_smpl_regression use_wandb=false
```

---

### Motion-X++ 三维动作与文本微调

Motion-X++ 支持直接读取 ZIP 或解压目录，先将官方 30 FPS、Y-up 的
`motion_generation/smplx322` 和 `semantic_label` 构建为可恢复的动作分片，再生成
T5-3B FP16 embedding 分片。Dataset 使用每个 worker 独立的小型 LRU，不会一次加载
全部动作或 embedding。

服务器数据审计：

```bash
python tools/data/motionxpp/inspect_motionxpp.py \
  --root inputs/Motion-Xplusplus \
  --output-dir outputs/motionxpp_inspect
```

完整动作构建：

```bash
python tools/data/motionxpp/build_motionxpp_genmo.py \
  --root inputs/Motion-Xplusplus \
  --output-root inputs/Motion-Xplusplus/genmo_support \
  --subsets-file outputs/motionxpp_inspect/recommended_subsets.txt \
  --records-per-shard 512 \
  --source-up-axis y \
  --target-fps 30 \
  --split-seed 20260724 \
  --strict
```

完成 T5 分片和预检后，从官方完整 checkpoint 做 20 步单卡 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/train.py \
  exp=gem_smpl_motionxpp \
  ckpt_path=inputs/pretrained/gem_smpl.ckpt \
  pl_trainer.devices=1 \
  pl_trainer.max_steps=20 \
  data.loader_opts.train.batch_size=2 \
  data.loader_opts.train.num_workers=0 \
  use_wandb=false
```

当前第一版只启用可靠的 SMPL-X 3D + semantic text。关键点归档没有图像宽高和校准
相机参数，所以正式配置保持 `condition_on_keypoints=false`，不会伪造 K 或把样本
标成纯 2D-only。完整的审计、T5、preflight、resume 和四卡训练命令见
[Motion-X++ 中文文档](docs/motionxpp.md)。

常见问题请参阅 [FAQ](docs/FAQ.md)。

---

## 🤝 NVIDIA 相关人形机器人项目

GEM 是 NVIDIA 人体动作数据、机器人和 Physical AI 研究生态的一部分。相关项目包括：

- [GEM-X](https://github.com/NVlabs/GEM-X)
- [SOMA Body Model](https://github.com/NVlabs/SOMA-X)
- [BONES-SEED Dataset](https://huggingface.co/datasets/bones-studio/seed)
- [ProtoMotions](https://github.com/NVlabs/ProtoMotions)
- [SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter)
- [SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl)
- [Kimodo](https://github.com/nv-tlabs/kimodo)

## 📖 引用

```bibtex
@inproceedings{genmo2025,
  title     = {GENMO: A GENeralist Model for Human MOtion},
  author    = {Li, Jiefeng and Cao, Jinkun and Zhang, Haotian and Rempe, Davis and Kautz, Jan and Iqbal, Umar and Yuan, Ye},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```

---

## 📄 许可证

本项目采用 NVIDIA OneWay Noncommercial License，详情请参阅 [LICENSE](LICENSE)。第三方组件遵循各自许可证，具体信息请参阅 [ATTRIBUTIONS.md](ATTRIBUTIONS.md)。
