# 音乐动作生成

`scripts/demo/demo_music.py` 将已有的 WAV、MP3 或 FLAC 音乐文件转换为 SMPL-X 人体动作。它**不会生成音乐音频**：

```text
音乐文件
  -> 30 FPS EDGE baseline35 特征
  -> 完整 PyTorch gem_smpl.ckpt
  -> DDIM + classifier-free guidance
  -> 原始 151 维 GEM 动作
  -> EnDecoder
  -> 零体型 global/incam SMPL-X 参数
  -> 原子发布 READY 动作目录
```

该路径不读取视频，也不会加载 YOLO、ByteTrack、ViTPose、HMR2 或 T5。音乐动作生成需要完整的 `inputs/pretrained/gem_smpl.ckpt`，并使用 `exp=gem_smpl` 配置。实时 ONNX 导出是 regression-only，不包含音乐条件 DDIM 采样，因此本路径不会使用这些 ONNX 模型。

## EDGE baseline35 输入契约

特征提取严格参考 EDGE 的 `data/audio_extraction/baseline_features.py`：

| 通道 | 特征 |
|---|---|
| `0` | onset strength（起始强度） |
| `1:21` | 20 维 MFCC |
| `21:33` | 12 维 chroma CENS |
| `33` | 二值 onset peak |
| `34` | 二值 beat peak |

时间参数固定为：

```text
FPS:         30
sample_rate: 15360
hop_length:  512
```

checkpoint、模型第一层 `music_embedder` 线性层和输入特征张量的维数必须全部为 35。任何维数不匹配都会立即报错；程序不会截断列、补零、执行 PCA、随机投影，也不会替换为其他音频表示。

EDGE 原始数据集脚本先把音频切成 5 秒片段，再保留 150 帧。本 Demo 支持显式指定 `--start_sec` 和 `--duration_sec`，因此移除了固定 5 秒裁剪。由于 Librosa 的边界分帧行为，输出有时会比 `duration × 30` 多一个边界帧；实际生成时长按 `feature_frames / 30` 记录。

特征提取器优先使用 Librosa 的 `offset` 和 `duration` 只解码所选范围，长 MP3 的短片段请求不再先读取整首歌曲。如果音频后端不支持范围读取，才回退到完整解码后切片；`feature_metadata["audio_decode_mode"]` 会记录 `range` 或 `full_fallback`。两种路径都保持相同的采样率、hop、帧率和 35 通道定义。

如当前环境缺少音频特征依赖，可执行：

```bash
python -m pip install "librosa>=0.10,<0.11"
```

## CPU 试运行（dry-run）

dry-run 会读取真实音频并提取真实 baseline35 特征，但不会加载 checkpoint、GEM、CUDA、SMPL-X、Open3D，也不会创建输出目录或 READY：

```bash
python scripts/demo/demo_music.py \
  --audio /path/to/song.wav \
  --duration_sec 10 \
  --dry_run
```

它会打印：

- 音频原始时长和选中范围；
- BPM、onset peak 和 beat peak 数量；
- 音乐特征 shape 与有限值检查；
- 合成相机张量 shape；
- 所有条件 mask 统计。

正常情况下只有 `has_music_mask` 为 True，图像、2D、相机、speech/audio 和文本条件全部关闭。

## 生成 READY 动作

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

使用 `--num_samples N` 可以发布 N 个独立 generation，第 `i` 个样本使用 `seed + i`。

`--save_features` 为兼容旧命令而保留；当前 READY 协议始终包含 `music_features.pt`。如果不指定 `--no_render`，程序会尝试生成 `motion_global.mp4`。指定 `--mux_audio` 后，还会尝试使用选中的原始音频范围生成 `motion_with_audio.mp4`。

Open3D 或 ffmpeg 失败只会产生 warning，有效的动作参数仍会正常发布。

第一版有意拒绝超过 `--max_frames` 的音频范围。默认值为 600，即 30 FPS 下约 20 秒。请选择更短的 `--start_sec` 和 `--duration_sec`；首次机器人测试推荐使用 5～10 秒。

长音乐生成需要专门处理动作重叠、根节点连续、脚接触连续和节拍连续。本实现不会进行未经验证的简单动作拼接。

## 常驻音乐动作生成服务

单次 `demo_music.py` 的 CLI 和输出协议保持不变。连续生成多个音乐片段时，可以启动独立的常驻音乐服务。服务启动时只做一次：

1. 审计完整 checkpoint 的 `music_embedder` 权重和 35 维输入；
2. 加载一次不含 T5 的完整 GEM-SMPL；
3. 固定 DDIM steps 和 CFG；
4. 初始化一次 DDIM；
5. 执行一次不发布文件的音乐 warmup；
6. 打印 `[ResidentMusic] SERVICE READY`。

正常请求不会重新加载 checkpoint/GEM、不会再次初始化 DDIM、不会卸载模型，也不会调用 `gc.collect()` 或 `torch.cuda.empty_cache()`。只有 CUDA OOM 恢复和服务关闭允许清理 CUDA cache；所有 GEM 请求通过同一把锁串行执行。

### stdin 模式

```bash
cd /home/weili/GENMO
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 \
python scripts/demo/demo_music_server.py \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt \
  --device cuda:0 \
  --output_root outputs/music_motion \
  --transport stdin \
  --start_sec 0 \
  --duration_sec 10 \
  --seed 42 \
  --ddim_steps 20 \
  --guidance_scale 2.5 \
  --shape_mode zero \
  --feature_cache_size 32 \
  --max_frames 600
```

看到 `music-motion>` 后，直接输入一行服务端音频路径：

```text
/home/weili/music/song.wav
"/home/weili/music/My Song.mp3"
```

也可以输入 JSON：

```json
{"request_id":"music-001","audio_path":"/home/weili/music/song.flac","start_sec":15,"duration_sec":10,"seed":7}
```

直接路径使用服务启动参数中的默认范围，默认是前 10 秒。JSON 的 `duration_sec=null` 表示从 `start_sec` 到文件末尾，但最终特征帧数仍不能超过 `--max_frames`。

管理命令：

```text
/status
/help
/clear-cache
/quit
```

`/clear-cache` 只清 CPU 音乐特征缓存，不卸载 GEM。每个 cache value 是 CPU float32 contiguous `[L,35]` 张量和一份独立元数据，不缓存 waveform。cache key 包含解析后的绝对路径、设备/inode、文件大小、mtime、选择范围、特征版本和 EDGE 时间参数；文件变化后会自动重新提取。

可重复指定服务端路径白名单：

```bash
--allowed_audio_root /home/weili/music \
--allowed_audio_root /mnt/shared/audio
```

路径会执行 `expanduser()` 和 `resolve(strict=True)`，并在解析符号链接后检查是否位于白名单内，从而拒绝 `../` 或 symlink 越界。

### ZMQ 模式与客户端

ZMQ 默认只绑定 loopback：

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

客户端：

```bash
python scripts/demo/music_motion_client.py \
  --endpoint tcp://127.0.0.1:7011 \
  --audio "/home/weili/music/My Song.wav" \
  --start_sec 0 \
  --duration_sec 10 \
  --seed 42 \
  --request_id music-001 \
  --timeout_seconds 60
```

客户端只发送 `audio_path`，不上传音频内容。该路径属于服务端机器；跨机器使用必须保证两端拥有相同共享挂载。`--full` 表示从 `start_sec` 到末尾，不能与 `--duration_sec` 同时使用。

成功响应包含输出路径、帧数、30 FPS、BPM、缓存命中状态、特征提取/输入构造/生成/保存/总耗时和 GPU 显存快照。单个错误返回结构化 JSON，服务继续处理后续请求。

### 常驻服务输出

常驻服务复用单次 Demo 的 body 参数校验、151 维诊断输出、zero shape、制品写入和 READY 原子发布函数。每个成功请求仍产生：

```text
smpl_params.pt
motion.npz
raw_motion_151d.pt
music_features.pt
metadata.json
source_audio.txt
READY
```

metadata 额外记录 `request_id`、`request_metadata` 和 `service=resident_music_motion`，`source` 仍为 `music_only`。

常驻服务 v1 不渲染、不执行 ffmpeg mux，避免 Open3D、额外 body model 和 ffmpeg 影响低延迟与显存稳定。需要渲染或合成视频时使用单次 `demo_music.py`。

## 原子输出协议

每个 sample 都是 `output_root` 的直接子目录：

```text
outputs/music_motion/
  song_start0p000_seed42_<UTC>_<uuid>/
    smpl_params.pt
    motion.npz
    raw_motion_151d.pt
    music_features.pt
    metadata.json
    source_audio.txt
    motion_global.mp4          # 可选
    motion_with_audio.mp4      # 可选
    READY                      # 始终最后创建
```

程序首先在 `outputs/music_motion/.tmp_<uuid>/` 中写入并刷新全部文件，然后通过 `os.replace` 原子重命名目录，最后才写入并刷新 `READY`。生成失败时会清理自己的临时目录，不会暴露 READY，也不会覆盖已有 generation。

`smpl_params.pt` 包含：

```text
body_pose:     [L, 63]
global_orient: [L, 3]
transl:        [L, 3]
betas:         [L, 10]
```

global 和 incam 两组参数还包含相机张量、FPS、音频选择范围、特征设置、checkpoint、seed 和元数据。两组 betas 均严格为全零，`motion.npz` 中的 betas 也为全零。

`raw_motion_151d.pt` 保留 GEM 扩散模型的原始诊断输出，其中的 shape 分量不会被伪装成已经覆盖；正式输出使用 `smpl_params.pt -> body_params_global`。

## AIST++ 数据准备

任意音乐文件的 `--audio` 推理不依赖官方 annotations 目录。批量准备 AIST++ 音乐特征时，还需要已经对齐、与 sequence 同名的 WAV 文件；只有一个共享 music-ID WAV 无法确定各 sequence 的时间偏移。

以下审计和准备工具不会修改原始 annotations 目录：

```bash
python tools/data/aistpp/audit_aistpp.py \
  --annotations-root /home/weili/datasets/AISTPP_official/annotations

python tools/data/aistpp/extract_musicfeat_v2.py \
  --annotations-root /home/weili/datasets/AISTPP_official/annotations \
  --aligned-wav-dir /path/to/per_sequence_wavs \
  --output-dir inputs/AIST++/musicfeat_v2
```
