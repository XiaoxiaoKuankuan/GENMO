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

每个 sample 都是 `outputs/music_motion/<generation>/` 的直接子目录，包含 SMPL 参数、原始诊断数据、音乐特征、元数据，以及最后创建的 READY。保存的 global/incam betas 和 streamer 每次 FK 使用的 betas 都是全零。没有动作时 streamer 继续定频发送 idle；动作结束后平滑返回 idle，不保持最后一帧。

终端 2 增加 `--audio_playback ffplay` 可同时进行尽力而为的本地音乐播放。音频失败不会中断控制流，这也不是硬实时音频时钟。robot 模式必须提供 `--idle_motion inputs/motions/smplx_idle_stand.pt`。先在 MuJoCo 中验证，再使用低速、吊装保护、物理急停和机器人正常硬件保护；软件 ESTOP 不能替代物理急停。

完整的 35 通道契约、dry-run、原子输出协议、长音频限制、直接播放命令和安全说明请参阅 [音乐动作生成与机器人实时播放](docs/MUSIC_DEMO.md)。

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

正式构建会先写入 `.tmp` 文件，重新加载并验证全部记录，再原子替换正式输出。详细审计报告写入 `outputs/humanml3d_build_report/`。本工具不提取 T5 embedding；后续需要根据最终 PTH 的 motion key 和 `text_data` 单独生成 `all_text_embed.pth`。

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
