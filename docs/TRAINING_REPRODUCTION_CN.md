# GENMO / GEM-SMPL 完整训练复现指南

本文记录的是当前仓库从一台新机器开始，完成环境安装、模型资源准备、数据集补齐、
训练数据构建、训练前检查、正式训练和断点恢复的完整过程。目标不是逐行解释训练代码，
而是让后来者能够按照同一套步骤把工程真正跑起来，并知道每一步完成后应该看到什么。

本文对应仓库：

```text
https://github.com/XiaoxiaoKuankuan/GENMO
```

当前实际开发目录为：

```text
/home/weili/GENMO
```

文中的路径以这个目录为例。换到服务器时，可以将 `/home/weili` 换成自己的用户名或
数据盘路径。建议不要把几十 GB 到几百 GB 的原始数据直接复制进 Git 仓库，而是统一
存放在外部数据盘，再通过软链接接入 `inputs/`。

## 阅读导航

- 第 1～3 节：项目定位、框架理解和正确复现顺序。
- 第 4～5 节：系统环境、Python 包、checkpoint 和身体模型。
- 第 6 节：拿到仓库后从安装到真实 Demo 的最短验证路线。
- 第 7～11 节：基础动作数据、HumanML3D、AIST++、BEAT2 和 Motion-X++。
- 第 12～14 节：新增前处理工具、最终目录和统一训练预检。
- 第 15～18 节：smoke、正式训练、多卡、微调、断点恢复和训练后验证。
- 第 19～22 节：常见故障、新服务器检查表、复现边界和推荐结论。

---

## 1. 项目简介

GENMO 的核心模型 GEM 是一个统一的人体动作模型。它既可以做视频、二维关键点和相机
条件下的人体动作回归，也可以在完整扩散配置下接收文本、音乐或语音条件生成动作。
仓库中的 GEM-SMPL 使用 SMPL-X 身体模型表达人体，并把不同来源的数据整理到统一的
151 维动作表示中训练。

复现时需要先区分两类模型：

- `gem_smpl_regression` 是回归模型，主要服务视频到 SMPL-X 的实时推理。当前导出的
  ONNX denoiser 也属于这条路径。
- `gem_smpl` 是完整模型，同时包含 regression、DDIM diffusion、classifier-free
  guidance、文本、音乐和语音条件。文本生成、音乐生成以及完整多模态训练必须使用它。

本仓库后来增加了两个更适合实际环境的训练入口：

- `gem_smpl_server`：完整 GEM-SMPL 训练配置，使用当前已经准备好的七个训练集和四个
  验证集。它只移除了本地缺失的 `3DPW-OCC` 额外遮挡增强数据，不改变模型主体。
- `gem_smpl_motionxpp`：在官方完整 `gem_smpl.ckpt` 基础上加入 Motion-X++ 的
  SMPL-X 三维动作和语义文本监督，适合做扩展数据微调。

训练阶段不会常驻加载 T5-3B。HumanML3D 和 Motion-X++ 的文本先离线编码为
`[50, 1024]` 特征，训练 DataLoader 直接读取这些预计算结果。因此训练机器需要足够的
磁盘和内存，但不需要在训练时再为 T5-3B 额外保留显存。

---

## 2. 快速理解仓库

只需要先记住下面几个目录：

```text
GENMO/
├── configs/                 Hydra 实验、模型、数据集和回调配置
├── gem/                     GEM 模型、数据集、训练管线和通用工具
├── scripts/                 训练入口和各种推理 Demo
├── tools/data/              本仓库新增的数据审计、转换和特征提取工具
├── tools/train/             正式训练前的统一预检工具
├── inputs/                  模型、训练制品和外部数据软链接
├── outputs/                 构建报告、训练日志、checkpoint 和推理结果
├── docs/                    复现、数据和运行说明
└── tests/                   数据工具、训练输入和运行时回归测试
```

训练配置由 Hydra 组合。运行：

```bash
python scripts/train.py exp=gem_smpl_server
```

时，`exp=...` 会选择一套完整实验配置，再组合模型、pipeline、dataset、optimizer、
scheduler 和 callback。实际调参通常不需要修改 YAML，可以直接在命令行覆盖，例如：

```bash
python scripts/train.py exp=gem_smpl_server \
  data.loader_opts.train.batch_size=8 \
  data.loader_opts.train.num_workers=8 \
  pl_trainer.devices=1
```

### 2.1 训练框架在做什么

不同数据集不需要同时具备所有条件。训练样本通过统一字段和 condition mask 表示“这一条
样本真正有什么”：

- AMASS 提供大规模三维人体动作基础；
- BEDLAM、H36M 和 3DPW 补充图像、二维关键点、相机和人体回归监督；
- HumanML3D 提供动作与自然语言描述；
- AIST++ 提供动作、音乐、关键点和相机；
- BEAT2 提供动作与语音；
- Motion-X++ 补充更多 SMPL-X 三维动作与语义文本。

EnDecoder 负责在 SMPL-X 参数和统一 151 维动作特征之间转换。回归模式直接预测动作
特征；完整模型还会在相同表示上执行扩散去噪。文本被预先编码为 50 个 token、每个
1024 维的 T5 特征，AIST++ 音乐被编码为每帧 35 维 EDGE baseline 特征。数据加载时，
不存在的条件保持关闭，不会用另一种模态伪装。

完整训练同时学习回归和扩散任务。也正因为如此，只导出单步回归 denoiser 的 ONNX
模型不能拿来替代 PyTorch DDIM 训练或文本、音乐生成。

---

## 3. 建议的复现顺序

不要一上来就启动 500,000 步训练。可靠的顺序是：

1. 安装系统依赖和 Python 环境。
2. 下载完整 GEM checkpoint、SMPL-X 身体模型和基础 GVHMR 预处理数据。
3. 准备 HumanML3D、AIST++、BEAT2；按需准备 Motion-X++。
4. 用新增工具生成训练真正读取的 PTH、音乐特征和 T5 特征。
5. 运行统一 preflight，让所有数据集各读一个真实样本并组成 batch。
6. 运行 20 步 smoke training。
7. 最后启动单卡或多卡正式训练。

建议预留：

- 系统盘至少 50 GB 可用空间；
- 数据盘至少 300 GB，可根据是否保留原视频和临时 T5 分片增加；
- 一张支持 CUDA 的 NVIDIA GPU；
- 完整训练推荐 24 GB 或更大显存；
- HumanML3D 单体动作文件约 1.4 GB，T5 特征约 6.5 GB；
- Motion-X++ 官方数据约 50 GB，完整转换还会额外生成动作和文本特征分片。

---

## 4. 从空机器配置环境

### 4.1 系统依赖

以下命令以 Ubuntu 22.04 为例：

```bash
sudo apt update
sudo apt install -y \
  git \
  git-lfs \
  curl \
  wget \
  ffmpeg \
  libsndfile1 \
  build-essential

git lfs install
```

确认 NVIDIA 驱动工作正常：

```bash
nvidia-smi
```

本机验证过的主要环境是：

```text
Ubuntu 22.04
Python 3.10.12
NVIDIA RTX 4090
PyTorch 2.6.0+cu124
TorchVision 0.21.0+cu124
CUDA runtime 12.4
Lightning 2.3.0
Transformers 5.13.1
Hydra 1.3.0
NumPy 1.23.5
Librosa 0.10.2.post1
```

不要求所有小版本完全相同，但 PyTorch、TorchVision 和 CUDA wheel 必须匹配。

### 4.2 克隆仓库

```bash
cd /home/weili
git clone git@github.com:XiaoxiaoKuankuan/GENMO.git
cd GENMO
git status
```

如果机器没有配置 GitHub SSH key，也可以使用：

```bash
git clone https://github.com/XiaoxiaoKuankuan/GENMO.git
```

### 4.3 使用 uv 创建虚拟环境

安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

创建 Python 3.10 环境：

```bash
cd /home/weili/GENMO
uv venv .venv --python 3.10
source .venv/bin/activate
```

先安装与 CUDA 12.4 对应的 PyTorch：

```bash
uv pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

再安装仓库及训练、测试依赖：

```bash
uv pip install -e ".[train,dev]"
```

这里不需要再手工逐个安装 Hydra、Lightning、Transformers、SMPL-X、Open3D、
Librosa、W&B 等包；它们已经声明在 `setup.cfg`。仓库当前对一些容易发生兼容变化的
包固定了版本或范围，例如：

```text
timm==0.6.7
lightning==2.3.0
hydra-core==1.3
numpy==1.23.5
imageio==2.34.1
av<14
ultralytics==8.3.50
librosa>=0.10,<0.11
```

如果后续安装了一个会强制升级 NumPy 的包，应重新检查 `numpy` 版本和全部数据工具，
不要在长时间特征提取任务中临时改变环境。

仓库也提供：

```bash
bash scripts/install_env.sh
```

该脚本执行的是可编辑安装。为了做完整训练和测试，仍建议安装 `train`、`dev` extras。

### 4.4 验证环境

```bash
python - <<'PY'
import torch
import torchvision
import lightning
import transformers
import hydra
import librosa

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("lightning:", lightning.__version__)
print("transformers:", transformers.__version__)
print("hydra:", hydra.__version__)
print("librosa:", librosa.__version__)
PY
```

如果 `torch.cuda.is_available()` 是 `False`，先解决驱动或 PyTorch wheel 问题，不要继续
执行完整模型预检。

---

## 5. 下载模型和公共资源

主要官方下载入口：

| 资源 | 官方入口 |
|---|---|
| GEM / GEM-X checkpoint | <https://huggingface.co/nvidia/GEM-X> |
| T5-3B | <https://huggingface.co/google-t5/t5-3b> |
| SMPL-X | <https://smpl-x.is.tue.mpg.de/> |
| GVHMR 预处理数据 | <https://drive.google.com/drive/folders/10sEef1V_tULzddFxzCmDUpsIqfv7eP-P?usp=drive_link> |
| HumanML3D | <https://github.com/EricGuo5513/HumanML3D> |
| AIST++ | <https://google.github.io/aistplusplus_dataset/download.html> |
| BEAT / BEAT2 | <https://pantomatrix.github.io/BEAT/> |
| Motion-X++ | <https://huggingface.co/datasets/YuhongZhang/Motion-Xplusplus> |

下载任何人体数据前都应先阅读对应许可。SMPL-X、AMASS、AIST Dance Video Database
等资源不能因为工程复现需要就跳过原始授权。

### 5.1 Hugging Face 命令行

安装依赖后可以直接使用 `hf`：

```bash
hf --help
```

需要鉴权的数据或模型先登录：

```bash
hf auth login
```

### 5.2 完整 GEM-SMPL checkpoint

完整 checkpoint 来自 Hugging Face 的 `nvidia/GEM-X`：

```bash
cd /home/weili/GENMO
mkdir -p inputs/pretrained

hf download nvidia/GEM-X \
  gem_smpl.ckpt \
  --local-dir inputs/pretrained
```

确认文件存在：

```bash
ls -lh inputs/pretrained/gem_smpl.ckpt
```

这个文件约 5.5 GB。文本、音乐生成和 Motion-X++ 微调都需要完整 checkpoint，不能用
实时 ONNX denoiser 代替。

### 5.3 T5-3B

离线提取文本特征需要 T5-3B。建议下载到外部模型目录：

```bash
mkdir -p /home/weili/models/t5-3b

hf download google-t5/t5-3b \
  --local-dir /home/weili/models/t5-3b
```

后续脚本统一写：

```text
--model-name-or-path /home/weili/models/t5-3b
--local-files-only
```

这样正式批处理不会因为 Hugging Face 网络、代理配置或缓存目录不同而中断。不要在
T5 加载失败时用全零 embedding 代替，零特征不等价于真实文本条件。

### 5.4 SMPL-X 身体模型

SMPL-X 模型受许可约束，需要在 SMPL-X 官方网站注册、同意协议并手动下载。准备：

```text
SMPLX_NEUTRAL.npz
SMPLX_MALE.npz
SMPLX_FEMALE.npz
```

放到：

```text
inputs/checkpoints/body_models/smplx/
```

最终至少应为：

```text
inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
inputs/checkpoints/body_models/smplx/SMPLX_MALE.npz
inputs/checkpoints/body_models/smplx/SMPLX_FEMALE.npz
```

验证：

```bash
find inputs/checkpoints/body_models/smplx -maxdepth 1 -type f -printf '%f\n' | sort
```

RICH 指标还会使用仓库中的关节回归矩阵和 SMPL-X 到 SMPL 稀疏转换矩阵。当前代码已经
移除了未使用的 SMPL mesh faces 初始化，因此不要通过复制、改名或伪造
`SMPL_NEUTRAL.pkl` 来解决 MetricRICH 初始化问题。

### 5.5 推理 Demo 的额外资源

这部分不是训练依赖。只有需要复现离线视频 Demo 时，才下载 HMR2 和 ViTPose：

```text
inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt
inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth
```

资源可从
[GVHMR 模型目录](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD)
下载。实时 webcam 还需要 ONNX Runtime：

```bash
uv pip install onnxruntime-gpu nvidia-cudnn-cu12
```

实时脚本缺少 ONNX 模型时会按现有逻辑从 `nvidia/GEM-X` 下载到 `inputs/onnx/`。这些
ONNX denoiser 是 regression-only，只用于实时视频回归，不是完整训练、文本生成或音乐
生成的替代品。

---

## 6. 拿到仓库后的快速 Demo 验证

这一节给出一套最短、可复制的上手流程。它的目的不是替代后面的完整数据准备，而是在
投入大量时间下载训练集之前，先确认仓库、Python、CUDA、checkpoint、T5 和基本推理
链路可以工作。

### 6.1 从克隆到进入环境

```bash
cd /home/weili

git clone https://github.com/XiaoxiaoKuankuan/GENMO.git
cd GENMO

sudo apt update
sudo apt install -y \
  git \
  git-lfs \
  curl \
  wget \
  ffmpeg \
  libsndfile1 \
  build-essential

git lfs install

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

uv venv .venv --python 3.10
source .venv/bin/activate

uv pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

uv pip install -e ".[train,dev]"
```

以后每次打开新终端，先执行：

```bash
cd /home/weili/GENMO
source .venv/bin/activate
```

验证安装：

```bash
python - <<'PY'
import torch
import gem

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("GENMO import: OK")
PY
```

### 6.2 准备最少模型资源

下载完整 checkpoint：

```bash
mkdir -p inputs/pretrained

hf download nvidia/GEM-X \
  gem_smpl.ckpt \
  --local-dir inputs/pretrained

ls -lh inputs/pretrained/gem_smpl.ckpt
```

从 SMPL-X 官网手动下载身体模型并放置：

```text
inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
inputs/checkpoints/body_models/smplx/SMPLX_MALE.npz
inputs/checkpoints/body_models/smplx/SMPLX_FEMALE.npz
```

下载本地 T5-3B：

```bash
mkdir -p /home/weili/models/t5-3b

hf download google-t5/t5-3b \
  --local-dir /home/weili/models/t5-3b

du -sh /home/weili/models/t5-3b
```

### 6.3 先运行不加载大模型的 dry-run

纯文本 dry-run 不加载 T5、GEM、checkpoint，也不要求 CUDA：

```bash
python scripts/demo/demo_smpl_text.py \
  --prompt "a person walks forward and waves" \
  --num_frames 60 \
  --fps 30 \
  --dry_run
```

关键输出应接近：

```text
kp2d:          (60, 17, 3)
bbx_xys:       (60, 3)
K_fullimg:     (60, 3, 3)
cam_angvel:    (60, 6)
f_imgseq:      (60, 1024)
text_embed:    (50, 1024)
has_img:       0 / 60
has_2d:        0 / 60
has_cam:       0 / 60
```

音乐 dry-run 会读取真实 WAV、MP3 或 FLAC 并提取 EDGE baseline35，但不加载 GEM：

```bash
python scripts/demo/demo_music.py \
  --audio /path/to/song.wav \
  --duration_sec 2 \
  --dry_run
```

关键输出应包括：

```text
music_embed: (约 60, 35)
has_music:   L / L
has_img:     0 / L
has_2d:      0 / L
has_cam:     0 / L
```

### 6.4 运行真实纯文本生成

先用 60 或 120 帧并关闭渲染，验证最核心的文本扩散和 SMPL-X 参数保存：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/demo/demo_smpl_text.py \
  --prompt "a person walks forward, turns left, and waves with the right hand" \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt \
  --t5_model /home/weili/models/t5-3b \
  --local_files_only \
  --num_frames 120 \
  --fps 30 \
  --seed 42 \
  --guidance_scale 2.5 \
  --ddim_steps 50 \
  --shape_mode zero \
  --no_render
```

成功结果位于：

```text
outputs/text_motion/<唯一动作目录>/
```

检查发布文件：

```bash
find outputs/text_motion \
  -maxdepth 2 \
  -type f \
  \( -name READY -o -name smpl_params.pt -o -name motion.npz \) \
  -print
```

检查最新一条动作的 shape、finite 和零体型：

```bash
python - <<'PY'
from pathlib import Path
import torch

files = sorted(
    Path("outputs/text_motion").glob("*/smpl_params.pt"),
    key=lambda path: path.stat().st_mtime,
)
if not files:
    raise SystemExit("No text-motion smpl_params.pt was found")

path = files[-1]
data = torch.load(path, map_location="cpu", weights_only=False)
body = data["body_params_global"]

print("file:", path)
print("body_pose:", tuple(body["body_pose"].shape))
print("global_orient:", tuple(body["global_orient"].shape))
print("transl:", tuple(body["transl"].shape))
print("betas:", tuple(body["betas"].shape))
print("all finite:", all(value.isfinite().all() for value in body.values()))
print("betas norm:", body["betas"].norm().item())
PY
```

120 帧时应为：

```text
body_pose:     (120, 63)
global_orient: (120, 3)
transl:        (120, 3)
betas:         (120, 10)
all finite:    True
betas norm:    0.0
```

得到 `smpl_params.pt` 和最后创建的 `READY`，说明 T5 编码、完整 GEM checkpoint、
DDIM、151 维动作解码、SMPL-X 参数生成和原子发布已经工作。

### 6.5 运行真实音乐生成

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/demo/demo_music.py \
  --audio /path/to/song.wav \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt \
  --start_sec 0 \
  --duration_sec 5 \
  --output_root outputs/music_motion \
  --seed 42 \
  --shape_mode zero \
  --guidance_scale 2.5 \
  --ddim_steps 50 \
  --save_features \
  --no_render
```

检查：

```bash
find outputs/music_motion \
  -maxdepth 2 \
  -type f \
  \( -name READY -o -name smpl_params.pt -o -name music_features.pt \) \
  -print
```

每个成功目录至少包含：

```text
smpl_params.pt
motion.npz
raw_motion_151d.pt
music_features.pt
metadata.json
source_audio.txt
READY
```

需要全局渲染和原音乐合成时，删除 `--no_render` 并增加：

```text
--mux_audio
```

例如：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/demo/demo_music.py \
  --audio /path/to/song.wav \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt \
  --duration_sec 5 \
  --output_root outputs/music_motion \
  --shape_mode zero \
  --mux_audio
```

渲染或 ffmpeg 失败不会改变已经生成的 SMPL-X 参数；排查动作生成时优先使用
`--no_render`，把模型问题和可视化依赖问题分开。

### 6.6 运行离线视频 Demo

视频 Demo 还需要：

```text
inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt
inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth
```

从第 5.5 节给出的 GVHMR 模型目录下载后运行：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/demo/demo_smpl.py \
  --input_list /path/to/video.mp4 \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt
```

视频和文本混合输入：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/demo/demo_smpl.py \
  --input_list \
    /path/to/video1.mp4 \
    "text:a person walks forward and raises both arms" \
    /path/to/video2.mp4 \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt
```

只保存参数、不渲染：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/demo/demo_smpl.py \
  --input_list /path/to/video.mp4 \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt \
  --no_render
```

### 6.7 运行实时摄像头 Demo

安装 GPU ONNX Runtime：

```bash
uv pip install onnxruntime-gpu nvidia-cudnn-cu12
```

检查执行提供器：

```bash
python - <<'PY'
import onnxruntime as ort
print(ort.get_available_providers())
PY
```

输出中应包含：

```text
CUDAExecutionProvider
```

摄像头运行：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/demo/demo_webcam.py \
  --camera_id 0 \
  --no_imgfeat \
  --display \
  --shape_mode zero
```

摄像头编号不是 0 时修改，例如：

```text
--camera_id 2
```

使用视频文件按实际 FPS 播放：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/demo/demo_webcam.py \
  --video /path/to/video.mp4 \
  --no_imgfeat \
  --display \
  --shape_mode zero
```

首次运行缺少 ONNX 文件时，脚本会按照现有逻辑下载到 `inputs/onnx/`。实时 ONNX 验证
成功只代表视频 regression 路径正常，文本和音乐仍应通过完整 PyTorch checkpoint 验证。

### 6.8 推荐的最短验证顺序

如果只想最快确认新机器能够运行，依次执行：

```bash
# 1. Python、CUDA 和仓库导入
python -c \
  "import torch, gem; print(torch.__version__, torch.cuda.is_available())"

# 2. 不加载大模型的文本输入检查
python scripts/demo/demo_smpl_text.py \
  --prompt "a person walks forward" \
  --num_frames 60 \
  --dry_run

# 3. 真实文本生成
CUDA_VISIBLE_DEVICES=0 \
python scripts/demo/demo_smpl_text.py \
  --prompt "a person walks forward" \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt \
  --t5_model /home/weili/models/t5-3b \
  --local_files_only \
  --num_frames 60 \
  --shape_mode zero \
  --no_render

# 4. 检查完整制品是否最后发布 READY
find outputs/text_motion -maxdepth 2 -name READY -print
```

这四步完成后，再继续下载数百 GB 训练数据、执行第 14 节 preflight 和第 15 节训练，
可以显著减少把环境问题拖到数据准备完成后才发现的情况。

---

## 7. 基础动作与验证数据

完整训练首先需要 AMASS、BEDLAM、H36M、3DPW，验证需要 EMDB 和 RICH。这些数据不是
直接把原始数据下载到仓库就能读取，而是使用 GVHMR 已经整理好的 `hmr4d_support`
训练制品。

GVHMR 官方数据说明中提供了
[Google Drive 预处理归档](https://drive.google.com/drive/folders/10sEef1V_tULzddFxzCmDUpsIqfv7eP-P?usp=drive_link)。
浏览器登录并下载所需的 `*_hmr4d_support.tar.gz` 后，把归档放进 `inputs/` 并解压：

```bash
cd /home/weili/GENMO/inputs

tar -xzvf AMASS_hmr4d_support.tar.gz
tar -xzvf BEDLAM_hmr4d_support.tar.gz
tar -xzvf H36M_hmr4d_support.tar.gz
tar -xzvf 3DPW_hmr4d_support.tar.gz
tar -xzvf EMDB_hmr4d_support.tar.gz
tar -xzvf RICH_hmr4d_support.tar.gz
```

不同批次的归档文件名可能略有变化，解压后的关键目录应是：

```text
inputs/AMASS/hmr4d_support/
inputs/BEDLAM/hmr4d_support/
inputs/H36M/hmr4d_support/
inputs/3DPW/hmr4d_support/
inputs/EMDB/hmr4d_support/
inputs/RICH/hmr4d_support/
```

其中 HumanML3D 构建还会复用：

```text
inputs/AMASS/hmr4d_support/smplxpose_v2.pth
```

GEM-X 仓库中补充的一些小型验证制品可以从 Hugging Face 下载：

```bash
cd /home/weili/GENMO

hf download nvidia/GEM-X \
  --include "gem_smpl/missing_hmr4d_support/**" \
  --local-dir .

cp -a gem_smpl/missing_hmr4d_support/inputs/. inputs/
```

复制前先检查目标，避免覆盖自己已经生成的同名文件：

```bash
find gem_smpl/missing_hmr4d_support/inputs -type f | sort
```

如果只想训练回归模型，准备 AMASS、BEDLAM、H36M、3DPW 和三个基础验证集后即可先走
`gem_smpl_regression`。完整多模态训练还必须继续准备下面三类数据。

---

## 8. HumanML3D：动作与文本训练数据

### 7.1 获取官方仓库

HumanML3D 官方仓库不能直接重新分发 AMASS 动作，但提供 `index.csv`、训练划分和文本
标注。先克隆：

```bash
mkdir -p /home/weili/datasets
cd /home/weili/datasets

git clone https://github.com/EricGuo5513/HumanML3D.git \
  HumanML3D_official
```

检查：

```bash
test -f /home/weili/datasets/HumanML3D_official/index.csv
test -f /home/weili/datasets/HumanML3D_official/HumanML3D/train.txt
test -f /home/weili/datasets/HumanML3D_official/HumanML3D/texts.zip
```

当前构建器可以只读访问 `texts.zip`，不要求把全部文本解压。如果已解压
`HumanML3D/texts/`，也会优先读取目录。

### 7.2 准备精确 AMASS 映射表

本仓库的转换不是根据名称模糊猜测 AMASS 动作，而是读取已经审计过的精确映射：

```text
outputs/humanml3d_amass_exact_coverage.csv
```

当前映射表 SHA256 为：

```text
eec909d987a1f57abbde81931470c99a00f18bd8ad7d1cd9ffd90061e2833ba4
```

新机器复现时必须把这份 CSV 作为小型构建资产一起归档或复制到相同位置，再校验：

```bash
sha256sum outputs/humanml3d_amass_exact_coverage.csv
```

当前仓库没有提供重新生成这份精确映射报告的脚本，所以不要把它误写成
`build_humanml3d_smpl.py` 的输出。构建器只消费该报告，不会用模糊匹配补齐 unmatched
动作。

### 7.3 小规模构建

先处理 20 条精确动作，验证环境和路径：

```bash
cd /home/weili/GENMO
source .venv/bin/activate

python tools/data/humanml3d/build_humanml3d_smpl.py \
  --humanml-root /home/weili/datasets/HumanML3D_official \
  --amass-file inputs/AMASS/hmr4d_support/smplxpose_v2.pth \
  --mapping-csv outputs/humanml3d_amass_exact_coverage.csv \
  --limit 20 \
  --output outputs/humanml3d_test/humanml3d_smplhpose_train.pth \
  --report-dir outputs/humanml3d_test/report \
  --overwrite
```

检查结果：

```bash
python - <<'PY'
import torch

path = "outputs/humanml3d_test/humanml3d_smplhpose_train.pth"
data = torch.load(path, map_location="cpu", weights_only=False)
print("records:", len(data))
for key in list(data)[:5]:
    item = data[key]
    print(
        key,
        item["pose"].shape,
        item["trans"].shape,
        item["beta"].shape,
        item["gender"],
        len(item["text_data"]),
    )
PY
```

### 7.4 全量 dry-run 和正式构建

先执行全量检查但不保存主 PTH：

```bash
python tools/data/humanml3d/build_humanml3d_smpl.py \
  --humanml-root /home/weili/datasets/HumanML3D_official \
  --amass-file inputs/AMASS/hmr4d_support/smplxpose_v2.pth \
  --mapping-csv outputs/humanml3d_amass_exact_coverage.csv \
  --dry-run \
  --report-dir outputs/humanml3d_full_dryrun
```

正式构建：

```bash
python tools/data/humanml3d/build_humanml3d_smpl.py \
  --humanml-root /home/weili/datasets/HumanML3D_official \
  --amass-file inputs/AMASS/hmr4d_support/smplxpose_v2.pth \
  --mapping-csv outputs/humanml3d_amass_exact_coverage.csv \
  --output inputs/HumanML3D_SMPL/hmr4d_support/humanml3d_smplhpose_train.pth \
  --report-dir outputs/humanml3d_build_report \
  --overwrite
```

这个步骤完成时间裁剪、原动作、官方镜像文本、确定性子片段分组和结构验证，只保存
SMPL-X 动作与原始 caption，不提取 T5。

当前真实构建结果是：

```text
index.csv 总行数                 14,616
exact_family_path 训练基础动作   10,600
跳过 HumanAct12                    959
跳过 unmatched                     133
原动作记录                       10,587
镜像记录                         10,587
子片段记录                        2,068
总记录                           23,242
总帧数                        5,044,068
按 30 FPS 计算总时长              46.7043 小时
输出大小                    1,435,215,526 字节
```

正式动作文件的 SHA256 为：

```text
22a222d41ea8d259ef4e3f340ba465c53ecb5e20830215f61cccd5477c4484ac
```

迁移时应对源机器和目标机器实际执行 `sha256sum` 并比较完整值。构建报告中还记录了
140 条非法文本行、14 个过短子片段和 107 个 duration mismatch；它们没有被静默隐藏。
当前没有缺失 AMASS key、缺失文本文件或非法动作 shape。

### 7.5 提取 T5-3B 特征

先估算空间，不加载 T5：

```bash
python tools/data/humanml3d/extract_t5_embeddings.py \
  --input \
    inputs/HumanML3D_SMPL/hmr4d_support/humanml3d_smplhpose_train.pth \
  --estimate-only
```

正式提取：

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/data/humanml3d/extract_t5_embeddings.py \
  --input \
    inputs/HumanML3D_SMPL/hmr4d_support/humanml3d_smplhpose_train.pth \
  --output \
    inputs/HumanML3D_SMPL/t5_embeddings_v1_half/all_text_embed.pth \
  --model-name-or-path /home/weili/models/t5-3b \
  --local-files-only \
  --device cuda:0 \
  --batch-size 16 \
  --motions-per-shard 256 \
  --resume \
  --cleanup-shards \
  --strict
```

为了复现当前已有制品，T5 模型推理保持脚本默认的 float32，最终 embedding 保存为 CPU
float16。不要额外指定 `--model-dtype float16` 来冒充完全相同的生成条件。

当前结果：

```text
motion key             23,242
caption                63,044
embedding shape        [caption_count, 50, 1024]
保存 dtype             CPU float16
中间分片               91
最终文件大小           约 6.46 GB
```

当前最终 T5 文件的 SHA256 为：

```text
d1672e30c92fcccf438ecbf6de17eb8bc58f3adb57f45e88c312ae84c5f23ff0
```

完成后检查：

```bash
ls -lh \
  inputs/HumanML3D_SMPL/hmr4d_support/humanml3d_smplhpose_train.pth \
  inputs/HumanML3D_SMPL/t5_embeddings_v1_half/all_text_embed.pth
```

---

## 9. AIST++：音乐条件动作训练数据

### 8.1 下载官方标注和视频

[AIST++ 官方下载页](https://google.github.io/aistplusplus_dataset/download.html)提供
motion、2D/3D keypoints、camera、split 和完整 annotation 包。下载前需要阅读并同意
AIST Dance Video Database 的使用条款。

官方视频下载器使用：

```bash
mkdir -p /home/weili/datasets/AISTPP_fullset
cd /home/weili/datasets/AISTPP_fullset

wget \
  https://raw.githubusercontent.com/google/aistplusplus_api/main/downloader.py

python downloader.py \
  --download_folder=/home/weili/datasets/AISTPP_fullset/videos \
  --num_processes=5
```

从 AIST++ 官方下载页取得完整 annotation 压缩包后，整理为：

```text
/home/weili/datasets/AISTPP_fullset/aist_plusplus_final/
├── motions/
├── cameras/
├── keypoints2d/
├── keypoints3d/
├── splits/
└── ignore_list.txt
```

官方 motion 和 keypoints 是 60 FPS。GENMO 的 AIST++ 构建工具会同步取 `[::2]` 转为
30 FPS。

### 8.2 准备同名对齐 WAV

EDGE baseline35 特征必须从与每个 sequence 时间对齐的同名 WAV 提取，例如：

```text
gBR_sBM_cAll_d04_mBR0_ch01.wav
```

不能拿 `mBR0.wav` 从第 0 秒开始替代所有使用同一音乐的舞蹈序列，也不能猜测切片偏移。
当前使用的目录是：

```text
/home/weili/datasets/AISTPP_fullset/music_prepare/aligned_wav_official
```

当前目录中有 981 个真实同名 WAV，另有 408 个链接到早期已经对齐的 WAV，共 1,389 个
可解析文件，仍有 19 个缺失项。若下载的是已经与 sequence 对齐的视频，可以用 ffmpeg
逐个抽取音轨：

```bash
ffmpeg -i <sequence_video.mp4> -vn -ac 1 <sequence_name>.wav
```

这条命令只负责抽取已经对齐的视频音轨，不负责估计偏移。仓库当前没有自动恢复
per-sequence 音乐对齐关系的工具，因此应把对齐 WAV 目录作为需要备份的数据资产。

### 8.3 审计 AIST++

```bash
cd /home/weili/GENMO

python tools/data/aistpp/audit_aistpp.py \
  --annotations-root \
    /home/weili/datasets/AISTPP_fullset/aist_plusplus_final \
  --aligned-wav-dir \
    /home/weili/datasets/AISTPP_fullset/music_prepare/aligned_wav_official
```

### 8.4 提取 EDGE baseline35 音乐特征

```bash
python tools/data/aistpp/extract_musicfeat_v2.py \
  --annotations-root \
    /home/weili/datasets/AISTPP_fullset/aist_plusplus_final \
  --aligned-wav-dir \
    /home/weili/datasets/AISTPP_fullset/music_prepare/aligned_wav_official \
  --output-dir inputs/AIST++/musicfeat_v2 \
  --allow-missing \
  --overwrite
```

`--allow-missing` 只表示把缺失 WAV 写进报告后继续提取，不能让正式官方 980/20/20
split 缺少任意一条必要特征。正式构建前仍需确认官方 split 的所有 sequence 都可用。

输出是 30 FPS 的 float32 Tensor：

```text
[L, 35]
```

35 维通道固定为：

```text
0       onset strength
1:21    20 MFCC
21:33   12 chroma CENS
33      binary onset peak
34      binary beat peak
```

提取固定使用：

```text
sample_rate = 15360
hop_length  = 512
target_fps  = 30
```

不要截断列、补列、PCA、随机投影或用 mel spectrogram 替代。

### 8.5 构建官方 980/20/20 训练文件

先 dry-run：

```bash
python tools/data/aistpp/build_annot_aist_official_30fps.py \
  --annotations-root \
    /home/weili/datasets/AISTPP_fullset/aist_plusplus_final \
  --musicfeat-dir inputs/AIST++/musicfeat_v2 \
  --output-root inputs/AIST++ \
  --view c01 \
  --dry-run \
  --allow-ignored-official \
  --report-dir outputs/aistpp_official_dryrun
```

当前官方 split 与 `ignore_list.txt` 有 28 条交集，已经人工检查后使用
`--allow-ignored-official` 明确保留。新数据版本不要无条件照抄这个参数，应先查看
dry-run 报告。

正式构建：

```bash
python tools/data/aistpp/build_annot_aist_official_30fps.py \
  --annotations-root \
    /home/weili/datasets/AISTPP_fullset/aist_plusplus_final \
  --musicfeat-dir inputs/AIST++/musicfeat_v2 \
  --output-root inputs/AIST++ \
  --view c01 \
  --allow-ignored-official \
  --report-dir outputs/aistpp_official_build \
  --overwrite
```

必须得到：

```text
inputs/AIST++/annot_aist_30fps.pt
inputs/AIST++/train.pt
inputs/AIST++/val.pt
inputs/AIST++/test.pt
inputs/AIST++/minitrain.pt
```

当前真实结果：

```text
annot 总序列       1,020
train                980
val                   20
test                  20
minitrain             16
总帧数            430,287
按 30 FPS 时长      3.984 小时
```

早期的 `build_annot_aist_30fps.py` 是 partial 数据构建器。本机曾用 411 条动作和 408
条音乐特征构建 408 条工程验证数据，但它不是官方 benchmark split。正式完整训练应使用
`build_annot_aist_official_30fps.py` 生成的无 `_partial` 文件。

---

## 10. BEAT2：语音条件动作训练数据

### 9.1 下载

BEAT2 使用 Git LFS 发布。可从官方 PantoMatrix Hugging Face 数据仓库克隆：

```bash
cd /home/weili/datasets
git lfs install

git clone https://huggingface.co/datasets/H-Liu1997/BEAT2 \
  BEAT2_official
```

确认文件不是 Git LFS pointer：

```bash
cd /home/weili/datasets/BEAT2_official
git lfs pull
du -sh .
```

当前数据根包含：

```text
Chinese/
English/
Japanese/
Spanish/
```

各语言目录下应有 split CSV、SMPL-X NPZ 和对应 WAV。

### 9.2 审计并构建 split 索引

先 dry-run：

```bash
cd /home/weili/GENMO

python tools/data/beat2/build_all_splits.py \
  --root /home/weili/datasets/BEAT2_official \
  --dry-run \
  --report-dir outputs/beat2_build_dryrun
```

当前本机中文子集有 7 条 CSV 记录缺少 NPZ。首选做法是重新下载缺失文件。如果确认
跳过并保留审计记录，可显式执行：

```bash
python tools/data/beat2/build_all_splits.py \
  --root /home/weili/datasets/BEAT2_official \
  --output /home/weili/datasets/BEAT2_official/all_splits.pth \
  --report-dir outputs/beat2_build_report \
  --allow-missing-pairs \
  --overwrite
```

默认不会把 `additional` 合并进 train。当前结果：

```text
有效序列       2,047
train          1,376
val              118
test             355
additional       198
minitrain         16
总时长        66.085 小时
缺失 WAV          0
缺失 NPZ          7
```

最后建立仓库软链接：

```bash
cd /home/weili/GENMO

if [ ! -e inputs/BEAT2 ]; then
  ln -s /home/weili/datasets/BEAT2_official inputs/BEAT2
fi

readlink -f inputs/BEAT2
test -f inputs/BEAT2/all_splits.pth
```

这个工具只生成索引，不预提取语音特征。训练 Dataset 会读取原始 WAV 和动作数据。

---

## 11. Motion-X++：后来补充的三维动作与文本数据

Motion-X++ 不是 `gem_smpl_server` 的必要数据，而是后来加入的扩展训练来源。当前第一版
只使用可靠的 SMPL-X 三维动作和 semantic text，不启用关键点条件。原因是公开关键点
归档没有配套图像尺寸和校准相机参数，不能伪造相机 K。

### 10.1 下载

从官方 Hugging Face 数据集下载到外部数据盘：

```bash
mkdir -p /home/weili/datasets/Motion-Xplusplus

hf download YuhongZhang/Motion-Xplusplus \
  --repo-type dataset \
  --local-dir /home/weili/datasets/Motion-Xplusplus
```

也可以使用 Git LFS：

```bash
cd /home/weili/datasets
git lfs install
git clone \
  https://huggingface.co/datasets/YuhongZhang/Motion-Xplusplus \
  Motion-Xplusplus
cd Motion-Xplusplus
git lfs pull
```

在仓库中建立软链接：

```bash
cd /home/weili/GENMO

if [ ! -e inputs/Motion-Xplusplus ]; then
  ln -s \
    /home/weili/datasets/Motion-Xplusplus \
    inputs/Motion-Xplusplus
fi

readlink -f inputs/Motion-Xplusplus
du -sh inputs/Motion-Xplusplus
```

服务器用户名为 `liwei` 时，实际使用过：

```bash
cd /home/liwei/GENMO

if [ ! -e inputs/Motion-Xplusplus ]; then
  ln -s \
    /home/liwei/datasets/Motion-Xplusplus \
    inputs/Motion-Xplusplus
fi
```

### 10.2 审计

```bash
python tools/data/motionxpp/inspect_motionxpp.py \
  --root inputs/Motion-Xplusplus \
  --output-dir outputs/motionxpp_inspect \
  --sample-count 3
```

当前完整归档统计：

| subset | SMPL-X | semantic text | keypoints |
|---|---:|---:|---:|
| animation | 559 | 559 | 559 |
| haa500 | 6,944 | 6,944 | 6,944 |
| humman | 971 | 971 | 971 |
| idea400 | 12,040 | 12,040 | 12,040 |
| kungfu | 1,031 | 1,031 | 1,032 |
| music | 3,394 | 3,394 | 3,394 |
| perform | 922 | 922 | 922 |
| 合计 | 25,861 | 25,861 | 25,862 |

审计会生成推荐 subset 列表：

```text
outputs/motionxpp_inspect/recommended_subsets.txt
```

### 10.3 先做 8 条 smoke

```bash
python tools/data/motionxpp/build_motionxpp_genmo.py \
  --root inputs/Motion-Xplusplus \
  --output-root outputs/motionxpp_smoke/genmo_support \
  --subsets animation \
  --limit 8 \
  --records-per-shard 4 \
  --source-up-axis y \
  --target-fps 30 \
  --strict
```

提取 8 条 T5：

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/data/motionxpp/extract_t5_embeddings.py \
  --manifest \
    outputs/motionxpp_smoke/genmo_support/manifests/train.jsonl \
  --output-root \
    outputs/motionxpp_smoke/t5_embeddings_v1_half \
  --batch-size 8 \
  --motions-per-shard 4 \
  --model-name-or-path /home/weili/models/t5-3b \
  --local-files-only \
  --device cuda:0 \
  --model-dtype float16 \
  --strict
```

预检：

```bash
python tools/data/motionxpp/preflight_motionxpp.py \
  --root inputs/Motion-Xplusplus \
  --motion-manifest \
    outputs/motionxpp_smoke/genmo_support/manifests/train.jsonl \
  --embedding-manifest \
    outputs/motionxpp_smoke/t5_embeddings_v1_half/manifests/train.json \
  --sample-records 8 \
  --dataset-samples 8 \
  --report outputs/motionxpp_smoke/preflight_report.json
```

### 10.4 全量构建

动作分片：

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

中断后使用完全相同的参数并增加：

```text
--resume
```

分别为 train、val、test 提取 T5 分片：

```bash
for SPLIT in train val test; do
  CUDA_VISIBLE_DEVICES=0 \
  python tools/data/motionxpp/extract_t5_embeddings.py \
    --manifest \
      inputs/Motion-Xplusplus/genmo_support/manifests/${SPLIT}.jsonl \
    --output-root \
      inputs/Motion-Xplusplus/t5_embeddings_v1_half \
    --batch-size 16 \
    --motions-per-shard 256 \
    --model-name-or-path /home/weili/models/t5-3b \
    --local-files-only \
    --device cuda:0 \
    --model-dtype float16 \
    --resume \
    --strict
done
```

完整预检时把 manifest 换成正式路径。Motion-X++ 使用动作和 embedding 小分片，避免每个
DataLoader worker 都加载一个巨大的单体文件。

更完整的 Motion-X++ 参数说明见：

```text
docs/motionxpp.md
```

---

## 12. 本仓库新增的数据准备工具

为了把各数据集整理成 GEM 训练直接读取的格式，本项目新增了以下几组工具。这里按任务
说明用途，不逐个展开代码实现。

| 数据 | 工具 | 用途 |
|---|---|---|
| HumanML3D | `build_humanml3d_smpl.py` | 用精确 AMASS 映射构建 30 FPS SMPL-X 动作、官方镜像和文本子片段 |
| HumanML3D | `extract_t5_embeddings.py` | 将所有 caption 离线编码为 T5-3B `[50,1024]` 特征 |
| AIST++ | `audit_aistpp.py` | 检查官方标注目录、动作数量、帧长和同名 WAV 覆盖率 |
| AIST++ | `extract_musicfeat_v2.py` | 提取 EDGE baseline35、30 FPS 音乐特征 |
| AIST++ | `build_annot_aist_official_30fps.py` | 构建官方 crossmodal 980/20/20 训练文件 |
| AIST++ | `build_annot_aist_30fps.py` | 构建不完整数据的 partial 工程验证集 |
| BEAT2 | `build_all_splits.py` | 审计四语言 NPZ/WAV 配对并建立 split 索引 |
| Motion-X++ | `inspect_motionxpp.py` | 审计 ZIP/目录、schema、配对和 subset 重叠 |
| Motion-X++ | `build_motionxpp_genmo.py` | 构建 30 FPS SMPL-X 动作分片和确定性 split |
| Motion-X++ | `extract_t5_embeddings.py` | 构建可恢复的 T5 文本特征分片 |
| Motion-X++ | `preflight_motionxpp.py` | 检查动作、embedding 和真实 Dataset batch |
| 全部正式数据 | `preflight_gem_smpl.py` | 在训练前验证配置、模型资源、所有数据集和混合 batch |

这些工具的共同原则是：

- 不修改原始数据；
- 不静默填补缺失条件；
- 不用模糊匹配替代精确映射；
- 先写临时文件、回读验证，再原子发布；
- 把跳过项和异常写进 JSON/CSV 报告；
- 大型数据尽量使用分片、resume 和 worker 本地小缓存。

---

## 13. 完整训练前应有的目录

执行 `gem_smpl_server` 前，关键结构至少是：

```text
inputs/
├── pretrained/
│   └── gem_smpl.ckpt
├── checkpoints/
│   └── body_models/
│       └── smplx/
│           ├── SMPLX_NEUTRAL.npz
│           ├── SMPLX_MALE.npz
│           └── SMPLX_FEMALE.npz
├── AMASS/hmr4d_support/
├── BEDLAM/hmr4d_support/
├── H36M/hmr4d_support/
├── 3DPW/hmr4d_support/
├── EMDB/hmr4d_support/
├── RICH/hmr4d_support/
├── HumanML3D_SMPL/
│   ├── hmr4d_support/
│   │   └── humanml3d_smplhpose_train.pth
│   └── t5_embeddings_v1_half/
│       └── all_text_embed.pth
├── AIST++/
│   ├── annot_aist_30fps.pt
│   ├── train.pt
│   ├── val.pt
│   ├── test.pt
│   ├── minitrain.pt
│   └── musicfeat_v2/
└── BEAT2 -> /home/weili/datasets/BEAT2_official
```

Motion-X++ 微调还需要：

```text
inputs/Motion-Xplusplus -> /home/weili/datasets/Motion-Xplusplus

inputs/Motion-Xplusplus/
├── genmo_support/
│   ├── manifests/
│   ├── shards/
│   └── reports/
└── t5_embeddings_v1_half/
    ├── manifests/
    └── shards/
```

快速检查软链接：

```bash
find inputs -maxdepth 1 -type l -printf '%p -> %l\n'
readlink -f inputs/BEAT2
readlink -f inputs/Motion-Xplusplus
```

---

## 14. 统一训练前预检

正式完整训练使用：

```bash
cd /home/weili/GENMO
source .venv/bin/activate

python tools/train/preflight_gem_smpl.py \
  --exp gem_smpl_server \
  --samples-per-dataset 1 \
  --batch-size 2 \
  --num-workers 0 \
  --instantiate-model \
  --check-pretrained \
  --strict
```

预检不是只检查文件是否存在，它还会：

- 通过 Hydra 组合真实训练配置；
- 检查完整 checkpoint 的文本和音乐权重；
- 确认音乐输入维数为 35，动作表示为 151 维；
- 检查 HumanML3D 动作 key 与 T5 key；
- 检查 AIST++ 官方 split 和音乐特征；
- 检查 BEAT2 的 NPZ/WAV 配对；
- 检查身体模型与验证集资源；
- 逐个实例化 Dataset 并读取真实样本；
- 使用现有 `collate_fn` 组成混合 batch；
- 可实例化完整 GEM 模型。

报告写入：

```text
outputs/preflight_gem_smpl/report.json
```

当前已经通过的真实统计：

| 训练集 | Dataset 长度 |
|---|---:|
| AMASS | 52,788 |
| BEDLAM | 37,537 |
| H36M | 6,196 |
| 3DPW | 88 |
| AIST++ | 980 |
| BEAT2 | 1,376 |
| HumanML3D | 34,534 |

验证集：

| 验证集 | Dataset 长度 |
|---|---:|
| EMDB1 | 17 |
| EMDB2 | 25 |
| 3DPW | 37 |
| RICH | 191 |

只有 preflight 状态为 `passed` 后才进入训练。若这里失败，优先修正数据路径、shape、
finite、split 或资源问题，不要通过删除 metric、吞掉异常或填零来强行开训。

---

## 15. 训练命令

### 14.1 先跑 20 步 smoke

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/train.py \
  exp=gem_smpl_server \
  pl_trainer.devices=1 \
  pl_trainer.max_steps=20 \
  data.loader_opts.train.batch_size=2 \
  data.loader_opts.train.num_workers=0 \
  use_wandb=false
```

这个步骤用真实模型和真实数据验证：

- DataLoader 能持续取样；
- 混合 batch 能进入模型；
- forward、loss、backward 和 optimizer step 正常；
- 验证 callback 能实例化；
- checkpoint 输出目录可写。

### 14.2 从随机初始化训练完整模型

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/train.py \
  exp=gem_smpl_server \
  use_wandb=false
```

默认配置：

```text
batch_size          4
num_workers         4
precision           16-mixed
gradient_clip_val   0.5
max_steps           500000
val_check_interval  3000
optimizer           AdamW
learning rate       2e-4
```

“随机初始化完整训练”才是严格意义上的从头训练。它计算量很大，先确认 smoke 训练稳定。

### 14.3 从官方 checkpoint 加载权重继续训练

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/train.py \
  exp=gem_smpl_server \
  ckpt_path=inputs/pretrained/gem_smpl.ckpt \
  use_wandb=false
```

这里的 `ckpt_path` 只加载模型权重，不恢复 optimizer、scheduler、global step 或原训练
日志状态。它适合迁移学习和基于官方权重继续训练，不等于真正的断点恢复。

### 14.4 多 GPU

四卡示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/train.py \
  exp=gem_smpl_server \
  pl_trainer.devices=4 \
  data.loader_opts.train.batch_size=4 \
  data.loader_opts.train.num_workers=4 \
  use_wandb=false
```

`batch_size` 是每个进程的 batch。四卡时全局有效 batch 通常为单卡 batch 乘设备数，
还要结合梯度累积设置判断。不要只看命令行中的一个数字。

### 14.5 W&B

需要在线记录：

```bash
wandb login

CUDA_VISIBLE_DEVICES=0 \
python scripts/train.py \
  exp=gem_smpl_server \
  use_wandb=true
```

不需要 W&B：

```text
use_wandb=false
```

### 14.6 回归模型

只训练视频、二维关键点和相机条件的回归路径：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/train.py \
  exp=gem_smpl_regression \
  use_wandb=false
```

这个 checkpoint 不具备文本或音乐 DDIM 生成能力。

### 14.7 原始完整配置

原始 `gem_smpl` 配置还包含本机未准备的 `3dpw_occ_v1`：

```bash
python scripts/train.py exp=gem_smpl
```

只有在 `inputs/3DPW/hmr4d_support/imgfeats/3dpw_occ_train` 等 3DPW-OCC 资源完整时才使用。
当前可靠复现入口是：

```text
exp=gem_smpl_server
```

---

## 16. Motion-X++ 微调训练

Motion-X++ 完整构建和预检通过后，先做 20 步：

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

正式单卡：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/train.py \
  exp=gem_smpl_motionxpp \
  ckpt_path=inputs/pretrained/gem_smpl.ckpt \
  pl_trainer.devices=1 \
  use_wandb=false
```

正式四卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/train.py \
  exp=gem_smpl_motionxpp \
  ckpt_path=inputs/pretrained/gem_smpl.ckpt \
  pl_trainer.devices=4 \
  data.loader_opts.train.batch_size=4 \
  data.loader_opts.train.num_workers=4 \
  use_wandb=false
```

当前 Motion-X++ 配置的主要设置：

```text
初始化权重       官方 gem_smpl.ckpt
学习率           2e-5
最大步数         20,000
精度             16-mixed
默认 batch       4
默认 workers     4
条件             SMPL-X 3D + semantic text
关键点条件       关闭
```

模型结构仍保留完整 audio/music 参数，以兼容官方 checkpoint，但这个实验没有加入 BEAT2
或 3DPW-OCC 数据源。

---

## 17. 断点恢复

训练输出通常位于：

```text
outputs/gem_mixed/<exp_name>/version_<N>/
```

其中包含配置快照、日志和 checkpoint。查看：

```bash
find outputs/gem_mixed -type f \
  \( -name '*.ckpt' -o -name 'last.ckpt' \) \
  -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort
```

真正恢复 optimizer、scheduler 和 global step 时，使用当前训练入口支持的：

```bash
python scripts/train.py \
  exp=gem_smpl_server \
  resume_mode=last
```

恢复时保持同一实验和输出路径设置。不要把：

```text
ckpt_path=inputs/pretrained/gem_smpl.ckpt
```

误当成断点恢复。前者只是载入权重，后者才会继续已有训练状态。

开始长任务前建议检查：

```bash
df -h .
nvidia-smi
ulimit -n
```

训练中监控：

```bash
watch -n 1 nvidia-smi
find outputs/gem_mixed -name '*.ckpt' -ls
```

---

## 18. 训练后快速验证

训练结束后先确认 checkpoint 可被 Lightning 读取，再根据目标选择 Demo。

视频回归路径：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/demo/demo_smpl.py \
  --input_list <video_path> \
  --ckpt_path <trained_checkpoint>
```

纯文本扩散路径：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/demo/demo_smpl_text.py \
  --prompt "a person walks forward and waves" \
  --ckpt_path <full_gem_smpl_checkpoint> \
  --num_frames 120 \
  --fps 30 \
  --seed 42 \
  --shape_mode zero \
  --no_render
```

纯音乐扩散路径：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/demo/demo_music.py \
  --audio <song.wav> \
  --ckpt_path <full_gem_smpl_checkpoint> \
  --duration_sec 5 \
  --output_root outputs/music_motion \
  --shape_mode zero \
  --no_render
```

文本和音乐生成必须测试完整 checkpoint。如果把 regression checkpoint 传给这些 Demo，
应当明确失败，而不是继续用随机文本层或全零条件。

---

## 19. 常见问题

### 18.1 T5 下载失败或代理报错

如果出现 Hugging Face 代理错误，例如不支持的 `socks://` scheme，先检查：

```bash
env | grep -i proxy
```

可在确认网络策略后临时清理错误代理：

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy
```

最稳定的方法是在网络正常的机器上执行 `hf download`，把完整本地目录复制到服务器，
然后使用：

```text
--model-name-or-path /absolute/path/to/t5-3b
--local-files-only
```

### 18.2 `SMPLX_NEUTRAL.npz` 不存在

这是许可证模型，不能靠 `pip install smplx` 自动获得。必须从 SMPL-X 官方下载后放到
指定目录。不要把其他 body model 改名冒充。

### 18.3 AIST++ 音乐特征少于动作数量

先查 `missing_wavs.json`。每条动作需要同 stem 且时间对齐的 WAV。不能把整首音乐从
第 0 秒复制给所有舞蹈，也不能用最后一帧 padding 掩盖大范围缺失。

### 18.4 HumanML3D 缺少映射 CSV

`build_humanml3d_smpl.py` 不生成映射表。必须从当前工程资产备份
`outputs/humanml3d_amass_exact_coverage.csv`，并校验 SHA256。没有映射表时不要启用
模糊匹配。

### 18.5 BEAT2 是 Git LFS pointer

如果文件只有几 KB 并以 Git LFS 文本开头，执行：

```bash
cd /home/weili/datasets/BEAT2_official
git lfs install
git lfs pull
```

### 18.6 DataLoader worker 被杀死

先把 worker 和 batch 调低：

```bash
python scripts/train.py exp=gem_smpl_server \
  data.loader_opts.train.batch_size=2 \
  data.loader_opts.train.num_workers=0 \
  use_wandb=false
```

确认可运行后再逐步增加。HumanML3D 是较大的单体 PTH，Motion-X++ 已采用分片和小型
LRU 以减少 worker 内存。

### 18.7 CUDA OOM

依次降低：

```text
data.loader_opts.train.batch_size
pl_trainer.devices
```

训练已经默认使用 `16-mixed`。不要在同一张卡上同时运行 T5 提取、推理 server 和正式
训练。

### 18.8 `gem_smpl` 启动时缺少 3DPW-OCC

这是原配置的数据依赖。当前数据条件下改用：

```text
exp=gem_smpl_server
```

不要创建空目录或伪造图像特征来让原配置越过检查。

### 18.9 预检通过但训练很慢

预检只证明契约和模型路径正确，不代表当前 I/O 和 worker 数已经最优。观察 GPU
utilization、CPU、磁盘吞吐和共享内存，再调 `batch_size`、`num_workers` 和数据盘位置。

---

## 20. 一台新服务器的最短检查清单

完成迁移后逐项运行：

```bash
cd /home/weili/GENMO
source .venv/bin/activate

git status
nvidia-smi

test -f inputs/pretrained/gem_smpl.ckpt
test -f inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
test -f inputs/AMASS/hmr4d_support/smplxpose_v2.pth
test -f \
  inputs/HumanML3D_SMPL/hmr4d_support/humanml3d_smplhpose_train.pth
test -f \
  inputs/HumanML3D_SMPL/t5_embeddings_v1_half/all_text_embed.pth
test -f inputs/AIST++/annot_aist_30fps.pt
test -f inputs/AIST++/train.pt
test -f inputs/AIST++/val.pt
test -f inputs/AIST++/test.pt
test -f inputs/BEAT2/all_splits.pth
```

然后：

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

20 步：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/train.py \
  exp=gem_smpl_server \
  pl_trainer.max_steps=20 \
  data.loader_opts.train.batch_size=2 \
  data.loader_opts.train.num_workers=0 \
  use_wandb=false
```

最后启动正式训练：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/train.py \
  exp=gem_smpl_server \
  use_wandb=false
```

---

## 21. 复现边界与应备份的资产

能够只依赖公开下载和仓库脚本重建的内容：

- Python 环境；
- 完整 GEM checkpoint；
- T5-3B；
- GVHMR 预处理数据；
- HumanML3D 官方文本和 split；
- HumanML3D 动作 PTH 与 T5 embedding；
- AIST++ baseline35 特征和正式训练 PTH；
- BEAT2 split 索引；
- Motion-X++ 动作和 T5 分片；
- 完整训练 preflight 和训练配置。

需要额外保存、不能假定仓库会自动重建的本地资产：

- `outputs/humanml3d_amass_exact_coverage.csv`；
- AIST++ 每个 sequence 的同名对齐 WAV；
- 受许可约束的 SMPL-X model 文件；
- 已训练 checkpoint；
- 数据构建报告和各正式制品的完整 SHA256；
- 若上游下载链接变化，GVHMR 预处理归档的本地备份。

建议为每次正式构建保存：

```bash
sha256sum <artifact> > <artifact>.sha256
git rev-parse HEAD > outputs/reproduction_git_commit.txt
python -m pip freeze > outputs/reproduction_pip_freeze.txt
nvidia-smi > outputs/reproduction_nvidia_smi.txt
```

这样以后才能区分“代码相同”与“代码、模型、数据和运行环境都相同”。

---

## 22. 当前推荐结论

如果目标是复现当前已经验证的数据组合和完整训练链路，使用：

```text
exp=gem_smpl_server
```

如果目标是在官方完整 checkpoint 上增加 Motion-X++ 三维动作和语义文本监督，使用：

```text
exp=gem_smpl_motionxpp
ckpt_path=inputs/pretrained/gem_smpl.ckpt
```

不要把实时 ONNX 模型用于文本、音乐或完整扩散训练；不要把 partial AIST++ 数据称为
官方 980/20/20 split；不要把 `ckpt_path` 当成 optimizer 断点恢复；不要为了通过预检
伪造缺失条件。

按照本文完成环境、模型、基础数据、HumanML3D、AIST++、BEAT2、preflight 和 20 步
smoke 后，才算真正具备可复现的 GEM-SMPL 完整训练条件。
