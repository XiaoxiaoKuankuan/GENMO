# GENMO 本地文本动作工作台

在仓库根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python scripts/demo/demo_smpl_text_web.py
```

打开 **http://127.0.0.1:8766**。服务只绑定本机回环地址，前端不需要 Node 或联网。
首次生成加载 T5 和 GENMO，此后同模型保持常驻；关闭终端中的服务（Ctrl+C）会释放模型。
如端口已占用，请使用现有服务，或通过 `--port 8767` 选择另一个端口。
同一输出目录只允许一个服务进程，另一个服务须同时指定不同的 `--output_root`。

## 使用方法

1. 下拉选择模型，默认使用本地 MotionMillion `s190000.ckpt`。需要其他模型时，展开
   “添加本地模型”，粘贴 `.ckpt` 路径并校验；路径注册后重启仍保留。
2. 输入 prompt、动作帧数和 DDIM 步数，点击“生成动作视频”。生成期间可以编辑下一次
   参数，但当前任务使用提交时的副本，同时只允许一个任务。
3. 完成后手动点击视频播放。支持暂停、拖动、全屏和下载；新视频从零开始且不会自动播放。
4. 历史中的“回看视频”切换播放器，“复用参数”回填模型、原文、帧数和 DDIM，均不会自动生成。

帧数为 **1–900** 的整数，默认 **120**；DDIM 为 **2–1000** 的整数，默认 **50**。
FPS 固定 **30**，120 帧即 4 秒，240 帧即 8 秒。CFG 为 2.5、seed 为 42、shape 为 zero，
沿用 demo 默认后处理，渲染分辨率为 1280×720。文本使用 T5，建议英文动作描述；服务不会翻译。
任务记录保留提交的 prompt 原文；T5 推理沿用原接口去除首尾空白，超长文本按模型 token 上限截断。

固定的本地 T5 路径：

```text
/home/weili/.cache/huggingface/hub/models--t5-3b/snapshots/bed96aab9ee46012a5046386105ee5fd0ac572f0
```

## 模型与依赖

自动扫描 `inputs/pretrained` 和 `inputs/checkpoints` 的 `.ckpt` 文件；CPU mmap 检查真实权重，
仅列出带文本编码层、交叉注意力和 151 维 SMPL 扩散输出的模型。已知 BUMI、音乐和回归权重
不会进入列表。模型契约自动解析旧版 50 token 和新版 150 token；路径、文件身份、大小、修改
及变更时间用于缓存失效，生成前也检查文件是否变化。请注册已经完整保存的 checkpoint。

Python 环境需包含本仓库原有推理依赖、CUDA、SMPL-X 资产、Open3D、PyAV 和 Flask 3.1。
Flask 作为可选 `web` 依赖声明；在已配置 GENMO 的环境可用 `pip install -e '.[web]'` 安装。
系统需有包含 `libx264` 的 `ffmpeg`。当前运行入口依赖仓库内配置和模型资产，应从本仓库启动。

## 结果保存与故障恢复

正式结果默认持久保存到 `outputs/text_motion_web/`，不会自动删除。每个任务结构为：

```text
outputs/text_motion_web/
  models.json                       # 手动注册的模型真实路径
  history.json                      # 原子保存的历史索引
  tasks/<任务 ID>/
    task.json                       # 原始参数、冻结模型、固定设置、阶段和耗时
    artifacts/<动作目录>/
      smpl_params.pt                # 全局与相机坐标的 SMPL 参数
      motion.npz                    # body_pose、global_orient、transl、betas、fps
      metadata.json / prompt.txt    # 原 demo 的推理元数据和规范化文本
      READY                         # 原动作产物契约，不代表网页视频完成
      video.mp4 / thumbnail.jpg
      media_checks.json             # 视频编码、帧数、FPS 和完整解码结果
      render.log                    # 本次独立渲染进程日志
```

网页只有在动作数组有限、零体型和 FPS 契约满足，且 H.264/yuv420p 视频完整解码、帧数正确后
才标记完成。动作生成但渲染失败时，动作文件仍保留，历史显示失败及所在阶段，上一条成功视频
保持可用。显存不足时可以降低帧数重试；切换 checkpoint 会释放旧引擎并重新加载 T5 和 GENMO。
服务重启保留完成记录，中断任务标记失败，不自动重跑。异常退出的 GPU 工作进程会在后续提交时
重建；Linux 父进程退出会结束其 GPU / 渲染子进程。

## 本地 HTTP 接口

| 方法与路径 | 内容 |
|---|---|
| `GET /api/models` | 已校验模型、扫描状态和固定参数 |
| `POST /api/models` | JSON `{"path":"/absolute/model.ckpt"}`，校验并注册 |
| `POST /api/jobs` | 仅接受 `model_id`、`prompt`、`num_frames`、`ddim_steps` |
| `GET /api/jobs/<id>` | 单任务状态、错误、冻结参数和结果 URL |
| `GET /api/history` | 倒序历史、当前活动任务和历史读取错误 |
| `GET /api/jobs/<id>/video` | 已完成视频，支持 HTTP Range 和拖动 |
| `GET /api/jobs/<id>/thumbnail` | 已完成任务缩略图 |

提交成功为 HTTP 202，非法参数为 400，已有活动任务为 409；POST 需要同源 JSON。
媒体路径由任务 ID 映射，不支持通用文件路径读取。原有命令行 demo、stdin/ZMQ 文本服务的
请求格式及默认参数保持兼容，网页新增 DDIM 更新方法仅由网页工作进程调用。
