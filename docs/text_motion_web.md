# GENMO 本地文本动作工作台

## 2026-09-28：服务器2 s215000 ONNX网页交付

- 本机验证入口：**http://127.0.0.1:8767/**；原8766继续提供UMR视频。
- 本次公网入口：**https://rivers-sunrise-sake-vip.trycloudflare.com/**。域名随临时隧道重建可能变化，
  当前进程/地址记录在`outputs/text_motion_web/share/share.json`，本机必须保持运行。
- 默认已注册模型：`BUMI ONNX · bumi_text_crop120_s215000 / deployment.json · step 215000`。
  实际使用ONNX Runtime CUDA，默认120帧、30FPS、DDIM50、CFG2.5、seed42，契约范围4–120帧。
- 完整checkpoint：`inputs/checkpoints/bumi_text_crop120_s215000/s215000.ckpt`，来源/SHA见同目录`source.json`。
  ONNX和整套资产：`inputs/deployments/bumi_text_crop120_s215000/`，主文件为`models/model.onnx`。
- 已保留走路、举双臂、下蹲三条实际生成视频，均为单视角1280×720、H.264、120帧；网页历史可回看。
  数值验收为同训练T5的三条文本×四种帧长共12组，报告在部署包`validation.json`。
  随机高斯文本条件压力测试未通过，独立报告保留在`outputs/text_motion_web/release/validation_synthetic_stress.json`；
  不将真实T5验收推广到任意随机文本特征。网页、来源和诊断证据见同目录`release.json`、`web_validation.json`。
- 浏览器模型选择/表单提交已实测；后续浏览器控制超时，因此不宣称自动播放交互验收通过。
  三个视频完整解码、公网下载SHA及HTTP Range均通过，生成后可手动播放检查。

在仓库根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python scripts/demo/demo_bumi_text_web.py
```

打开 **http://127.0.0.1:8766**。服务只绑定本机回环地址，前端不需要 Node 或联网。
首次生成加载 T5 和 GENMO，此后同模型保持常驻；关闭终端中的服务（Ctrl+C）会释放模型。
如端口已占用，请使用现有服务，或通过 `--port 8767` 选择另一个端口。
同一输出目录只允许一个服务进程，另一个服务须同时指定不同的 `--output_root`。

## 使用方法

1. 下拉选择模型，支持训练 `.ckpt` 和由 `tools/export/bumi_text.py package`
   生成的 ONNX 部署包 `deployment.json`。默认 checkpoint 候选路径为
   `inputs/checkpoints/bumi_text/model.ckpt`。需要其他模型时，展开“添加本地模型”，
   粘贴 checkpoint 或部署清单路径并校验；路径注册后重启仍保留。
   模型列表明确显示 `TORCH` / `ONNX`，同训练步数优先列出 ONNX；选择 ONNX 后
   网页实际使用 ONNX Runtime CUDA 执行单步去噪，T5、DDIM 调度和 qpos 解码仍由公共运行时执行。
2. 输入 prompt、动作帧数和 DDIM 步数，点击“生成动作视频”。生成期间可以编辑下一次
   参数，但当前任务使用提交时的副本，同时只允许一个任务。
3. 完成后手动点击视频播放。支持暂停、拖动、全屏和下载；新视频从零开始且不会自动播放。
4. 历史中的“回看视频”切换播放器，“复用参数”回填模型、原文、帧数和 DDIM，均不会自动生成。

帧数按所选模型的真实契约校验：当前 crop120 为 **4–120**，历史 full300 为 **60–300**，默认 **120**；DDIM 为 **2–1000** 的整数，默认 **50**。
FPS 固定 **30**，120 帧即 4 秒，240 帧即 8 秒。CFG 为 2.5、seed 为 42、
沿用 demo 默认后处理，渲染分辨率为 1280×720。文本使用 T5，建议英文动作描述；服务不会翻译。
任务记录保留提交的 prompt 原文；T5 推理沿用原接口去除首尾空白，超长文本按模型 token 上限截断。

固定的本地 T5 路径：

```text
/home/weili/.cache/huggingface/hub/models--t5-3b/snapshots/bed96aab9ee46012a5046386105ee5fd0ac572f0
```

## 模型与依赖

自动扫描 `inputs/pretrained`、`inputs/checkpoints`、`inputs/deployments` 的 `.ckpt` 和
`deployment.json`；CPU mmap 检查 checkpoint 真实权重，
仅列出包含有效 `bumi_text_contract`、30D动作输出、2D接触输出且资产指纹匹配的BUMI文本模型。
SMPL、音乐和缺少契约的权重不会进入列表。文本固定150 token；路径、文件身份、大小、修改
及变更时间用于缓存失效，生成前也检查文件是否变化。部署包另外校验 ONNX、外部权重、
统计和机器人资产的 SHA，缓存绑定全部部署资产的文件身份；任何资产变化后重新校验，
不复用旧引擎。请注册已经完整保存并通过数值对照的模型。

Python 环境需包含本仓库文本推理依赖、CUDA、当前BUMI资产、MuJoCo、PyAV 和 Flask 3.1。
Flask 与公开分享入口使用的 Waitress 作为可选 `web` 依赖声明；在已配置 GENMO 的环境
可用 `pip install -e '.[web]'` 安装。
系统需有包含 `libx264` 的 `ffmpeg`。当前运行入口依赖仓库内配置和模型资产，应从本仓库启动。

## 结果保存与故障恢复

结果保存在 `outputs/text_motion_web/`，**最多保留最近 60 条已结束记录**（成功与失败均计入），
按任务创建时间保留最新记录。新任务结束及服务启动时，自动删除超额的最旧记录、视频、
动作参数、缩略图和日志，实际释放磁盘空间。正在生成的任务不参与清理，因此生成期间
最多临时显示 60 条历史加 1 条活动任务，结束后恢复为 60 条。

清理只作用于本站 `tasks/<任务 ID>`，不删除模型 checkpoint、T5、训练数据或其他网页结果。
目录先原子移到 `.expired_tasks/<任务 ID>` 再删除；清理中断后重启继续，不恢复已淘汰历史。
文件系统错误会显示在历史读取提示中，并在下一次任务结束或重启时重试；遇到路径异常时
保留文件并报告，不能用扩大删除范围的方式强行达成条数上限。每个任务结构为：

```text
outputs/text_motion_web/
  models.json                       # 手动注册的模型真实路径
  history.json                      # 原子保存的历史索引
  tasks/<任务 ID>/
    task.json                       # 原始参数、冻结模型、固定设置、阶段和耗时
    artifacts/<动作目录>/
      motion.npz                    # qpos、qpos_raw、接触logits、关节顺序、fps
      metadata.json                 # 文本、模型SHA、实际后端及ORT providers、模型契约
      video.mp4 / thumbnail.jpg
      media_checks.json             # 视频编码、帧数、FPS 和完整解码结果
      render.log                    # 本次独立渲染进程日志
```

网页只有在 qpos28 动作数组有限、关节顺序/四元数/FPS 契约满足，且 H.264/yuv420p 视频完整解码、帧数正确后
才标记完成。动作生成但渲染失败时，动作文件仍保留，历史显示失败及所在阶段，上一条成功视频
在未超过保留上限时保持可用。若当前选中的旧视频被淘汰，页面切换到仍保留的最新成功视频，
仍需手动播放；没有成功视频时显示空预览。显存不足时可以降低帧数重试；切换 checkpoint
会释放旧引擎并重新加载 T5 和 GENMO。服务重启保留限额内的记录，中断任务标记失败，
不自动重跑。异常退出的 GPU 工作进程会在后续提交时
重建；Linux 父进程退出会结束其 GPU / 渲染子进程。

## 本地 HTTP 接口

| 方法与路径 | 内容 |
|---|---|
| `GET /api/models` | 已校验模型、扫描状态和固定参数 |
| `POST /api/models` | JSON `{"path":"/absolute/model.ckpt"}` 或部署包 `deployment.json` 路径，校验并注册 |
| `POST /api/jobs` | 仅接受 `model_id`、`prompt`、`num_frames`、`ddim_steps` |
| `GET /api/jobs/<id>` | 单任务状态、错误、冻结参数和结果 URL |
| `GET /api/history` | 倒序历史、当前活动任务和历史读取错误 |
| `GET /api/jobs/<id>/video` | 已完成视频，支持 HTTP Range 和拖动 |
| `GET /api/jobs/<id>/thumbnail` | 已完成任务缩略图 |

提交成功为 HTTP 202，非法参数为 400，已有活动任务为 409；POST 需要同源 JSON。
媒体路径由任务 ID 映射，不支持通用文件路径读取。原有命令行 demo、stdin/ZMQ 文本服务的
请求格式及默认参数保持兼容，网页新增 DDIM 更新方法仅由网页工作进程调用。

## 无密码公网分享

公开入口复用本地 8766 的 GPU 推理服务、已注册模型和历史。访客无需账户或密码，
可以选择模型、输入文本、生成动作、播放和下载视频、回看历史。所有访客共用一个
任务位；生成中另一请求返回 409。公开页面明确提示：文本与生成历史对其他访客可见。

共享网关仅在回环地址 `127.0.0.1:8768` 监听，由 HTTPS 隧道转发。访客不能注册模型，
也不会得到 checkpoint、T5、任务目录的本机绝对路径、文件指纹和内部异常堆栈。
本机维护者通过本地服务端口添加 checkpoint 或 ONNX 部署清单，注册后公网页面刷新可选。
本分支只接受 BUMI 150 token 文本模型；共享网关不改变模型契约或固定推理参数。

公网提交只接收原有四个生成字段，帧数和 DDIM 边界不变；额外限制 prompt 最多
4096 个字符、JSON 请求体最多 16 KiB。后台继续保留原文，编码器按模型 token 上限截断。
公开媒体接口保留 Range、HEAD、ETag 和条件请求，浏览器可以拖动进度。

### 启动分享入口

先确认本机 8766 服务已启动，再从
[Cloudflare 官方下载页](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/)
安装 `cloudflared`。当前本机位于 `$HOME/.local/bin/cloudflared`。
终端一启动隧道：

```bash
"$HOME/.local/bin/cloudflared" tunnel --no-autoupdate --protocol http2 \
  --edge-ip-version 4 --url http://127.0.0.1:8768 --metrics 127.0.0.1:18768
```

日志出现 `https://….trycloudflare.com` 后，在终端二将它填入 `--public-origin`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/demo/share_bumi_text_web.py \
  --public-origin https://实际生成的域名.trycloudflare.com
```

在网关启动前隧道短暂返回 502 是因为 8768 尚未监听；网关启动后即可分享 HTTPS 地址。
运行中的分享服务无需重复启动。首次交付已将两个进程独立于终端运行，状态保存在：

```text
outputs/text_motion_web/share/
  share.json       # 当前实际公网 URL、上游地址以及两个进程的 PID
  gateway.log      # Waitress 网关日志
  gateway.pid
  tunnel.log       # 隧道连接状态与随机域名
  tunnel.pid
```

上述文件描述本次已启动的进程，手动重建时须以新进程和新隧道日志为准，不把旧 PID 或旧 URL
当作当前状态。停止公开分享时，先用 `ps -p <PID> -o pid,lstart,args` 核对对应命令，再结束
分享网关与 cloudflared 两个进程；保留 8766 本地推理服务及当前限额内的用户历史。前台启动时可在
两个分享终端分别按 Ctrl+C。

本机需保持开机，8766 服务、分享网关和隧道都需保持运行。
[Quick Tunnel](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)
无需 Cloudflare 账户，提供随机临时域名，隧道重新创建时网址可能改变，不承诺生产可用性。
需要固定域名时，可让具备域名权限的维护者配置正式 Cloudflare Tunnel，并把固定 HTTPS
origin 传给同一网关；无需迁移 GPU 模型或复制网页实现。本实现不修改训练服务器任务。
