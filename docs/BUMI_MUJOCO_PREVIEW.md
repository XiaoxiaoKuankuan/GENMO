# BUMI MuJoCo 动作预览

常驻 GENMO 可以自动打开 MuJoCo 窗口，查看模型生成的机器人参考动作。窗口只写入
`qpos` 并执行正向运动学 `mj_forward`，没有 `mj_step`、重力积分、PD 控制或控制策略。
因此画面用于检查动作和对比跟踪效果；机器人是否能跟随，应看 GMT 仿真或实物。

## 1. 安装与更新

在部署目录运行一次：

```bash
bash install.sh
```

安装器使用项目自身 `.venv`，增加固定的 MuJoCo 3.2.3 和相关依赖。无需手工激活
环境。`bash run.sh check` 会核验模型文件、TensorRT、MuJoCo 资源和关节排列，不打开
窗口、不连接 GMT、不发送动作。已有 s350000 ONNX/engine 可直接继续使用，无需重新导出。

运行图形窗口需要本机桌面会话及可用的 NVIDIA/OpenGL 驱动，当前基线为 Ubuntu 22.04
x86_64 / RTX 4090。没有 `DISPLAY` 的纯 SSH 终端不能直接弹出本机桌面窗口。
独立预览会明确报错；GMT 模式会提示预览不可用，并保留原有生成与控制链路。

## 2. 在配置文件里选择模式

用编辑器修改项目根目录 `deployment.ini`，重启 GENMO 后生效。

```ini
[runtime]
mode = gmt

[preview]
enabled = true
robot_manifest = assets/bumi_viewer/manifest.json
```

| 配置 | 行为 |
|---|---|
| `mode = gmt`，`enabled = true` | 连接已有 Bridge，同时显示其当前发出的参考姿态；默认配置 |
| `mode = gmt`，`enabled = false` | 继续使用原有 GMT 链路，不打开窗口 |
| `mode = preview`，`enabled = true` | 独立生成与播放，不创建 Bridge/Redis/ROS 连接 |

`preview` 模式要求开启窗口，不允许 `enabled = false`。旧配置没有这两个配置段时，
默认仍为 GMT 模式、不打开窗口；原来的长命令行不加 `--preview` 也保持原行为。

### GMT 联动

终端一沿用已有 GMT 启动方式。终端二：

```bash
cd /home/weili/GENMO-deploy-bumi
bash run.sh bridge
```

终端三：

```bash
cd /home/weili/GENMO-deploy-bumi
bash run.sh genmo
```

窗口显示 Bridge **已经发布的当前参考姿态**，包括站姿、缓入、舞蹈和返回站姿。
音乐仍由 Bridge 播放，查看器不会播放第二份音频。参考姿态不是 GMT 策略输出或实物反馈；
控制器的响应迟延、跟踪误差不会被写回本窗口。

当 ACK 中断导致 Bridge 返回站姿时，窗口跟随这一实际参考变化。查看器不会绕过 ACK。

### 独立预览

把配置改为 `mode = preview`，仅运行：

```bash
cd /home/weili/GENMO-deploy-bumi
bash run.sh genmo
```

不启动 GMT 或 Bridge，不要求 Redis、ROS、Docker 和 GMT policy。配置文件里保留的
GMT/Redis 字段不用于建立连接。此模式运行 `bash run.sh bridge` 或 `check-gmt` 会提示
无需启动/检查控制器，避免误以为两个模式需要同时运行。

## 3. 输入音乐与控制播放

在 `bumi>` 提示符后输入，例如：

```text
play "/你的音乐目录/song.wav" 10
play "/你的音乐目录/另一首.wav" 20 --start 5 --seed 42
status
stand
quit
```

单独输入带引号的音乐路径默认生成整首。再次 `play` 会取消旧任务、平滑返回站姿后
启动新任务；`stand` 停止音乐并返回站姿。`quit` 退出控制台并清理它创建的窗口及本地
播放器；GMT 模式下保留原有 Bridge 的站姿待机行为。

窗口可用鼠标旋转、缩放，观察中心随机器人根位置移动。关闭窗口只停止显示，不停止
GENMO/GMT 或独立播放器；需要停止动作时在控制台输入 `stand`。关闭后要再次打开窗口，
重新启动控制台。本版不增加暂停、拖动进度或录像控件。

独立模式默认预生成两个有效块后播放，短音乐的最后一块可提前满足条件。继承
120/30/90 滑窗、DDIM20、CFG2.5、seed42、足锁和 30→50 Hz 插值，采用本地单调时钟。
起始姿态来自模型 kinematics，缓入 0.8 秒、返回 1 秒。音频在动作主体开始时触发，
仍采用现有 ffplay；这不是对音频设备输出延迟的测量或补偿。

待机根高度在运行时按实际关节姿态做 FK 后计算，使最低脚底代理点离地 2mm；当前
BUMI 可见脚底对应约 1mm 余量。独立预览默认根高度约 0.48121m，当前 GMT 默认
屈膝姿态约 0.47546m；初始化与返回站姿使用同一高度。原始 kinematics JSON 的
default_qpos 和指纹保留不变。生成的舞蹈保持自身高度，跳跃等动作不会逐帧被拉到地面。

## 4. 数据接口、进程和状态

```text
GMT 模式：
GENMO → qpos30 → Bridge → 50 Hz 轨迹包 → Redis → GMT
                    └─ preview_frame 只读快照 → 查看器

独立模式：
GENMO → qpos30 → 本地播放器（插值、过渡、时钟、音频）→ 查看器
```

查看器是单独的 Python 进程，使用匿名管道传输最新姿态，不增加网络端口。GMT 模式
每秒最多 50 次轮询既有 7022 ZeroMQ 接口，使用独立客户端，不占用 Console 的生成/
心跳锁。Bridge 控制接口新增：

```json
{"command": "preview_frame"}
```

首次成功发布前返回 `available: false`。随后返回原生顺序 `qpos[28]`、`frame_index`、
`request_id`、`revision`、`state`、预览 `sequence`、`published_monotonic`、
`age_seconds`、`kinematics_sha256`，以及用于对照发送包的 `stream_id/packet_sequence`。
只读请求不推进游标、不刷新生成心跳、不操作 ACK。原有协议和请求继续兼容。

画面超过 0.2 秒没有新参考时冻结并提示；连接恢复后跟随最新参考，不补播旧帧。
管道满时丢弃显示帧，窗口不会对推理或发布线程施加反压。窗口异常退出只报告显示错误。
`status` 新增 `runtime_mode`、`preview.alive/sent_frames/dropped_frames/last_error`；
独立播放器标记 `playback_clock=local_monotonic`、`gmt_acked=null`，表示 ACK 不适用。

## 5. 资源搬迁与验证边界

`assets/bumi_viewer/` 包含完整 XML、22 个引用的 mesh 和指纹清单，已纳入 Git。
XML 原文 SHA256 必须等于 kinematics 的 `source_mjcf_sha256`；每个 mesh 也逐一校验。
禁止只拿同名 BUMI XML 替换。模型、资源和配置都按相对路径定位，不依赖
`/home/weili/robot_retargeter` 或其他电脑上的同名目录。

新电脑仍需携带 `models/bumi_v5_s350000/`；Git 只包含代码和渲染资源，不包含 ONNX/engine。
部署压缩包包含两者；解压后执行一次 `bash install.sh`，不要复制旧机器的 `.venv`。

单元测试覆盖模式、无网络本地播放、任务版本、缓冲不足、错误姿态、快照与慢渲染。
真实 GPU/桌面验收入口为 `tests/bumi/validate_bumi_preview_runtime.py`，需要显式指定
模型清单、两首音乐和报告路径；可选 GMT policy 仅供隔离测试接收端使用，不属于用户
启动命令。它只创建临时 Redis 端口/专用 key，不启动 GMT、仿真或实机。

本机 X11/NVIDIA 环境在退出窗口时仍可能输出 `NV-GLX missing` 提示；已加入渲染线程
等待及 GLFW 显式回收，并要求真实验收中的查看器以退出码 0 结束。该提示与动作生成、
窗口播放或 GMT ACK 无关；其他电脑的桌面驱动仍需按实际环境验证。
