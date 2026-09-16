# BUMI music-only 独立部署

本部署使用 BUMI v5 scratch s350000，直接从音乐生成 BUMI qpos，再通过独立 Bridge
交给 GMT。原 GENMO 仓库负责 checkpoint→ONNX、TensorRT 构建和 PyTorch 对照；
`deploy/bumi-music-only-gmt` 分支只保留运行代码，不携带训练 checkpoint 或模型构造器。

## 目录与资产

部署工作目录为 `/home/weili/GENMO-deploy-bumi`；搬到其他电脑可以改成任意目录。
代码由 Git 分支交付，大文件放在忽略跟踪的 `models/bumi_v5_s350000/`，必须一起复制：

```text
models/bumi_v5_s350000/
  deployment.json
  onnx/      # 自包含 ONNX 与同名 .onnx.json
  engine/    # bumi_music_denoiser.engine 与 engine.json
  assets/    # kinematics 与 stats JSON
  gmt/       # model_135000_stage2.onnx，用于 Bridge 读取策略契约
```

清单中的路径均相对于 `deployment.json`，启动不依赖导出元数据里记录的历史绝对路径。
checkpoint 的 SHA256 只作为来源记录，实际模型与资产逐文件验证完整 SHA256；这不等于
在目标机重新验证一个并不存在的 checkpoint。ONNX/engine/stats/kinematics 必须来自同套
发布。损坏文件、错误来源、越界路径、资产参数与清单混用都会拒绝启动。

## 运行环境

已选基线：Ubuntu 22.04 x86_64、Python 3.10、RTX 4090、NVIDIA 580.159.03，
PyTorch 2.6.0+cu124、TensorRT 10.13.3.9。Python 的完整运行依赖闭包在
`requirements/deployment/runtime.lock`；TensorRT Python 包和系统库单列，避免把原
训练环境里的 Lightning、Hydra、T5、SMPL、GMR、渲染和训练数据加载依赖带入部署。

在新电脑创建环境，不复制原机器的 `.venv`：

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements/deployment/runtime.lock
# 先按 system-packages.txt 配好同版 libnvinfer/libnvinfer-plugin，再安装对应 binding。
.venv/bin/python -m pip install --no-deps -r requirements/deployment/tensorrt-bindings.lock
```

宿主机另需 Redis Server、FFmpeg/ffplay。GMT 继续运行在原 `noetic` 容器里。
当前容器为 host 网络，GENMO 与 GMT 共用宿主机 `127.0.0.1:6379`。GMT CUDA 模式使用
它自己的隔离 ORT 1.19.0/CUDA 12.4/cuDNN 9 运行库；它与 GENMO TensorRT 是两个进程。

engine 与 GPU 类型、TensorRT/libnvinfer、精度及模型指纹绑定。换 GPU 或不兼容版本时，
由原 GENMO 仓库在目标构建环境重新生成并验证，再重新发布资产包；部署分支不构建模型。
不要删除 `engine.json` 或绕过指纹检查。原仓库的构建器支持 `--fp32-sensitive`：保留
Transformer MLP 的 FP16 加速，关键浮点计算强制 FP32，并单独记录精度策略/缓存指纹。
实际交付精度及验证结论以 `engine.json` 和原仓库验收报告为准，不能仅凭 `.engine` 扩展名判断。

## 启动前检查

```bash
.venv/bin/python -B scripts/demo/check_bumi_deployment.py \
  --deployment-manifest models/bumi_v5_s350000/deployment.json --inference
```

检查文件指纹、配套 stats/kinematics、GMT 关节名称和一次真实 engine 推理。该命令不连接
Redis，不启动 Bridge 或机器人。单步成功只代表安装与模型接口可用，完整采样及通信报告
在原仓库 `outputs/deployment/bumi_music_v5_s350000/`。

## 三个终端

终端一进入当前真正使用的 GMT 工作区：

```bash
docker exec -it noetic bash
cd /host/Documents/bumi_GMT_deployment_obs
```

按场景选择一个入口，仿真使用：

```bash
./simulation.sh
```

实机沿用已有入口及现场操作流程：

```bash
./real.sh robot_model:=bumi_4340 gmt_onnx_provider:=cuda gmt_cuda_device_id:=0
```

这两条命令不是依次执行的步骤。仿真参数为 2000/40=50 Hz，实机为 500/10=50 Hz。
GMT 模式的启用沿用现有控制流程。部署包中的 GMT policy 必须与控制器加载的
`model_135000_stage2.onnx` 相同，不能只检查文件名。

终端二进入部署目录，启动独立 Bridge：

```bash
.venv/bin/python -u scripts/demo/demo_bumi_gmt_bridge.py \
  --kinematics models/bumi_v5_s350000/assets/bumi_kinematics_robot_retargeter_fe934_v1.json \
  --gmt-policy models/bumi_v5_s350000/gmt/model_135000_stage2.onnx \
  --verbose
```

终端三进入同一个部署目录，启动常驻 GENMO：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/demo/demo_music_bumi_console.py \
  --backend tensorrt \
  --deployment-manifest models/bumi_v5_s350000/deployment.json
```

交互示例：

```text
play "/path/音乐.wav" full --seed 42
play "/path/另一首.mp3" 10 --start 5 --seed 42
status
stand
quit
```

默认 DDIM20、CFG2.5、seed42，GENMO 30 Hz，120 帧窗口/30 帧重叠/90 帧新增，
Bridge 50 Hz。换歌通过新 revision 取消旧任务；`stand` 平滑回站姿；音频由 Bridge
在获得对应 GMT ACK 后启动。`quit` 退出 Console；`shutdown` 还请求关闭 Bridge。

原 GENMO 仓库的 `--checkpoint --onnx --engine --kinematics --stats` 方式仍兼容，
不能与 `--deployment-manifest` 混用。资产清单不修改 DDIM/CFG 的既有 CLI 调参方式。

## 通信和错误定位

```text
Console → ZeroMQ REQ/REP tcp://127.0.0.1:7022 → Bridge
Bridge  → Redis DB0 / gmt_online_frame_bumi → GMT MotionLoaderRedis
GMT     → Redis / gmt_online_frame_bumi_ack → Bridge
```

GENMO 发出 30 Hz qpos28，Bridge 完成 30→50 Hz 插值与关节重排，再发送 110×55
的 `trajectory_v1`。GMT 形成 21×52=1092 的 command window。Redis 使用 SET/GET，
轨迹 TTL250 ms；ACK TTL1000 ms。ACK 是接收确认，不是实际机器人跟踪成功的反馈。
只与本功能相关的 GMT 修改见 [GMT 接入说明](BUMI_GMT_GENMO_INTERFACE.md)。

- 找不到模型或哈希不匹配：重新复制完整模型目录，先运行独立检查器。
- GPU/TensorRT 不匹配：在匹配环境重新构建发布，保持版本/指纹校验。
- Bridge 未连接：检查 Console 与 Bridge 的 `--bridge`/`--bind` 是否一致。
- 等待 ACK：检查 Redis、GMT online 模式、轨迹/ACK key 和控制器是否正在消费轨迹。
- 生成失败或缓冲不足：检查 `status` 的 `last_error`、窗口耗时及 Bridge 输出。

## 原仓库的发布与验收入口

导出/构建复用 `tools/export/export_bumi_music_onnx.py` 和
`tools/export/build_bumi_music_tensorrt.py`。数值对照使用
`tools/eval/validate_bumi_music_tensorrt.py`，不得通过放宽阈值掩盖精度问题。
通过后调用 `tools/export/package_bumi_deployment.py` 发布，必填参数为
`--checkpoint --onnx --engine --kinematics --stats --gmt-policy --output-dir`。
输出目录必须不存在，发布前完整校验并原子移动，不覆盖既有结果。

`tools/eval/validate_bumi_gmt_deployment.py` 在随机非生产端口启动临时 Redis/Bridge，
编译指定 `obs` 工作区的真实 C++ 接收器，对接部署目录的 Console。它只验证模型生成、
协议、窗口、ACK、超时及实时性，不启动 ROS/Gazebo/实机。所有临时产物自动清理。
上述发布/验收工具保留在原 GENMO 仓库，不进入最小部署分支。
