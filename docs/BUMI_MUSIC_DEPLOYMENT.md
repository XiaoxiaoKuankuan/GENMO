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

部署分支保留34个Python文件（含测试和包入口），具体闭包见 `DEPLOYMENT_FILES.json`。
核心职责如下，网络构造、训练配置、checkpoint读取和导出工具仍在原仓库：

| 代码 | 部署职责 |
|---|---|
| `scripts/demo/demo_music_bumi_console.py` | 音乐命令、常驻模型、特征缓存、换歌/取消/状态和滑窗提交 |
| `scripts/demo/demo_bumi_gmt_bridge.py` | 独立播放与轨迹发布、ACK同步、站姿过渡、输入检查及故障处理 |
| `gem/utils/music_features.py` | 本地音频解码与EDGE35提取 |
| `gem/runtime/bumi_music_deploy.py`、`music_only_trt.py` | ORT/TRT单步执行、DDIM、120/30/90滑窗及确定性seed派生 |
| `gem/robots/bumi/` | qpos30反归一化、qpos28解码、纯Torch FK、contact和因果足锁 |
| `gem/runtime/bumi_online_stream.py`、`bumi_gmt_plan.py`、`qpos_timeline.py` | 在线身份/帧序、轨迹计划、增量时间轴和30→50 Hz重采样 |
| `gem/runtime/gmt_trajectory.py`及桥接实际引用的辅助模块 | GMT policy关节契约、trajectory_v1/ACK编码和Redis发布 |
| `gem/runtime/bumi_deployment_bundle.py`、`scripts/demo/check_bumi_deployment.py` | 相对路径资产清单、完整哈希、契约及安装检查 |

`gem/diffusion_utils/` 只保留实际DDIM依赖的数学与采样模块；部署分支的
`gem/runtime/__init__.py`不导入文本/人体/视频引擎。原仓库的包级接口不作此裁剪。

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

若使用uv安装第一份锁，需要显式选择 `--index-strategy unsafe-best-match`，因为本清单
同时使用PyPI和PyTorch cu124索引；所有包仍按锁中版本安装。pytest仅用于开发回归，
不属于运行依赖，本次干净环境另装pytest9.1.1完成测试。

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

## 2026-09-16 实际交付与验收

正式资产包约1.50 GB（十进制，不含Python环境），没有训练checkpoint。部署清单保留
原仓库发布提交 `9e1706bd8977ed047d498a97b8e690bc8641abd9` 及 `source_git_dirty=false`。
完整文件SHA均在清单中，其中模型身份为：

| 项目 | SHA256 |
|---|---|
| 源s350000 checkpoint（不交付） | `fdf3bd67910b76b252d77932b485445258262fa24b5286850812fbe33aca51cc` |
| ONNX | `b2d0ed2fba436459f57597dfb3976338426e063953d5a7d0b5770284abdb5481` |
| 最终TensorRT engine | `ae5bc2eaab8c81b4652c064036e8501ed31010df91d0e92b98d24d97cb602a88` |
| GMT policy | `d2e176657d72d1b0efcb04406abd17145fc45a3a5755b62aae7b866e0a6e3d1b` |

- 原仓库重新导出结果与既有s350000 ONNX逐字节一致。
- 普通FP16首次未通过既定数值阈值，最终交付采用 `attention_norm_heads_fp32_v1` 混合精度。
  PyTorch/ONNX/TensorRT单步和完整20步DDIM全部通过原阈值；完整采样TRT对PyTorch的
  qpos最大绝对差0.00343883，FK位置最大绝对差0.000800386米。contact最大绝对差
  0.284624按既有 `atol=0.03, rtol=0.03` 联合容差通过，不是逐项绝对误差都小于0.03。
- 原仓库回归83通过、3项条件跳过；最小部署目录全新环境45通过。真实engine单步输出
  为 `[1,120,30]` 和 `[1,120,2]`，数值有限。
- 将代码和模型迁到含中文/空格的系统临时目录，工作目录设在项目外，模块来源与资产解析
  均通过；环境中Hydra、Lightning、Transformers、SMPL-X、Open3D均不存在。
- 真实AIST mLH0前10秒生成3个窗口并提交300帧；部署目录Console→Bridge→真实GMT C++
  接收器通过。续窗两次耗时0.08788/0.09449秒，P95为0.09416秒；这是本机本段音乐的
  小样本，不是所有音乐的延迟保证。首次EDGE35提取（含冷启动）12.40秒，需与去噪耗时区分。
- 隔离Redis端口59351、ZMQ47127，C++原有5项协议测试全部通过；实际接收1339个新包
  （包括站姿与过渡），1092维窗口有限、centered true/false一致，ACK同步和停止发送后
  0.2秒过期处理通过。Bridge发布频率约50.013 Hz。
- 以上只验收导出、推理和通信；没有启动ROS控制器、仿真或实机，也没有修改GMT工作树。

正式证据在原仓库 `outputs/deployment/bumi_music_v5_s350000/`：
`export_verification.json`、`parity_sensitive.json`、`communication.json`、
`relocation.json`、`unit_tests.txt`、`minimal_unit_tests.txt`。首版失败的数值报告保留供追溯，
不合格engine及全部本轮临时目录/缓存已清理。

便携交付包 `bumi_music_only_gmt_s350000.tar.gz` 放在同一证据目录，包含部署代码、模型和
`validation/`验收结论，不包含`.git`工作树指针、`.venv`、训练checkpoint或训练数据。
另一台电脑解压后按本说明创建环境，再执行安装检查和三个终端启动；Git分支仅含代码，
单独clone分支后仍需另行复制整个 `models/bumi_v5_s350000/`。
