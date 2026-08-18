# GENMO TensorRT 滑窗实物部署

这套入口与现有完整序列 PyTorch demo 并存，不替换旧接口。固定数据流为：

```text
EDGE35 -> TensorRT 20-step DDIM -> 120/30 hard inpainting
       -> streaming SMPL -> 同步 GMR -> BUMI qpos 30->50 Hz
       -> trajectory_v1[过去10 + 当前1 + 未来99] -> GMT
```

部署由两个 GENMO 常驻进程组成：

- `demo_music_robot_bridge.py`：独立安全桥，拥有 GMR、GMT、音频、站姿和 50 Hz
  发布时钟。即使控制台退出或 TensorRT 不可用，它仍持续发送 GENMO 合成站姿经
  GMR 得到的 BUMI 站姿。
- `demo_music_robot_console.py`：前台交互终端，TensorRT engine、151D 解码器和
  EDGE35 缓存只加载一次。

## 1. 一次性准备

### GMR 同步服务

部署要求 `/home/weili/GMR-CPP_e1jump_lowdpi` 包含本方案新增的同步 GMR 服务与
实时 qpos viewer。建议使用独立 build 目录，不改仓库已有生成文件：

```bash
cmake -S /home/weili/GMR-CPP_e1jump_lowdpi \
  -B /home/weili/GMR-CPP_e1jump_lowdpi/build-genmo-stream
cmake --build /home/weili/GMR-CPP_e1jump_lowdpi/build-genmo-stream \
  --target smplx_bumi3_batch_server bumi3_qpos_viewer -j2
```

协议为同步二进制 `GMRQ/GMRA v1`。每个 412-byte SMP1 `FRAME` 必须收到相同
sequence 的 `qpos[28] + CRC32`；丢帧、重复、乱序或子进程退出都会取消当前动作。

### TensorRT Python 版本预检

实物模式没有 PyTorch fallback。Python binding 必须和本机实际加载的
`libnvinfer` 主、次版本一致：

```bash
cd /home/weili/GENMO
source .venv/bin/activate

python - <<'PY'
import tensorrt as trt
from gem.runtime.music_only_trt import validate_tensorrt_installation
print("binding:", trt.__version__)
print("libnvinfer:", validate_tensorrt_installation(trt))
PY
```

本机系统包为 `libnvinfer 10.13.3.9 + CUDA 13`。若 `import tensorrt` 失败，在
GENMO 自己的 venv 安装精确匹配版本；不要复用其他项目的不同版本环境：

```bash
cd /home/weili/GENMO
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install "packaging==24.2"
python -m pip install --no-cache-dir "tensorrt-cu13==10.13.3.9"
```

不要把 `packaging` 升级到 25 以上；当前 Lightning 2.3.0 要求
`packaging<25.0`。

如果完整安装在构建已弃用的 `nvidia-cuda-runtime-cu13` 时失败，并且系统已经通过
Debian 包安装了完全匹配的 `libnvinfer.so.10.13.3`，使用系统 runtime，只安装
Python frontend/bindings：

```bash
python -m pip install --no-deps "tensorrt-cu13-bindings==10.13.3.9"
python -m pip install --no-deps "tensorrt-cu13==10.13.3.9"
```

这种安装会让 `pip check` 报告未安装 `tensorrt-cu13-libs`，因为 runtime 实际由系统
包提供；这不是运行错误。必须通过上面的 binding/libnvinfer/Builder 预检确认版本
一致且 Builder 能创建，不能在没有系统 `libnvinfer` 的机器上照搬该绕过方式。

### 导出固定 ONNX 和构建 engine

```bash
cd /home/weili/GENMO
source .venv/bin/activate

CKPT=/home/weili/GENMO/outputs/gem_smpl_music_only_4set_physics_v1/version_0/checkpoints/s050000.ckpt
TRT_ROOT=/home/weili/GENMO/outputs/tensorrt/music_only_physics_s050000
mkdir -p "$TRT_ROOT"

python -u tools/export/export_music_only_onnx.py \
  --trt-deployment \
  --ckpt "$CKPT" \
  --exp gem_smpl_music_only_4set_physics_v1 \
  --seq-len 120 \
  --device cuda:0 \
  --output "$TRT_ROOT/music_only_denoiser.onnx" \
  --overwrite

python -u tools/export/build_music_only_tensorrt.py \
  --onnx "$TRT_ROOT/music_only_denoiser.onnx" \
  --checkpoint "$CKPT" \
  --output-dir "$TRT_ROOT/engines" \
  --device cuda:0 \
  --precision fp16
```

构建脚本输出最终 engine 绝对路径。缓存目录名包含 ONNX/checkpoint SHA256、
TensorRT/CUDA、GPU 型号与 compute capability、精度和固定 shape。engine 不提交
Git；运行时会再次校验同目录的 `engine.json` 和 engine SHA256。

### FP16 验证门槛

从四个训练音乐集之一选择至少 4 秒音乐：

```bash
ENGINE=/absolute/path/from/build/music_only_denoiser.engine
python -u tools/eval/validate_music_only_tensorrt.py \
  --audio /absolute/path/to/test.wav \
  --checkpoint "$CKPT" \
  --onnx "$TRT_ROOT/music_only_denoiser.onnx" \
  --engine "$ENGINE" \
  --ddim-steps 20 \
  --output "$TRT_ROOT/fp16_validation.json"
```

单步 PyTorch/ONNX/TensorRT 必须满足 `atol=rtol=0.03`；20-step 的 151D、Root
轨迹和 FK 必须有限，TensorRT 的 Root/FK 速度、加速度、jerk RMS 相对 PyTorch
变化不得超过 5%。失败时用 `--precision fp32` 重新构建并再次验证，不能忽略报告
继续进入实物模式。

## 2. 三个 tmux 终端

启动前清除旧急停文件，并确认没有 legacy publisher 同时写
`gmt_online_frame_bumi`：

```bash
rm -f /tmp/genmo_estop
```

### 终端 A：GMT

```bash
tmux new -s gmt
cd /home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_listao
./simulation.sh
```

GMT 必须支持现有 `trajectory_v1` ACK 合约。先在仿真中完成全部验收，再切实物。

### 终端 B：安全桥

```bash
tmux new -s genmo_bridge
cd /home/weili/GENMO
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python -u scripts/demo/demo_music_robot_bridge.py \
  --bind tcp://127.0.0.1:7021 \
  --gmr-root /home/weili/GMR-CPP_e1jump_lowdpi \
  --gmr-binary /home/weili/GMR-CPP_e1jump_lowdpi/build-genmo-stream/smplx_bumi3_batch_server \
  --ik-config /home/weili/GMR-CPP_e1jump_lowdpi/config/ik_configs/smplx_to_bumi3_auto.json \
  --robot-xml /home/weili/GMR-CPP_e1jump_lowdpi/assets/bumi3/mjcf/bumi3.xml \
  --gmr-vis \
  --gmr-viewer-binary /home/weili/GMR-CPP_e1jump_lowdpi/build-genmo-stream/bumi3_qpos_viewer \
  --gmr-viewer-width 640 \
  --gmr-viewer-height 480 \
  --gmt-policy /home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_listao/src/legged_rl/rl_controller/rl_controllers/policy/bumi/0724_lab_148500.onnx \
  --redis-key gmt_online_frame_bumi \
  --audio-playback ffplay
```

`--gmr-vis` 是可选开关。窗口显示安全桥在同一个 50 Hz tick 交给 GMT 的当前原生
BUMI `qpos[28]`，而不是 GMR 提前求解的未来窗口，因此可以与 Gazebo 中 GMT 的实际
跟踪结果按时间直接比较。关闭 MuJoCo 窗口不会终止安全桥或 GMT；状态中的
`gmr_viewer_alive` 会变为 `false`。

启动成功后应看到 GMR reset、控制 endpoint 和持续站姿发布。安全桥独占 GMT Redis
key；不要再启动旧 GMR Redis publisher。

### 终端 C：常驻交互控制台

```bash
tmux new -s genmo_console
cd /home/weili/GENMO
source .venv/bin/activate

CKPT=/home/weili/GENMO/outputs/gem_smpl_music_only_4set_physics_v1/version_0/checkpoints/s050000.ckpt
ENGINE=/absolute/path/from/build/music_only_denoiser.engine

CUDA_VISIBLE_DEVICES=0 python -u scripts/demo/demo_music_robot_console.py \
  --engine "$ENGINE" \
  --checkpoint "$CKPT" \
  --bridge tcp://127.0.0.1:7021 \
  --ddim-steps 20 \
  --guidance-scale 2.5
```

控制台命令：

```text
robot> "/abs/path/song.wav"
robot> play "/abs/path/song.wav" 60
robot> "/abs/path/song.wav" 60
robot> play "/abs/path/song.wav" full
robot> play "/abs/path/song.wav" 20 --start 10 --seed 42
robot> stand
robot> status
robot> quit
robot> shutdown
```

只输入音频路径时默认从头滚动生成到音乐结束。`20` 秒严格生成 600 帧；指定时长
超过可用音乐会报错，不会静默缩短。
`quit/exit` 先回站姿，只退出控制台；安全桥继续 50 Hz 站姿。`shutdown` 先回站姿，
再关闭安全桥。`Ctrl-C` 等价于 `stand` 后退出控制台。

## 3. 滑窗与安全语义

- 第一窗提交 120 帧；后续窗用上一窗最后 30 帧作为每个 DDIM timestep 的
  `q_sample(known_x0)`，最终 bitwise 覆盖，然后只提交新增 90 帧。
- 每窗噪声由 `(request_seed, chunk_index)` 确定。取消后旧 revision 的迟到 chunk
  会被丢弃。
- Root 第一帧严格为零，只用上一帧朝向和局部速度积分新增帧；overlap 不重复积分，
  也不重复进入有状态 GMR。
- 安全桥队列最多两个 chunk。未来缓冲达到 12 秒时控制台暂停继续生成，降到 4 秒
  后立即连续生成下一批窗口。低于 2.2 秒会停止音频并平滑回站姿。
- 安全桥只在具有真实未来 99 帧时推进舞蹈 cursor，不复制舞蹈末帧填未来。动作
  完整结束后才附加 1 秒回站和稳定站姿。
- 每次新 plan 先等待 GMT 对 stream/revision/plan 的 ACK。站姿到首帧过渡 0.8 秒，
  动作首帧到达时才启动音频。
- 控制台 heartbeat 为 0.5 秒；1.5 秒失联、GMR 异常、CRC/sequence/帧号错误、
  GMT ACK 中断都会取消旧动作并回合成站姿。

急停优先级最高：

```bash
touch /tmp/genmo_estop
```

删除文件只解除急停锁存，不恢复旧动作；必须重新发送 `play`。

## 4. 验收顺序

先运行单测：

```bash
cd /home/weili/GENMO
source .venv/bin/activate
pytest -q \
  tests/test_music_only_onnx.py \
  tests/test_music_only_trt_streaming.py \
  tests/test_robot_stream.py \
  tests/test_gmt_trajectory.py \
  tests/test_stream_smpl_params_to_gmr.py
```

然后按固定顺序验证：仿真持续站姿 → 低幅 10 秒 → 完整音乐 → 播放中 `stand` →
控制台崩溃 → GMR 崩溃 → ACK 中断 → 连续播放 10 分钟。任一安全检查失败，不进入
下一阶段或实物测试。
