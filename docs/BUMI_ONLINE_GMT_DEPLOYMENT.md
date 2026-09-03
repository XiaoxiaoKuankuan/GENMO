# BUMI GENMO 常驻滚动生成到 GMT

本方案是在线双进程部署，不会先生成整首动作：

```text
常驻 BUMI GENMO TensorRT
  -> 120 帧独立窗口 / 30 帧重叠 / 90 帧步长
  -> overlap-add + quaternion SLERP
  -> root XY 单次状态化积分 + 因果足锁
  -> bumi_online_qpos_stream_v1 (qpos28@30 Hz)
  -> 安全桥 + 增量 30->50 Hz
  -> trajectory_v1 (110x55) -> Redis -> GMT
```

旧 `demo_music_robot_console.py`、`demo_music_robot_bridge.py`、`robot_stream_v1`、
SMP1/GMR 链路以及整首离线基准 `demo_bumi_onnx_gmt.py` 均保持不变。新链路默认使用
`tcp://127.0.0.1:7022`，避免与旧桥默认的 7021 冲突。

## 当前 s100000 资产

```bash
export BUMI_CKPT=/home/weili/GENMO/inputs/checkpoints/bumi_4set_robot_retargeter_pass_v2_qpos30_contact_v3_continue/s100000_20260903/s100000_weights.ckpt
export BUMI_KIN=/home/weili/GENMO/inputs/checkpoints/bumi_4set_robot_retargeter_pass_v2_qpos30_contact_v3_continue/s100000_20260903/bumi_kinematics.json
export BUMI_STATS=/home/weili/GENMO/inputs/checkpoints/bumi_4set_robot_retargeter_pass_v2_qpos30_contact_v3_continue/s100000_20260903/bumi_qpos30_stats_train_4set_pass_v2.json
export BUMI_ONNX=/home/weili/GENMO/outputs/onnx/bumi_music/rr_pass_v2_qpos30_v3_s100000_20260903/bumi_music_denoiser_t120.onnx
export BUMI_GMT_POLICY=/home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_listao/src/legged_rl/rl_controller/rl_controllers/policy/bumi/policy_mix80old20new.onnx
```

`s100000` 目前只作为接口和 21 关节映射样本，不是实机安全候选。仓库现有
`outputs/tensorrt/bumi/s430000/...` 引擎绑定的是另一 checkpoint，不能拿来启动上述
`s100000`。正式 TensorRT 联调必须先在部署 GPU 上构建与 s100000 ONNX/checkpoint 匹配
的新引擎：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/export/build_bumi_music_tensorrt.py \
  --onnx "$BUMI_ONNX" \
  --checkpoint "$BUMI_CKPT" \
  --output-dir outputs/tensorrt/bumi/s100000_20260903 \
  --device cuda:0 \
  --precision fp16
```

记录命令输出的实际 `.engine` 绝对路径为 `BUMI_ENGINE`，并先运行既有 TensorRT parity
检查。控制台启动时仍会重新校验 engine.json、checkpoint SHA、ONNX SHA、TensorRT ABI、
GPU 指纹以及固定 qpos/contact 双输出，不能绕过。

## 启动顺序

先确认 GMT 的 `simulation.sh` 实际加载的就是下方显式传入的 policy，再启动 GMT。随后
启动新安全桥：

```bash
cd /home/weili/GENMO
.venv/bin/python -u scripts/demo/demo_bumi_gmt_bridge.py \
  --bind tcp://127.0.0.1:7022 \
  --kinematics "$BUMI_KIN" \
  --gmt-policy "$BUMI_GMT_POLICY" \
  --redis-host 127.0.0.1 \
  --redis-port 6379 \
  --redis-key gmt_online_frame_bumi \
  --audio-playback ffplay \
  --verbose
```

正式 TensorRT 常驻控制台：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/demo/demo_music_bumi_console.py \
  --backend tensorrt \
  --checkpoint "$BUMI_CKPT" \
  --onnx "$BUMI_ONNX" \
  --engine "$BUMI_ENGINE" \
  --kinematics "$BUMI_KIN" \
  --stats "$BUMI_STATS" \
  --bridge tcp://127.0.0.1:7022 \
  --device cuda:0 \
  --ddim-steps 50
```

只做接口诊断时可用 ONNX 后端；它不是正式低时延后端：

```bash
.venv/bin/python -u scripts/demo/demo_music_bumi_console.py \
  --backend onnx \
  --checkpoint "$BUMI_CKPT" \
  --onnx "$BUMI_ONNX" \
  --kinematics "$BUMI_KIN" \
  --stats "$BUMI_STATS" \
  --bridge tcp://127.0.0.1:7022 \
  --device cpu \
  --onnx-provider cpu \
  --ddim-steps 2
```

交互命令：

```text
play "/absolute/song.wav" full --start 0 --seed 42
status
stand
quit
shutdown
```

`status` 同时给出当前 revision、窗口耗时、生成窗口数、已提交帧、暂存重叠帧、未来
缓冲秒数、后续窗口 P95、全部模型/资产/后处理指纹、GMT ACK 和 50 Hz 发布状态。默认
两个有效块预生成后才允许 ACK 启动播放；播放中使用 12 秒高水位和 4 秒低水位控制生成。
后续窗口 P95 达到 3 秒会触发性能门并请求返回站姿。控制台使用严格 REQ/REP，每次只允许
一个 qpos 块在途；安全检查和增量计划完成并回复后才会生成下一块，因此桥端处理本身形成
有界同步背压，不存在无界块队列。每个通过完整安全检查并成功纳入计划的 qpos 块也会刷新
桥端存活时间，避免独立心跳线程短时调度延迟造成假超时；控制台只允许同一
`request_id/revision` 的心跳响应清理活动请求。桥端 `status.last_stand_reason` 独立保留
最近一次安全返回的首要原因，迟到块产生的 `last_error` 不会再覆盖它。

## 后处理边界

没有新增低通或 Savitzky-Golay 滤波。运行时平滑只包括：30 帧 overlap-add、根四元数
最短弧 SLERP、接触迟滞与足底锚点驱动的因果 root XY 修正、30→50 Hz 插值，以及启停
smoothstep。训练阶段的速度/加速度/jerk 等导数损失属于模型训练约束，不是部署时额外
滤波。

本阶段验收仅限纯 CPU 合约回归、ONNX/资产身份和关节映射静态检查；没有启动 Gazebo，
更没有执行实机。首次进入仿真或机器人前仍需单独完成动力学、控制稳定性和硬件安全验收。
