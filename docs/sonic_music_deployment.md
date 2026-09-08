# GENMO SMPL music-only → SONIC → G1 MuJoCo

本入口在本机运行 GENMO、SONIC C++ 与 MuJoCo 三个持续进程。GENMO 读取音频文件，先提取完整 EDGE35 特征，在播放过程中继续按窗口生成 SMPL。SONIC 固定使用当前 release 的 SMPL mode 2 控制 G1 的 29 个身体关节；手指沿用默认姿态。机器人跟踪人体局部姿态与根朝向，允许自然移动，GENMO 世界平移只用于诊断。

## 启动和退出

在 `/home/weili/GENMO` 中执行：

```bash
.venv/bin/python -B scripts/demo/demo_music_sonic.py \
  --audio "/absolute/path/song.wav" --launch-local
```

程序自动完成模型预热、前两窗预缓冲、PD 初始化以及 SMPL 站立准备。预约时刻释放虚拟弹力带，先站立 1 秒，再用 1 秒过渡进入舞蹈；音乐第 0 个采样与会话第 100 帧对齐。正式音乐结束后，用 1 秒收尾并保持站立画面，按 Ctrl+C 关闭。播放中按 Ctrl+C 会在约 0.5 秒之后进入收尾，音频在切换前 250 ms 淡出。故障立即停止音乐并冻结物理推进。

常用选项：

| 选项 | 用途 |
|---|---|
| `--start-sec 10 --duration-sec 30` | 选段后再提取特征，保持原始播放速度 |
| `--seed 42` | 固定滚动窗口的派生随机种子 |
| `--output-dir /absolute/new/directory` | 使用新的输出目录；拒绝覆盖已有目录 |
| `--audio-device pulse` | 选择 PortAudio 输出设备，也可传数字设备编号 |
| `--exit-on-finish` | 自然收尾后自动退出，适合验收 |
| `--headless --audio-output off` | 显式无窗口、无声测试，不属于真实声卡验收 |
| `--audit-observations` | 保存实际 C++ 编码器输入，供独立重组比较 |
| `--smpl-npz .../generated_smpl.npz` | 固定片段回放，隔离生成速度对控制的影响 |
| `--sonic-root ... --sim-python ...` | 覆盖 SONIC 路径和仿真 Python；默认使用 SONIC 的 `.venv_sim/bin/python` |

不使用 `--launch-local` 时，需先按 SONIC 仓库 `docs/sonic_music_input.md` 的命令启动两个服务。本入口只连接 `127.0.0.1`；活动会话拒绝被另一个会话覆盖，退出时只终止自己启动的子进程。音乐模式由协调器管理，不使用摄像头模式的 Enter/T/R 或编码器切换按键。

## 固定资产与环境

| 资产 | 本次基线 |
|---|---|
| Checkpoint | `outputs/gem_smpl_music_only_4set_manual_q1_physics_v3_100k/version_0/checkpoints/s100000.ckpt` |
| Checkpoint SHA256 | `98d70a145fb8f430ab557cdd0bde4af5f71e881b16f6976b385db6d172683136` |
| 专用 ONNX | `outputs/tensorrt/sonic_music_physics_v3_s100000/music_only_denoiser.onnx` |
| TensorRT FP16 engine | `outputs/tensorrt/sonic_music_physics_v3_s100000/engines/f4f522f9352162851b01e948440bc7e5f5899877eeabafc843359dd7dcb21a95/music_only_denoiser.engine` |
| GENMO 默认参数 | DDIM 20、CFG 2.5、seed 42、120 帧窗口、30 帧重叠 |
| EnDecoder | `MM_V1_AMASS_LOCAL_BEDLAM_CAM`、`gvhmr`、151 维、`clip_std=True` |
| SONIC encoder | `gear_sonic_deploy/policy/release/model_encoder.onnx`，1762→64，SHA256 前缀 `013ab0287236` |
| SONIC decoder | `gear_sonic_deploy/policy/release/model_decoder.onnx`，994→29，SHA256 前缀 `c7241a123eaa` |
| 配套观测 | 同目录 `observation_config.yaml`，SHA256 前缀 `466d05947c78`；不要替换成 1751 维配置 |
| MuJoCo 场景 | SONIC `scene_43dof.xml` 包含 `g1_29dof_with_hand.xml`，29 身体关节加 Dex3 手指 |

启动会校验 checkpoint、engine manifest/GPU 指纹及三个 SONIC release 文件散列；每个会话记录源代码、二进制、人体 FK、G1 XML、统计量的实际指纹。模型和生成数据不进入 Git。

GENMO 附加 Python 依赖已在 `setup.cfg` 的 `sonic_music` extra 中声明：

```bash
.venv/bin/python -m pip install 'sounddevice==0.5.3' 'nvidia-ml-py==13.610.43'
```

Linux 还需要 PortAudio 动态库。本机缺少系统 `libportaudio2`，已从 Ubuntu 包 `libportaudio2_19.6.0-1.1_amd64.deb` 提取到 `.venv/lib/libportaudio.so.2.0.0`，并建立 `.so.2` 和 `.so` 链接；动态库 SHA256 为 `a7b79691ead40552b032fb5aa63ad90774d00903aeea2c5547e472329ed47415`。入口发现该库时会设置本进程的库搜索路径并重启一次，不修改系统包。其他机器可安装发行版的 `libportaudio2`。仿真使用已有 `.venv_sim`，并需要 Unitree SDK、CycloneDDS、MuJoCo；本机 `env_isaaclab` 不具备完整仿真依赖。

需要重新构建 GENMO 专用产物时，使用既有导出工具；生成器拥有 DDIM 循环，engine 只导出单次 `pred_motion`：

```bash
.venv/bin/python -B tools/export/export_music_only_onnx.py \
  --ckpt outputs/gem_smpl_music_only_4set_manual_q1_physics_v3_100k/version_0/checkpoints/s100000.ckpt \
  --exp gem_smpl_music_only_4set_manual_q1_physics_v3_100k \
  --seq-len 120 --trt-deployment --device cuda:0 \
  --output outputs/tensorrt/sonic_music_physics_v3_s100000/music_only_denoiser.onnx
```

```bash
.venv/bin/python -B tools/export/build_music_only_tensorrt.py \
  --onnx outputs/tensorrt/sonic_music_physics_v3_s100000/music_only_denoiser.onnx \
  --checkpoint outputs/gem_smpl_music_only_4set_manual_q1_physics_v3_100k/version_0/checkpoints/s100000.ckpt \
  --precision fp16 --output-dir outputs/tensorrt/sonic_music_physics_v3_s100000/engines
```

在生成的 engine 目录上运行 `tools/eval/validate_music_only_tensorrt.py`，传入相同的 `--checkpoint`、`--exp`、`--onnx`、`--engine` 和 `--audio`，保持 DDIM 20、CFG 2.5、seed 42。已有构建及一致性结果随本次验收归档。

## 数据与时间契约

1. 音频统一解码为 48 kHz、双声道 float32 PCM；选段和实际采样数确定音乐时长。EDGE35 在 30 Hz 时间轴提取，动作在播放期间逐窗生成。
2. 第一窗提交 120 帧；后续用上一窗末尾 30 帧硬约束续接，每窗最多新增 90 帧。`StreamingSmplDecoder` 保留跨窗根状态，不重复解码重叠区域。
3. 在整首音乐的全局 50 Hz 网格插值，身体与根姿态用最短弧 SLERP。只保留一个 30 Hz 插值端点，避免逐窗舍入漂移；最终输出 `ceil(时长 × 50)` 帧。随后执行 SONIC FK 和六腕映射，六腕按真实 G1 XML 限位。
4. Prefix 为 100 帧，正常收尾为 50 帧，另加 10 帧站立参考供编码器未来窗口使用。正常播放当前帧到未来第 9 帧均必须存在。
5. 预缓冲约 7 秒音乐，短片段全部生成；低水位 4 秒、高水位 12 秒、未来缓冲容量 15 秒。10 帧历史上下文另计。已提交帧在正常播放中不可改写，显式用户停止允许替换尚未消费的未来段。
6. 声卡静音预启动，在回调之外校准 PortAudio 与 `CLOCK_MONOTONIC`，回调只复制 PCM 并记录每块 DAC 时间和采样编号。SONIC 复用 SDK 的 CLOCK_MONOTONIC 周期 timerfd（20 ms）；MuJoCo 使用 5 ms 固定物理步长与绝对墙钟期限。
7. 监测采样时刻的实际控制帧、声卡进度、控制计算耗时、仿真时间和 GPU 使用。欠载低于 0.5 秒、心跳超过 1.5 秒、音频 underflow、跌倒、仿真偏差超过 100 ms 或音频/参考偏差持续超过 100 ms 会终止会话。故障后通过新会话重新开始。

## 验证与录像

```bash
# 新增单元测试与原滚动生成契约测试
.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_sonic_music.py tests/test_music_only_trt_streaming.py

# 复核实际 C++ 观测、物理步进、倾角、足底滑动和双膝跟踪相位
.venv/bin/python -B tools/eval/evaluate_sonic_music_session.py outputs/sonic_music/SESSION

# 按真实状态时间戳绘制带音乐录像（会话内 dance_with_music.mp4，拒绝覆盖）
.venv/bin/python -B tools/eval/evaluate_sonic_music_session.py outputs/sonic_music/SESSION --render

# 心跳中断、六秒生成延迟和用户停止；使用同一八秒音频与固定 SMPL
.venv/bin/python -B tools/eval/validate_sonic_music_failures.py \
  --audio /path/to/eight_second_source.wav \
  --smpl-npz /path/to/eight_second_session/generated_smpl.npz \
  --output-dir /absolute/new/fault_directory
```

每次会话包含 `manifest.json`、`audio.wav`、`generated_smpl.npz`、`reference_50hz.npz`、`windows.json`、`timeline.jsonl`、`sim_state.jsonl`、两个子进程日志和 `report.json`。用户淡出时另存 `played_audio.wav`。`evaluation.json` 为复算结果；录像来自真实 qpos/qvel 的时间戳回放，不是把生成参考直接贴到机器人上，也不替代现场主观视听复核。

2026-09-08 首轮实测已经完成：固定片段 8 秒、在线 Compas3D 30 秒、在线 FineDance 原曲选段 180 秒以及三个故障用例。180 秒会话为 `outputs/sonic_music/20260908_190755_36ae39a4`：58 次播放中生产耗时平均 74.2 ms、P95 78.8 ms；音频/参考误差 P95 8.68 ms、最大 9.23 ms；完整控制周期 P99 1.76 ms，实际控制 50.00 Hz，仿真 200.000 Hz，RTF 1.000000。无弹力带、无跌倒和自动重置，最低基座 0.363 m、最大倾角 25.58°。

固定片段 436 次 C++ 编码观测与 Python 重组最大误差 `8.47e-16`。180 秒双膝屈曲的相位估计为滞后约 20 ms，相关系数 0.958；这只是双膝指标，不能解释成全身固定延迟。足底接触点切向速度 RMS 约 0.124 m/s，仍有滑动，舞蹈质量结论以本次音乐、seed 和录像为限，不是零脚滑或真机验证。

## 最终代码回归（2026-09-08）

最终会话为 `outputs/sonic_music/20260908_192304_56097e25`，源曲 `outputs/server_music_wav_4set_10_20260818/finedance/100.wav` 的前 180 秒，seed 42。5400 帧 30 Hz SMPL、9000 帧音乐参考，加前后缀共 9160 帧，编号完全连续；17 个代码/模型/资产指纹与当前文件一致。

| 指标 | 最终实测 |
|---|---|
| 播放中生产 | 58 窗，平均 75.02 ms，P95 77.62 ms |
| 音频/实际参考时间误差 | P95 10.53 ms，最大 17.39 ms |
| 完整控制周期计算 | P99 1.304 ms，墙钟 50.00006 Hz |
| 物理步进和实时因子 | 200.00055 Hz，RTF 1.000003 |
| GPU 总已用显存峰值 | 3.041 GiB，含显示及同时运行的模型 |
| 控制状态 | 无弹力带、无跌倒、无 reset；最低基座 0.368 m，最大倾角 23.66° |
| 双膝跟踪相位 | 约 40 ms 滞后，相关 0.959，只代表双膝信号 |
| 足底接触点切向速度 | RMS 0.108 m/s，P95 0.117 m/s，存在脚滑 |

同目录 `dance_with_music.mp4` 是完整状态回放录像；`report.json`、`evaluation.json` 和 `manifest.json` 与该录像使用同一会话。`outputs/sonic_music/validation_20260908` 保存模型一致性、26 项单元测试、短尾窗、指纹复核及现场窗口截图。三进程故障注入结果另见 `outputs/sonic_music/fault_validation_20260908/results.json`。主观舞蹈与音乐观感仍需观看录像复核；本次没有开展真机或训练验证。
