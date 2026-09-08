# GENMO SMPL music-only → SONIC → G1 MuJoCo

本入口在本机运行 GENMO、SONIC C++ 与 MuJoCo 三个持续进程。常驻模式在没有音乐时持续发送双臂略微打开的站姿，收到音频文件后提取完整 EDGE35 特征，在播放过程中继续按窗口生成 SMPL；结束或主动停止后回到站姿，继续接收下一首。SONIC 固定使用当前 release 的 SMPL mode 2 控制 G1 的 29 个身体关节；手指沿用默认姿态。机器人跟踪人体局部姿态与根朝向，允许自然移动，GENMO 世界平移只用于诊断。

## 启动和退出

在 `/home/weili/GENMO` 中执行：

```bash
.venv/bin/python -B scripts/demo/demo_music_sonic.py --resident --launch-local
```

也可以省略 `--resident` 和 `--audio`，无音频启动自动进入常驻模式。默认手臂外展 15°，可用 `--arm-open-degrees 10` 调整到 0～45°。先保持 MuJoCo 吊绳开启并完成原有 PD 初始化，在 **MuJoCo 窗口按 `]` 进入 SONIC 控制，再按 `9` 松开吊绳落地**。没有音乐、提取特征和预缓冲时都会继续跟踪站姿；程序不自动起控或松绳。

常驻准备默认把吊绳锚点平滑降到 0.82 m（`--hang-height`），减少策略长时间悬空摆腿后高处落地的相位敏感性；实际测试发现原 1 m 悬挂并非每次松绳都能稳定落地。按 `9` 后外部拉力实际归零，正式表演始终没有吊绳辅助。窗口按 `]` 时会恢复全身跟踪相机，避免 MuJoCo 原生同名快捷键把视角切到机载相机。

在启动 GENMO 的终端输入以下命令并回车：

```text
play /absolute/path/song.wav
stop
status
quit
```

直接输入完整文件路径也可以；带空格的路径可以加引号。`play` 开始准备当前音乐，`stop` 或 **MuJoCo 窗口的 `P`** 平滑停止表演，`status` 查看站姿/播放及吊绳状态，`quit` 收尾后退出全部自有进程。终端也接受 `]` 起控和 `9` 切换吊绳。表演期间 Ctrl+C 只停止当前表演；站姿待机时 Ctrl+C 退出常驻程序。正在播放时不会把新输入的歌曲自动插入队列，应先停止当前表演。

每首音乐仍采用前两窗预缓冲和单独的媒体零点；松绳至少一秒后，先保持站姿一秒，再用一秒过渡接入首帧舞蹈，音乐第 0 个采样与本首歌第 100 帧参考对齐。结束后用一秒回到站姿，继续闭环控制；主动停止时提前 250 ms 淡出音乐。换歌不 reset MuJoCo、不重新吊起、不重新校准 heading，也不重新加载 TensorRT 模型。每首歌的首朝向接续当前站姿，收尾只保留末端 yaw 并恢复直立。故障停止音频并冻结仿真，不能用换歌清除故障。

需要保留原来的单首自动起控/自动松绳验收行为时，显式提供 `--audio` 且不加 `--resident`：

```bash
.venv/bin/python -B scripts/demo/demo_music_sonic.py \
  --audio "/absolute/path/song.wav" --launch-local --exit-on-finish
```

常用选项：

| 选项 | 用途 |
|---|---|
| `--start-sec 10 --duration-sec 30` | 选段后再提取特征，保持原始播放速度 |
| `--seed 42` | 固定滚动窗口的派生随机种子 |
| `--resident --audio /path/song.wav` | 常驻启动时预先选择一首歌，仍等待人工起控和松绳 |
| `--output-dir /absolute/new/directory` | 使用新的输出目录；拒绝覆盖已有目录 |
| `--audio-device pulse` | 选择 PortAudio 输出设备，也可传数字设备编号 |
| `--exit-on-finish` | 单首模式自然收尾后退出；常驻模式继续站立 |
| `--headless --audio-output off` | 显式无窗口、无声测试，不属于真实声卡验收 |
| `--audit-observations` | 保存实际 C++ 编码器输入，供独立重组比较 |
| `--smpl-npz .../generated_smpl.npz` | 固定片段回放，隔离生成速度对控制的影响 |
| `--sonic-root ... --sim-python ...` | 覆盖 SONIC 路径和仿真 Python；默认使用 SONIC 的 `.venv_sim/bin/python` |

不使用 `--launch-local` 时，需先按 SONIC 仓库 `docs/sonic_music_input.md` 的命令启动两个服务。本入口只连接 `127.0.0.1`；活动会话拒绝被另一个会话覆盖，退出时只终止自己启动的子进程。音乐模式不使用摄像头模式的 Enter/T/R 或编码器切换按键。常驻控制台负责转发上述窗口起控和停止事件。

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

常驻目录的 `resident_timeline.jsonl` 记录待机、人工起控、吊绳和持续站姿包计数；`tracks/` 下每首歌保存独立的上述动作、音频和报告，公共 SONIC/MuJoCo 进程日志放在常驻目录。每首歌的 `sim_state.jsonl` 按预约 epoch 提取，避免混入前一首动作。`resident_report.json` 与 `tracks.json` 记录最终退出状态和各首歌目录。

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


## 常驻模式回归（2026-09-08）

最终记录为 `outputs/sonic_music/20260908_resident_final_v3`。同一组进程运行 121.763 秒，完成准备取消、人工起控松绳、30 秒自然结束、窗口 P 中断、中断后再次播放 30 秒，以及结束后继续站姿 15 秒；七项验收通过。起控与 heading 校准均只有一次，始终 mode 2，无跌倒或仿真 reset，累计收到 1178 个站姿包。另通过初始化稳定后额外悬挂等待 2/10/20 秒的三次真实窗口松绳检查。

两次完整播放使用真实声卡，音频/参考误差 P95 为 7.85/13.10 ms，最大 15.23/19.53 ms；生成耗时 P95 为 85.38/85.40 ms，控制计算 P99 为 1.82/1.77 ms。默认 15° 站姿和 0.82 m 吊绳锚点下，整场最低基座 0.682 m、最大倾角 14.09°；地面待机样本最大倾角 7.04°。这些结果只覆盖本次音乐、seed 和仿真场景，不能解释成无限时长或所有悬挂高度均已验证。

该目录有 `standing_before_music.png`、`standing_after_music.png`、`acceptance.json` 和分曲报告；第一首完整音乐的 33 秒状态回放录像位于 `tracks/0002_25042946/dance_with_music.mp4`。30 项单元测试、编译、落地检查及旧单首兼容审计见 `outputs/sonic_music/validation_20260908_resident`。
